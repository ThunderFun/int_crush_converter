"""GPTQ: Accurate Post-Training Quantization (Frantar et al., ICLR 2023).

Block-wise quantization using Hessian information to optimally redistribute
quantization error to remaining unquantized columns.
"""

import os
import torch

from .log import logger
from .scales import (
    calculate_scales, quantize_weights,
    calculate_scales_int8, quantize_weights_int8,
    calculate_scales_asymmetric, quantize_weights_asymmetric,
    calculate_scales_int8_asymmetric, quantize_weights_int8_asymmetric,
)
from .config import DIAG_MEAN_FLOOR, SCALE_FLOOR, FP16_SCALE_FLOOR, MAX_FP16_SCALE, INT4_SCALE_DIVISOR, INT8_SCALE_DIVISOR
from .rounding import _invert_hessian, _gptq_block, _round_half_away_from_zero
from .types import QuantizationResult
from .gptq_triton import gptq_loop_triton, _HAS_GPTQ_TRITON

_DISABLE_TRITON = os.environ.get("GPTQ_DISABLE_TRITON", "0") == "1"


def _gptq_block_triton(
    W_work: torch.Tensor,
    quantized_W: torch.Tensor,
    row_scales: torch.Tensor,
    H_inv_block: torch.Tensor,
    col_start: int,
    n_cols: int,
    block_size: int,
    clamp_min: int,
    clamp_max: int,
    row_zp: torch.Tensor | None = None,
) -> bool:
    """Try to run GPTQ block via Triton kernel. Returns True on success.

    Falls back silently to the caller's PyTorch path if Triton is unavailable,
    CUDA is not present, or the kernel raises an exception.

    Args:
        W_work: [M, N] working weights, modified in-place
        quantized_W: [M, N] output int8 tensor, modified in-place
        row_scales: [M, 1] per-row scales
        H_inv_block: [n_cols, n_cols] inverse Hessian for this column range
                     (already sliced, not the full Hessian)
        col_start: first column index for this block
        n_cols: number of columns in this block
        block_size: GPTQ sub-block size for column processing
        clamp_min: minimum quantized value (-8 for INT4, -128 for INT8)
        clamp_max: maximum quantized value (7 for INT4, 127 for INT8)
        row_zp: [M, 1] per-row zero-points (optional, for asymmetric quantization)

    Returns:
        True if the Triton kernel succeeded, False otherwise.
    """
    if not _HAS_GPTQ_TRITON:
        return False
    if not torch.cuda.is_available():
        return False

    try:
        M = W_work.shape[0]
        scales_1d = row_scales.squeeze(1).contiguous()

        W_slice = W_work[:, col_start:col_start + n_cols].contiguous()
        H_gpu = H_inv_block.contiguous().to("cuda")
        s_gpu = scales_1d.to("cuda")

        zp_gpu = None
        if row_zp is not None:
            zp_gpu = row_zp.squeeze(1).contiguous().to("cuda")

        W_gpu = W_slice.to("cuda")
        Q_gpu = torch.zeros(M, n_cols, dtype=torch.int8, device="cuda")

        result = gptq_loop_triton(W_gpu, H_gpu, s_gpu, Q_gpu, n_cols, M, block_size, clamp_min, clamp_max, row_zp=zp_gpu)
        if result is not None:
            W_work[:, col_start:col_start + n_cols] = W_gpu.to(W_work.device)
            quantized_W[:, col_start:col_start + n_cols] = Q_gpu.to(quantized_W.device)
            return True
        return False
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        logger.debug("GPTQ Triton fallback: %s", e)
        return False


def _prepare_hessian(
    H_block: torch.Tensor,
    damping: float,
    device: torch.device,
) -> torch.Tensor:
    """Compute Cholesky of H⁻¹ per GPTQ paper Algorithm 1.

    Intuition for *why* H⁻¹ and not H: the GPTQ column-update rule
    needs the inverse Hessian to compute δ = −H⁻¹[:,j] · err / H⁻¹[j,j].
    Rather than forming the full dense H⁻¹, Algorithm 1 (Frantar et al.,
    arXiv:2210.17323) stores only Cholesky(H⁻¹) and recovers individual
    columns of H⁻¹ via forward-solve — O(n²) per column instead of O(n³).
    The upper-triangular factor L.T is cached here; the per-column
    extraction happens in the main loop below.
    Falls back to full inverse if Cholesky fails (non-positive-definite H⁻¹).
    """
    diag_mean = H_block.diagonal().mean().clamp(min=DIAG_MEAN_FLOOR)
    H_damped = H_block + damping * diag_mean * torch.eye(H_block.shape[0], dtype=torch.float32)
    H_inv = _invert_hessian(H_damped)
    try:
        L = torch.linalg.cholesky(H_inv)
        return L.T.to(device)  # upper triangular
    except torch.linalg.LinAlgError:
        return H_inv.to(device)


def _prepare_hinv(
    H_block: torch.Tensor,
    damping: float,
    device: torch.device,
) -> torch.Tensor:
    """Return the true inverse Hessian (no Cholesky decomposition).

    This is the production implementation used by default via
    ``_gptq_prepare_hessian``.  Unlike the legacy :func:`_prepare_hessian`
    (which stores only the Cholesky factor), this returns the full H⁻¹
    so that column extraction is a simple slice rather than a forward-solve.
    """
    diag_mean = H_block.diagonal().mean().clamp(min=DIAG_MEAN_FLOOR)
    H_damped = H_block + damping * diag_mean * torch.eye(H_block.shape[0], dtype=torch.float32)
    return _invert_hessian(H_damped).to(device)


# Module-level reference that can be monkey-patched by tests.
# Default is _prepare_hinv (full inverse H⁻¹ — the production path).
# Tests can swap this to _prepare_hessian to exercise the legacy
# Cholesky-based path.
_gptq_prepare_hessian = _prepare_hinv


_HESSIAN_METHODS = {
    "hinv": _prepare_hinv,
    "cholesky": _prepare_hessian,
}


def gptq_quantize_layer(
    W: torch.Tensor,
    hessian: torch.Tensor,
    block_size: int = 128,
    damping: float = 0.01,
    int_bits: int = 4,
    asymmetric: bool = False,
    hessian_method: str = "hinv",
    piso_scales: torch.Tensor | None = None,
) -> QuantizationResult:
    """Quantize a weight matrix using GPTQ.

    INT4 always uses asymmetric quantization (scale + zero-point) so all 16
    integer levels are utilized. INT8 defaults to symmetric (scale only,
    range [-127, 127]); pass ``asymmetric=True`` to use all 256 levels
    asymmetric instead.

    The scale is clamped to ``[FP16_SCALE_FLOOR, MAX_FP16_SCALE]`` BEFORE
    the GPTQ column loop runs. This is essential: if the clamp happened
    after quantization, the stored (Q, scale) pair would be internally
    inconsistent — q values were computed with one scale but the stored
    scale differs, so dequantization cannot recover the original weight
    range.

    Supports both full Hessians [in, in] and block-diagonal Hessians
    [num_blocks, bs, bs]. For block-diagonal, each block is processed
    independently.

    Args:
        W: [out_features, in_features] weight tensor
        hessian: Hessian matrix — 2D [in, in] or 3D [num_blocks, bs, bs]
        block_size: GPTQ block size for column processing
        damping: damping ratio as fraction of mean diagonal (default: 0.01)
        int_bits: quantization bit-width (4 or 8)
        asymmetric: use asymmetric quantization. INT4 ignores this flag
                   (always asymmetric). INT8 defaults to symmetric; pass
                   True to use asymmetric.
        hessian_method: "hinv" (full inverse H⁻¹) or "cholesky" (Cholesky
                       factor of H⁻¹). Default "hinv".
        piso_scales: Optional [out_features, 1] PiSO-optimal per-row scales.
                     When provided, these are used instead of computing absmax
                     scales.  The scales are clamped to the valid FP16 range
                     before use.  See arXiv:2606.10890.

    Returns:
        QuantizationResult with quantized_W, scales, zero_points, method_used, and fallbacks.
    """
    if int_bits not in (4, 8):
        raise ValueError(f"int_bits must be 4 or 8, got {int_bits}")
    if W.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor, got {W.dim()}D")
    if hessian_method not in _HESSIAN_METHODS:
        raise ValueError(f"hessian_method must be one of {list(_HESSIAN_METHODS)}, got '{hessian_method}'")

    prepare_fn = _HESSIAN_METHODS[hessian_method]

    W = W.float()
    out_features, in_features = W.shape

    clamp_min = -128 if int_bits == 8 else -8
    clamp_max = 127 if int_bits == 8 else 7

    # ── PiSO scales: use data-aware optimal scales when provided ──
    if piso_scales is not None and int_bits == 8:
        piso = piso_scales.float().to(W.device)
        if piso.shape == (out_features, 1):
            row_scales = piso.clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE)
            row_zp_int8 = None  # PiSO + symmetric for now
            if asymmetric:
                # Compute zero-points from PiSO scales
                w_min = W.amin(dim=1, keepdim=True)
                row_zp_raw = -128 - _round_half_away_from_zero(w_min / row_scales)
                row_zp = row_zp_raw.clamp(-128, 127)
                zero_rows = (w_min == 0) & (W.amax(dim=1, keepdim=True) == 0)
                row_zp = row_zp.masked_fill(zero_rows, 0)
                row_zp_int8 = row_zp.to(torch.int8)
            logger.debug("GPTQ: using PiSO scales for %dx%d", out_features, in_features)
        else:
            logger.warning("PiSO scales shape %s != (%d, 1); falling back to absmax",
                           tuple(piso.shape), out_features)
            piso_scales = None  # fall through to absmax

    if piso_scales is None:
        if int_bits == 4:
            # INT4 always uses asymmetric (uses all 16 levels)
            w_min = W.amin(dim=1, keepdim=True)
            w_max = W.amax(dim=1, keepdim=True)
            # Clamp scale BEFORE quantization so Q values are consistent with the
            # stored scale. Without this, a row whose (max-min) > 975K would have
            # its scale silently clamped to 65000 AFTER quantization, leaving Q
            # computed against the original (unclamped) scale — which makes the
            # saved (Q, scale) pair internally inconsistent.
            row_scales = ((w_max - w_min).float() / 15.0).clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE)
            row_zp_raw = -8 - _round_half_away_from_zero(w_min / row_scales)
            # When zp would be clamped, the range-based scale is too small for the
            # weight's absolute position.  Increase the scale so the quantization
            # range covers the actual weight values with the clamped zp.
            zp_clamp_lo = row_zp_raw < -8
            zp_clamp_hi = row_zp_raw > 7
            if zp_clamp_lo.any():
                fix_scale = (w_max.float() / 15.0).clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE)
                row_scales = torch.maximum(row_scales, torch.where(zp_clamp_lo, fix_scale, row_scales))
            if zp_clamp_hi.any():
                fix_scale = (-w_min.float() / 15.0).clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE)
                row_scales = torch.maximum(row_scales, torch.where(zp_clamp_hi, fix_scale, row_scales))
            # Recompute zp with the (possibly adjusted) scale
            row_zp = (-8 - _round_half_away_from_zero(w_min / row_scales)).clamp(-8, 7)
            # Zero rows: set zp to 0 so zero maps to zero
            zero_rows = (w_min == 0) & (w_max == 0)
            row_zp = row_zp.masked_fill(zero_rows, 0)
            row_zp_int8 = row_zp.to(torch.int8)
        else:
            # INT8
            if asymmetric:
                # Asymmetric: scale = (max - min) / 255, zp = -128 - round(min / scale).
                # Uses all 256 levels. Clamp scale BEFORE quantization for consistency.
                w_min = W.amin(dim=1, keepdim=True)
                w_max = W.amax(dim=1, keepdim=True)
                row_scales = ((w_max - w_min).float() / 255.0).clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE)
                row_zp_raw = -128 - _round_half_away_from_zero(w_min / row_scales)
                # When zp would be clamped, the range-based scale is too small for
                # the weight's absolute position.  E.g., weights near 5.0 with tiny
                # variance produce scale ≈ 1e-5 and zp ≈ -500000, clamped to -128.
                # With the tiny scale, all Q values saturate at 127 and dequant ≈ 0,
                # giving MSE ≈ 25 and causing GPTQ error propagation to blow up to
                # inf.  The fix: increase the scale so the quantization range covers
                # the actual weight values with the clamped zp.
                zp_clamp_lo = row_zp_raw < -128
                zp_clamp_hi = row_zp_raw > 127
                if zp_clamp_lo.any():
                    fix_scale = (w_max.float() / 255.0).clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE)
                    row_scales = torch.maximum(row_scales, torch.where(zp_clamp_lo, fix_scale, row_scales))
                if zp_clamp_hi.any():
                    fix_scale = (-w_min.float() / 255.0).clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE)
                    row_scales = torch.maximum(row_scales, torch.where(zp_clamp_hi, fix_scale, row_scales))
                # Recompute zp with the (possibly adjusted) scale
                row_zp = (-128 - _round_half_away_from_zero(w_min / row_scales)).clamp(-128, 127)
                zero_rows = (w_min == 0) & (w_max == 0)
                row_zp = row_zp.masked_fill(zero_rows, 0)
                row_zp_int8 = row_zp.to(torch.int8)
            else:
                # Symmetric: scale = max(|W|) / 127, range [-127, 127].
                # Uses 255 of 256 levels (the -128 level is unused). Clamp scale
                # BEFORE quantization for consistency.
                row_scales = (W.abs().amax(dim=1, keepdim=True) / INT8_SCALE_DIVISOR).clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE)
                row_zp_int8 = None

    quantized_W = torch.zeros_like(W, dtype=torch.int8)
    W_work = W.clone()
    used_triton = False

    if hessian.dim() == 3:
        num_blocks, bs, _ = hessian.shape
        for bi in range(num_blocks):
            col_start = bi * bs
            col_end = min(col_start + bs, in_features)
            actual_bs = col_end - col_start

            H_block = hessian[bi, :actual_bs, :actual_bs].float()
            H_chol = prepare_fn(H_block, damping, W.device)

            if _gptq_block_triton(
                W_work, quantized_W, row_scales, H_chol,
                col_start, actual_bs, block_size, clamp_min, clamp_max,
                row_zp=row_zp_int8,
            ):
                used_triton = True
            else:
                _gptq_block(
                    W_work, quantized_W, row_scales, H_chol,
                    col_start, col_end, block_size, clamp_min, clamp_max,
                    row_zp=row_zp_int8,
                )

    elif hessian.dim() == 2:
        if hessian.shape != (in_features, in_features):
            raise ValueError(
                f"Hessian shape {hessian.shape} != ({in_features}, {in_features})"
            )

        H = hessian.float()
        H_chol = prepare_fn(H, damping, W.device)

        if _gptq_block_triton(
            W_work, quantized_W, row_scales, H_chol,
            0, in_features, block_size, clamp_min, clamp_max,
            row_zp=row_zp_int8,
        ):
            used_triton = True
        else:
            _gptq_block(
                W_work, quantized_W, row_scales, H_chol,
                0, in_features, block_size, clamp_min, clamp_max,
                row_zp=row_zp_int8,
            )
    else:
        raise ValueError(f"Expected 2D or 3D Hessian, got {hessian.dim()}D")

    accel = "Triton" if used_triton else "PyTorch"
    logger.debug("GPTQ: %s path, %dx%d, block_size=%d", accel, out_features, in_features, block_size)

    # Compute MSE and max error for the result against the *original* weight.
    # (W_work is mutated in-place by the GPTQ column loop, so using it would
    # report artificially near-zero error.)
    dequant = quantized_W.float() * row_scales.float()
    if row_zp_int8 is not None:
        dequant = (quantized_W.float() - row_zp_int8.float()) * row_scales.float()
    mse = (W - dequant).pow(2).mean().item()
    max_err = (W - dequant).abs().max().item()

    method_used = "gptq_triton" if used_triton else "gptq_pytorch"

    return QuantizationResult(
        quantized_W=quantized_W,
        scales=row_scales.to(torch.float16),
        zero_points=row_zp_int8,
        mse=mse,
        max_err=max_err,
        method_used=method_used,
        fallbacks=[],
    )


def gptq_quantize_layer_rtn(
    W: torch.Tensor,
    int_bits: int = 4,
    asymmetric: bool = False,
    clipping_ratios: list[float] | None = None,
    piso_scales: torch.Tensor | None = None,
) -> QuantizationResult:
    """Fallback: RTN quantization (no Hessian correction).

    Used when calibration data is not available for a layer.

    Args:
        W: [out_features, in_features] weight tensor
        int_bits: quantization bit-width (4 or 8)
        asymmetric: use asymmetric quantization (scale + zero-point)
        clipping_ratios: optional list of clipping ratios to search

    Returns:
        QuantizationResult with quantized_W, scales, zero_points (if asymmetric), method_used.
    """
    if int_bits not in (4, 8):
        raise ValueError(f"int_bits must be 4 or 8, got {int_bits}")

    fallbacks: list[str] = []

    # ── PiSO scales: use data-aware optimal scales when provided ──
    if piso_scales is not None and int_bits == 8 and not asymmetric:
        piso = piso_scales.float().to(W.device)
        if piso.shape[0] == W.shape[0] and piso.shape[1] == 1:
            scales = piso.clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE).to(torch.float16)
            quantized = quantize_weights_int8(W, scales)
            dequant = quantized.float() * scales.float()
            mse = (W - dequant).pow(2).mean().item()
            max_err = (W - dequant).abs().max().item()
            return QuantizationResult(
                quantized_W=quantized,
                scales=scales,
                zero_points=None,
                mse=mse,
                max_err=max_err,
                method_used="rtn_piso",
                fallbacks=fallbacks,
            )

    if asymmetric:
        if int_bits == 8:
            scales, zps = calculate_scales_int8_asymmetric(W, clipping_ratios=clipping_ratios)
            quantized = quantize_weights_int8_asymmetric(W, scales, zps)
        else:
            in_features = W.shape[1]
            scales, zps = calculate_scales_asymmetric(W, in_features, clipping_ratios=clipping_ratios)
            quantized = quantize_weights_asymmetric(W, scales, zps, in_features)

        dequant = (quantized.float() - zps.float()) * scales.float()
        mse = (W - dequant).pow(2).mean().item()
        max_err = (W - dequant).abs().max().item()

        return QuantizationResult(
            quantized_W=quantized,
            scales=scales,
            zero_points=zps,
            mse=mse,
            max_err=max_err,
            method_used="rtn",
            fallbacks=fallbacks,
        )

    if int_bits == 8:
        scales = calculate_scales_int8(W, clipping_ratios=clipping_ratios)
        quantized = quantize_weights_int8(W, scales)
    else:
        in_features = W.shape[1]
        scales = calculate_scales(W, in_features, clipping_ratios=clipping_ratios)
        quantized = quantize_weights(W, scales, in_features)

    dequant = quantized.float() * scales.float()
    mse = (W - dequant).pow(2).mean().item()
    max_err = (W - dequant).abs().max().item()

    return QuantizationResult(
        quantized_W=quantized,
        scales=scales,
        zero_points=None,
        mse=mse,
        max_err=max_err,
        method_used="rtn",
        fallbacks=fallbacks,
    )
