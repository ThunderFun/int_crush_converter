"""LDLQ: Weight-only quantization using W^T W as the Hessian.

Implements the LDLQ algorithm from QuIP (Chee et al., 2023) for INT4/INT8
quantization without calibration data. The Hessian is computed as
H = W^T @ W / M, making this a weight-only method.

The column loop is identical to GPTQ (Frantar et al., 2022) but uses
per-element scales with sign-flip correction and round-half-away-from-zero.
"""

import math

import torch

from .dlr import is_dlr, woodbury_inverse
from .log import logger
from .config import (
    ABS_SCALE_FLOOR, CONVERGENCE_EPS, CONVERGENCE_IMPROVEMENT_THRESHOLD,
    DENOMINATOR_FLOOR, DIAG_MEAN_FLOOR, HINV_DIAG_FLOOR, SCALE_CEIL_MULTIPLIER,
    SCALE_FLOOR, SCALE_MIN, SCALE_FLOOR_MULTIPLIER, SANITIZE_CEIL, SANITIZE_FLOOR,
    INT4_SCALE_DIVISOR, INT8_SCALE_DIVISOR, SCALE_MAX, SCALE_DTYPE,
)
from .rounding import _invert_hessian, _ldlq_round_column
from .types import QuantizationResult
from .greedy import greedy_local_search_pytorch, greedy_lowrank_triton, greedy_local_search_triton, _HAS_TRITON
from .ldlq_triton import ldlq_loop_triton


_ldlq_compiled_fn = None
_HAS_TORCH_COMPILE = hasattr(torch, "compile")  # PyTorch 2.0+


def _get_ldlq_compiled_fn():
    """Lazily compile LDLQ column body with torch.compile (fuses ~10 ops → 1-2 kernels)."""
    global _ldlq_compiled_fn
    if _ldlq_compiled_fn is not None:
        return _ldlq_compiled_fn
    if not _HAS_TORCH_COMPILE:
        return None
    try:
        @torch.compile(mode="default")
        def _compiled_ldlq_column(W_col, W_work_col, scale_col, hinv_diag_val,
                                  clamp_min, clamp_max):
            scale_safe = scale_col.clamp(min=1e-8)
            y = W_col / scale_safe
            q = torch.where(y >= 0, torch.floor(y + 0.5), torch.ceil(y - 0.5))
            q = q.clamp(clamp_min, clamp_max)
            W_sign = torch.where(W_work_col.abs() > 1e-8, W_work_col.sign(), q.sign())
            should_flip = (W_sign * q.sign()) < 0
            can_flip = should_flip & (q > clamp_min)
            q = torch.where(should_flip & (q == clamp_min),
                            torch.full_like(q, clamp_max), q)
            q = torch.where(can_flip, -q, q)
            q = q.clamp(clamp_min, clamp_max)
            err = W_col - q * scale_safe
            err_norm = err / hinv_diag_val
            return q.to(torch.int8), err_norm
        _ldlq_compiled_fn = _compiled_ldlq_column
    except (RuntimeError, ImportError):
        _ldlq_compiled_fn = None
    return _ldlq_compiled_fn


def _invert_block_diagonal_hessian(
    hessian: torch.Tensor, N: int, damping: float,
) -> torch.Tensor:
    """Invert a block-diagonal Hessian into a full N×N inverse matrix.

    Each block is damped and inverted independently.  The result is a dense
    ``[N, N]`` matrix whose off-block-diagonal entries are zero, ready for
    the LDLQ column loop.

    Args:
        hessian: ``[num_blocks, bs, bs]`` block-diagonal Hessian.
        N: total column dimension (may be larger than ``num_blocks * bs``
            if the last block is partial).
        damping: damping ratio as fraction of per-block mean diagonal.

    Returns:
        ``[N, N]`` float32 inverse Hessian.
    """
    num_blocks, bs, _ = hessian.shape
    H_inv = torch.zeros(N, N, dtype=torch.float32)
    for bi in range(num_blocks):
        col_start = bi * bs
        col_end = min(col_start + bs, N)
        actual_bs = col_end - col_start
        H_block = hessian[bi, :actual_bs, :actual_bs].float()
        diag_mean = H_block.diagonal().mean().clamp(min=DIAG_MEAN_FLOOR)
        H_block_damped = H_block + damping * diag_mean * torch.eye(actual_bs, dtype=torch.float32)
        H_inv[col_start:col_end, col_start:col_end] = _invert_hessian(H_block_damped)
    return H_inv


def ldlq_quantize_layer(
    W: torch.Tensor,
    hessian: torch.Tensor | None = None,
    block_size: int = 128,
    damping: float = 0.01,
    int_bits: int = 4,
    iterations: int = 1,
    greedy_passes: int = 0,
    rank_threshold: float = 0.01,
) -> QuantizationResult:
    """Quantize a weight matrix using LDLQ.

    Supports two Hessian sources:
    - **Calibration Hessian** (recommended): ``hessian = E_x[xx^T]`` from
      calibration data, same as GPTQ.  This is what the QuIP papers use and
      gives genuinely better results than RTN.
    - **Weight-only** (fallback): when ``hessian=None``, computes
      ``H = W^T @ W / M``.  This is *not* what the papers describe and
      can perform worse than RTN because the error propagation targets
      the wrong directions.  Pass a calibration Hessian whenever possible.

    Args:
        W: [out_features, in_features] weight tensor
        hessian: Proxy Hessian — 2D [in, in], 3D [blocks, bs, bs], or DLR
            dict ``{"format": "dlr", "D": (n,), "U": (n, r)}``.
            If None, falls back to H = W^T @ W / M (weight-only, not recommended).
        block_size: block size for column processing
        damping: damping ratio as fraction of mean diagonal (default: 0.01)
        int_bits: quantization bit-width (4 or 8)
        iterations: number of LDLQ iterations with scale refinement (1 = single pass)
        greedy_passes: greedy coordinate descent passes after LDLQ (0 = disabled)
        rank_threshold: eigenvalue threshold for low-rank greedy (default: 0.01).
            Lower values keep more eigenvalues (better quality, slower).
            Only used when greedy_passes > 0 and Triton is available.

    Returns:
        QuantizationResult with quantized_W, scales, method_used, and fallbacks populated.
    """
    if int_bits not in (4, 8):
        raise ValueError(f"int_bits must be 4 or 8, got {int_bits}")
    if W.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor, got {W.dim()}D")

    W = W.float()
    M, N = W.shape

    scale_divisor = INT8_SCALE_DIVISOR if int_bits == 8 else INT4_SCALE_DIVISOR
    clamp_min = -128 if int_bits == 8 else -8
    clamp_max = 127 if int_bits == 8 else 7

    fallbacks: list[str] = []

    # ── Compute or accept Hessian ──
    if hessian is not None:
        if is_dlr(hessian):
            # DLR: H⁻¹ via Woodbury identity.
            D = hessian["D"].float()
            U = hessian["U"].float()
            if D.shape[0] != N:
                raise ValueError(
                    f"DLR Hessian dim {D.shape[0]} != in_features {N}"
                )
            H_inv = woodbury_inverse(D, U, damping=damping)
        else:
            hessian = hessian.float()
            H = hessian  # Keep H available for greedy search
            if hessian.dim() == 3:
                # Block-diagonal Hessian: invert per-block into a full N×N matrix.
                H_inv = _invert_block_diagonal_hessian(hessian, N, damping)
            elif hessian.dim() == 2:
                if hessian.shape != (N, N):
                    raise ValueError(
                        f"Hessian shape {hessian.shape} != ({N}, {N})"
                    )
                diag_mean = hessian.diagonal().mean().clamp(min=DIAG_MEAN_FLOOR)
                H_damped = hessian + damping * diag_mean * torch.eye(N, dtype=torch.float32)
                H_inv = _invert_hessian(H_damped)
            else:
                raise ValueError(
                    f"Expected 2D tensor, 3D tensor, or DLR dict Hessian, "
                    f"got {hessian.dim()}D"
                )
    else:
        # Fallback: weight-only Hessian (not what the papers describe).
        import warnings
        warnings.warn(
            "LDLQ: no calibration Hessian provided; using H = W^T W / M. "
            "This can perform worse than RTN. Pass a calibration Hessian "
            "from GPTQ calibration data for best results.",
            stacklevel=2,
        )
        H = W.T @ W / M

        # Guard: if H has inf/nan (from extreme weight values), fall back to RTN
        if not torch.isfinite(H).all():
            logger.warning("LDLQ: Hessian has inf/nan (weight abs_max=%.2e), falling back to RTN", W.abs().max().item())
            row_scales = (W.abs().amax(dim=1, keepdim=True) / scale_divisor).clamp(min=SCALE_MIN, max=SCALE_MAX)
            q = (W / row_scales).round().clamp(clamp_min, clamp_max).to(torch.int8)
            dequant = q.float() * row_scales.float()
            mse = (W - dequant).double().pow(2).mean().item()
            max_err = (W - dequant).abs().max().item()
            fallbacks.append("hessian_nan_rtn")
            return QuantizationResult(
                quantized_W=q,
                scales=row_scales.to(SCALE_DTYPE),
                zero_points=None,
                mse=mse,
                max_err=max_err,
                method_used="rtn",
                fallbacks=fallbacks,
            )

        diag_mean = H.diagonal().mean().clamp(min=DIAG_MEAN_FLOOR)
        H = H + damping * diag_mean * torch.eye(N, dtype=torch.float32)
        H_inv = _invert_hessian(H)

    # Initial per-row scales
    row_scales = (W.abs().amax(dim=1, keepdim=True) / scale_divisor).clamp(min=SCALE_MIN)

    # Expand per-row scales to per-element for the LDLQ column loop.
    flat_scales = row_scales.expand(M, N).reshape(-1).clone()

    if iterations > 1:
        quantized_W, _, method_used = _run_iterative_ldlq(
            W, H_inv, flat_scales, iterations, block_size, clamp_min, clamp_max,
        )
    else:
        quantized_W, method_used = _single_ldlq_pass(W, H_inv, flat_scales, block_size, clamp_min, clamp_max)

    # Greedy local search: coordinate descent on the quantization grid.
    if greedy_passes > 0:
        # Compute per-row least-squares scales for the greedy step
        Q_float = quantized_W.float()
        numerator = (Q_float * W).sum(dim=1, keepdim=True)
        denominator = (Q_float * Q_float).sum(dim=1, keepdim=True).clamp(min=DENOMINATOR_FLOOR)
        greedy_scales = (numerator / denominator).clamp(min=SCALE_MIN)
        scale_2d = greedy_scales.expand(M, N)

        # Move to GPU for Triton if available (matching _single_ldlq_pass pattern)
        greedy_device = torch.device("cuda") if torch.cuda.is_available() else W.device
        oom_fallback = False
        try:
            W_g = W.to(greedy_device)
            Q_g = quantized_W.to(greedy_device)
            s_g = scale_2d.to(greedy_device)
            H_g = H.to(greedy_device)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            greedy_device = W.device
            W_g, Q_g, s_g, H_g = W, quantized_W, scale_2d, H
            oom_fallback = True
            fallbacks.append("greedy_oom_cpu")

        quantized_W = _greedy_local_search(
            W_g, Q_g, s_g, H_g, greedy_passes, clamp_min, clamp_max,
            rank_threshold=rank_threshold,
        )

        # Move back to original device
        if quantized_W.device != W.device:
            quantized_W = quantized_W.to(W.device)

    # Compute the per-row scale that best fits the quantized integers to the
    # original weights.  For each row i the least-squares optimum is:
    #   s_i = (Q · W).sum(dim=1) / (Q · Q).sum(dim=1)
    # This keeps Q unchanged while finding the best dequantization scale.
    Q_float = quantized_W.float()
    numerator = (Q_float * W).sum(dim=1, keepdim=True)
    denominator = (Q_float * Q_float).sum(dim=1, keepdim=True).clamp(min=DENOMINATOR_FLOOR)
    final_scales = (numerator / denominator).clamp(min=SCALE_MIN)

    dequant = quantized_W.float() * final_scales.float()
    mse = (W - dequant).double().pow(2).mean().item()
    max_err = (W - dequant).abs().max().item()

    return QuantizationResult(
        quantized_W=quantized_W,
        scales=final_scales.to(SCALE_DTYPE),
        zero_points=None,
        mse=mse,
        max_err=max_err,
        method_used=method_used,
        fallbacks=fallbacks,
    )


def _single_ldlq_pass(
    W: torch.Tensor,
    H_inv: torch.Tensor,
    flat_scales: torch.Tensor,
    block_size: int,
    clamp_min: int,
    clamp_max: int,
) -> torch.Tensor:
    """Single pass of LDLQ quantization with OOM fallback.

    Allocates on CUDA when available, falls back to CPU on OOM.
    Backend priority: Triton (fused kernel) → torch.compile → eager PyTorch.

    Args:
        W: [M, N] weight tensor (float32)
        H_inv: [N, N] inverse Hessian
        flat_scales: [M*N] per-element scales
        block_size: column block size
        clamp_min, clamp_max: quantization range

    Returns:
        Q_prime: [M, N] quantized values as int8
    """
    M, N = W.shape
    original_device = W.device

    # Prefer CUDA if available, fall back to CPU on OOM
    target_device = torch.device("cuda") if torch.cuda.is_available() else original_device
    try:
        Q_prime = torch.zeros(M, N, dtype=torch.int8, device=target_device)
        W_work = W.to(target_device).clone()
        H_inv = H_inv.to(target_device)
        flat_scales = flat_scales.to(target_device)
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        target_device = torch.device("cpu")
        W_work = W.cpu().clone()
        H_inv = H_inv.cpu()
        flat_scales = flat_scales.cpu()
        Q_prime = torch.zeros(M, N, dtype=torch.int8, device="cpu")

    # Reshape scales to [M, N]
    scale_2d = flat_scales.reshape(M, N)

    # Sanitize non-finite scales
    if not torch.isfinite(scale_2d).all():
        scale_2d = torch.nan_to_num(scale_2d, nan=SANITIZE_FLOOR, posinf=SANITIZE_CEIL, neginf=SANITIZE_FLOOR)
        scale_2d = scale_2d.clamp(min=SANITIZE_FLOOR)

    # Choose processing strategy: Triton > torch.compile > eager
    actual_method = "ldlq_cpu"
    if target_device.type == "cpu":
        logger.debug("LDLQ: CPU path, %dx%d, block_size=%d", M, N, block_size)
        Q_prime = _ldlq_loop_cpu(W_work, H_inv, scale_2d, Q_prime, N, M, block_size, clamp_min, clamp_max)
    else:
        # Try Triton block kernel first (fastest — fuses entire block into one launch)
        if _HAS_TRITON and block_size <= N:
            result = ldlq_loop_triton(W_work, H_inv, scale_2d, Q_prime, N, M, block_size, clamp_min, clamp_max)
            if result is not None:
                logger.debug("LDLQ: GPU path (Triton), %dx%d, block_size=%d", M, N, block_size)
                Q_prime = result
                actual_method = "ldlq_triton"
            else:
                compiled_fn = _get_ldlq_compiled_fn()
                accel = "torch.compile" if compiled_fn is not None else "eager"
                logger.debug("LDLQ: GPU path (%s), %dx%d, block_size=%d", accel, M, N, block_size)
                Q_prime = _ldlq_loop_gpu(W_work, H_inv, scale_2d, Q_prime, N, M, block_size, clamp_min, clamp_max,
                                         compiled_fn=compiled_fn)
                actual_method = "ldlq_compile" if compiled_fn is not None else "ldlq_cpu"
        else:
            compiled_fn = _get_ldlq_compiled_fn()
            accel = "torch.compile" if compiled_fn is not None else "eager"
            logger.debug("LDLQ: GPU path (%s), %dx%d, block_size=%d", accel, M, N, block_size)
            Q_prime = _ldlq_loop_gpu(W_work, H_inv, scale_2d, Q_prime, N, M, block_size, clamp_min, clamp_max,
                                      compiled_fn=compiled_fn)
            actual_method = "ldlq_compile" if compiled_fn is not None else "ldlq_cpu"

    # Move result back to original device
    if Q_prime.device != original_device:
        Q_prime = Q_prime.to(original_device)

    # Clamp to valid range
    Q_prime = Q_prime.clamp(clamp_min, clamp_max)
    return Q_prime, actual_method


def _ldlq_loop_cpu(
    W_work: torch.Tensor,
    H_inv: torch.Tensor,
    scale_2d: torch.Tensor,
    Q_prime: torch.Tensor,
    N: int,
    M: int,
    block_size: int,
    clamp_min: int,
    clamp_max: int,
) -> torch.Tensor:
    """CPU LDLQ loop with immediate intra-block error propagation.

    Processes columns one at a time within each block, then applies a lazy
    batch update to columns outside the block. This is the standard GPTQ
    algorithm (Frantar et al., 2022) with lazy batch updates.
    """
    for i1 in range(0, N, block_size):
        i2 = min(i1 + block_size, N)
        block_count = i2 - i1

        Hinv_block = H_inv[i1:i2, i1:i2]
        hinv_diag = torch.diag(Hinv_block).clamp(min=HINV_DIAG_FLOOR)
        hinv_diag_inv = (1.0 / hinv_diag).to(torch.float32)

        err_accum = torch.zeros(M, block_count, dtype=torch.float32)

        for j_local in range(block_count):
            j = i1 + j_local

            q_j, err_j = _ldlq_round_column(W_work[:, j], scale_2d[:, j], clamp_min, clamp_max)
            Q_prime[:, j] = q_j.to(torch.int8)

            err_j_norm = err_j * hinv_diag_inv[j_local]
            err_accum[:, j_local] = err_j_norm

            # Intra-block error propagation
            if j_local + 1 < block_count:
                hinv_row = Hinv_block[j_local, j_local + 1:block_count]
                W_work[:, j + 1:i2] -= err_j_norm.unsqueeze(1) * hinv_row.unsqueeze(0)

        # Inter-block batch update
        if i2 < N:
            update_slice = H_inv[i1:i2, i2:]
            W_work[:, i2:] -= err_accum @ update_slice

        del Hinv_block, hinv_diag, hinv_diag_inv, err_accum

    return Q_prime


def _ldlq_loop_gpu(
    W_work: torch.Tensor,
    H_inv: torch.Tensor,
    scale_2d: torch.Tensor,
    Q_prime: torch.Tensor,
    N: int,
    M: int,
    block_size: int,
    clamp_min: int,
    clamp_max: int,
    compiled_fn=None,
) -> torch.Tensor:
    """GPU LDLQ loop with optional torch.compile acceleration.

    When compiled_fn is provided, uses torch.compile to fuse the per-column
    quantize+sign-flip+error-compute operations, eliminating ~10 kernel
    launches per column.
    """
    for i1 in range(0, N, block_size):
        i2 = min(i1 + block_size, N)
        block_count = i2 - i1

        Hinv_block = H_inv[i1:i2, i1:i2]
        hinv_diag = torch.diag(Hinv_block).clamp(min=HINV_DIAG_FLOOR)

        err_accum = torch.zeros(M, block_count, device=W_work.device, dtype=torch.float32)

        for j_local in range(block_count):
            j = i1 + j_local

            if compiled_fn is not None:
                q_int8, err_j_norm = compiled_fn(
                    W_work[:, j], W_work[:, j], scale_2d[:, j], hinv_diag[j_local],
                    float(clamp_min), float(clamp_max),
                )
                Q_prime[:, j] = q_int8
                err_accum[:, j_local] = err_j_norm
            else:
                q_j, err_j = _ldlq_round_column(W_work[:, j], scale_2d[:, j], clamp_min, clamp_max)
                Q_prime[:, j] = q_j.to(torch.int8)

                err_j_norm = err_j / hinv_diag[j_local]
                err_accum[:, j_local] = err_j_norm

            # Intra-block error propagation
            if j_local + 1 < block_count:
                hinv_row = Hinv_block[j_local, j_local + 1:block_count]
                W_work[:, j + 1:i2] -= err_j_norm.unsqueeze(1) * hinv_row.unsqueeze(0)

        # Inter-block batch update
        if i2 < N:
            update_slice = H_inv[i1:i2, i2:]
            W_work[:, i2:] -= err_accum @ update_slice

        del Hinv_block, hinv_diag, err_accum

    return Q_prime


def _run_iterative_ldlq(
    W: torch.Tensor,
    H_inv: torch.Tensor,
    flat_scales: torch.Tensor,
    iterations: int,
    block_size: int,
    clamp_min: int,
    clamp_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run iterative LDLQ with scale refinement.

    Each iteration re-estimates per-element scales from the quantized result,
    then re-runs LDLQ. Converges when MSE improvement < 0.01%. Divergence
    detection reverts if MSE increases by > 5%.
    """
    M, N = W.shape
    current_scales = flat_scales.clone()
    Q_result = None
    prev_mse = float("inf")

    # Sanitize non-finite scales
    if not torch.isfinite(current_scales).all():
        current_scales = torch.nan_to_num(current_scales, nan=SANITIZE_FLOOR, posinf=SANITIZE_CEIL, neginf=SANITIZE_FLOOR)
        current_scales = current_scales.clamp(min=SANITIZE_FLOOR)

    SCALE_MOMENTUM = 0.3
    MSE_DIVERGENCE_THRESHOLD = 1.05
    W_NONZERO_THRESHOLD = 1e-8

    mse_values = []
    prev_scales = None
    actual_method = "ldlq_cpu"

    for iter_idx in range(iterations):
        Q_iter, iter_method = _single_ldlq_pass(W, H_inv, current_scales, block_size, clamp_min, clamp_max)
        Q_result = Q_iter
        actual_method = iter_method  # Track the actual backend used

        # Compute MSE
        mse = (W - Q_iter.float() * current_scales.reshape(M, N)).double().abs().pow(2).mean().item()
        mse_values.append(mse)
        logger.debug("LDLQ iteration %d/%d: MSE=%.6f", iter_idx + 1, iterations, mse)

        if not math.isfinite(mse):
            logger.warning("LDLQ: non-finite MSE at iteration %d, reverting", iter_idx + 1)
            if prev_scales is not None:
                current_scales = prev_scales
            break

        if iter_idx > 0 and mse > prev_mse * MSE_DIVERGENCE_THRESHOLD:
            logger.warning("LDLQ: diverged at iteration %d (MSE increased), reverting", iter_idx + 1)
            if prev_scales is not None:
                current_scales = prev_scales
            break

        improvement = (prev_mse - mse) / (prev_mse + CONVERGENCE_EPS)
        if iter_idx > 0 and improvement < CONVERGENCE_IMPROVEMENT_THRESHOLD:
            logger.debug("LDLQ: converged at iteration %d (improvement=%.6f)", iter_idx + 1, improvement)
            break

        prev_mse = mse

        if iter_idx < iterations - 1:
            prev_scales = current_scales.clone()
            current_scales = _update_ldlq_scales(
                W, Q_iter, current_scales, flat_scales, SCALE_MOMENTUM, W_NONZERO_THRESHOLD
            )

        # Q_result holds a reference to Q_iter; on the next iteration
        # Q_iter is reassigned (not mutated), so the old tensor is freed.

    return Q_result, current_scales, actual_method


def _update_ldlq_scales(
    W: torch.Tensor,
    Q_iter: torch.Tensor,
    current_scales: torch.Tensor,
    flat_scales: torch.Tensor,
    momentum: float,
    nonzero_threshold: float,
) -> torch.Tensor:
    """Update scales for next LDLQ iteration.

    Valid (nonzero, sign-consistent) elements: new_scale = |W| / |Q|.
    Mismatched elements: revert to original flat scales.
    Momentum blending (default 0.3) prevents oscillation.
    """
    W_flat = W.reshape(-1)
    Q_flat = Q_iter.reshape(-1).float()

    nonzero_mask = Q_flat != 0
    W_nonzero = W_flat.abs() > nonzero_threshold
    sign_consistent = W_nonzero & ((W_flat * Q_flat) > 0)
    valid = nonzero_mask & sign_consistent
    mismatched = nonzero_mask & ~sign_consistent

    new_scales = current_scales.clone()
    scale_min, scale_max = flat_scales.aminmax()
    scale_floor = max(scale_min.item() * SCALE_FLOOR_MULTIPLIER, ABS_SCALE_FLOOR)
    scale_ceil = scale_max.item() * SCALE_CEIL_MULTIPLIER

    if valid.any():
        try:
            computed = (W_flat[valid].abs() / Q_flat[valid].abs()).clamp(scale_floor, scale_ceil)
            if momentum > 0:
                new_scales[valid] = momentum * current_scales[valid] + (1 - momentum) * computed
            else:
                new_scales[valid] = computed
        except RuntimeError as e:
            # Indexing or shape mismatch — revert to original scales
            logger.warning("LDLQ: scale update failed (%s), reverting to original scales", e)
            new_scales[valid] = flat_scales[valid].clamp(scale_floor, scale_ceil)

    if mismatched.any():
        try:
            new_scales[mismatched] = flat_scales[mismatched].clamp(scale_floor, scale_ceil)
        except RuntimeError as e:
            logger.warning("LDLQ: mismatched scale update failed (%s)", e)

    return new_scales


def _greedy_local_search(
    W: torch.Tensor,
    Q: torch.Tensor,
    scale_2d: torch.Tensor,
    H: torch.Tensor,
    num_passes: int,
    clamp_min: int,
    clamp_max: int,
    rank_threshold: float = 0.01,
) -> torch.Tensor:
    """Greedy local search with automatic backend selection.

    Delegates to :mod:`converter.greedy` which selects the fastest available
    backend: low-rank Triton → full-rank Triton v2 → PyTorch.

    Converts Q to float32 contiguous once here so that each backend
    does not need to clone independently.
    """
    # Prepare Q once as float32 contiguous — all backends modify it in-place
    Q = Q.clone().float().contiguous()

    # Try low-rank Triton first (fastest for low-rank Hessians)
    if _HAS_TRITON and W.is_cuda:
        result = greedy_lowrank_triton(
            W, Q, scale_2d, H, num_passes, clamp_min, clamp_max,
            rank_threshold=rank_threshold,
        )
        if result is not None:
            Q_out, k, used_lowrank = result
            if Q_out is not None:
                path = "lowrank" if used_lowrank else "v2"
                logger.debug("Greedy: GPU path (Triton %s), k=%d", path, k)
                return Q_out

    # Fall back to full-rank Triton v2
    if _HAS_TRITON and W.is_cuda:
        result = greedy_local_search_triton(
            W, Q, scale_2d, H, num_passes, clamp_min, clamp_max,
        )
        if result is not None:
            logger.debug("Greedy: GPU path (Triton v2)")
            return result

    # PyTorch fallback
    logger.debug("Greedy: CPU path")
    return greedy_local_search_pytorch(W, Q, scale_2d, H, num_passes, clamp_min, clamp_max)
