"""GPTQ: Accurate Post-Training Quantization (Frantar et al., ICLR 2023).

Block-wise quantization using Hessian information to optimally redistribute
quantization error to remaining unquantized columns.
"""

import os
import torch

from .scales import INT4_SCALE_DIVISOR, INT8_SCALE_DIVISOR, calculate_scales, quantize_weights, calculate_scales_int8, quantize_weights_int8
from .rounding import _invert_hessian, _gptq_block
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

        W_gpu = W_slice.to("cuda")
        Q_gpu = torch.zeros(M, n_cols, dtype=torch.int8, device="cuda")

        result = gptq_loop_triton(W_gpu, H_gpu, s_gpu, Q_gpu, n_cols, M, block_size, clamp_min, clamp_max)
        if result is not None:
            W_work[:, col_start:col_start + n_cols] = W_gpu.to(W_work.device)
            quantized_W[:, col_start:col_start + n_cols] = Q_gpu.to(quantized_W.device)
            return True
        return False
    except Exception as e:
        print(f"    GPTQ Triton fallback: {e}")
        return False


def gptq_quantize_layer(
    W: torch.Tensor,
    hessian: torch.Tensor,
    block_size: int = 128,
    damping: float = 0.01,
    int_bits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight matrix using GPTQ with per-row quantization.

    Supports both full Hessians [in, in] and block-diagonal Hessians [num_blocks, bs, bs].
    For block-diagonal, each block is processed independently.

    Args:
        W: [out_features, in_features] weight tensor
        hessian: Hessian matrix — 2D [in, in] or 3D [num_blocks, bs, bs]
        block_size: GPTQ block size for column processing
        damping: damping ratio as fraction of mean diagonal (default: 0.01)
        int_bits: quantization bit-width (4 or 8)

    Returns:
        (quantized_W, scales):
            quantized_W: [out_features, in_features] quantized values as int8
            scales: [out_features, 1] float16 per-row scales
    """
    if int_bits not in (4, 8):
        raise ValueError(f"int_bits must be 4 or 8, got {int_bits}")
    if W.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor, got {W.dim()}D")

    W = W.float()
    out_features, in_features = W.shape

    scale_divisor = INT8_SCALE_DIVISOR if int_bits == 8 else INT4_SCALE_DIVISOR
    clamp_min = -128 if int_bits == 8 else -8
    clamp_max = 127 if int_bits == 8 else 7

    row_scales = (W.abs().amax(dim=1, keepdim=True) / scale_divisor).clamp(min=1e-6)

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
            diag_mean = H_block.diagonal().mean().clamp(min=1e-6)
            H_block = H_block + damping * diag_mean * torch.eye(actual_bs, dtype=torch.float32)

            H_inv = _invert_hessian(H_block)
            H_inv = H_inv.to(W.device)

            if _gptq_block_triton(
                W_work, quantized_W, row_scales, H_inv,
                col_start, actual_bs, block_size, clamp_min, clamp_max,
            ):
                used_triton = True
            else:
                _gptq_block(
                    W_work, quantized_W, row_scales, H_inv,
                    col_start, col_end, block_size, clamp_min, clamp_max,
                )

    elif hessian.dim() == 2:
        if hessian.shape != (in_features, in_features):
            raise ValueError(
                f"Hessian shape {hessian.shape} != ({in_features}, {in_features})"
            )

        H = hessian.float()
        diag_mean = H.diagonal().mean().clamp(min=1e-6)
        H = H + damping * diag_mean * torch.eye(in_features, dtype=torch.float32)

        H_inv = _invert_hessian(H)
        H_inv = H_inv.to(W.device)

        if _gptq_block_triton(
            W_work, quantized_W, row_scales, H_inv,
            0, in_features, block_size, clamp_min, clamp_max,
        ):
            used_triton = True
        else:
            _gptq_block(
                W_work, quantized_W, row_scales, H_inv,
                0, in_features, block_size, clamp_min, clamp_max,
            )
    else:
        raise ValueError(f"Expected 2D or 3D Hessian, got {hessian.dim()}D")

    accel = "Triton" if used_triton else "PyTorch"
    print(f"    GPTQ: {accel} path, {out_features}x{in_features}, block_size={block_size}")

    return quantized_W, row_scales.to(torch.float16)


def gptq_quantize_layer_rtn(
    W: torch.Tensor,
    int_bits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback: RTN quantization (no Hessian correction).

    Used when calibration data is not available for a layer.

    Args:
        W: [out_features, in_features] weight tensor
        int_bits: quantization bit-width (4 or 8)

    Returns:
        Same as gptq_quantize_layer
    """
    if int_bits not in (4, 8):
        raise ValueError(f"int_bits must be 4 or 8, got {int_bits}")

    if int_bits == 8:
        scales = calculate_scales_int8(W)
        quantized = quantize_weights_int8(W, scales)
    else:
        in_features = W.shape[1]
        scales = calculate_scales(W, in_features)
        quantized = quantize_weights(W, scales, in_features)
    return quantized, scales
