"""Per-row scale calculation and quantization for INT-Crush INT4/INT8.

Per-row quantization computes one scale per row:
    scale[row] = max(|W[row, :]) / Q

This is the foundation for PermuQuant channel reordering, which improves
quantization by placing similar channels adjacent to each other.
"""

import torch

INT4_SCALE_DIVISOR = 7.0
INT8_SCALE_DIVISOR = 127.0


def calculate_scales(W: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Calculate per-row scales for INT4 quantization.

    Args:
        W: [out_features, in_features] weight tensor
        group_size: quantization group size (pass in_features for per-row)

    Returns:
        scales: [out_features, num_groups] float16 scales
    """
    if W.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {W.dim()}D")

    out_features, in_features = W.shape
    if in_features % group_size != 0:
        pad = group_size - (in_features % group_size)
        W = torch.nn.functional.pad(W, (0, pad))
        in_features = W.shape[1]

    num_groups = in_features // group_size
    W_grouped = W.reshape(out_features, num_groups, group_size)
    max_vals = W_grouped.abs().amax(dim=2)
    # Compute in float32 to avoid float16 underflow (1e-8 rounds to 0 in float16)
    scales = (max_vals.float() / INT4_SCALE_DIVISOR).clamp(min=1e-6).to(torch.float16)
    return scales


def quantize_weights(W: torch.Tensor, scales: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Quantize weight matrix to INT4 using precomputed scales.

    Args:
        W: [out_features, in_features] rotated weight tensor
        scales: [out_features, num_groups] scales
        group_size: quantization group size (pass in_features for per-row)

    Returns:
        quantized: [out_features, in_features] INT4 values in [-8, 7]
    """
    out_features, in_features = W.shape
    if in_features % group_size != 0:
        pad = group_size - (in_features % group_size)
        W = torch.nn.functional.pad(W, (0, pad))
        in_features = W.shape[1]

    num_groups = in_features // group_size
    W_grouped = W.reshape(out_features, num_groups, group_size)
    W_scaled = W_grouped / scales.unsqueeze(2).to(W.dtype)
    W_rounded = W_scaled.round().clamp(-8, 7)
    return W_rounded.reshape(out_features, in_features).to(torch.int8)


def calculate_scales_int8(W: torch.Tensor) -> torch.Tensor:
    """Calculate per-row scales for INT8 quantization.

    Args:
        W: [out_features, in_features] weight tensor

    Returns:
        scales: [out_features, 1] float16 per-row scales
    """
    if W.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {W.dim()}D")

    max_vals = W.abs().amax(dim=1, keepdim=True)
    # Compute in float32 to avoid float16 underflow (1e-8 rounds to 0 in float16)
    scales = (max_vals.float() / INT8_SCALE_DIVISOR).clamp(min=1e-6).to(torch.float16)
    return scales


def quantize_weights_int8(W: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Quantize weight matrix to INT8 using precomputed per-row scales.

    Args:
        W: [out_features, in_features] rotated weight tensor
        scales: [out_features, 1] per-row scales

    Returns:
        quantized: [out_features, in_features] INT8 values in [-128, 127]
    """
    if W.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {W.dim()}D")

    W_scaled = W / scales.to(W.dtype)
    W_rounded = W_scaled.round().clamp(-128, 127)
    return W_rounded.to(torch.int8)
