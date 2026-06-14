"""Per-row scale calculation and quantization for INT-Crush INT4/INT8.

Per-row quantization computes one scale per row:
    scale[row] = max(|W[row, :]) / Q

This is the foundation for PermuQuant channel reordering, which improves
quantization by placing similar channels adjacent to each other.
"""

import torch

from .config import INT4_SCALE_DIVISOR, INT8_SCALE_DIVISOR, MAX_FP16_SCALE, FP16_SCALE_FLOOR
from .rounding import _round_half_away_from_zero


def _pad_to_group_size(W: torch.Tensor, group_size: int) -> torch.Tensor:
    """Pad W's in_features dimension to be divisible by group_size."""
    _, in_features = W.shape
    if in_features % group_size != 0:
        pad = group_size - (in_features % group_size)
        W = torch.nn.functional.pad(W, (0, pad))
    return W


def _search_clipping_ratio(
    clipping_ratios: list[float],
    compute_fn,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Search clipping ratios for the lowest-MSE scales (and optional zero-points).

    Iterates over each candidate ratio, calls ``compute_fn(ratio)`` to obtain
    ``(candidate_scales, candidate_zp, mse)``, and keeps the per-group/row
    best scales based on MSE.  ``candidate_zp`` is ``None`` for symmetric
    quantization.

    Args:
        clipping_ratios: list of ratios to try, e.g. [0.8, 0.85, 0.9, 0.95, 1.0]
        compute_fn: callable(ratio) -> (candidate_scales, candidate_zp | None, mse)

    Returns:
        (best_scales, best_zp): best scales and zero-points.
        ``best_zp`` is ``None`` when ``compute_fn`` always returns ``None`` for zp.
    """
    best_scales = None
    best_zp = None
    best_mse = None
    for ratio in clipping_ratios:
        candidate_scales, candidate_zp, mse = compute_fn(ratio)
        if best_scales is None:
            best_scales = candidate_scales
            best_zp = candidate_zp
            best_mse = mse
        else:
            better = mse < best_mse
            best_scales = torch.where(better, candidate_scales, best_scales)
            if candidate_zp is not None:
                best_zp = torch.where(better, candidate_zp, best_zp)
            best_mse = torch.where(better, mse, best_mse)
    return best_scales, best_zp


def calculate_scales(W: torch.Tensor, group_size: int = 128,
                     clipping_ratios: list[float] | None = None) -> torch.Tensor:
    """Calculate per-row scales for INT4 quantization.

    Args:
        W: [out_features, in_features] weight tensor
        group_size: quantization group size (pass in_features for per-row)
        clipping_ratios: optional list of ratios to search for lowest-MSE
                         scale per group. e.g. [0.8, 0.85, 0.9, 0.95, 1.0]

    Returns:
        scales: [out_features, num_groups] float16 scales
    """
    if W.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {W.dim()}D")

    W_padded = _pad_to_group_size(W, group_size)
    out_features, in_features = W_padded.shape
    num_groups = in_features // group_size
    W_grouped = W_padded.reshape(out_features, num_groups, group_size)
    max_vals = W_grouped.abs().amax(dim=2)

    if clipping_ratios is None:
        scales = (max_vals.float() / INT4_SCALE_DIVISOR).clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE).to(torch.float16)
        return scales

    def _compute(ratio):
        candidate_scales = (max_vals.float() * ratio / INT4_SCALE_DIVISOR).clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE)
        q = (W_grouped / candidate_scales.unsqueeze(2)).round().clamp(-8, 7)
        dequant = q * candidate_scales.unsqueeze(2)
        mse = (W_grouped - dequant).pow(2).mean(dim=2)
        return candidate_scales, None, mse

    best_scales, _ = _search_clipping_ratio(clipping_ratios, _compute)
    return best_scales.to(torch.float16)


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


def calculate_scales_int8(W: torch.Tensor,
                          clipping_ratios: list[float] | None = None) -> torch.Tensor:
    """Calculate per-row scales for INT8 quantization.

    Args:
        W: [out_features, in_features] weight tensor
        clipping_ratios: optional list of ratios to search for lowest-MSE
                         scale per row. e.g. [0.8, 0.85, 0.9, 0.95, 1.0]

    Returns:
        scales: [out_features, 1] float16 per-row scales
    """
    if W.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {W.dim()}D")

    max_vals = W.abs().amax(dim=1, keepdim=True)

    if clipping_ratios is None:
        scales = (max_vals.float() / INT8_SCALE_DIVISOR).clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE).to(torch.float16)
        return scales

    W_3d = W.unsqueeze(1)  # [out, 1, in]

    def _compute(ratio):
        candidate_scales = (max_vals.float() * ratio / INT8_SCALE_DIVISOR).clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE)
        q = (W_3d / candidate_scales.unsqueeze(2)).round().clamp(-128, 127)
        dequant = q * candidate_scales.unsqueeze(2)
        mse = (W_3d - dequant).pow(2).mean(dim=2)
        return candidate_scales, None, mse

    best_scales, _ = _search_clipping_ratio(clipping_ratios, _compute)
    return best_scales.to(torch.float16)


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


# --- Asymmetric quantization (scale + zero-point per group) ---


def calculate_scales_asymmetric(W: torch.Tensor, group_size: int = 128,
                                clipping_ratios: list[float] | None = None
                                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate per-group asymmetric scales and zero-points for INT4.

    Args:
        W: [out_features, in_features] weight tensor
        group_size: quantization group size
        clipping_ratios: optional list of ratios to search for lowest-MSE
                         per group. e.g. [0.8, 0.85, 0.9, 0.95, 1.0]

    Returns:
        (scales, zero_points):
            scales:      [out_features, num_groups] float16
            zero_points: [out_features, num_groups] int8
    """
    if W.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {W.dim()}D")

    W_padded = _pad_to_group_size(W, group_size)
    out_features, in_features = W_padded.shape
    num_groups = in_features // group_size
    W_grouped = W_padded.reshape(out_features, num_groups, group_size)
    w_min = W_grouped.amin(dim=2)
    w_max = W_grouped.amax(dim=2)

    if clipping_ratios is None:
        scales = ((w_max - w_min).float() / 15.0).clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE)
        zero_points = (-8 - _round_half_away_from_zero(w_min / scales)).clamp(-8, 7)
        # Zero groups: set zp to 0 so zero maps to zero
        zero_groups = (w_min == 0) & (w_max == 0)
        zero_points = zero_points.masked_fill(zero_groups, 0)
        return scales.to(torch.float16), zero_points.to(torch.int8)

    def _compute(ratio):
        range_vals = (w_max - w_min).float() * ratio
        candidate_scales = (range_vals / 15.0).clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE)
        candidate_zp = (-8 - _round_half_away_from_zero(w_min / candidate_scales)).clamp(-8, 7)
        q = (W_grouped / candidate_scales.unsqueeze(2)
             + candidate_zp.unsqueeze(2)).round().clamp(-8, 7)
        dequant = (q - candidate_zp.unsqueeze(2)) * candidate_scales.unsqueeze(2)
        mse = (W_grouped - dequant).pow(2).mean(dim=2)
        return candidate_scales, candidate_zp, mse

    best_scales, best_zp = _search_clipping_ratio(clipping_ratios, _compute)
    return best_scales.to(torch.float16), best_zp.to(torch.int8)


def quantize_weights_asymmetric(
    W: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Quantize weight matrix to INT4 using asymmetric scales and zero-points.

    Args:
        W: [out_features, in_features] rotated weight tensor
        scales: [out_features, num_groups] scales
        zero_points: [out_features, num_groups] zero-points
        group_size: quantization group size

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
    W_scaled = (W_grouped / scales.unsqueeze(2).to(W.dtype)
                + zero_points.unsqueeze(2).to(W.dtype))
    W_rounded = W_scaled.round().clamp(-8, 7)
    return W_rounded.reshape(out_features, in_features).to(torch.int8)


def calculate_scales_int8_asymmetric(
    W: torch.Tensor,
    clipping_ratios: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate per-row asymmetric scales and zero-points for INT8.

    Args:
        W: [out_features, in_features] weight tensor
        clipping_ratios: optional list of ratios to search for lowest-MSE
                         per row. e.g. [0.8, 0.85, 0.9, 0.95, 1.0]

    Returns:
        (scales, zero_points):
            scales:      [out_features, 1] float16
            zero_points: [out_features, 1] int16 (int8 range [-128,127])
    """
    if W.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {W.dim()}D")

    w_min = W.amin(dim=1, keepdim=True)
    w_max = W.amax(dim=1, keepdim=True)

    if clipping_ratios is None:
        scales = ((w_max - w_min).float() / 255.0).clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE)
        zero_points = (-128 - _round_half_away_from_zero(w_min / scales)).clamp(-128, 127)
        return scales.to(torch.float16), zero_points.to(torch.int16)

    W_3d = W.unsqueeze(1)  # [out, 1, in]

    def _compute(ratio):
        range_vals = (w_max - w_min).float() * ratio
        candidate_scales = (range_vals / 255.0).clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE)
        candidate_zp = (-128 - _round_half_away_from_zero(w_min / candidate_scales)).clamp(-128, 127)
        q = (W_3d / candidate_scales.unsqueeze(2)
             + candidate_zp.unsqueeze(2)).round().clamp(-128, 127)
        dequant = (q - candidate_zp.unsqueeze(2)) * candidate_scales.unsqueeze(2)
        mse = (W_3d - dequant).pow(2).mean(dim=2)
        return candidate_scales, candidate_zp, mse

    best_scales, best_zp = _search_clipping_ratio(clipping_ratios, _compute)
    return best_scales.to(torch.float16), best_zp.to(torch.int16)


def quantize_weights_int8_asymmetric(
    W: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
) -> torch.Tensor:
    """Quantize weight matrix to INT8 using asymmetric scales and zero-points.

    Args:
        W: [out_features, in_features] rotated weight tensor
        scales: [out_features, 1] per-row scales
        zero_points: [out_features, 1] per-row zero-points

    Returns:
        quantized: [out_features, in_features] INT8 values in [-128, 127]
    """
    if W.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {W.dim()}D")

    W_scaled = (W / scales.to(W.dtype) + zero_points.to(W.dtype))
    W_rounded = W_scaled.round().clamp(-128, 127)
    return W_rounded.to(torch.int8)
