"""SmoothQuant: per-channel activation smoothing for quantization.

Implements the smoothing transformation from SmoothQuant (arXiv:2211.10438):

    Y = X @ W = (X @ diag(1/s)) @ (diag(s) @ W)

where s_j = max|X_j|^alpha / max|W_j|^(1-alpha) are per-input-channel
smoothing factors that balance quantization difficulty between activations
and weights.

At inference, the activation-side inverse scaling (1/s) is absorbed into
the preceding normalization or linear layer — zero runtime overhead.
"""

import torch

from .config import SMOOTHQUANT_AMAX_FLOOR


def compute_smoothing_factors(
    act_amax: torch.Tensor,
    weight: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Compute SmoothQuant per-channel smoothing factors.

    s_i = max|X[:,i]|^alpha / max|W[:,i]|^(1-alpha)

    After smoothing:
        X_smooth = X @ diag(1/s)   (absorbed into preceding module at inference)
        W_smooth = diag(s) @ W     (column-wise scaling, applied offline)

    Args:
        act_amax: [in_features] per-channel max absolute activation value
            from calibration data.  Typically collected by the calibration
            node as ``max_t |X[t, i]|`` across all calibration samples.
        weight: [out_features, in_features] weight matrix.
        alpha: Migration strength in [0, 1].
            0.0 = all difficulty to weights (s_i = 1/max|W_i|),
            1.0 = all difficulty to activations (s_i = max|X_i|),
            0.5 = even split (default, works well for most models).
            Use 0.75 for models with severe activation outliers (e.g., GLM-130B).

    Returns:
        [in_features] smoothing factors s (float32).
    """
    if act_amax.dim() != 1:
        raise ValueError(f"act_amax must be 1D [in_features], got {act_amax.dim()}D")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D [out, in], got {weight.dim()}D")
    if act_amax.shape[0] != weight.shape[1]:
        raise ValueError(
            f"act_amax length {act_amax.shape[0]} != weight in_features {weight.shape[1]}"
        )

    act_scale = act_amax.float().clamp(min=SMOOTHQUANT_AMAX_FLOOR)
    w_scale = weight.abs().amax(dim=0).float().clamp(min=SMOOTHQUANT_AMAX_FLOOR)

    return (act_scale ** alpha) / (w_scale ** (1.0 - alpha))


def apply_smoothing_to_weight(
    weight: torch.Tensor,
    smoothing: torch.Tensor,
) -> torch.Tensor:
    """Apply per-channel smoothing to a weight matrix.

    W_smooth[:, j] = W[:, j] * s_j

    Args:
        weight: [out_features, in_features] weight tensor.
        smoothing: [in_features] per-channel smoothing factors.

    Returns:
        [out_features, in_features] smoothed weight tensor (same dtype as input).
    """
    return weight * smoothing.unsqueeze(0).to(weight.dtype)


def compute_smoothing_from_hessian_diag(
    hessian_diag: torch.Tensor,
    weight: torch.Tensor,
    alpha: float = 0.5,
    num_calibration_samples: int = 128,
) -> torch.Tensor:
    """Approximate SmoothQuant factors from the Hessian diagonal.

    When per-channel activation amax is unavailable, the Hessian diagonal
    provides a proxy: ``H[i,i] = sum_t X[t,i]^2``, so
    ``sqrt(H[i,i] / n_samples) ≈ RMS(X[:,i])``.

    This is RMS rather than amax, so the smoothing factors will differ
    from the exact SmoothQuant formula, but the relative ordering of
    channel magnitudes is preserved and the result is usable.

    Args:
        hessian_diag: [in_features] diagonal of H = X^T X.
        weight: [out_features, in_features] weight matrix.
        alpha: Migration strength (see :func:`compute_smoothing_factors`).
        num_calibration_samples: Number of calibration samples used to
            compute the Hessian.  Divides out the accumulation so the
            result is a per-sample RMS.

    Returns:
        [in_features] smoothing factors s (float32).
    """
    n = max(1, num_calibration_samples)
    act_rms = (hessian_diag.float() / n).clamp(min=SMOOTHQUANT_AMAX_FLOOR).sqrt()
    w_scale = weight.abs().amax(dim=0).float().clamp(min=SMOOTHQUANT_AMAX_FLOOR)
    return (act_rms ** alpha) / (w_scale ** (1.0 - alpha))


def compute_smoothing_weight_only(
    weight: torch.Tensor,
) -> torch.Tensor:
    """Weight-only smoothing: normalize columns so max|W[:,j]| = 1.

    Equivalent to SmoothQuant with alpha=0 (all difficulty to weights).
    Reduces per-row dynamic range by equalizing column magnitudes,
    but does not use any activation statistics.

    Useful as a fallback when no calibration data is available.

    Args:
        weight: [out_features, in_features] weight matrix.

    Returns:
        [in_features] smoothing factors s (float32).
    """
    w_scale = weight.abs().amax(dim=0).float().clamp(min=SMOOTHQUANT_AMAX_FLOOR)
    return 1.0 / w_scale
