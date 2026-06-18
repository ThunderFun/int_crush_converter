"""PermuQuant: Channel reordering for quantization (arXiv:2605.09503).

Core logic: second-moment based channel reordering, joint activation+weight
statistics, calibration-based acceptance rule. Permutation is absorbed offline
into weights and adjacent modules — zero runtime overhead.
"""

import torch
import torch.nn.functional as F
from .scales import calculate_scales, quantize_weights, calculate_scales_int8, quantize_weights_int8


def channel_second_moments(W: torch.Tensor) -> torch.Tensor:
    """Compute per-channel second moments of a weight matrix.

    Args:
        W: [out_features, in_features] weight tensor

    Returns:
        mu2: [in_features] second moment per input channel
    """
    return (W ** 2).mean(dim=0)


def find_permutation_weight(W: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Find optimal channel permutation using weight statistics only.

    Sorts channels by descending second moment, which minimizes the upper
    bound on quantization error (Proposition 1 in PermuQuant).

    Args:
        W: [out_features, in_features] weight tensor
        group_size: quantization group size

    Returns:
        perm: [in_features] channel permutation indices
    """
    mu2 = channel_second_moments(W)
    # Sort descending by second moment
    perm = mu2.argsort(descending=True)
    return perm


def find_permutation_joint(
    W: torch.Tensor,
    act_mu2: torch.Tensor,
    alpha: float = 0.5,
    group_size: int = 128,
) -> torch.Tensor:
    """Find optimal channel permutation using joint activation + weight statistics.

    v_i = (act_mu2_i)^alpha * (weight_mu2_i)^(1 - alpha)

    Args:
        W: [out_features, in_features] weight tensor
        act_mu2: [in_features] activation second moments per channel
        alpha: balance between activation (1.0) and weight (0.0) importance
        group_size: quantization group size

    Returns:
        perm: [in_features] channel permutation indices
    """
    weight_mu2 = channel_second_moments(W)
    # Joint criterion: geometric mean weighted by alpha
    v = (act_mu2.clamp(min=1e-12) ** alpha) * (weight_mu2.clamp(min=1e-12) ** (1.0 - alpha))
    perm = v.argsort(descending=True)
    return perm


def compute_group_quant_error(
    W: torch.Tensor,
    group_size: int = 128,
    int_bits: int = 4,
) -> float:
    """Compute total quantization error for a weight matrix.

    This is the sum of squared errors after quantize-then-dequantize.

    Args:
        W: [out_features, in_features] weight tensor
        group_size: quantization group size
        int_bits: quantization bit-width (4 or 8)

    Returns:
        error: total squared quantization error (scalar)
    """
    if int_bits == 8:
        scales = calculate_scales_int8(W)
        quantized = quantize_weights_int8(W, scales)
    else:
        scales = calculate_scales(W, group_size)
        quantized = quantize_weights(W, scales, group_size)
    # Dequantize
    out_features, in_features = W.shape
    in_features_padded = quantized.shape[1]
    if int_bits == 8:
        # Per-row scales: [out, 1]
        dq = (quantized.float() * scales.float())[:, :in_features]
    else:
        # Per-group scales: [out, num_groups]
        num_groups = in_features_padded // group_size
        q_grouped = quantized.reshape(out_features, num_groups, group_size).float()
        dq_grouped = q_grouped * scales.unsqueeze(2).float()
        dq = dq_grouped.reshape(out_features, in_features_padded)[:, :in_features]
    return ((W - dq) ** 2).sum().item()


def find_permutation_with_acceptance(
    W: torch.Tensor,
    act_mu2: torch.Tensor | None = None,
    alpha: float = 0.5,
    group_size: int = 128,
    tau: float = 0.0,
    int_bits: int = 4,
) -> tuple[torch.Tensor, bool]:
    """Find optimal permutation and apply acceptance rule.

    Only accepts the permutation if it sufficiently reduces quantization error.

    Args:
        W: [out_features, in_features] weight tensor
        act_mu2: [in_features] activation second moments (None for weight-only)
        alpha: balance between activation and weight importance
        group_size: quantization group size
        tau: acceptance threshold (0 = accept if any improvement)
        int_bits: quantization bit-width (4 or 8)

    Returns:
        perm: [in_features] channel permutation indices
        accepted: whether the permutation was accepted
    """
    # Find candidate permutation
    if act_mu2 is not None:
        perm = find_permutation_joint(W, act_mu2, alpha, group_size)
    else:
        perm = find_permutation_weight(W, group_size)

    # Compute errors before and after
    error_orig = compute_group_quant_error(W, group_size, int_bits)
    W_reordered = W[:, perm]
    error_reorder = compute_group_quant_error(W_reordered, group_size, int_bits)

    # Acceptance rule: relative improvement must exceed tau
    if error_orig > 0:
        relative_improvement = (error_orig - error_reorder) / error_orig
    else:
        relative_improvement = 0.0

    accepted = relative_improvement > tau
    return perm, accepted


def sweep_alpha(
    W: torch.Tensor,
    act_mu2: torch.Tensor | None = None,
    group_size: int = 128,
    tau: float = 0.0,
    num_alpha: int = 11,
    int_bits: int = 4,
) -> tuple[torch.Tensor, float, bool]:
    """Sweep alpha values to find the best permutation.

    Args:
        W: [out_features, in_features] weight tensor
        act_mu2: [in_features] activation second moments (None for weight-only)
        group_size: quantization group size
        tau: acceptance threshold
        num_alpha: number of alpha values to try in [0, 1]
        int_bits: quantization bit-width (4 or 8)

    Returns:
        best_perm: [in_features] best channel permutation
        best_alpha: best alpha value
        accepted: whether any permutation was accepted
    """
    if num_alpha < 1:
        raise ValueError(f"num_alpha must be >= 1, got {num_alpha}")

    if act_mu2 is None or num_alpha == 1:
        # Weight-only or single alpha: no sweep needed
        perm, accepted = find_permutation_with_acceptance(
            W, act_mu2, 0.0, group_size, tau, int_bits
        )
        return perm, 0.0, accepted

    best_perm = None
    best_alpha = 0.0
    best_error = float('inf')
    best_accepted = False

    for i in range(num_alpha):
        alpha = i / (num_alpha - 1)
        perm, accepted = find_permutation_with_acceptance(
            W, act_mu2, alpha, group_size, tau, int_bits
        )

        if accepted:
            W_reordered = W[:, perm]
            error = compute_group_quant_error(W_reordered, group_size, int_bits)
            if error < best_error:
                best_error = error
                best_perm = perm
                best_alpha = alpha
                best_accepted = True

    if best_perm is None:
        # No permutation accepted; return identity
        best_perm = torch.arange(W.shape[1])
        best_accepted = False

    return best_perm, best_alpha, best_accepted


def apply_permutation_to_weight(W: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Apply channel permutation to weight matrix columns.

    Args:
        W: [out_features, in_features] weight tensor
        perm: [in_features] permutation indices

    Returns:
        W_permuted: [out_features, in_features] permuted weight tensor
    """
    return W[:, perm]


def apply_permutation_to_norm(norm_weight: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Apply channel permutation to LayerNorm/RMSNorm weight.

    Args:
        norm_weight: [in_features] norm weight (gamma)
        perm: [in_features] permutation indices

    Returns:
        norm_weight_permuted: [in_features] permuted norm weight
    """
    return norm_weight[perm]


def apply_permutation_to_norm_bias(
    norm_bias: torch.Tensor, perm: torch.Tensor
) -> torch.Tensor:
    """Apply channel permutation to LayerNorm bias.

    Args:
        norm_bias: [in_features] norm bias (beta)
        perm: [in_features] permutation indices

    Returns:
        norm_bias_permuted: [in_features] permuted norm bias
    """
    return norm_bias[perm]


def apply_permutation_to_linear_output(W: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Apply channel permutation to a linear layer's output channels.

    Used to absorb the activation-side permutation into the preceding linear layer.

    Args:
        W: [out_features, in_features] weight tensor of preceding linear
        perm: [in_features of next layer] permutation indices

    Returns:
        W_permuted: [out_features, in_features] weight with permuted output rows
    """
    return W[perm, :]


def permute_state_dict(
    state_dict: dict[str, torch.Tensor],
    layer_perms: dict[str, torch.Tensor],
    norm_map: dict[str, str] | None = None,
    linear_map: dict[str, str] | None = None,
) -> dict[str, torch.Tensor]:
    """Apply channel permutations to a state dict.

    For each layer in layer_perms:
    1. Permute the weight columns
    2. If the preceding module is a LayerNorm, permute its weight/bias
    3. If the preceding module is a linear, permute its output rows

    Args:
        state_dict: model state dict
        layer_perms: {layer_name: permutation_indices} for each layer to permute
        norm_map: {linear_name: norm_name} mapping linear layers to their preceding norms
        linear_map: {linear_name: prev_linear_name} mapping linear layers to preceding linears

    Returns:
        state_dict: modified state dict with permutations applied
    """
    if norm_map is None:
        norm_map = {}
    if linear_map is None:
        linear_map = {}

    for layer_name, perm in layer_perms.items():
        weight_key = f"{layer_name}.weight"

        if weight_key not in state_dict:
            continue

        # Permute weight columns
        state_dict[weight_key] = apply_permutation_to_weight(
            state_dict[weight_key], perm
        )

        # Absorb activation permutation into preceding module
        if layer_name in norm_map:
            norm_name = norm_map[layer_name]
            norm_weight_key = f"{norm_name}.weight"
            norm_bias_key = f"{norm_name}.bias"

            if norm_weight_key in state_dict:
                state_dict[norm_weight_key] = apply_permutation_to_norm(
                    state_dict[norm_weight_key], perm
                )
            if norm_bias_key in state_dict:
                state_dict[norm_bias_key] = apply_permutation_to_norm_bias(
                    state_dict[norm_bias_key], perm
                )

        elif layer_name in linear_map:
            prev_name = linear_map[layer_name]
            prev_weight_key = f"{prev_name}.weight"

            if prev_weight_key in state_dict:
                state_dict[prev_weight_key] = apply_permutation_to_linear_output(
                    state_dict[prev_weight_key], perm
                )

    return state_dict
