"""Benchmark module: side-by-side quantization quality comparison.

Runs every valid (method × bits × features) combo on the same weights
and returns per-layer MSE for comparison.

Public API:
    benchmark_method, benchmark_matrix, make_synthetic_model,
    make_synthetic_calibration, FEATURE_PRESETS, VALID_COMBOS
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F

from .config import DEFAULT_SKIP_PATTERNS, SCALE_DTYPE, SCALE_MIN, SCALE_MAX
from .gptq import gptq_quantize_layer, gptq_quantize_layer_rtn
from .ldlq import ldlq_quantize_layer
from .log import logger
from .permuquant import sweep_alpha
from .rotation import rotate_weights, get_hadamard
from .scales import (
    calculate_scales,
    quantize_weights,
    calculate_scales_int8,
    quantize_weights_int8,
    calculate_scales_asymmetric,
    quantize_weights_asymmetric,
    calculate_scales_int8_asymmetric,
    quantize_weights_int8_asymmetric,
)
from .smoothquant import (
    compute_smoothing_factors,
    apply_smoothing_to_weight,
    compute_smoothing_from_hessian_diag,
    compute_smoothing_weight_only,
)
from .smoothrot import detect_ffn_pairs
from .svd import decompose_weight
from .dlr import is_dlr, permute_dlr, make_dlr_dict, transform_dlr_for_smoothquant
from .types import (
    BenchmarkConfig,
    BenchmarkReport,
    LayerBenchmarkResult,
    MethodBenchmarkResult,
)

# ── Feature presets ──────────────────────────────────────────────────────────

FEATURE_PRESETS: dict[str, dict[str, Any]] = {
    "plain": {
        "rot_size": 0,
        "smoothquant": False,
        "smoothrot": False,
        "use_permuquant": False,
        "svd_rank": 0,
    },
    "convrot": {
        "rot_size": 64,
        "smoothquant": False,
        "smoothrot": False,
        "use_permuquant": False,
        "svd_rank": 0,
    },
    "smoothquant": {
        "rot_size": 0,
        "smoothquant": True,
        "smoothrot": False,
        "use_permuquant": False,
        "svd_rank": 0,
    },
    "smoothrot": {
        "rot_size": 64,
        "smoothquant": True,
        "smoothrot": True,
        "use_permuquant": False,
        "svd_rank": 0,
    },
    "permuquant": {
        "rot_size": 0,
        "smoothquant": False,
        "smoothrot": False,
        "use_permuquant": True,
        "svd_rank": 0,
    },
    "svd": {
        "rot_size": 0,
        "smoothquant": False,
        "smoothrot": False,
        "use_permuquant": False,
        "svd_rank": 16,
    },
    "convrot+sq": {
        "rot_size": 64,
        "smoothquant": True,
        "smoothrot": False,
        "use_permuquant": False,
        "svd_rank": 0,
    },
    "convrot+pq": {
        "rot_size": 64,
        "smoothquant": False,
        "smoothrot": False,
        "use_permuquant": True,
        "svd_rank": 0,
    },
    "smoothrot+pq": {
        "rot_size": 64,
        "smoothquant": True,
        "smoothrot": True,
        "use_permuquant": True,
        "svd_rank": 0,
    },
    "svd+convrot": {
        "rot_size": 64,
        "smoothquant": False,
        "smoothrot": False,
        "use_permuquant": False,
        "svd_rank": 16,
    },
    "everything": {
        "rot_size": 64,
        "smoothquant": True,
        "smoothrot": True,
        "use_permuquant": True,
        "svd_rank": 8,
    },
}


def _apply_feature_preset(
    config: BenchmarkConfig,
    presets: dict[str, dict[str, Any]] | None = None,
) -> BenchmarkConfig:
    """Apply the feature preset string to a BenchmarkConfig.

    Returns a new BenchmarkConfig with fields overridden by the preset.
    Raises ValueError for unknown preset names.

    Args:
        config: The benchmark config with a ``features`` string.
        presets: Optional override for the presets dict.  When ``None``,
            uses the module-level :data:`FEATURE_PRESETS`.  Pass adapted
            presets from :func:`_adapt_presets_for_calibration` to use
            the calibration's ``rot_size``.
    """
    if presets is None:
        presets = FEATURE_PRESETS
    if config.features not in presets:
        raise ValueError(
            f"Unknown feature preset '{config.features}'. "
            f"Valid presets: {list(presets.keys())}"
        )
    preset = presets[config.features]
    # Create a copy and apply overrides
    import copy
    cfg = copy.copy(config)
    cfg.rot_size = preset["rot_size"]
    cfg.svd_rank = preset["svd_rank"]
    # Store feature flags as attributes on the config for use in benchmark_method
    cfg._smoothquant = preset["smoothquant"]
    cfg._smoothrot = preset["smoothrot"]
    cfg._use_permuquant = preset["use_permuquant"]
    return cfg


def _adapt_presets_for_calibration(
    presets: dict[str, dict[str, Any]],
    calibration: dict | None,
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Adapt feature presets to match the calibration's rot_size.

    When the calibration Hessian was collected in a rotated space
    (``hessian_rotated=True``), presets that use rotation must match
    the calibration's ``rot_size``.  Presets with ``rot_size=0``
    (no rotation) are incompatible with a rotated Hessian and are
    flagged for skipping.

    Returns:
        ``(adapted_presets, skip_for_gptq)`` — the adapted presets dict,
        and a set of feature names that should be skipped for GPTQ/LDLQ
        (because they can't use the rotated Hessian).
    """
    if calibration is None:
        return presets, set()

    cal_meta = calibration.get("metadata", {})
    cal_rotated = bool(cal_meta.get("hessian_rotated", False))
    cal_rot_size = int(cal_meta.get("rot_size", 0))

    if not cal_rotated:
        return presets, set()

    # Calibration is rotated — adapt presets
    adapted = {}
    skip_for_gptq: set[str] = set()
    for name, preset in presets.items():
        p = dict(preset)  # shallow copy
        preset_rot = p.get("rot_size", 0)
        if preset_rot == 0:
            # No-rotation preset can't use a rotated Hessian
            skip_for_gptq.add(name)
        else:
            # Rotation preset — match the calibration's rot_size
            p["rot_size"] = cal_rot_size
        adapted[name] = p

    return adapted, skip_for_gptq


def _override_preset_rot_size(
    presets: dict[str, dict[str, Any]],
    rot_size: int,
) -> dict[str, dict[str, Any]]:
    """Override rot_size in presets that use rotation.

    Presets with ``rot_size=0`` (no rotation) are left unchanged.
    Presets with ``rot_size > 0`` are updated to use the given value.

    Args:
        presets: Feature presets dict.
        rot_size: The user-specified rot_size to apply.

    Returns:
        New presets dict with updated rot_sizes.
    """
    if rot_size <= 0:
        return presets
    adapted = {}
    for name, preset in presets.items():
        p = dict(preset)
        if p.get("rot_size", 0) > 0:
            p["rot_size"] = rot_size
        adapted[name] = p
    return adapted


def _should_skip(name: str, skip_patterns: list[str]) -> bool:
    """Return True if a layer name matches any skip pattern."""
    name_lower = name.lower()
    return any(pattern in name_lower for pattern in skip_patterns)


def _dequantize(
    quantized_W: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor | None,
) -> torch.Tensor:
    """Dequantize: recover float weights from quantized integers + scales."""
    W_f = quantized_W.float()
    s_f = scales.float()
    if zero_points is not None:
        return (W_f - zero_points.float()) * s_f
    return W_f * s_f


def _compute_layer_metrics(
    W_orig: torch.Tensor,
    W_dequant_orig: torch.Tensor,
    activations: torch.Tensor | None,
) -> dict[str, float | None]:
    """Compute per-layer metrics in original weight space.

    Returns dict with weight_mse, weight_max_err, snr_db, output_mse.
    All MSE values are relative: ||W - W_deq||² / ||W||².
    """
    err = W_orig - W_dequant_orig
    w_norm_sq = W_orig.double().pow(2).sum().item()
    err_norm_sq = err.double().pow(2).sum().item()

    if w_norm_sq < 1e-30:
        weight_mse = 0.0
        snr_db = float("inf")
    else:
        weight_mse = err_norm_sq / w_norm_sq
        snr_db = 10.0 * math.log10(w_norm_sq / max(err_norm_sq, 1e-30))

    w_max = W_orig.abs().max().item()
    if w_max < 1e-30:
        weight_max_err = 0.0
    else:
        weight_max_err = err.abs().max().item() / w_max

    output_mse = None
    if activations is not None:
        Y_orig = activations @ W_orig.T
        Y_deq = activations @ W_dequant_orig.T
        y_norm_sq = Y_orig.double().pow(2).sum().item()
        y_err_sq = (Y_orig - Y_deq).double().pow(2).sum().item()
        if y_norm_sq < 1e-30:
            output_mse = 0.0
        else:
            output_mse = y_err_sq / y_norm_sq

    return {
        "weight_mse": weight_mse,
        "weight_max_err": weight_max_err,
        "snr_db": snr_db,
        "output_mse": output_mse,
    }


def _inverse_rotate(W_rotated: torch.Tensor, rot_size: int, orig_in_features: int) -> torch.Tensor:
    """Apply inverse Hadamard rotation (self-inverse) and remove padding.

    The Hadamard matrix H is self-inverse (H @ H = I), so the inverse of
    W_rot = W @ H^T is W = W_rot @ H^T.  But since H = H^T for regular
    Hadamard, the forward and inverse are the same operation.
    """
    from .rotation import _is_power_of_four, make_hadamard_sylvester
    if _is_power_of_four(rot_size):
        H = get_hadamard(rot_size, dtype=torch.float32, device=str(W_rotated.device))
    else:
        H = make_hadamard_sylvester(rot_size, dtype=torch.float32, device=str(W_rotated.device))
    out_features, in_features_padded = W_rotated.shape
    n_groups = in_features_padded // rot_size
    W_grouped = W_rotated.reshape(out_features, n_groups, rot_size)
    W_orig = (W_grouped @ H).reshape(out_features, in_features_padded)
    return W_orig[:, :orig_in_features]


# ── Core benchmark functions ─────────────────────────────────────────────────


def benchmark_method(
    weights: dict[str, torch.Tensor],
    config: BenchmarkConfig,
    *,
    calibration: dict | None = None,
    activations: dict[str, torch.Tensor] | None = None,
    progress_callback: Any | None = None,
    presets: dict[str, dict[str, Any]] | None = None,
) -> MethodBenchmarkResult:
    """Run one method config on a set of weights.

    Iterates over all 2D tensors in ``weights``, applies the feature
    pipeline (rotation, smoothquant, etc.), quantizes with the
    specified method, dequantizes, and computes MSE in *original*
    weight space via the inverse-transform chain.

    Args:
        weights: State dict-like mapping of layer names to weight tensors.
        config: Benchmark configuration.
        calibration: Optional calibration data (required for GPTQ).
        activations: Optional dict of {layer_name: [tokens, in_features]}
            tensors for output MSE computation.
        progress_callback: Optional callable ``(layer_name, layer_idx, total)``
            invoked after each layer is processed.

    Returns:
        :class:`MethodBenchmarkResult` with per-layer metrics.

    Raises:
        ValueError: For invalid configs (unknown method, invalid int_bits,
            missing calibration for GPTQ).
    """
    # ── Validate config ──
    if config.method not in ("rtn", "gptq", "ldlq"):
        raise ValueError(f"Unknown method '{config.method}'. Must be 'rtn', 'gptq', or 'ldlq'.")
    if config.int_bits not in (4, 8):
        raise ValueError(f"int_bits must be 4 or 8, got {config.int_bits}")
    if config.method == "gptq" and calibration is None:
        raise ValueError("GPTQ requires calibration data")

    # ── Apply feature preset ──
    cfg = _apply_feature_preset(config, presets=presets)
    smoothquant = cfg._smoothquant
    smoothrot = cfg._smoothrot
    use_permuquant = cfg._use_permuquant
    rot_size = cfg.rot_size
    svd_rank = cfg.svd_rank
    is_int8 = cfg.int_bits == 8

    # ── Seed ──
    if cfg.seed >= 0:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

    # ── SmoothRot W4 safeguard ──
    if smoothrot and cfg.int_bits == 4 and not cfg.force_smoothrot_w4:
        logger.debug(
            "SmoothRot auto-disabled for INT4; continuing as plain ConvRot."
        )
        smoothrot = False
        # Fall through to ConvRot (rot_size > 0 path)

    # ── SmoothRot: detect FFN pairs ──
    smoothrot_pairs: dict = {}
    smoothrot_factors: dict[str, torch.Tensor] = {}
    if smoothrot:
        smoothrot_pairs = detect_ffn_pairs(weights)
        if smoothrot_pairs and calibration is not None:
            # We need to compute smoothing factors for down-projections.
            # For synthetic calibration, this is done per-layer below.
            pass

    # ── Calibration name map ──
    name_map: dict[str, str] = {}
    if calibration is not None:
        cal_keys = list(calibration.get("hessians", {}).keys())
        # Build name map: state_dict key → calibration key
        for key in weights:
            if key.endswith(".weight"):
                module_name = key[: -len(".weight")]
                if module_name in cal_keys:
                    name_map[key] = module_name

    skip_patterns = cfg.skip_patterns if cfg.skip_patterns is not None else DEFAULT_SKIP_PATTERNS

    # Pre-count quantizable layers for progress reporting
    total_layers = sum(
        1 for _name, _tensor in weights.items()
        if not _should_skip(_name, skip_patterns) and _tensor.dim() == 2
    )

    layer_results: list[LayerBenchmarkResult] = []
    t0 = time.monotonic()
    layer_idx = 0

    for name, tensor in weights.items():
        # Skip non-matching layers
        if _should_skip(name, skip_patterns):
            continue
        if tensor.dim() != 2:
            continue

        W_orig = tensor.float().clone()  # clone to prevent in-place mutation by PermuQuant etc.
        out_features, orig_in_features = W_orig.shape
        layer_acts = activations.get(name) if activations else None

        # ── Step 1: SVD ──
        svd_result = None
        W_work = W_orig.clone()  # always clone so PermuQuant/quantize don't mutate W_orig
        if svd_rank > 0:
            try:
                svd_result = decompose_weight(W_orig, rank=svd_rank)
                W_work = svd_result.residual
            except Exception as e:
                logger.warning("SVD failed for %s: %s", name, e)
                # W_work is already cloned above

        # ── Step 2: Rotation + smoothing ──
        smoothing_factors: torch.Tensor | None = None
        smooth_source: str | None = None
        is_smoothrot_down = smoothrot and name in smoothrot_pairs

        if is_smoothrot_down:
            # SmoothRot: smooth first, then rotate
            # Compute smoothing factors for the down-projection
            layer_name = name_map.get(name)
            act_amax = None
            if layer_name is not None and calibration is not None:
                from .calibration_io import get_per_channel_amax, get_hessian_diag
                act_amax = get_per_channel_amax(calibration, layer_name, orig_in_features)

            if act_amax is not None:
                if act_amax.shape[0] < W_work.shape[1]:
                    act_amax = F.pad(act_amax, (0, W_work.shape[1] - act_amax.shape[0]))
                smoothing_factors = compute_smoothing_factors(
                    act_amax, W_work, alpha=cfg.smooth_alpha
                )
                smooth_source = "smoothrot"
            elif calibration is not None and layer_name is not None:
                from .calibration_io import get_hessian_diag
                h_diag = get_hessian_diag(calibration, layer_name, W_work.shape[1])
                if h_diag is not None:
                    if h_diag.shape[0] < W_work.shape[1]:
                        h_diag = F.pad(h_diag, (0, W_work.shape[1] - h_diag.shape[0]))
                    n_samples = calibration.get("metadata", {}).get("num_samples", 128)
                    smoothing_factors = compute_smoothing_from_hessian_diag(
                        h_diag, W_work, alpha=cfg.smooth_alpha,
                        num_calibration_samples=n_samples,
                    )
                    smooth_source = "smoothrot"
                else:
                    # No hessian diag — skip SmoothRot for this layer
                    is_smoothrot_down = False
                    smooth_source = None
            else:
                # No calibration — skip SmoothRot entirely
                is_smoothrot_down = False
                smooth_source = None

            if is_smoothrot_down and smoothing_factors is not None:
                W_smooth = apply_smoothing_to_weight(W_work, smoothing_factors)
                W_work = rotate_weights(W_smooth, rot_size)
            elif rot_size > 0:
                # Fallback to plain ConvRot
                W_work = rotate_weights(W_work, rot_size)
        elif rot_size > 0:
            # Plain ConvRot
            W_work = rotate_weights(W_work, rot_size)

        # ── Step 3: PermuQuant ──
        perm_applied = False
        perm_orig: torch.Tensor | None = None
        if use_permuquant:
            W_pq = W_work[:, :orig_in_features]
            actual_group_size = W_pq.shape[1] if cfg.int_bits == 4 else cfg.perm_group_size
            perm, alpha, accepted = sweep_alpha(
                W_pq, act_mu2=None, group_size=actual_group_size,
                tau=cfg.tau, int_bits=cfg.int_bits,
            )
            if accepted:
                W_work[:, :orig_in_features] = W_pq[:, perm]
                perm_orig = perm
                perm_applied = True

        # ── Step 4: SmoothQuant (standalone, not smoothrot) ──
        if smoothquant and not is_smoothrot_down:
            layer_name = name_map.get(name)
            act_amax = None
            if layer_name is not None and calibration is not None:
                from .calibration_io import get_per_channel_amax, get_hessian_diag
                act_amax = get_per_channel_amax(calibration, layer_name, orig_in_features)

            if act_amax is not None:
                if act_amax.shape[0] < W_work.shape[1]:
                    act_amax = F.pad(act_amax, (0, W_work.shape[1] - act_amax.shape[0]))
                if perm_applied and perm_orig is not None:
                    orig_f = min(orig_in_features, perm_orig.shape[0], act_amax.shape[0])
                    act_reordered = act_amax.clone()
                    act_reordered[:orig_f] = act_amax[:orig_f][perm_orig[:orig_f]]
                    act_amax = act_reordered
                smoothing_factors = compute_smoothing_factors(
                    act_amax, W_work, alpha=cfg.smooth_alpha
                )
                smooth_source = "per_channel_amax"
            elif calibration is not None and layer_name is not None:
                from .calibration_io import get_hessian_diag
                h_diag = get_hessian_diag(calibration, layer_name, W_work.shape[1])
                if h_diag is not None:
                    if perm_applied and perm_orig is not None:
                        orig_f = min(orig_in_features, perm_orig.shape[0], h_diag.shape[0])
                        h_reordered = h_diag.clone()
                        h_reordered[:orig_f] = h_diag[:orig_f][perm_orig[:orig_f]]
                        h_diag = h_reordered
                    if h_diag.shape[0] < W_work.shape[1]:
                        h_diag = F.pad(h_diag, (0, W_work.shape[1] - h_diag.shape[0]))
                    n_samples = calibration.get("metadata", {}).get("num_samples", 128)
                    smoothing_factors = compute_smoothing_from_hessian_diag(
                        h_diag, W_work, alpha=cfg.smooth_alpha,
                        num_calibration_samples=n_samples,
                    )
                    smooth_source = "hessian_diag"
                else:
                    smoothing_factors = compute_smoothing_weight_only(W_work)
                    smooth_source = "weight_only"
            else:
                smoothing_factors = compute_smoothing_weight_only(W_work)
                smooth_source = "weight_only"

            if smoothing_factors is not None:
                W_work = apply_smoothing_to_weight(W_work, smoothing_factors)

        # ── Step 5: Quantize ──
        quantized_W: torch.Tensor
        scales: torch.Tensor
        zero_points: torch.Tensor | None = None
        method_used = ""
        fallbacks: list[str] = []

        if cfg.method == "gptq":
            layer_name = name_map.get(name)
            hessian = None
            if layer_name is not None and calibration is not None:
                from .calibration_io import get_hessian
                hessian = get_hessian(calibration, layer_name, tensor.shape)

            if hessian is not None:
                # Rotate Hessian if needed
                already_rotated = calibration.get("metadata", {}).get("hessian_rotated", False)
                if not already_rotated and rot_size > 0:
                    from .rotation import rotate_hessian
                    hessian = rotate_hessian(hessian, rot_size)

                # Re-permute Hessian for self-computed permutations
                if perm_applied and perm_orig is not None:
                    if is_dlr(hessian):
                        D_new, U_new = permute_dlr(hessian["D"], hessian["U"], perm_orig)
                        hessian = make_dlr_dict(D_new, U_new)
                    elif hessian.dim() == 2:
                        hessian = hessian[perm_orig][:, perm_orig]
                    else:
                        # Block-diagonal: un-permute W_work
                        inv_perm = perm_orig.argsort()
                        orig_f = min(orig_in_features, inv_perm.shape[0])
                        W_work[:, :orig_f] = W_work[:, :orig_f][:, inv_perm]
                        if smoothing_factors is not None:
                            s_orig = smoothing_factors[:orig_f].clone()
                            smoothing_factors[:orig_f] = s_orig[inv_perm]
                        perm_applied = False

                # Transform Hessian for SmoothQuant/SmoothRot
                if smoothing_factors is not None and smooth_source is not None:
                    s = smoothing_factors.float()
                    if is_dlr(hessian):
                        D_new, U_new = transform_dlr_for_smoothquant(
                            hessian["D"], hessian["U"], s
                        )
                        hessian = make_dlr_dict(D_new, U_new)
                    else:
                        orig_dtype = hessian.dtype
                        if hessian.dim() == 2:
                            s_outer = (s.unsqueeze(0) * s.unsqueeze(1)).clamp(min=1e-16)
                            hessian = (hessian.float() / s_outer).clamp(max=1e30).to(orig_dtype)
                        elif hessian.dim() == 3:
                            hessian_f = hessian.float().clone()
                            for b in range(hessian.shape[0]):
                                bs = hessian.shape[1]
                                col_start = b * bs
                                col_end = min(col_start + bs, s.shape[0])
                                actual_bs = col_end - col_start
                                s_block = s[col_start:col_end]
                                s_outer = (s_block.unsqueeze(0) * s_block.unsqueeze(1)).clamp(min=1e-16)
                                hessian_f[b, :actual_bs, :actual_bs] = (
                                    hessian[b, :actual_bs, :actual_bs].float() / s_outer
                                ).clamp(max=1e30)
                            hessian = hessian_f.to(orig_dtype)

                result = gptq_quantize_layer(
                    W_work, hessian,
                    block_size=cfg.gptq_block_size,
                    damping=cfg.damping,
                    int_bits=cfg.int_bits,
                    asymmetric=cfg.asymmetric,
                    hessian_method=cfg.hessian_method,
                )
                quantized_W = result.quantized_W
                scales = result.scales
                zero_points = result.zero_points
                method_used = result.method_used
                fallbacks = list(result.fallbacks)
            else:
                # RTN fallback for GPTQ without Hessian
                result = gptq_quantize_layer_rtn(
                    W_work, int_bits=cfg.int_bits, asymmetric=cfg.asymmetric,
                )
                quantized_W = result.quantized_W
                scales = result.scales
                zero_points = result.zero_points
                method_used = result.method_used
                fallbacks = list(result.fallbacks)
                fallbacks.append("rtn_fallback")

        elif cfg.method == "ldlq":
            # LDLQ: same Hessian loading as GPTQ when calibration is available.
            layer_name = name_map.get(name)
            ldlq_hessian = None
            if layer_name is not None and calibration is not None:
                from .calibration_io import get_hessian
                ldlq_hessian = get_hessian(calibration, layer_name, tensor.shape)

            if ldlq_hessian is not None:
                # Rotate Hessian if needed
                already_rotated = calibration.get("metadata", {}).get("hessian_rotated", False)
                if not already_rotated and rot_size > 0:
                    from .rotation import rotate_hessian
                    ldlq_hessian = rotate_hessian(ldlq_hessian, rot_size)

                # Re-permute Hessian for self-computed permutations
                if perm_applied and perm_orig is not None:
                    if is_dlr(ldlq_hessian):
                        D_new, U_new = permute_dlr(ldlq_hessian["D"], ldlq_hessian["U"], perm_orig)
                        ldlq_hessian = make_dlr_dict(D_new, U_new)
                    elif ldlq_hessian.dim() == 2:
                        ldlq_hessian = ldlq_hessian[perm_orig][:, perm_orig]
                    else:
                        inv_perm = perm_orig.argsort()
                        orig_f = min(orig_in_features, inv_perm.shape[0])
                        W_work[:, :orig_f] = W_work[:, :orig_f][:, inv_perm]
                        if smoothing_factors is not None:
                            s_orig = smoothing_factors[:orig_f].clone()
                            smoothing_factors[:orig_f] = s_orig[inv_perm]
                        perm_applied = False

                # Transform Hessian for SmoothQuant/SmoothRot
                if smoothing_factors is not None and smooth_source is not None:
                    s = smoothing_factors.float()
                    if is_dlr(ldlq_hessian):
                        D_new, U_new = transform_dlr_for_smoothquant(
                            ldlq_hessian["D"], ldlq_hessian["U"], s
                        )
                        ldlq_hessian = make_dlr_dict(D_new, U_new)
                    else:
                        orig_dtype = ldlq_hessian.dtype
                        if ldlq_hessian.dim() == 2:
                            s_outer = (s.unsqueeze(0) * s.unsqueeze(1)).clamp(min=1e-16)
                            ldlq_hessian = (ldlq_hessian.float() / s_outer).clamp(max=1e30).to(orig_dtype)
                        elif ldlq_hessian.dim() == 3:
                            hessian_f = ldlq_hessian.float().clone()
                            for b in range(ldlq_hessian.shape[0]):
                                bs_blk = ldlq_hessian.shape[1]
                                col_start = b * bs_blk
                                col_end = min(col_start + bs_blk, s.shape[0])
                                actual_bs = col_end - col_start
                                s_block = s[col_start:col_end]
                                s_outer = (s_block.unsqueeze(0) * s_block.unsqueeze(1)).clamp(min=1e-16)
                                hessian_f[b, :actual_bs, :actual_bs] = (
                                    ldlq_hessian[b, :actual_bs, :actual_bs].float() / s_outer
                                ).clamp(max=1e30)
                            ldlq_hessian = hessian_f.to(orig_dtype)

            result = ldlq_quantize_layer(
                W_work,
                hessian=ldlq_hessian,
                block_size=cfg.gptq_block_size,
                damping=cfg.damping,
                int_bits=cfg.int_bits,
                iterations=cfg.ldlq_iterations,
                greedy_passes=cfg.greedy_passes,
                rank_threshold=cfg.rank_threshold,
            )
            quantized_W = result.quantized_W
            scales = result.scales
            zero_points = result.zero_points
            method_used = result.method_used
            fallbacks = list(result.fallbacks)

        else:  # RTN
            if cfg.asymmetric:
                if is_int8:
                    scales, zero_points = calculate_scales_int8_asymmetric(W_work)
                    quantized_W = quantize_weights_int8_asymmetric(W_work, scales, zero_points)
                else:
                    in_feat = W_work.shape[1]
                    scales, zero_points = calculate_scales_asymmetric(W_work, in_feat)
                    quantized_W = quantize_weights_asymmetric(W_work, scales, zero_points, in_feat)
            else:
                if is_int8:
                    scales = calculate_scales_int8(W_work)
                    quantized_W = quantize_weights_int8(W_work, scales)
                else:
                    in_feat = W_work.shape[1]
                    scales = calculate_scales(W_work, in_feat)
                    quantized_W = quantize_weights(W_work, scales, in_feat)
            method_used = "rtn"

        # ── Step 6: Dequantize in working space ──
        W_deq_work = _dequantize(quantized_W, scales, zero_points)

        # ── Step 7: Inverse transform back to original weight space ──
        W_deq_orig = W_deq_work

        # 7a. Inverse SmoothQuant (if applied, not smoothrot)
        if smoothquant and not is_smoothrot_down and smoothing_factors is not None:
            W_deq_orig = W_deq_orig / smoothing_factors.unsqueeze(0).to(W_deq_orig.dtype)

        # 7b. Inverse PermuQuant (if applied)
        if perm_applied and perm_orig is not None:
            inv_perm = perm_orig.argsort()
            orig_f = min(orig_in_features, inv_perm.shape[0])
            W_deq_orig[:, :orig_f] = W_deq_orig[:, :orig_f][:, inv_perm]

        # 7c. Inverse rotation (if applied)
        if rot_size > 0:
            W_deq_orig = _inverse_rotate(W_deq_orig.float(), rot_size, orig_in_features)

        # 7d. Inverse SmoothRot smoothing (if applied)
        if is_smoothrot_down and smoothing_factors is not None:
            # After inverse rotation, divide by smoothing factors
            W_deq_orig = W_deq_orig / smoothing_factors.unsqueeze(0).to(W_deq_orig.dtype)

        # 7e. Add SVD low-rank (if applied)
        if svd_result is not None:
            W_deq_orig = svd_result.L1.float() @ svd_result.L2.float() + W_deq_orig

        # ── Step 8: Compute metrics against original weight ──
        metrics = _compute_layer_metrics(W_orig, W_deq_orig.float(), layer_acts)

        layer_results.append(LayerBenchmarkResult(
            name=name,
            shape=(out_features, orig_in_features),
            weight_mse=metrics["weight_mse"],
            weight_max_err=metrics["weight_max_err"],
            output_mse=metrics["output_mse"],
            snr_db=metrics["snr_db"],
            method_used=method_used,
            fallbacks=fallbacks,
            smooth_source=smooth_source,
            permutation_applied=perm_applied,
            svd_rank=svd_result.L1.shape[1] if svd_result is not None else 0,
        ))

        layer_idx += 1
        if progress_callback is not None:
            progress_callback(name, layer_idx, total_layers)

    elapsed = time.monotonic() - t0

    # ── Aggregate metrics ──
    if not layer_results:
        return MethodBenchmarkResult(
            config=config,
            layers=[],
            mse_mean=float("nan"),
            mse_p95=float("nan"),
            max_err=float("nan"),
            output_mse_mean=None,
            elapsed_seconds=elapsed,
            error="No quantizable layers found",
        )

    mse_values = [lr.weight_mse for lr in layer_results]
    mse_mean = sum(mse_values) / len(mse_values)
    sorted_mse = sorted(mse_values)
    p95_idx = int(math.ceil(0.95 * len(sorted_mse))) - 1
    mse_p95 = sorted_mse[max(0, p95_idx)]
    max_err = max(lr.weight_max_err for lr in layer_results)

    output_mse_values = [lr.output_mse for lr in layer_results if lr.output_mse is not None]
    output_mse_mean = sum(output_mse_values) / len(output_mse_values) if output_mse_values else None

    return MethodBenchmarkResult(
        config=config,
        layers=layer_results,
        mse_mean=mse_mean,
        mse_p95=mse_p95,
        max_err=max_err,
        output_mse_mean=output_mse_mean,
        elapsed_seconds=elapsed,
        error=None,
    )


def benchmark_matrix(
    weights: dict[str, torch.Tensor],
    *,
    calibration: dict | None = None,
    activations: dict[str, torch.Tensor] | None = None,
    methods: list[str] | None = None,
    int_bits: list[int] | None = None,
    features: list[str] | None = None,
    feature_presets: dict[str, dict[str, Any]] | None = None,
    progress_callback: Any | None = None,
) -> BenchmarkReport:
    """Run all valid (method, bits, features) combos on the same weights.

    Skips invalid combos gracefully: if a combo raises ``ValueError``,
    records the error in ``MethodBenchmarkResult.error`` and continues.

    When calibration data is available and has ``hessian_rotated=True``,
    the calibration's ``rot_size`` is compared against each feature
    preset's ``rot_size``.  A mismatch means the Hessian is in the wrong
    rotation space, producing **incorrect** GPTQ/LDLQ results.  These
    combos are skipped with a descriptive error message.

    Args:
        weights: State dict-like mapping of layer names to weight tensors.
        calibration: Optional calibration data.
        activations: Optional dict of per-layer activations.
        methods: Methods to test (default: all three).
        int_bits: Bit-widths to test (default: [4, 8]).
        features: Feature presets to test (default: all valid combos).
        feature_presets: Override for :data:`FEATURE_PRESETS`.  When
            ``None``, uses the module-level defaults.  Pass a modified
            dict to override ``rot_size`` or other preset values.
        progress_callback: Optional callable ``(method, bits, features, combo_idx, total_combos)``
            invoked before each combo starts.

    Returns:
        :class:`BenchmarkReport` sorted by mse_mean.
    """
    if methods is None:
        methods = ["rtn", "gptq", "ldlq"]
    if int_bits is None:
        int_bits = [4, 8]
    if features is None:
        features = list(FEATURE_PRESETS.keys())
    if feature_presets is None:
        feature_presets = FEATURE_PRESETS

    # ── Adapt presets to match calibration rot_size ──
    # When calibration is rotated, presets with rot_size > 0 must match
    # the calibration's rot_size.  Presets with rot_size=0 are skipped
    # for GPTQ/LDLQ (can't use a rotated Hessian without rotation).
    adapted_presets, skip_gptq_feats = _adapt_presets_for_calibration(
        feature_presets, calibration,
    )

    # Build valid combos
    combos: list[tuple[str, int, str]] = []
    for method in methods:
        for bits in int_bits:
            for feat in features:
                # GPTQ without calibration is invalid
                if method == "gptq" and calibration is None:
                    continue
                # SmoothRot + LDLQ is excluded (covered by ConvRot)
                preset = adapted_presets.get(feat, {})
                if preset.get("smoothrot") and method == "ldlq":
                    continue
                # Skip GPTQ/LDLQ for presets incompatible with calibration
                if method in ("gptq", "ldlq") and feat in skip_gptq_feats:
                    continue
                combos.append((method, bits, feat))

    # Collect layer shapes from first quantizable layer
    skip_patterns = DEFAULT_SKIP_PATTERNS
    layer_shapes: dict[str, tuple] = {}
    for name, tensor in weights.items():
        if not _should_skip(name, skip_patterns) and tensor.dim() == 2:
            layer_shapes[name] = tuple(tensor.shape)

    total_combos = len(combos)
    results: list[MethodBenchmarkResult] = []
    for combo_idx, (method, bits, feat) in enumerate(combos, 1):
        config = BenchmarkConfig(
            method=method,
            int_bits=bits,
            features=feat,
        )
        try:
            result = benchmark_method(
                weights, config,
                calibration=calibration,
                activations=activations,
                presets=adapted_presets,
            )
        except Exception as e:
            result = MethodBenchmarkResult(
                config=config,
                layers=[],
                mse_mean=float("inf"),
                mse_p95=float("inf"),
                max_err=float("inf"),
                output_mse_mean=None,
                elapsed_seconds=0.0,
                error=str(e),
            )
        results.append(result)

        if progress_callback is not None:
            progress_callback(result, combo_idx, total_combos)

    # Sort by mse_mean (errors sort last)
    results.sort(key=lambda r: (r.error is not None, r.mse_mean))

    return BenchmarkReport(
        model_path=None,
        calibration_path=None,
        num_layers=len(layer_shapes),
        layer_shapes=layer_shapes,
        results=results,
    )


# ── Synthetic data generators ────────────────────────────────────────────────


def make_synthetic_model(
    num_layers: int = 8,
    shape: tuple[int, int] = (64, 256),
    *,
    outlier_channels: int = 0,
    outlier_scale: float = 50.0,
    low_rank: int = 0,
    seed: int = 42,
    ff_pairs: bool = False,
) -> dict[str, torch.Tensor]:
    """Generate synthetic model weights.

    Args:
        num_layers: Number of layers to generate.
        shape: ``(out_features, in_features)`` for each layer.
        outlier_channels: Number of outlier channels per layer.
        outlier_scale: Scale factor for outlier channels.
        low_rank: If > 0, generate low-rank weights with this rank.
        seed: Random seed.
        ff_pairs: Generate FFN-like pairs for SmoothRot testing.

    Returns:
        Dict of ``{"layers.0.q_proj.weight": tensor, ...}``.

    With ``ff_pairs=True``, generates PAIRS of tensors per block so that
    ``detect_ffn_pairs()`` can match them.
    """
    g = torch.Generator().manual_seed(seed)
    out_features, in_features = shape

    if ff_pairs:
        weights: dict[str, torch.Tensor] = {}
        for i in range(num_layers):
            # Up-projection: (out, in)
            W_up = torch.randn(out_features, in_features, generator=g) / math.sqrt(in_features)
            # Down-projection: (in, out) — transposed shape
            W_down = torch.randn(in_features, out_features, generator=g) / math.sqrt(out_features)
            weights[f"double_blocks.{i}.img_mlp.0.weight"] = W_up
            weights[f"double_blocks.{i}.img_mlp.2.weight"] = W_down
        return weights

    weights = {}
    for i in range(num_layers):
        if low_rank > 0:
            r = min(low_rank, min(out_features, in_features))
            A = torch.randn(out_features, r, generator=g) / math.sqrt(r)
            B = torch.randn(r, in_features, generator=g) / math.sqrt(in_features)
            W = A @ B
        else:
            W = torch.randn(out_features, in_features, generator=g) / math.sqrt(in_features)

        if outlier_channels > 0:
            chans = torch.randint(0, in_features, (outlier_channels,), generator=g)
            scale = outlier_scale / math.sqrt(in_features)
            W[:, chans] += torch.randn(out_features, outlier_channels, generator=g) * scale

        weights[f"layers.{i}.q_proj.weight"] = W

    return weights


def make_synthetic_calibration(
    layer_names: list[str],
    in_features: int,
    *,
    block_size: int | None = None,
    num_samples: int = 128,
    with_amax: bool = True,
    with_permutations: bool = False,
    seed: int = 42,
) -> dict:
    """Generate synthetic calibration data.

    Returns a dict matching the calibration .pt format::

        {
            "hessians": {name: tensor, ...},
            "shapes": {name: [in, in], ...},
            "layer_types": {name: "linear", ...},
            "metadata": {"num_samples": 128, ...},
            "amax_per_channel": {name: tensor, ...},  # if with_amax=True
            "permuquant": {name: tensor, ...},          # if with_permutations=True
        }

    Args:
        layer_names: Layer names (without .weight suffix) to generate data for.
        in_features: Number of input features.
        block_size: If not None, generate block-diagonal Hessians.
        num_samples: Number of calibration samples.
        with_amax: Include per-channel activation amax.
        with_permutations: Include channel permutations.
        seed: Random seed.
    """
    g = torch.Generator().manual_seed(seed)
    hessians: dict[str, torch.Tensor] = {}
    shapes: dict[str, list[int]] = {}
    layer_types: dict[str, str] = {}

    for name in layer_names:
        # Strip .weight suffix if present
        cal_name = name[:-len(".weight")] if name.endswith(".weight") else name

        if block_size is not None:
            num_blocks = (in_features + block_size - 1) // block_size
            blocks = []
            for _ in range(num_blocks):
                X = torch.randn(num_samples, block_size, generator=g)
                blocks.append(X.T @ X / float(num_samples))
            hessians[cal_name] = torch.stack(blocks)
        else:
            X = torch.randn(num_samples, in_features, generator=g)
            hessians[cal_name] = X.T @ X / float(num_samples)

        shapes[cal_name] = [in_features, in_features]
        layer_types[cal_name] = "linear"

    data: dict[str, Any] = {
        "hessians": hessians,
        "shapes": shapes,
        "layer_types": layer_types,
        "metadata": {"num_samples": num_samples},
    }

    if with_amax:
        amax_per_channel: dict[str, torch.Tensor] = {}
        for name in layer_names:
            cal_name = name[:-len(".weight")] if name.endswith(".weight") else name
            amax_per_channel[cal_name] = torch.rand(in_features, generator=g).clamp(min=0.1) * 10.0
        data["amax_per_channel"] = amax_per_channel

    if with_permutations:
        perms: dict[str, torch.Tensor] = {}
        for name in layer_names:
            cal_name = name[:-len(".weight")] if name.endswith(".weight") else name
            perms[cal_name] = torch.randperm(in_features)
        data["permuquant"] = perms

    return data


# ── Valid combos ─────────────────────────────────────────────────────────────

def _build_valid_combos() -> list[tuple[str, int, str]]:
    """Build list of all valid (method, bits, features) tuples."""
    combos = []
    for method in ("rtn", "gptq", "ldlq"):
        for bits in (4, 8):
            for feat in FEATURE_PRESETS:
                preset = FEATURE_PRESETS[feat]
                # SmoothRot + LDLQ is excluded
                if preset.get("smoothrot") and method == "ldlq":
                    continue
                combos.append((method, bits, feat))
    return combos


VALID_COMBOS: list[tuple[str, int, str]] = _build_valid_combos()
