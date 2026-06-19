"""Quantization pipeline for INT-Crush with PermuQuant and GPTQ.

Usage:
    python -m converter -i in.safetensors -o out/ --rot-size 256 --permuquant
    python -m converter -i in.safetensors -o out/ --rot-size 256 -c cal.pt --quant-method gptq
"""

import json
import os
import time

import torch
from safetensors.torch import load_file, save_file

from .log import logger
from .types import ProgressCallback, ProgressInfo, ProgressSummary, QuantizeConfig
from .rotation import rotate_weights, _is_power_of_two, rotate_hessian
from .dlr import (
    is_dlr, transform_dlr_for_smoothquant, permute_dlr, make_dlr_dict,
)
from .scales import (
    calculate_scales, quantize_weights,
    calculate_scales_int8, quantize_weights_int8,
    calculate_scales_asymmetric, quantize_weights_asymmetric,
    calculate_scales_int8_asymmetric, quantize_weights_int8_asymmetric,
)
from .config import (
    INT4_SCALE_DIVISOR, INT8_SCALE_DIVISOR, SCALE_MIN, SCALE_MAX, SCALE_DTYPE,
    SMOOTH_FACTOR_DTYPE, SMOOTHQUANT_AMAX_FLOOR, SMOOTHQUANT_HESSIAN_FLOOR, SANITIZE_CEIL,
    WEIGHT_OUTLIER_CLAMP, DEFAULT_SKIP_PATTERNS,
)
from .packing import pack_int4
from .permuquant import sweep_alpha
from .calibration_io import load_calibration, build_name_map, get_hessian, get_permutation, get_hessian_diag, get_per_channel_amax
from .gptq import gptq_quantize_layer, gptq_quantize_layer_rtn
from .ldlq import ldlq_quantize_layer
from .piso import compute_piso_scales_int8
from .rounding import _round_half_away_from_zero
from .smoothquant import (
    compute_smoothing_factors,
    apply_smoothing_to_weight,
    compute_smoothing_from_hessian_diag,
    compute_smoothing_weight_only,
)
from .smoothrot import detect_ffn_pairs, FFNPair
from .svd import decompose_weight


def _compute_smoothing_for_layer(
    name: str,
    name_map: dict[str, str],
    calibration: dict | None,
    W: torch.Tensor,
    orig_in_features: int,
    alpha: float,
) -> tuple[torch.Tensor | None, str]:
    """Compute SmoothQuant smoothing factors for a layer.

    Same logic as the main-loop SmoothQuant block, but without PermuQuant
    column reordering. Used by the PiSO pre-pass so PiSO scales are
    computed on the smoothed weight.

    Returns:
        (smoothing_factors, source) — [in_features] tensor or None;
        source is ``"per_channel_amax"``, ``"hessian_diag"``, or ``"weight_only"``.
    """
    layer_name = name_map.get(name)

    act_amax = None
    if layer_name is not None and calibration is not None:
        act_amax = get_per_channel_amax(calibration, layer_name, orig_in_features)

    if act_amax is not None:
        if act_amax.shape[0] < W.shape[1]:
            act_amax = torch.nn.functional.pad(
                act_amax, (0, W.shape[1] - act_amax.shape[0])
            )
        return compute_smoothing_factors(act_amax, W, alpha=alpha), "per_channel_amax"

    if calibration is not None and layer_name is not None:
        h_diag = get_hessian_diag(calibration, layer_name, W.shape[1])
        if h_diag is not None:
            if h_diag.shape[0] < W.shape[1]:
                h_diag = torch.nn.functional.pad(
                    h_diag, (0, W.shape[1] - h_diag.shape[0])
                )
            n_samples = calibration.get("metadata", {}).get("num_samples", 128)
            return (
                compute_smoothing_from_hessian_diag(
                    h_diag, W, alpha=alpha, num_calibration_samples=n_samples
                ),
                "hessian_diag",
            )

    return compute_smoothing_weight_only(W), "weight_only"


def _transform_hessian_for_smoothquant(
    hessian,
    smoothing_factors: torch.Tensor,
):
    """Transform Hessian for SmoothQuant: ``Ĥ = diag(1/s) H diag(1/s)``.

    For DLR, transforms factors directly (preserves structure).
    """
    s = smoothing_factors.float()

    # DLR Hessian: transform factors directly (preserves DLR structure).
    if is_dlr(hessian):
        D_new, U_new = transform_dlr_for_smoothquant(
            hessian["D"], hessian["U"], s
        )
        return make_dlr_dict(D_new, U_new)

    orig_dtype = hessian.dtype

    if hessian.dim() == 2:
        s_outer = (s.unsqueeze(0) * s.unsqueeze(1)).clamp(min=SMOOTHQUANT_HESSIAN_FLOOR)
        H_smooth = (hessian.float() / s_outer).clamp(max=SANITIZE_CEIL)
        return H_smooth.to(orig_dtype)

    if hessian.dim() == 3:
        num_blocks, bs, _ = hessian.shape
        H_out = hessian.float().clone()
        for b in range(num_blocks):
            col_start = b * bs
            col_end = min(col_start + bs, s.shape[0])
            actual_bs = col_end - col_start
            s_block = s[col_start:col_end]
            s_outer = (s_block.unsqueeze(0) * s_block.unsqueeze(1)).clamp(min=SMOOTHQUANT_HESSIAN_FLOOR)
            H_out[b, :actual_bs, :actual_bs] = (
                hessian[b, :actual_bs, :actual_bs].float() / s_outer
            ).clamp(max=SANITIZE_CEIL)
        return H_out.to(orig_dtype)

    return hessian


def should_skip(name: str, skip_patterns: list[str]) -> bool:
    """Return True if a layer name matches any skip pattern (case-insensitive).

    Skipped layers (embeddings, norms, output projections, etc.) are stored
    in the output safetensors without quantization.
    """
    name_lower = name.lower()
    return any(pattern in name_lower for pattern in skip_patterns)


def quantize_model(
    config: QuantizeConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Quantize a model from safetensors using INT-Crush + PermuQuant + GPTQ/LDLQ.

    Args:
        config: A :class:`QuantizeConfig` with all quantization parameters.
        progress_callback: Optional callable invoked after each quantized layer
            with a :class:`ProgressInfo` and once at the end with a
            :class:`ProgressSummary`.  Pass ``None`` (default) to disable.
            The callback is synchronous and called from the quantization thread.
            If it raises, the exception propagates and aborts quantization.
    """
    skip_patterns = config.skip_patterns
    if skip_patterns is None:
        skip_patterns = DEFAULT_SKIP_PATTERNS

    if config.exclude_patterns:
        skip_patterns = list(skip_patterns) + config.exclude_patterns
        logger.info("Excluding additional patterns: %s", config.exclude_patterns)

    if config.quant_method not in ("rtn", "gptq", "ldlq"):
        raise ValueError(f"quant_method must be 'rtn', 'gptq', or 'ldlq', got '{config.quant_method}'")

    if config.quant_method == "gptq" and not config.calibration_path:
        raise ValueError("GPTQ requires --calibration path to a .pt file")

    if config.quant_method == "ldlq" and not config.calibration_path:
        import warnings
        warnings.warn(
            "LDLQ: no calibration data provided; will fall back to "
            "H = W^T W / M (weight-only). This can perform worse than RTN. "
            "Pass --calibration for best results.",
        )

    if config.rot_size > 0 and not _is_power_of_two(config.rot_size):
        raise ValueError(f"rot_size must be 0 (no rotation) or a power of 2, got {config.rot_size}")

    if config.perm_group_size < 32 or (config.perm_group_size & (config.perm_group_size - 1)) != 0:
        raise ValueError(f"perm_group_size must be a power of 2 >= 32, got {config.perm_group_size}")

    if config.quant_group_size < 32 or (config.quant_group_size & (config.quant_group_size - 1)) != 0:
        raise ValueError(f"quant_group_size must be a power of 2 >= 32, got {config.quant_group_size}")

    if config.int_bits not in (4, 8):
        raise ValueError(f"int_bits must be 4 or 8, got {config.int_bits}")

    if config.smoothquant and config.int_bits == 4:
        logger.warning(
            "SmoothQuant is designed for W8A8 (INT8) quantization. "
            "With INT4 weight-only quantization, the benefit is limited. "
            "Consider using --int-bits 8 with --smoothquant."
        )

    # SmoothRot W4 safeguard: auto-disable SmoothRot for INT4 unless overridden.
    # At W4, the per-channel dynamic range of the smoothed weight can exceed
    # INT4's 16-level representable range, degrading quality.
    if config.smoothrot and config.int_bits == 4 and not config.force_smoothrot_w4:
        logger.warning(
            "SmoothRot is designed for W8A8 quantization. At W4 (INT4), "
            "the per-channel dynamic range of the smoothed weight can exceed "
            "INT4's 16-level representable range, DEGRADING quality. "
            "SmoothRot is automatically disabled. Use --force-smoothrot-w4 "
            "to override (not recommended)."
        )
        config.smoothrot = False

    # When smoothquant + rot_size > 0, automatically use smooth-then-rotate
    # order (the correct order per the SmoothRot paper).
    if config.smoothquant and config.rot_size > 0:
        config.smoothrot = True

    if config.smoothrot:
        if config.rot_size == 0:
            raise ValueError("SmoothRot requires rot_size > 0")
        if not config.smoothquant:
            config.smoothquant = True

    if config.svd_rank < 0:
        raise ValueError(f"svd_rank must be >= 0, got {config.svd_rank}")
    if config.svd_rank > 0:
        logger.info("SVD-absorbed enabled: rank=%d", config.svd_rank)

    is_int8 = config.int_bits == 8

    # Seed CPU (Mersenne Twister) and CUDA (Philox) RNGs for reproducibility.
    # Affects torch.svd_lowrank and other stochastic ops. seed=-1 disables.
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        logger.debug("RNG seeded with %d", config.seed)

    os.makedirs(config.output_dir, exist_ok=True)

    logger.info("Loading %s...", config.input_path)
    state_dict = load_file(config.input_path)
    logger.info("Loaded %d tensors", len(state_dict))

    # Load calibration data if GPTQ (or SmoothQuant needs it)
    calibration = None
    name_map = {}
    needs_calibration = config.quant_method in ("gptq", "ldlq") or config.smoothquant
    if needs_calibration and config.calibration_path:
        logger.info("Loading calibration from %s...", config.calibration_path)
        calibration = load_calibration(config.calibration_path)
        cal_keys = list(calibration["hessians"].keys())
        name_map = build_name_map(list(state_dict.keys()), cal_keys)
        logger.info("Matched %d/%d calibration layers to model layers", len(name_map), len(cal_keys))
    elif config.quant_method == "gptq" and not config.calibration_path:
        # GPTQ requires calibration; validated below
        pass

    # ── PiSO: precompute data-aware optimal scales ──
    piso_scales_dict: dict[str, torch.Tensor] = {}
    if config.piso_scales and calibration is not None and is_int8:
        # Count eligible layers first for progress reporting
        piso_eligible = [
            (name, tensor) for name, tensor in state_dict.items()
            if not should_skip(name, skip_patterns)
            and tensor.dim() == 2
            and name_map.get(name) is not None
            and get_hessian_diag(calibration, name_map.get(name, ""), tensor.shape[1]) is not None
        ]
        logger.info("PiSO: computing optimal scales for %d layers...", len(piso_eligible))
        piso_count = 0
        piso_start = time.monotonic()
        for idx, (name, tensor) in enumerate(piso_eligible, 1):
            layer_name = name_map[name]
            h_diag = get_hessian_diag(calibration, layer_name, tensor.shape[1])
            W = tensor.float()
            if config.rot_size > 0:
                W = rotate_weights(W, config.rot_size)

            # Apply SmoothQuant *before* PiSO so scales are computed on the
            # smoothed weight (the same weight GPTQ will quantize).
            if config.smoothquant:
                piso_sq, piso_sq_src = _compute_smoothing_for_layer(
                    name, name_map, calibration, W, tensor.shape[1], config.smooth_alpha
                )
                if piso_sq is not None:
                    W = apply_smoothing_to_weight(W, piso_sq)
                    # Transform Hessian diagonal: ĥ_j = h_j / s_j²
                    h_diag_raw = h_diag.float()
                    h_diag_smooth = h_diag_raw / piso_sq.float().pow(2).clamp(min=SMOOTHQUANT_HESSIAN_FLOOR)
                    if torch.isfinite(h_diag_smooth).all():
                        h_diag = h_diag_smooth
                        logger.debug("PiSO: SmoothQuant (%s) applied to %s", piso_sq_src, name)
                    else:
                        # Extreme smoothing factors caused non-finite Hessian
                        # diagonal — keep untransformed h_diag but still use
                        # smoothed weight (PiSO scales will be suboptimal but
                        # not catastrophic).
                        logger.warning(
                            "PiSO: SmoothQuant Hessian diagonal has non-finite "
                            "values for %s; using untransformed h_diag", name
                        )

            layer_start = time.monotonic()
            try:
                scales = compute_piso_scales_int8(W, h_diag)
                piso_scales_dict[name] = scales
                piso_count += 1
            except Exception as exc:
                logger.warning("PiSO failed for %s: %s", layer_name, exc)
            layer_elapsed = time.monotonic() - layer_start
            total_elapsed = time.monotonic() - piso_start
            avg = total_elapsed / idx
            remaining = avg * (len(piso_eligible) - idx)
            logger.info("  PiSO [%d/%d] %s %s (%.1fs, ETA %.0fs)",
                        idx, len(piso_eligible), name, list(tensor.shape),
                        layer_elapsed, remaining)
        piso_total = time.monotonic() - piso_start
        logger.info("PiSO: computed optimal scales for %d/%d layers in %.1fs",
                     piso_count, len(piso_eligible), piso_total)

    # ── SmoothRot: detect FFN pairs and precompute smoothing factors ──
    smoothrot_pairs: dict[str, FFNPair] = {}
    smoothrot_factors: dict[str, torch.Tensor] = {}
    smoothrot_alpha = config.smoothrot_alpha if config.smoothrot_alpha is not None else config.smooth_alpha
    if config.smoothrot:
        smoothrot_pairs = detect_ffn_pairs(state_dict)
        for down_name, pair in smoothrot_pairs.items():
            layer_name = name_map.get(down_name) if name_map else None
            W_down = state_dict[down_name].float()
            orig_in = W_down.shape[1]

            # Compute smoothing factors from activation statistics
            act_amax = None
            if layer_name is not None and calibration is not None:
                act_amax = get_per_channel_amax(calibration, layer_name, orig_in)

            if act_amax is not None:
                if act_amax.shape[0] < W_down.shape[1]:
                    act_amax = torch.nn.functional.pad(
                        act_amax, (0, W_down.shape[1] - act_amax.shape[0])
                    )
                s = compute_smoothing_factors(act_amax, W_down, alpha=smoothrot_alpha)
                smoothrot_factors[down_name] = s
                logger.debug("SmoothRot: %s smoothing from per_channel_amax (alpha=%.2f)",
                             down_name, smoothrot_alpha)
            elif calibration is not None and layer_name is not None:
                h_diag = get_hessian_diag(calibration, layer_name, W_down.shape[1])
                if h_diag is not None:
                    if h_diag.shape[0] < W_down.shape[1]:
                        h_diag = torch.nn.functional.pad(
                            h_diag, (0, W_down.shape[1] - h_diag.shape[0])
                        )
                    n_samples = calibration.get("metadata", {}).get("num_samples", 128)
                    s = compute_smoothing_from_hessian_diag(
                        h_diag, W_down, alpha=smoothrot_alpha,
                        num_calibration_samples=n_samples,
                    )
                    smoothrot_factors[down_name] = s
                    logger.debug("SmoothRot: %s smoothing from hessian_diag (alpha=%.2f)",
                                 down_name, smoothrot_alpha)
                else:
                    # No Hessian diagonal available — skip SmoothRot for this layer.
                    # Weight-only smoothing without calibration data amplifies
                    # quantization error at inference (1/s compensation hurts
                    # more than the smoothing helps).  Plain ConvRot is better.
                    logger.debug("SmoothRot: %s skipped (no hessian_diag, "
                                 "weight-only smoothing is harmful)", down_name)
            else:
                # No calibration data at all — skip SmoothRot entirely.
                # Weight-only smoothing is harmful without activation statistics.
                logger.debug("SmoothRot: %s skipped (no calibration data)", down_name)

        logger.info("SmoothRot: precomputed smoothing factors for %d FFN pairs (alpha=%.2f)",
                     len(smoothrot_factors), smoothrot_alpha)

    output_dict = {}
    meta_prefix = "int_crush"
    method_base = "int_crush_permuq" if config.use_permuquant else "int_crush"
    if config.smoothquant:
        method_base = f"{method_base}_smoothq"
    method = f"{method_base}_{config.quant_method}" if config.quant_method in ("gptq", "ldlq") else method_base
    # GPTQ INT4 always uses asymmetric internally (hardcoded in gptq.py);
    # GPTQ INT8 with --asymmetric also uses asymmetric; otherwise symmetric.
    # RTN follows config.asymmetric directly.
    gptq_int4_uses_asymmetric = (config.quant_method == "gptq" and not is_int8)
    scale_type_is_asymmetric = config.asymmetric or gptq_int4_uses_asymmetric
    metadata = {
        f"{meta_prefix}.format_version": "3" if not is_int8 else "1",
        f"{meta_prefix}.method": method,
        f"{meta_prefix}.rot_size": str(config.rot_size),
        f"{meta_prefix}.scale_type": "per_row_asymmetric" if scale_type_is_asymmetric else "per_row",
        f"{meta_prefix}.hadamard_type": "regular",
    }
    if not is_int8:
        metadata["int_crush.packing_order"] = "little"
    else:
        metadata["int_crush.packing_order"] = "native"
    if config.exclude_patterns:
        metadata[f"{meta_prefix}.exclude_patterns"] = ",".join(config.exclude_patterns)

    # Detect layers whose in_features will be padded by rotation.
    # ComfyUI reads in_channels from img_in.weight.shape[1], so padded
    # weights cause wrong model configuration. Store originals so the
    # loader can fix the model after creation.
    padded_layers = {}
    if config.rot_size > 0:
        for name, tensor in state_dict.items():
            if tensor.dim() == 2 and not should_skip(name, skip_patterns):
                orig_in = tensor.shape[1]
                if orig_in % config.rot_size != 0:
                    padded_in = config.rot_size * ((orig_in + config.rot_size - 1) // config.rot_size)
                    padded_layers[name] = str(orig_in)
                    logger.debug("Tracking padded layer: %s in_features %d -> %d", name, orig_in, padded_in)
        if padded_layers:
            metadata["int_crush.padded_layers"] = ";".join(
                f"{k}={v}" for k, v in padded_layers.items()
            )
    if config.use_permuquant:
        metadata["int_crush.permuquant"] = "true"
        metadata["int_crush.permuquant_tau"] = str(config.tau)
    if config.piso_scales and piso_scales_dict:
        metadata["int_crush.piso"] = "true"
        metadata["int_crush.piso_layers"] = str(len(piso_scales_dict))
    if config.quant_method == "gptq":
        metadata[f"{meta_prefix}.gptq_block_size"] = str(config.gptq_block_size)
        metadata[f"{meta_prefix}.damping_ratio"] = str(config.damping)
    elif config.quant_method == "ldlq":
        metadata[f"{meta_prefix}.ldlq_block_size"] = str(config.gptq_block_size)
        metadata[f"{meta_prefix}.damping_ratio"] = str(config.damping)
        metadata[f"{meta_prefix}.ldlq_iterations"] = str(config.ldlq_iterations)
        if config.greedy_passes > 0:
            metadata[f"{meta_prefix}.greedy_passes"] = str(config.greedy_passes)
            metadata[f"{meta_prefix}.rank_threshold"] = str(config.rank_threshold)
    if config.smoothquant:
        metadata["int_crush.smoothquant"] = "true"
        metadata["int_crush.smooth_alpha"] = str(config.smooth_alpha)
    if config.smoothrot:
        metadata["int_crush.smoothrot"] = "true"
        metadata["int_crush.smoothrot_alpha"] = str(smoothrot_alpha)
        metadata["int_crush.smoothrot_transform_order"] = "smooth_then_rotate"
        metadata["int_crush.smoothrot_pairs"] = str(len(smoothrot_pairs))
    if config.svd_rank > 0:
        metadata["int_crush.svd_absorbed"] = "true"
        metadata["int_crush.svd_rank"] = str(config.svd_rank)

    skipped = 0
    quantized = 0
    permuted = 0
    gptq_count = 0
    rtn_fallback = 0
    piso_count = 0
    smoothquant_count = 0
    svd_count = 0
    rot_size_mismatch_warned = False
    layer_reports: list[dict] = []

    # Pre-count quantizable layers for progress reporting.
    # Same skip logic as the main loop (should_skip + dim != 2).
    total_quantizable = sum(
        1 for _name, _tensor in state_dict.items()
        if not should_skip(_name, skip_patterns) and _tensor.dim() == 2
    )
    logger.debug("Quantizable layers: %d / %d total tensors", total_quantizable, len(state_dict))

    _loop_start: float | None = None
    if progress_callback is not None:
        _loop_start = time.monotonic()

    for name, tensor in state_dict.items():
        if should_skip(name, skip_patterns):
            output_dict[name] = tensor
            skipped += 1
            continue

        if tensor.dim() != 2:
            output_dict[name] = tensor
            skipped += 1
            continue

        logger.info("Quantizing %s (%s)...", name, tensor.shape)

        W = tensor.float()
        orig_in_features = W.shape[1]

        # 0.5 Sanitize outlier weights (corrupted data) before rotation/quantization.
        outlier_count = 0
        if WEIGHT_OUTLIER_CLAMP > 0:
            outlier_mask = W.abs() > WEIGHT_OUTLIER_CLAMP
            outlier_count = outlier_mask.sum().item()
            if outlier_count > 0:
                W = W.masked_fill(outlier_mask, 0.0)
                logger.warning(
                    "Zeroed %d outlier weight values in %s (|W| > %.0f)",
                    outlier_count, name, WEIGHT_OUTLIER_CLAMP,
                )

        # 0.7 SVD-absorbed: decompose W into low-rank FP16 + quantizable residual.
        svd_result = None
        if config.svd_rank > 0:
            svd_result = decompose_weight(W, rank=config.svd_rank)
            W = svd_result.residual  # downstream pipeline quantizes this
            svd_count += 1
            logger.debug("SVD: rank=%d for %s, residual norm=%.4f",
                         svd_result.L1.shape[1], name,
                         svd_result.residual.norm().item())

        # 1. Rotate weights (skip if rot_size == 0).
        #    Clone only when downstream mutates W_work in-place (GPTQ/LDLQ,
        #    PermuQuant). RTN with rot_size=0 and no PermuQuant is read-only.
        #
        #    SmoothRot: smooth FIRST (diag(s) @ W), rotate SECOND (W_smooth @ R^T).
        #    Correct order per SmoothRot paper — smoothing tames outliers before Hadamard.
        is_smoothrot_down = name in smoothrot_factors
        if is_smoothrot_down:
            # SmoothRot: smooth first, then rotate
            s = smoothrot_factors[name]
            W_smooth = apply_smoothing_to_weight(W, s)
            W_work = rotate_weights(W_smooth, config.rot_size)
            logger.debug("SmoothRot: smooth-then-rotate for %s", name)
        elif config.rot_size > 0:
            W_work = rotate_weights(W, config.rot_size)
        elif config.quant_method in ("gptq", "ldlq") or config.use_permuquant:
            W_work = W.clone()
        else:
            W_work = W  # RTN, no rotation, no PermuQuant — read-only

        # 2. PermuQuant: channel reordering.
        #    If calibration provides a permutation (already in Hessians), use it directly.
        #    Otherwise, compute from weight statistics.
        perm_applied = False
        perm_orig = None
        cal_perm = None
        if config.use_permuquant:
            layer_name = name_map.get(name) if name_map else None
            if layer_name is not None and calibration is not None:
                cal_perm = get_permutation(calibration, layer_name)

            if cal_perm is not None:
                # Apply calibration permutation to weight only (Hessian already permuted).
                orig_features = min(orig_in_features, cal_perm.shape[0])
                perm_slice = cal_perm[:orig_features]
                if (perm_slice.max() < orig_features
                        and perm_slice.min() >= 0
                        and perm_slice.unique().numel() == perm_slice.numel()):
                    W_orig = W_work[:, :orig_features]
                    W_work[:, :orig_features] = W_orig[:, perm_slice]
                    perm_orig = perm_slice
                    perm_applied = True
                    permuted += 1
                    logger.debug("PermuQuant applied (from calibration)")
                else:
                    logger.warning("Calibration permutation out of bounds, falling back to weight-only")

            if not perm_applied:
                W_orig = W_work[:, :orig_in_features]
                # For INT4, all methods (RTN, GPTQ, LDLQ) use per-row
                # quantization.  PermuQuant's acceptance check must use the
                # same granularity or it evaluates a different scheme.
                actual_group_size = W_orig.shape[1] if config.int_bits == 4 else config.perm_group_size
                perm, alpha, accepted = sweep_alpha(
                    W_orig, act_mu2=None, group_size=actual_group_size, tau=config.tau, int_bits=config.int_bits
                )
                if accepted:
                    W_work[:, :orig_in_features] = W_orig[:, perm]
                    perm_orig = perm
                    perm_applied = True
                    permuted += 1
                    logger.debug("PermuQuant applied (alpha=%.2f)", alpha)
                else:
                    logger.debug("PermuQuant skipped (no improvement)")

        # 2.5 SmoothQuant: per-channel smoothing.
        #    Skip for SmoothRot down-projections — already applied in step 1.
        #    Stored as {name}_smoothrot_factors for inference-side 1/s.
        smoothing_applied = False
        smoothing_factors = None
        smooth_source = None
        if is_smoothrot_down:
            # SmoothRot: smoothing already applied in step 1.
            # Store factors for inference-side 1/s application.
            smoothing_factors = smoothrot_factors[name]
            smoothing_applied = True
            smoothquant_count += 1
            smooth_source = "smoothrot"
        elif config.smoothquant and not config.smoothrot:
            # Old SmoothQuant path: only when SmoothRot is NOT active.
            # When --smoothrot is used, only FFN down-projections get smoothing.
            # Non-SmoothRot layers skip smoothing — weight-only smoothing on
            # rotated weights can amplify quantization error via inference-side 1/s.
            layer_name = name_map.get(name) if name_map else None

            # Determine activation statistics source
            act_amax = None
            if layer_name is not None and calibration is not None:
                act_amax = get_per_channel_amax(calibration, layer_name, orig_in_features)

            if act_amax is not None:
                # Primary path: per-channel amax from calibration
                # If weight was rotated/padded, pad act_amax to match
                if act_amax.shape[0] < W_work.shape[1]:
                    act_amax = torch.nn.functional.pad(
                        act_amax, (0, W_work.shape[1] - act_amax.shape[0])
                    )
                # If PermuQuant reordered columns, reorder act_amax to match
                if perm_applied and perm_orig is not None:
                    orig_features = min(orig_in_features, perm_orig.shape[0], act_amax.shape[0])
                    act_amax_reordered = act_amax.clone()
                    act_amax_reordered[:orig_features] = act_amax[:orig_features][perm_orig[:orig_features]]
                    act_amax = act_amax_reordered
                smoothing_factors = compute_smoothing_factors(
                    act_amax, W_work, alpha=config.smooth_alpha
                )
                smooth_source = "per_channel_amax"
            elif calibration is not None and layer_name is not None:
                # Fallback: approximate from Hessian diagonal
                h_diag = get_hessian_diag(calibration, layer_name, W_work.shape[1])
                if h_diag is not None:
                    # If PermuQuant reordered columns, reorder h_diag to match
                    if perm_applied and perm_orig is not None:
                        orig_features = min(orig_in_features, perm_orig.shape[0], h_diag.shape[0])
                        h_diag_reordered = h_diag.clone()
                        h_diag_reordered[:orig_features] = h_diag[:orig_features][perm_orig[:orig_features]]
                        h_diag = h_diag_reordered
                    # Pad to match rotated weight size
                    if h_diag.shape[0] < W_work.shape[1]:
                        h_diag = torch.nn.functional.pad(
                            h_diag, (0, W_work.shape[1] - h_diag.shape[0])
                        )
                    # Estimate n_samples from calibration metadata
                    n_samples = calibration.get("metadata", {}).get("num_samples", 128)
                    smoothing_factors = compute_smoothing_from_hessian_diag(
                        h_diag, W_work, alpha=config.smooth_alpha,
                        num_calibration_samples=n_samples,
                    )
                    smooth_source = "hessian_diag"
                    logger.debug("SmoothQuant: using Hessian-diagonal approximation for %s", name)
                else:
                    smoothing_factors = compute_smoothing_weight_only(W_work)
                    smooth_source = "weight_only"
                    logger.debug("SmoothQuant: no activation stats for %s, using weight-only", name)
            else:
                # No calibration at all: weight-only smoothing
                smoothing_factors = compute_smoothing_weight_only(W_work)
                smooth_source = "weight_only"
                logger.debug("SmoothQuant: no calibration, using weight-only smoothing for %s", name)

            if smoothing_factors is not None:
                W_work = apply_smoothing_to_weight(W_work, smoothing_factors)
                smoothing_applied = True
                smoothquant_count += 1
                logger.debug("SmoothQuant applied to %s (alpha=%.2f, source=%s)",
                             name, config.smooth_alpha, smooth_source)

        zero_points = None

        # 3. Quantize
        if config.quant_method == "gptq":
            # GPTQ: use Hessian-based error compensation
            layer_name = name_map.get(name)
            hessian = None
            if layer_name is not None:
                hessian = get_hessian(calibration, layer_name, tensor.shape)

            if hessian is not None:
                already_rotated = calibration.get("metadata", {}).get("hessian_rotated", False)
                cal_rot_size = calibration.get("metadata", {}).get("rot_size", 0)

                # If calibration was rotated with a different rot_size,
                # the Hessian is in the wrong space. Warn the user.
                if already_rotated and cal_rot_size and int(cal_rot_size) != config.rot_size:
                    if not rot_size_mismatch_warned:
                        logger.warning("Calibration rot_size=%s != converter rot_size=%d. "
                                      "Hessian is in the wrong rotation space. "
                                      "Re-run calibration with rot_size=%d for best results.",
                                      cal_rot_size, config.rot_size, config.rot_size)
                        rot_size_mismatch_warned = True

                if not already_rotated:
                    # Rotate: H_rot = R^T @ H @ R
                    if config.rot_size > 0:
                        hessian = rotate_hessian(hessian, config.rot_size)

                # Re-permute only for self-computed permutations.
                if perm_applied and cal_perm is None:
                    if is_dlr(hessian):
                        # DLR Hessian: permute D and U rows directly.
                        D_new, U_new = permute_dlr(
                            hessian["D"], hessian["U"], perm_orig
                        )
                        hessian = make_dlr_dict(D_new, U_new)
                    elif hessian.dim() == 2:
                        hessian = hessian[perm_orig][:, perm_orig]
                    else:
                        # Block-diagonal Hessian cannot be permuted in-place.
                        # Un-permute W_work to restore original column order
                        # so that W_work and hessian are in the same space.
                        inv_perm = perm_orig.argsort()
                        orig_features = min(orig_in_features, inv_perm.shape[0])
                        W_orig = W_work[:, :orig_features]
                        W_work[:, :orig_features] = W_orig[:, inv_perm]
                        # Also un-permute smoothing factors so they stay
                        # aligned with their columns.
                        if smoothing_applied and smoothing_factors is not None:
                            s_orig = smoothing_factors[:orig_features].clone()
                            smoothing_factors[:orig_features] = s_orig[inv_perm]
                        perm_applied = False
                        logger.debug(
                            "PermuQuant: un-permuted W_work for block-diagonal Hessian "
                            "for %s (%dD, shape %s)",
                            name, hessian.dim(), hessian.shape,
                        )

                # Transform Hessian for SmoothQuant/SmoothRot.
                #
                # SmoothRot path: Ĥ = R^T @ diag(1/s) @ H @ diag(1/s) @ R.
                #   → smooth Hessian first, then rotate.
                # Standalone SmoothQuant: Ĥ = diag(1/s) @ R^T @ H @ R @ diag(1/s).
                #   → rotate Hessian first (already done), then smooth.
                # Both produce H[i,j]/(s[i]·s[j]) but SmoothRot must compose
                # transforms in the correct mathematical order.
                if smoothing_applied and smoothing_factors is not None:
                    if is_smoothrot_down:
                        # SmoothRot: smooth first, then rotate
                        hessian = _transform_hessian_for_smoothquant(
                            hessian, smoothing_factors
                        )
                        if not already_rotated and config.rot_size > 0:
                            hessian = rotate_hessian(hessian, config.rot_size)
                        logger.debug("SmoothRot: smooth-then-rotate Hessian for %s", name)
                    else:
                        # Standalone: rotate first (already done above), then smooth
                        hessian = _transform_hessian_for_smoothquant(
                            hessian, smoothing_factors
                        )
                        logger.debug("SmoothQuant: transformed Hessian for %s", name)

                # Quantize full rotated weight; zero-energy padded columns handled by damping.
                result = gptq_quantize_layer(
                    W_work, hessian,
                    block_size=config.gptq_block_size,
                    damping=config.damping,
                    int_bits=config.int_bits,
                    asymmetric=config.asymmetric,
                    hessian_method=config.hessian_method,
                    piso_scales=piso_scales_dict.get(name),
                )
                quantized_W = result.quantized_W
                scales = result.scales
                zero_points = result.zero_points
                gptq_count += 1
            else:
                # Fallback to RTN if no calibration data or Hessian is missing/mismatched
                result = gptq_quantize_layer_rtn(
                    W_work, int_bits=config.int_bits,
                    asymmetric=config.asymmetric, clipping_ratios=config.clipping_ratios,
                    piso_scales=piso_scales_dict.get(name),
                )
                quantized_W = result.quantized_W
                scales = result.scales
                zero_points = result.zero_points
                rtn_fallback += 1
                logger.debug("No calibration data, using RTN fallback")
        elif config.quant_method == "ldlq":
            # LDLQ: adaptive rounding using the same Hessian as GPTQ.
            # When calibration data is available, uses H = E_x[xx^T].  Without calibration,
            # falls back to H = W^T W / M (weight-only, worse than RTN).
            layer_name = name_map.get(name) if name_map else None
            ldlq_hessian = None
            if layer_name is not None and calibration is not None:
                ldlq_hessian = get_hessian(calibration, layer_name, tensor.shape)

            if ldlq_hessian is not None:
                already_rotated = calibration.get("metadata", {}).get("hessian_rotated", False)
                cal_rot_size = calibration.get("metadata", {}).get("rot_size", 0)

                if already_rotated and cal_rot_size and int(cal_rot_size) != config.rot_size:
                    if not rot_size_mismatch_warned:
                        logger.warning(
                            "Calibration rot_size=%s != converter rot_size=%d. "
                            "Hessian is in the wrong rotation space.",
                            cal_rot_size, config.rot_size,
                        )
                        rot_size_mismatch_warned = True

                # Rotate Hessian if needed
                if not already_rotated and config.rot_size > 0:
                    ldlq_hessian = rotate_hessian(ldlq_hessian, config.rot_size)

                # Re-permute for self-computed permutations
                if perm_applied and cal_perm is None:
                    if is_dlr(ldlq_hessian):
                        # DLR Hessian: permute D and U rows directly.
                        D_new, U_new = permute_dlr(
                            ldlq_hessian["D"], ldlq_hessian["U"], perm_orig
                        )
                        ldlq_hessian = make_dlr_dict(D_new, U_new)
                    elif ldlq_hessian.dim() == 2:
                        ldlq_hessian = ldlq_hessian[perm_orig][:, perm_orig]
                    else:
                        inv_perm = perm_orig.argsort()
                        orig_features = min(orig_in_features, inv_perm.shape[0])
                        W_orig = W_work[:, :orig_features]
                        W_work[:, :orig_features] = W_orig[:, inv_perm]
                        if smoothing_applied and smoothing_factors is not None:
                            s_orig = smoothing_factors[:orig_features].clone()
                            smoothing_factors[:orig_features] = s_orig[inv_perm]
                        perm_applied = False
                        logger.debug(
                            "PermuQuant: un-permuted W_work for block-diagonal Hessian "
                            "for %s (%dD, shape %s)",
                            name, ldlq_hessian.dim(), ldlq_hessian.shape,
                        )

                # Transform Hessian for SmoothQuant/SmoothRot
                if smoothing_applied and smoothing_factors is not None:
                    if is_smoothrot_down:
                        ldlq_hessian = _transform_hessian_for_smoothquant(
                            ldlq_hessian, smoothing_factors
                        )
                        if not already_rotated and config.rot_size > 0:
                            ldlq_hessian = rotate_hessian(ldlq_hessian, config.rot_size)
                        logger.debug("SmoothRot: smooth-then-rotate Hessian for %s (LDLQ)", name)
                    else:
                        ldlq_hessian = _transform_hessian_for_smoothquant(
                            ldlq_hessian, smoothing_factors
                        )
                        logger.debug("SmoothQuant: transformed Hessian for %s (LDLQ)", name)

            result = ldlq_quantize_layer(
                W_work,
                hessian=ldlq_hessian,
                block_size=config.gptq_block_size,
                damping=config.damping,
                int_bits=config.int_bits,
                iterations=config.ldlq_iterations,
                greedy_passes=config.greedy_passes,
                rank_threshold=config.rank_threshold,
            )
            quantized_W = result.quantized_W
            scales = result.scales
            zero_points = result.zero_points
            gptq_count += 1
        else:
            # RTN: simple round-to-nearest
            result = None
            zero_points = None
            if config.asymmetric:
                if is_int8:
                    scales, zero_points = calculate_scales_int8_asymmetric(
                        W_work, clipping_ratios=config.clipping_ratios
                    )
                    quantized_W = quantize_weights_int8_asymmetric(
                        W_work, scales, zero_points
                    )
                else:
                    in_features = W_work.shape[1]
                    scales, zero_points = calculate_scales_asymmetric(
                        W_work, in_features, clipping_ratios=config.clipping_ratios
                    )
                    quantized_W = quantize_weights_asymmetric(
                        W_work, scales, zero_points, in_features
                    )
            else:
                if is_int8:
                    scales = calculate_scales_int8(W_work, clipping_ratios=config.clipping_ratios)
                    quantized_W = quantize_weights_int8(W_work, scales)
                else:
                    in_features = W_work.shape[1]
                    scales = calculate_scales(W_work, in_features, clipping_ratios=config.clipping_ratios)
                    quantized_W = quantize_weights(W_work, scales, in_features)

        # 4. Pack (INT4) or store directly (INT8)
        if is_int8:
            stored_weights = quantized_W
        else:
            stored_weights = pack_int4(quantized_W)

        # 5. Verify: dequantize and compare to working weights.
        #    Use float64 for MSE to prevent overflow when scales are very large.
        if zero_points is not None:
            dequant = (quantized_W.float() - zero_points.float()) * scales.float()
        else:
            dequant = quantized_W.float() * scales.float()
        diff = W_work - dequant
        del dequant
        max_err = diff.abs().max().item()
        # Compute MSE in float64 to prevent overflow when weights or scales
        # are very large.  (scale/2)² can exceed 3.4e38 (float32 max) for
        # scales > ~3.7e19, producing inf even when quantization quality is
        # reasonable. float64 has 53 bits of mantissa so precision
        # is not lost for normal-magnitude errors.
        mse = diff.double().pow(2).mean().item()
        del diff
        bad_scales = scales.isinf().any() or scales.isnan().any() or (scales == 0).any()
        logger.debug("MSE=%.6f  max_err=%.4f  scale_range=[%.6f, %.6f]  bad_scales=%s",
                     mse, max_err, scales.min().item(), scales.max().item(), bad_scales)
        if bad_scales:
            logger.warning("Bad scales detected in %s! inf=%s nan=%s zero=%s",
                           name, scales.isinf().any().item(), scales.isnan().any().item(), (scales == 0).any().item())
            n_bad = scales.isinf().sum() + scales.isnan().sum() + (scales == 0).sum()
            bad_mask = scales.isinf() | scales.isnan() | (scales == 0)
            # Recompute scales from W_work using the correct formula for the
            # quantization mode.
            fix_divisor = INT8_SCALE_DIVISOR if is_int8 else INT4_SCALE_DIVISOR
            sym_scales = (W_work.float().abs().amax(dim=1, keepdim=True) / fix_divisor).clamp(min=SCALE_MIN, max=SCALE_MAX).to(SCALE_DTYPE)
            if zero_points is not None:
                w_min = W_work.amin(dim=1, keepdim=True)
                w_max = W_work.amax(dim=1, keepdim=True)
                if is_int8:
                    asym_scales = ((w_max - w_min).float() / 255.0).clamp(min=SCALE_MIN, max=SCALE_MAX).to(SCALE_DTYPE)
                else:
                    asym_scales = ((w_max - w_min).float() / 15.0).clamp(min=SCALE_MIN, max=SCALE_MAX).to(SCALE_DTYPE)
                # Use asymmetric scale where it produced valid values; fall back
                # to symmetric otherwise (e.g. when W_work itself has NaN/Inf).
                asym_ok = bad_mask & torch.isfinite(asym_scales) & (asym_scales > 0)
                fix_scales = torch.where(asym_ok, asym_scales, sym_scales)
                scales = torch.where(bad_mask, fix_scales, scales)
                # Re-quantize affected rows with the repaired scales so the
                # stored (Q, scale) pair is consistent.
                clamp_min = -128 if is_int8 else -8
                clamp_max = 127 if is_int8 else 7
                bad_row_idx = bad_mask.squeeze(1).nonzero(as_tuple=True)[0]
                for r in bad_row_idx:
                    r_s = scales[r].to(W_work.dtype)
                    # Recompute zero-point for the repaired scale.
                    r_w_min = W_work[r].amin()
                    r_zp = (clamp_min - _round_half_away_from_zero(r_w_min / r_s)).clamp(clamp_min, clamp_max)
                    zero_points[r] = r_zp.to(zero_points.dtype)
                    quantized_W[r] = ((W_work[r] / r_s + r_zp.to(W_work.dtype)).round().clamp(clamp_min, clamp_max)).to(torch.int8)
            else:
                scales = torch.where(bad_mask, sym_scales, scales)
                # Re-quantize affected rows with the repaired scales.
                clamp_min = -128 if is_int8 else -8
                clamp_max = 127 if is_int8 else 7
                bad_row_idx = bad_mask.squeeze(1).nonzero(as_tuple=True)[0]
                for r in bad_row_idx:
                    r_s = scales[r].to(W_work.dtype)
                    quantized_W[r] = ((W_work[r] / r_s).round().clamp(clamp_min, clamp_max)).to(torch.int8)
            logger.warning("Replaced %d bad scale values in %s (Inf/NaN/zero); quality may be degraded", n_bad, name)

        # Collect per-layer metrics for quality report
        if config.quality_report_path is not None:
            layer_method = result.method_used if result is not None else "rtn"
            layer_fallbacks = list(result.fallbacks) if result is not None else []
            if bad_scales:
                layer_fallbacks.append("scale_repair")
            layer_reports.append({
                "name": name,
                "method": layer_method,
                "mse": mse,
                "max_err": max_err,
                "scale_range": [scales.min().item(), scales.max().item()],
                "fallbacks": layer_fallbacks,
                "shape": list(tensor.shape),
                "smoothquant": smoothing_applied,
                "smoothrot": is_smoothrot_down,
                "smooth_source": smooth_source if smoothing_applied else None,
                "svd_rank": svd_result.L1.shape[1] if svd_result is not None else 0,
            })

        # Store weights and scales
        output_dict[name] = stored_weights
        output_dict[f"{name}_scale"] = scales
        if zero_points is not None:
            output_dict[f"{name}_zp"] = zero_points

        # Store SVD low-rank factors for inference
        if svd_result is not None:
            output_dict[f"{name}_L1"] = svd_result.L1  # [out, r] FP16
            output_dict[f"{name}_L2"] = svd_result.L2  # [r, in]  FP16

        # Store smoothing factors for inference-side inverse absorption
        if is_smoothrot_down and smoothing_factors is not None:
            # SmoothRot: store as smoothrot_factors for inference-side 1/s
            # (applied BEFORE the online Hadamard, not after).
            output_dict[f"{name}_smoothrot_factors"] = smoothing_factors[:orig_in_features].to(SMOOTH_FACTOR_DTYPE)
        elif smoothing_applied and smoothing_factors is not None:
            # Standalone SmoothQuant: store as {name}_smooth
            # Store only the original (unpadded) portion
            output_dict[f"{name}_smooth"] = smoothing_factors[:orig_in_features].to(SMOOTH_FACTOR_DTYPE)

        # ComfyUI-INT8-Fast compatibility: write comfy_quant metadata tensor
        if config.comfy_compat and config.rot_size > 0 and is_int8:
            layer_prefix = name.rsplit(".weight", 1)[0]
            comfy_quant_data = {
                "convrot": True,
                "convrot_groupsize": config.rot_size,
                "per_row": True,
                "smoothquant": config.smoothquant and not is_smoothrot_down,
                "smooth_alpha": config.smooth_alpha if (config.smoothquant and not is_smoothrot_down) else None,
            }
            if is_smoothrot_down:
                comfy_quant_data["smoothrot"] = True
                comfy_quant_data["smoothrot_transform_order"] = "smooth_then_rotate"
                comfy_quant_data["smooth_alpha"] = smoothrot_alpha
            if svd_result is not None:
                comfy_quant_data["svd_absorbed"] = True
                comfy_quant_data["svd_rank"] = svd_result.L1.shape[1]
            comfy_quant = json.dumps(comfy_quant_data).encode("utf-8")
            output_dict[f"{layer_prefix}.comfy_quant"] = torch.tensor(
                list(comfy_quant), dtype=torch.uint8
            )

        # Store permutation if applied (only original columns, not padding)
        if perm_applied:
            output_dict[f"{name}_perm"] = perm_orig.to(torch.int32)

        quantized += 1

        if progress_callback is not None:
            elapsed = time.monotonic() - _loop_start
            if quantized > 1:
                avg_per_layer = elapsed / quantized
                remaining = avg_per_layer * (total_quantizable - quantized)
            else:
                remaining = 0.0
            layer_method = result.method_used if result is not None else "rtn"
            progress_callback(ProgressInfo(
                current_layer=quantized,
                total_layers=total_quantizable,
                layer_name=name,
                layer_shape=tuple(tensor.shape),
                elapsed_seconds=elapsed,
                estimated_remaining_seconds=max(remaining, 0.0),
                method=layer_method,
                mse=mse,
            ))

        # Store bias if present
        if name.endswith(".weight"):
            bias_key = name[:-len(".weight")] + ".bias"
            if bias_key in state_dict:
                output_dict[bias_key] = state_dict[bias_key]

    # Ensure all tensors are contiguous (SVD-absorbed splits can be non-contiguous views)
    output_dict = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in output_dict.items()}

    # Save
    output_path = os.path.join(config.output_dir, "model.safetensors")
    logger.info("Saving to %s...", output_path)
    save_file(output_dict, output_path, metadata=metadata)

    # Summary
    input_size = sum(t.numel() * t.element_size() for t in state_dict.values()) / (1024**3)
    output_size = sum(
        t.numel() * t.element_size()
        for k, t in output_dict.items()
        if k != "__metadata__"
    ) / (1024**3)

    logger.info("Done! Quantized %d layers, skipped %d layers", quantized, skipped)
    if config.use_permuquant:
        logger.info("PermuQuant applied to %d/%d layers", permuted, quantized)
    if config.smoothquant:
        logger.info("SmoothQuant applied to %d/%d layers (alpha=%.2f)", smoothquant_count, quantized, config.smooth_alpha)
    if config.smoothrot and smoothrot_factors:
        logger.info("SmoothRot: %d FFN pairs processed (alpha=%.2f)", len(smoothrot_factors), smoothrot_alpha)
    if config.piso_scales and piso_scales_dict:
        logger.info("PiSO: optimal scales computed for %d layers", len(piso_scales_dict))
    if config.svd_rank > 0:
        logger.info("SVD-absorbed: %d/%d layers decomposed (rank=%d)", svd_count, quantized, config.svd_rank)
    if config.quant_method in ("gptq", "ldlq"):
        logger.info("%s: %d layers, RTN fallback: %d layers", config.quant_method.upper(), gptq_count, rtn_fallback)
    logger.info("Input size: %.2f GB", input_size)
    logger.info("Output size: %.2f GB", output_size)
    logger.info("Compression ratio: %.2fx", input_size / output_size)

    if progress_callback is not None and _loop_start is not None:
        total_elapsed = time.monotonic() - _loop_start
        progress_callback(ProgressSummary(
            total_layers=quantized,
            skipped_layers=skipped,
            permuted_layers=permuted,
            gptq_layers=gptq_count,
            rtn_fallback_layers=rtn_fallback,
            smoothquant_layers=smoothquant_count,
            svd_layers=svd_count,
            elapsed_seconds=total_elapsed,
            input_size_gb=round(input_size, 2),
            output_size_gb=round(output_size, 2),
            compression_ratio=round(input_size / output_size, 2),
        ))

    # Write quality report
    if config.quality_report_path is not None:
        report = {
            "layers": layer_reports,
            "summary": {
                "total_layers": quantized + skipped,
                "quantized": quantized,
                "skipped": skipped,
                "input_gb": round(input_size, 2),
                "output_gb": round(output_size, 2),
                "compression_ratio": round(input_size / output_size, 2),
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(config.quality_report_path)), exist_ok=True)
        with open(config.quality_report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Quality report written to %s", config.quality_report_path)
