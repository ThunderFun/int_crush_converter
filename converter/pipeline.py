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
from .scales import (
    calculate_scales, quantize_weights,
    calculate_scales_int8, quantize_weights_int8,
    calculate_scales_asymmetric, quantize_weights_asymmetric,
    calculate_scales_int8_asymmetric, quantize_weights_int8_asymmetric,
)
from .config import INT4_SCALE_DIVISOR, INT8_SCALE_DIVISOR, FP16_SCALE_FLOOR, MAX_FP16_SCALE
from .packing import pack_int4
from .permuquant import sweep_alpha
from .calibration_io import load_calibration, build_name_map, get_hessian, get_permutation
from .gptq import gptq_quantize_layer, gptq_quantize_layer_rtn
from .ldlq import ldlq_quantize_layer
from .rounding import _round_half_away_from_zero


DEFAULT_SKIP_PATTERNS = [
    "embed",
    "norm",
    "modulation",
    "lm_head",
    "output",
    "proj_out",
]


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

    if config.quant_method == "ldlq" and config.calibration_path:
        import warnings
        warnings.warn("LDLQ does not use calibration data; --calibration path will be ignored")

    if config.rot_size > 0 and not _is_power_of_two(config.rot_size):
        raise ValueError(f"rot_size must be 0 (no rotation) or a power of 2, got {config.rot_size}")

    if config.perm_group_size < 32 or (config.perm_group_size & (config.perm_group_size - 1)) != 0:
        raise ValueError(f"perm_group_size must be a power of 2 >= 32, got {config.perm_group_size}")

    if config.quant_group_size < 32 or (config.quant_group_size & (config.quant_group_size - 1)) != 0:
        raise ValueError(f"quant_group_size must be a power of 2 >= 32, got {config.quant_group_size}")

    if config.int_bits not in (4, 8):
        raise ValueError(f"int_bits must be 4 or 8, got {config.int_bits}")

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

    # Load calibration data if GPTQ
    calibration = None
    name_map = {}
    if config.quant_method == "gptq":
        logger.info("Loading calibration from %s...", config.calibration_path)
        calibration = load_calibration(config.calibration_path)
        cal_keys = list(calibration["hessians"].keys())
        name_map = build_name_map(list(state_dict.keys()), cal_keys)
        logger.info("Matched %d/%d calibration layers to model layers", len(name_map), len(cal_keys))

    output_dict = {}
    meta_prefix = "int_crush"
    method_base = "int_crush_permuq" if config.use_permuquant else "int_crush"
    method = f"{method_base}_{config.quant_method}" if config.quant_method in ("gptq", "ldlq") else method_base
    metadata = {
        f"{meta_prefix}.format_version": "3" if not is_int8 else "1",
        f"{meta_prefix}.method": method,
        f"{meta_prefix}.rot_size": str(config.rot_size),
        f"{meta_prefix}.scale_type": "per_row_asymmetric" if (config.asymmetric and not (config.quant_method == "gptq" and is_int8)) else "per_row",
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

    skipped = 0
    quantized = 0
    permuted = 0
    gptq_count = 0
    rtn_fallback = 0
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
        # Skip non-quantizable layers
        if should_skip(name, skip_patterns):
            output_dict[name] = tensor
            skipped += 1
            continue

        # Only quantize 2D weight tensors
        if tensor.dim() != 2:
            output_dict[name] = tensor
            skipped += 1
            continue

        logger.info("Quantizing %s (%s)...", name, tensor.shape)

        W = tensor.float()
        orig_in_features = W.shape[1]

        # 1. Rotate weights (skip if rot_size == 0)
        #    Clone only when the downstream path mutates W_work in-place:
        #    GPTQ/LDLQ always do, and PermuQuant reorders columns.
        #    RTN with rot_size=0 and no PermuQuant only reads W_work.
        if config.rot_size > 0:
            W_work = rotate_weights(W, config.rot_size)
        elif config.quant_method in ("gptq", "ldlq") or config.use_permuquant:
            W_work = W.clone()
        else:
            W_work = W  # RTN, no rotation, no PermuQuant — read-only

        # 2. PermuQuant: apply channel reordering
        #    If calibration provides a permutation (from ComfyUI-GPTQ-Calibration),
        #    use it directly — the Hessians are already in permuted space.
        #    Otherwise, compute our own from weight statistics.
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

                # If the calibration was rotated with a different rot_size
                # than the converter is using, the Hessian is in the wrong
                # space.  We can't re-rotate from the original space, but
                # we should warn the user.
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
                    if hessian.dim() == 2:
                        hessian = hessian[perm_orig][:, perm_orig]
                    else:
                        # Block-diagonal Hessian cannot be permuted in-place.
                        # Un-permute W_work to restore original column order
                        # so that W_work and hessian are in the same space.
                        inv_perm = perm_orig.argsort()
                        orig_features = min(orig_in_features, inv_perm.shape[0])
                        W_orig = W_work[:, :orig_features]
                        W_work[:, :orig_features] = W_orig[:, inv_perm]
                        perm_applied = False
                        logger.debug(
                            "PermuQuant: un-permuted W_work for block-diagonal Hessian "
                            "for %s (%dD, shape %s)",
                            name, hessian.dim(), hessian.shape,
                        )

                # Quantize full rotated weight; zero-energy padded columns handled by damping.
                result = gptq_quantize_layer(
                    W_work, hessian,
                    block_size=config.gptq_block_size,
                    damping=config.damping,
                    int_bits=config.int_bits,
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
                )
                quantized_W = result.quantized_W
                scales = result.scales
                zero_points = result.zero_points
                rtn_fallback += 1
                logger.debug("No calibration data, using RTN fallback")
        elif config.quant_method == "ldlq":
            # LDLQ: weight-only quantization using H = W^T @ W (no calibration needed)
            # Quantize the full rotated weight (no truncation).
            # LDLQ computes H = W^T @ W / M internally from the weight,
            # so it naturally handles the padded columns.
            result = ldlq_quantize_layer(
                W_work,
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

        # 4. Pack (INT4 only) or store directly (INT8)
        if is_int8:
            stored_weights = quantized_W
        else:
            stored_weights = pack_int4(quantized_W)

        # 5. Verify: dequantize and compare to working weights
        #    Compute diff once to avoid materialising (W_work - dequant) twice
        #    and free intermediates eagerly to reduce peak memory.
        #    Use non-in-place subtraction because W_work may alias W
        #    (optimisation: skip clone for RTN with rot_size=0).
        if zero_points is not None:
            dequant = (quantized_W.float() - zero_points.float()) * scales.float()
        else:
            dequant = quantized_W.float() * scales.float()
        diff = W_work - dequant
        del dequant
        max_err = diff.abs().max().item()
        mse = diff.pow(2).mean().item()
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
            sym_scales = (W_work.float().abs().amax(dim=1, keepdim=True) / fix_divisor).clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE).to(torch.float16)
            if zero_points is not None:
                w_min = W_work.amin(dim=1, keepdim=True)
                w_max = W_work.amax(dim=1, keepdim=True)
                if is_int8:
                    asym_scales = ((w_max - w_min).float() / 255.0).clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE).to(torch.float16)
                else:
                    asym_scales = ((w_max - w_min).float() / 15.0).clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE).to(torch.float16)
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
            })

        # Store weights and scales
        output_dict[name] = stored_weights
        output_dict[f"{name}_scale"] = scales
        if zero_points is not None:
            output_dict[f"{name}_zp"] = zero_points

        # ComfyUI-INT8-Fast compatibility: write comfy_quant metadata tensor
        if config.comfy_compat and config.rot_size > 0 and is_int8:
            layer_prefix = name.rsplit(".weight", 1)[0]
            comfy_quant = json.dumps({
                "convrot": True,
                "convrot_groupsize": config.rot_size,
                "per_row": True,
            }).encode("utf-8")
            output_dict[f"{layer_prefix}.comfy_quant"] = torch.tensor(
                list(comfy_quant), dtype=torch.uint8
            )

        # Store permutation if applied (only original columns, not padding)
        if perm_applied:
            output_dict[f"{name}.perm"] = perm_orig.to(torch.int32)

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
