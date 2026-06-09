"""CLI entry point for INT-Crush quantization with PermuQuant and GPTQ.

Usage:
    python -m converter -i in.safetensors -o out/ --rot-size 256 --permuquant
    python -m converter -i in.safetensors -o out/ --rot-size 256 -c cal.pt --quant-method gptq
"""

import argparse
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .rotation import rotate_weights, _is_power_of_two, rotate_hessian
from .scales import calculate_scales, quantize_weights, calculate_scales_int8, quantize_weights_int8
from .packing import pack_int4
from .permuquant import find_permutation_weight, sweep_alpha
from .calibration_io import load_calibration, build_name_map, get_hessian, get_permutation
from .gptq import gptq_quantize_layer, gptq_quantize_layer_rtn
from .ldlq import ldlq_quantize_layer


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
    input_path: str,
    output_dir: str,
    rot_size: int = 256,
    group_size: int = 128,
    use_permuquant: bool = False,
    tau: float = 0.0,
    skip_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    calibration_path: str | None = None,
    quant_method: str = "rtn",
    gptq_block_size: int = 128,
    damping: float = 0.01,
    int_bits: int = 4,
    ldlq_iterations: int = 1,
    comfy_compat: bool = False,
) -> None:
    """Quantize a model from safetensors using INT-Crush + PermuQuant + GPTQ/LDLQ.

    Args:
        input_path: Path to input safetensors file
        output_dir: Output directory for quantized model
        rot_size: Regular Hadamard group size (power of 4: 16/64/256)
        group_size: Quantization group size (default: 128)
        use_permuquant: Whether to apply PermuQuant channel reordering
        tau: PermuQuant acceptance threshold (default: 0.0)
        skip_patterns: Custom skip patterns (default: common patterns)
        calibration_path: Path to .pt calibration file (required for GPTQ)
        quant_method: Quantization method: "rtn", "gptq", or "ldlq"
        gptq_block_size: GPTQ/LDLQ block size (default: 128)
        damping: GPTQ/LDLQ damping ratio (default: 0.01)
        int_bits: quantization bit-width (4 or 8)
        ldlq_iterations: Number of LDLQ iterations with scale refinement (default: 1)
    """
    if skip_patterns is None:
        skip_patterns = DEFAULT_SKIP_PATTERNS

    if exclude_patterns:
        skip_patterns = list(skip_patterns) + exclude_patterns
        print(f"Excluding additional patterns: {exclude_patterns}")

    if quant_method not in ("rtn", "gptq", "ldlq"):
        raise ValueError(f"quant_method must be 'rtn', 'gptq', or 'ldlq', got '{quant_method}'")

    if quant_method == "gptq" and not calibration_path:
        raise ValueError("GPTQ requires --calibration path to a .pt file")

    if quant_method == "ldlq" and calibration_path:
        import warnings
        warnings.warn("LDLQ does not use calibration data; --calibration path will be ignored")

    if rot_size > 0 and not _is_power_of_two(rot_size):
        raise ValueError(f"rot_size must be 0 (no rotation) or a power of 2, got {rot_size}")

    if group_size < 32 or (group_size & (group_size - 1)) != 0:
        raise ValueError(f"group_size must be a power of 2 >= 32, got {group_size}")

    if int_bits not in (4, 8):
        raise ValueError(f"int_bits must be 4 or 8, got {int_bits}")

    is_int8 = int_bits == 8

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {input_path}...")
    state_dict = load_file(input_path)
    print(f"Loaded {len(state_dict)} tensors")

    # Load calibration data if GPTQ
    calibration = None
    name_map = {}
    if quant_method == "gptq":
        print(f"Loading calibration from {calibration_path}...")
        calibration = load_calibration(calibration_path)
        cal_keys = list(calibration["hessians"].keys())
        name_map = build_name_map(list(state_dict.keys()), cal_keys)
        print(f"Matched {len(name_map)}/{len(cal_keys)} calibration layers to model layers")

    output_dict = {}
    meta_prefix = "int_crush"
    method_base = "int_crush_permuq" if use_permuquant else "int_crush"
    method = f"{method_base}_{quant_method}" if quant_method in ("gptq", "ldlq") else method_base
    metadata = {
        f"{meta_prefix}.format_version": "3" if not is_int8 else "1",
        f"{meta_prefix}.method": method,
        f"{meta_prefix}.rot_size": str(rot_size),
        f"{meta_prefix}.scale_type": "per_row",
        f"{meta_prefix}.hadamard_type": "regular",
    }
    if not is_int8:
        metadata["int_crush.packing_order"] = "little"
    else:
        metadata["int_crush.packing_order"] = "native"
    if exclude_patterns:
        metadata[f"{meta_prefix}.exclude_patterns"] = ",".join(exclude_patterns)

    # Detect layers whose in_features will be padded by rotation.
    # ComfyUI reads in_channels from img_in.weight.shape[1], so padded
    # weights cause wrong model configuration. Store originals so the
    # loader can fix the model after creation.
    padded_layers = {}
    if rot_size > 0:
        for name, tensor in state_dict.items():
            if tensor.dim() == 2 and not should_skip(name, skip_patterns):
                orig_in = tensor.shape[1]
                if orig_in % rot_size != 0:
                    padded_in = rot_size * ((orig_in + rot_size - 1) // rot_size)
                    padded_layers[name] = str(orig_in)
                    print(f"  Tracking padded layer: {name} in_features {orig_in} -> {padded_in}")
        if padded_layers:
            metadata["int_crush.padded_layers"] = ";".join(
                f"{k}={v}" for k, v in padded_layers.items()
            )
    if use_permuquant:
        metadata["int_crush.permuquant"] = "true"
        metadata["int_crush.permuquant_tau"] = str(tau)
    if quant_method == "gptq":
        metadata[f"{meta_prefix}.gptq_block_size"] = str(gptq_block_size)
        metadata[f"{meta_prefix}.damping_ratio"] = str(damping)
    elif quant_method == "ldlq":
        metadata[f"{meta_prefix}.ldlq_block_size"] = str(gptq_block_size)
        metadata[f"{meta_prefix}.damping_ratio"] = str(damping)

    skipped = 0
    quantized = 0
    permuted = 0
    gptq_count = 0
    rtn_fallback = 0

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

        print(f"Quantizing {name} ({tensor.shape})...")

        W = tensor.float()
        orig_in_features = W.shape[1]

        # 1. Rotate weights (skip if rot_size == 0)
        if rot_size > 0:
            W_work = rotate_weights(W, rot_size)
        else:
            W_work = W.clone()

        # 2. PermuQuant: apply channel reordering
        #    If calibration provides a permutation (from ComfyUI-GPTQ-Calibration),
        #    use it directly — the Hessians are already in permuted space.
        #    Otherwise, compute our own from weight statistics.
        perm_applied = False
        perm_orig = None
        cal_perm = None
        if use_permuquant:
            layer_name = name_map.get(name) if name_map else None
            if layer_name is not None and calibration is not None:
                cal_perm = get_permutation(calibration, layer_name)

            if cal_perm is not None:
                # Calibration permutation: apply to weight only.
                # Hessian is already in permuted space from calibration.
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
                    print(f"  PermuQuant applied (from calibration)")
                else:
                    print(f"  WARNING: calibration permutation out of bounds, falling back to weight-only")

            if not perm_applied:
                W_orig = W_work[:, :orig_in_features]
                # For INT4, all methods (RTN, GPTQ, LDLQ) use per-row
                # quantization.  PermuQuant's acceptance check must use the
                # same granularity or it evaluates a different scheme.
                actual_group_size = W_orig.shape[1] if int_bits == 4 else group_size
                perm, alpha, accepted = sweep_alpha(
                    W_orig, act_mu2=None, group_size=actual_group_size, tau=tau, int_bits=int_bits
                )
                if accepted:
                    W_work[:, :orig_in_features] = W_orig[:, perm]
                    perm_orig = perm
                    perm_applied = True
                    permuted += 1
                    print(f"  PermuQuant applied (alpha={alpha:.2f})")
                else:
                    print(f"  PermuQuant skipped (no improvement)")

        # 3. Quantize
        if quant_method == "gptq":
            # GPTQ: use Hessian-based error compensation
            layer_name = name_map.get(name)
            hessian = None
            if layer_name is not None:
                hessian = get_hessian(calibration, layer_name, tensor.shape)

            if hessian is not None:
                already_rotated = calibration.get("metadata", {}).get("hessian_rotated", False)

                if not already_rotated:
                    if cal_perm is not None:
                        # Calibration Hessian: P^T @ H @ P = H[inv_perm][:, inv_perm].
                        # To recover H[perm][:, perm] (the permuted weight space),
                        # index by cal_perm composed with itself.
                        if hessian.dim() == 2:
                            perm_sq = cal_perm[cal_perm]
                            hessian = hessian[perm_sq][:, perm_sq]
                        else:
                            raise ValueError(
                                "Calibration permutation + block-diagonal Hessian "
                                "is not supported. Regenerate calibration data "
                                "with the permutation applied."
                            )

                    # Rotate: H_rot = R^T @ H @ R
                    if rot_size > 0:
                        hessian = rotate_hessian(hessian, rot_size)

                # Re-permute only when we computed our own permutation
                # (cal_perm is None).  When calibration provides the
                # permutation the un-permute + rotate already produced
                # the correct Hessian for the permuted-rotated weight.
                if perm_applied and cal_perm is None:
                    if hessian.dim() == 2:
                        hessian = hessian[perm_orig][:, perm_orig]
                    else:
                        raise ValueError(
                            f"PermuQuant permutation cannot be applied to "
                            f"block-diagonal Hessian ({hessian.dim()}D, shape "
                            f"{hessian.shape}). Regenerate calibration data "
                            f"with the permutation applied."
                        )

                # Quantize the full rotated weight (no truncation).
                # If rotation padded the weight, the padded columns will have
                # zero Hessian diagonal — damping prevents div-by-zero, and
                # error compensation naturally skips zero-energy columns.
                quantized_W, scales = gptq_quantize_layer(
                    W_work, hessian,
                    block_size=gptq_block_size,
                    damping=damping,
                    int_bits=int_bits,
                )
                gptq_count += 1
            else:
                # Fallback to RTN if no calibration data or Hessian is missing/mismatched
                quantized_W, scales = gptq_quantize_layer_rtn(W_work, int_bits=int_bits)
                rtn_fallback += 1
                print(f"  No calibration data, using RTN fallback")
        elif quant_method == "ldlq":
            # LDLQ: weight-only quantization using H = W^T @ W (no calibration needed)
            # Quantize the full rotated weight (no truncation).
            # LDLQ computes H = W^T @ W / M internally from the weight,
            # so it naturally handles the padded columns.
            quantized_W, scales = ldlq_quantize_layer(
                W_work,
                block_size=gptq_block_size,
                damping=damping,
                int_bits=int_bits,
                iterations=ldlq_iterations,
            )
            gptq_count += 1
        else:
            # RTN: simple round-to-nearest
            if is_int8:
                scales = calculate_scales_int8(W_work)
                quantized_W = quantize_weights_int8(W_work, scales)
            else:
                in_features = W_work.shape[1]
                scales = calculate_scales(W_work, in_features)
                quantized_W = quantize_weights(W_work, scales, in_features)

        # 4. Pack (INT4 only) or store directly (INT8)
        if is_int8:
            stored_weights = quantized_W
        else:
            stored_weights = pack_int4(quantized_W)

        # 5. Verify: dequantize and compare to working weights
        dequant = quantized_W.float() * scales.float()
        mse = (W_work - dequant).pow(2).mean().item()
        max_err = (W_work - dequant).abs().max().item()
        bad_scales = scales.isinf().any() or scales.isnan().any() or (scales == 0).any()
        print(f"  MSE={mse:.6f}  max_err={max_err:.4f}  scale_range=[{scales.min():.6f}, {scales.max():.6f}]  bad_scales={bad_scales}")
        if bad_scales:
            print(f"  WARNING: bad scales detected! inf={scales.isinf().any()} nan={scales.isnan().any()} zero={(scales == 0).any()}")

        # Store weights and scales
        output_dict[name] = stored_weights
        output_dict[f"{name}_scale"] = scales

        # ComfyUI-INT8-Fast compatibility: write comfy_quant metadata tensor
        if comfy_compat and rot_size > 0 and is_int8:
            import json
            layer_prefix = name.rsplit(".weight", 1)[0]
            comfy_quant = json.dumps({
                "convrot": True,
                "convrot_groupsize": rot_size,
                "per_row": True,
            }).encode("utf-8")
            output_dict[f"{layer_prefix}.comfy_quant"] = torch.tensor(
                list(comfy_quant), dtype=torch.uint8
            )

        # Store permutation if applied (only original columns, not padding)
        if perm_applied:
            output_dict[f"{name}.perm"] = perm_orig.to(torch.int32)

        quantized += 1

        # Store bias if present
        if name.endswith(".weight"):
            bias_key = name[:-len(".weight")] + ".bias"
            if bias_key in state_dict:
                output_dict[bias_key] = state_dict[bias_key]

    # Save
    output_path = os.path.join(output_dir, "model.safetensors")
    print(f"Saving to {output_path}...")
    save_file(output_dict, output_path, metadata=metadata)

    # Summary
    input_size = sum(t.numel() * t.element_size() for t in state_dict.values()) / (1024**3)
    output_size = sum(
        t.numel() * t.element_size()
        for k, t in output_dict.items()
        if k != "__metadata__"
    ) / (1024**3)

    print(f"\nDone! Quantized {quantized} layers, skipped {skipped} layers")
    if use_permuquant:
        print(f"PermuQuant applied to {permuted}/{quantized} layers")
    if quant_method in ("gptq", "ldlq"):
        print(f"{quant_method.upper()}: {gptq_count} layers, RTN fallback: {rtn_fallback} layers")
    print(f"Input size: {input_size:.2f} GB")
    print(f"Output size: {output_size:.2f} GB")
    print(f"Compression ratio: {input_size / output_size:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize a model using INT-Crush + PermuQuant + GPTQ/LDLQ"
    )
    parser.add_argument("-i", "--input", required=True, help="Input safetensors file")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--rot-size", type=int, default=0, choices=[0, 16, 64, 256],
                        help="Regular Hadamard group size (0=no rotation, default: 0)")
    parser.add_argument("--int-bits", type=int, default=4, choices=[4, 8],
                        help="Quantization bit-width: 4 (INT4) or 8 (INT8, default: 4)")
    parser.add_argument("--group-size", type=int, default=128,
                        help="PermuQuant group size for acceptance evaluation (default: 128). "
                             "INT4 quantization always uses per-row scales for speed.")
    parser.add_argument("--permuquant", action="store_true",
                        help="Enable PermuQuant channel reordering")
    parser.add_argument("--tau", type=float, default=0.0,
                        help="PermuQuant acceptance threshold (default: 0.0)")
    parser.add_argument("--skip-patterns", type=str, default=None,
                        help="Comma-separated skip patterns (default: embed,norm,modulation,lm_head,output,proj_out)")
    parser.add_argument("--exclude-patterns", type=str, default=None,
                        help="Comma-separated additional patterns to exclude from quantization (added to defaults)")
    parser.add_argument("-c", "--calibration", type=str, default=None,
                        help="Path to .pt calibration file (for GPTQ)")
    parser.add_argument("--quant-method", type=str, default="rtn", choices=["rtn", "gptq", "ldlq"],
                        help="Quantization method: 'rtn' (round-to-nearest), 'gptq' (calibration-based), or 'ldlq' (weight-only, no calibration) (default: rtn)")
    parser.add_argument("--gptq-block-size", type=int, default=128,
                        help="GPTQ block size (default: 128)")
    parser.add_argument("--damping", type=float, default=0.01,
                        help="GPTQ/LDLQ damping ratio (default: 0.01)")
    parser.add_argument("--ldlq-iterations", type=int, default=1,
                        help="LDLQ iterations with scale refinement (default: 1)")
    parser.add_argument("--comfy-compat", action="store_true",
                        help="Write comfy_quant metadata for ComfyUI-INT8-Fast compatibility (INT8 + ConvRot only)")

    args = parser.parse_args()

    skip_patterns = None
    if args.skip_patterns:
        skip_patterns = [p.strip() for p in args.skip_patterns.split(",")]

    exclude_patterns = None
    if args.exclude_patterns:
        exclude_patterns = [p.strip() for p in args.exclude_patterns.split(",")]

    quantize_model(
        input_path=args.input,
        output_dir=args.output,
        rot_size=args.rot_size,
        group_size=args.group_size,
        use_permuquant=args.permuquant,
        tau=args.tau,
        skip_patterns=skip_patterns,
        exclude_patterns=exclude_patterns,
        calibration_path=args.calibration,
        quant_method=args.quant_method,
        gptq_block_size=args.gptq_block_size,
        damping=args.damping,
        int_bits=args.int_bits,
        ldlq_iterations=args.ldlq_iterations,
        comfy_compat=args.comfy_compat,
    )


if __name__ == "__main__":
    main()
