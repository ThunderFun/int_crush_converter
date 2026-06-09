"""Load and validate GPTQ calibration data from ComfyUI-GPTQ-Calibration .pt files."""

import torch


def load_calibration(path: str) -> dict:
    """Load a calibration .pt file and validate its structure.

    Args:
        path: Path to the .pt calibration file

    Returns:
        Calibration dict with keys: metadata, hessians, amax, shapes, layer_types

    Raises:
        KeyError: If required keys are missing
    """
    data = torch.load(path, map_location="cpu", weights_only=True)

    required_keys = {"hessians", "shapes", "layer_types"}
    missing = required_keys - set(data.keys())
    if missing:
        raise KeyError(f"Calibration file missing required keys: {missing}")

    return data


def build_name_map(state_dict_keys: list[str], calibration_keys: list[str]) -> dict[str, str]:
    """Build mapping from state_dict keys to calibration layer names.

    State dict keys have '.weight' suffix (e.g. 'blocks.0.attn.q_proj.weight').
    Calibration keys do not (e.g. 'blocks.0.attn.q_proj').

    Returns:
        Dict mapping state_dict_key -> calibration_key for matched layers
    """
    cal_set = set(calibration_keys)
    mapping = {}

    for key in state_dict_keys:
        # Strip .weight suffix to get module name
        if key.endswith(".weight"):
            module_name = key[: -len(".weight")]
        else:
            continue

        if module_name in cal_set:
            mapping[key] = module_name

    return mapping


def get_hessian(calibration: dict, layer_name: str, weight_shape: torch.Size) -> torch.Tensor | None:
    """Get the Hessian tensor for a layer.

    Returns the raw Hessian — either 2D [in, in] (full) or 3D [blocks, bs, bs] (block-diagonal).
    The caller (gptq_quantize_layer) handles both formats.

    Supports both the legacy stacked-tensor format and the newer list-of-blocks
    format from ComfyUI-GPTQ-Calibration (where the last block retains its
    true size instead of being zero-padded).  Lists are auto-stacked into a
    3D tensor with zero-padded last block; downstream code already slices
    ``[:actual_bs, :actual_bs]`` so the padding is harmless.

    Args:
        calibration: The loaded calibration dict
        layer_name: Calibration layer name (without .weight suffix)
        weight_shape: Shape of the weight tensor [out_features, in_features]

    Returns:
        Hessian tensor, or None if not found
    """
    hessians = calibration["hessians"]
    if layer_name not in hessians:
        return None

    H_raw = hessians[layer_name]

    # ── list-of-blocks format (ComfyUI-GPTQ-Calibration) ──────────────
    if isinstance(H_raw, list):
        if not H_raw or not isinstance(H_raw[0], torch.Tensor):
            return None
        max_bs = max(t.shape[0] for t in H_raw)
        padded = []
        for t in H_raw:
            if t.shape[0] == max_bs and t.shape[1] == max_bs:
                padded.append(t)
            else:
                block = torch.zeros(max_bs, max_bs, dtype=t.dtype)
                block[: t.shape[0], : t.shape[1]] = t
                padded.append(block)
        H = torch.stack(padded)  # [num_blocks, max_bs, max_bs]
    elif isinstance(H_raw, torch.Tensor):
        H = H_raw
    else:
        return None

    in_features = weight_shape[1]

    if H.dim() == 2:
        if H.shape[0] != in_features or H.shape[1] != in_features:
            return None
        return H

    if H.dim() == 3:
        num_blocks, block_size, _ = H.shape
        expected_blocks = (in_features + block_size - 1) // block_size
        if num_blocks != expected_blocks or block_size != H.shape[2]:
            return None
        return H
    return None


def get_permutation(calibration: dict, layer_name: str) -> torch.Tensor | None:
    """Get the PermuQuant permutation for a layer from calibration data.

    When the ComfyUI-GPTQ-Calibration node runs with ``permuquant=True``,
    it stores per-layer channel permutations under the ``"permuquant"`` key.
    These permutations were already applied to the activations during Hessian
    accumulation, so the Hessians are in permuted-channel space.

    Args:
        calibration: The loaded calibration dict
        layer_name: Calibration layer name (without .weight suffix)

    Returns:
        Permutation indices as int64, or None if not available
    """
    permuquant = calibration.get("permuquant")
    if not permuquant or layer_name not in permuquant:
        return None
    return permuquant[layer_name].to(torch.int64)
