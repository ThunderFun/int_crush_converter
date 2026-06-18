"""SmoothRot: FFN pair detection for smooth-then-rotate quantization.

SmoothRot applies smoothing FIRST (SmoothQuant-style), then Hadamard rotation SECOND.
Reverse of legacy rotate-then-smooth; produces lower output MSE on models with
activation outliers.

This module detects FFN up/down-projection pairs so the pipeline can compute
smoothing factors for down-projections and store them alongside quantized weights
for inference-side 1/s application (before the online Hadamard).

Design: smoothing factors stored as ``{name}_smoothrot_factors`` tensors
(Option A from implementation plan) — avoids architecture-specific row-split
knowledge, works for any model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .log import logger


@dataclass
class FFNPair:
    """An FFN up/down-projection pair detected from state-dict keys."""

    up_name: str
    """State-dict key for the up-projection weight (e.g. ``double_blocks.0.img_mlp.0.weight``)."""

    down_name: str
    """State-dict key for the down-projection weight (e.g. ``double_blocks.0.img_mlp.2.weight``)."""

    block_id: str
    """Block identifier (e.g. ``double_blocks.0.img_mlp``)."""

    up_out_features: int
    """W_up.shape[0] — may be larger than down_in_features for fused projections."""

    down_in_features: int
    """W_down.shape[1] — the dimension that smoothing factors ``s`` have."""


# ── Architecture-specific patterns ──────────────────────────────────────────

# FLUX: double_blocks.{N}.{img,txt}_mlp.{0,2}.weight
_DOUBLE_BLOCK_RE = re.compile(
    r"^(double_blocks\.\d+\.(?:img|txt)_mlp)\.([02])\.weight$"
)

# FLUX: single_blocks.{N}.linear{1,2}.weight
_SINGLE_BLOCK_RE = re.compile(
    r"^(single_blocks\.\d+)\.linear([12])\.weight$"
)

# IG4 (SwiGLU): layers.{N}.feed_forward.w{1,2,3}.weight
#   w1 = gate_proj (expanding), w3 = up_proj (expanding), w2 = down_proj (contracting)
_IG4_FFN_RE = re.compile(
    r"^(layers\.\d+\.feed_forward)\.w([123])\.weight$"
)


def detect_ffn_pairs(state_dict: dict) -> dict[str, FFNPair]:
    """Detect FFN up/down-projection pairs from state-dict keys.

    Scans the state dict for patterns matching known FFN block structures
    and pairs the up-projection (``mlp.0`` / ``linear1``) with the
    down-projection (``mlp.2`` / ``linear2``).

    Args:
        state_dict: Model state dict (keys are layer names, values are tensors).

    Returns:
        Mapping from down-projection name to :class:`FFNPair`.
        Layers that don't match any FFN pattern are silently skipped
        (attention QKV, projections, norms, etc.).
    """
    # Collect potential up and down projections by block_id
    up_candidates: dict[str, tuple[str, int]] = {}   # block_id -> (name, out_features)
    down_candidates: dict[str, tuple[str, int]] = {}  # block_id -> (name, in_features)

    for name, tensor in state_dict.items():
        if not hasattr(tensor, "dim") or tensor.dim() != 2:
            continue

        m = _DOUBLE_BLOCK_RE.match(name)
        if m:
            block_id = m.group(1)
            idx = m.group(2)
            if idx == "0":
                up_candidates[block_id] = (name, tensor.shape[0])
            elif idx == "2":
                down_candidates[block_id] = (name, tensor.shape[1])
            continue

        m = _SINGLE_BLOCK_RE.match(name)
        if m:
            block_id = m.group(1)
            idx = m.group(2)
            if idx == "1":
                up_candidates[block_id] = (name, tensor.shape[0])
            elif idx == "2":
                down_candidates[block_id] = (name, tensor.shape[1])
            continue

        m = _IG4_FFN_RE.match(name)
        if m:
            block_id = m.group(1)
            idx = m.group(2)
            if idx in ("1", "3"):
                # w1 (gate) and w3 (up) are both up-projections.
                # Prefer w3 as the canonical up_name (matches LLaMA up_proj).
                up_candidates[block_id] = (name, tensor.shape[0])
            elif idx == "2":
                down_candidates[block_id] = (name, tensor.shape[1])
            continue

    # Pair up and down projections that share the same block_id
    pairs: dict[str, FFNPair] = {}
    for block_id in sorted(set(up_candidates) & set(down_candidates)):
        up_name, up_out = up_candidates[block_id]
        down_name, down_in = down_candidates[block_id]
        pairs[down_name] = FFNPair(
            up_name=up_name,
            down_name=down_name,
            block_id=block_id,
            up_out_features=up_out,
            down_in_features=down_in,
        )

    # Log orphan up-projections (no matching down-projection)
    for block_id in sorted(set(up_candidates) - set(down_candidates)):
        up_name, _ = up_candidates[block_id]
        logger.debug("SmoothRot: orphan up-projection %s (no matching down-proj)", up_name)

    # Log standalone down-projections (no matching up-projection)
    for block_id in sorted(set(down_candidates) - set(up_candidates)):
        down_name, _ = down_candidates[block_id]
        logger.debug("SmoothRot: orphan down-projection %s (no matching up-proj)", down_name)

    logger.info("SmoothRot: detected %d FFN pairs", len(pairs))
    return pairs
