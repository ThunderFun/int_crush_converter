"""Shared test fixtures and helpers for INT-Crush converter tests.

Install the package in editable mode so tests can import ``converter``::

    pip install -e .[dev]

No sys.path manipulation needed.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import pytest
from safetensors.torch import save_file

from converter.config import (
    INT8_SCALE_DIVISOR,
    SCALE_MIN,
    SCALE_MAX,
    SCALE_DTYPE,
)


# ── Fixture factories ────────────────────────────────────────────────────────


@pytest.fixture
def tmp_safetensors(tmp_path):
    """Write a small safetensors file with random 2D weight tensors.

    Returns a **callable** so tests can customise layer names and shapes::

        path = tmp_safetensors()
        path = tmp_safetensors(layer_names=["a.weight"], shapes=[(8, 32)])
    """
    def _make(
        layer_names: list[str] | None = None,
        shapes: list[tuple[int, ...]] | None = None,
        state: dict[str, torch.Tensor] | None = None,
    ) -> str:
        path = str(tmp_path / "model.safetensors")
        if state is not None:
            save_file(state, path)
            return path
        if layer_names is None:
            layer_names = ["layers.0.q_proj.weight", "layers.0.k_proj.weight"]
        if shapes is None:
            shapes = [(16, 64), (16, 64)]
        tensors = {
            name: torch.randn(*shape, dtype=torch.float32)
            for name, shape in zip(layer_names, shapes)
        }
        save_file(tensors, path)
        return path
    return _make


@pytest.fixture
def tmp_calibration(tmp_path):
    """Write a mock calibration ``.pt`` file.

    Returns a **callable**::

        path = tmp_calibration(["layer.0"], [64])
        path = tmp_calibration(["layer.0"], [64], block_size=32)  # 3D Hessian
    """
    def _make(
        layer_names: list[str],
        in_features_list: list[int],
        block_size: int | None = None,
    ) -> str:
        path = str(tmp_path / "cal.pt")
        hessians: dict[str, torch.Tensor] = {}
        shapes: dict[str, list[int]] = {}
        layer_types: dict[str, str] = {}
        for name, in_feat in zip(layer_names, in_features_list):
            if block_size is not None:
                num_blocks = (in_feat + block_size - 1) // block_size
                blocks = []
                for _ in range(num_blocks):
                    X = torch.randn(32, block_size)
                    blocks.append(X.T @ X)
                hessians[name] = torch.stack(blocks)
            else:
                X = torch.randn(32, in_feat)
                hessians[name] = X.T @ X / 32.0
            shapes[name] = [in_feat, in_feat]
            layer_types[name] = "linear"
        data = {
            "hessians": hessians,
            "shapes": shapes,
            "layer_types": layer_types,
            "metadata": {},
        }
        torch.save(data, path)
        return path
    return _make


# ── Reusable helpers (importable) ────────────────────────────────────────────


def make_hessian(
    in_features: int,
    seed: int | None = 42,
    num_samples: int = 128,
) -> torch.Tensor:
    """Create a realistic positive-definite Hessian matrix.

    Returns a 2-D ``[in_features, in_features]`` tensor.
    Pass ``seed=None`` to use the current RNG state without re-seeding.
    """
    if seed is not None:
        torch.manual_seed(seed)
    X = torch.randn(num_samples, in_features)
    return X.T @ X


def make_weight(
    out_features: int,
    in_features: int,
    seed: int = 42,
) -> torch.Tensor:
    """Create a random weight matrix with a fixed seed."""
    torch.manual_seed(seed)
    return torch.randn(out_features, in_features)


def make_block_diagonal_hessian(
    in_features: int,
    block_size: int = 32,
    seed: int = 42,
    num_samples: int = 32,
) -> torch.Tensor:
    """Create a 3-D block-diagonal Hessian ``[num_blocks, bs, bs]``."""
    torch.manual_seed(seed)
    num_blocks = (in_features + block_size - 1) // block_size
    blocks = []
    for _ in range(num_blocks):
        X = torch.randn(num_samples, block_size)
        blocks.append(X.T @ X)
    return torch.stack(blocks)


# ── Synthetic data helpers ────────────────────────────────────────────────────


def make_realistic_flux_activation(
    tokens: int,
    in_features: int,
    seed: int = 1,
    outlier_channels: int = 5,
    outlier_scale: float = 20.0,
    heavy_tail: float = 2.0,
) -> torch.Tensor:
    """Synthesize activations with FLUX-style outlier structure."""
    g = torch.Generator().manual_seed(seed)
    gamma_shape = torch.empty((tokens, in_features)).uniform_(0, 1, generator=g)
    gamma = torch._standard_gamma(
        torch.full_like(gamma_shape, heavy_tail)
    )
    X = torch.randn(tokens, in_features, generator=g) * torch.sqrt(gamma / heavy_tail)
    X /= math.sqrt(in_features)
    if outlier_channels > 0:
        chans = torch.randint(0, in_features, (outlier_channels,), generator=g)
        X[:, chans] *= outlier_scale
    return X


def quantize_int8_per_row(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric INT8. Returns (quantized_W_int8, scales_fp16)."""
    max_vals = W.abs().amax(dim=1, keepdim=True).clamp(min=SCALE_MIN)
    scales = (max_vals.float() / INT8_SCALE_DIVISOR).clamp(min=SCALE_MIN, max=SCALE_MAX)
    scales = scales.to(SCALE_DTYPE)
    q = (W / scales.to(W.dtype)).round().clamp(-128, 127).to(torch.int8)
    return q, scales


def dequantize_int8_per_row(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_int8_per_row."""
    return q.float() * scales.float()


# ── Benchmark fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def synthetic_model():
    """8-layer synthetic model with (64, 256) weights."""
    from converter.benchmark import make_synthetic_model
    return make_synthetic_model(num_layers=8, shape=(64, 256), seed=42)


@pytest.fixture
def synthetic_calibration(synthetic_model):
    """Calibration data matching synthetic_model's layer names."""
    from converter.benchmark import make_synthetic_calibration
    names = [k for k in synthetic_model if k.endswith(".weight")]
    return make_synthetic_calibration(names, in_features=256, seed=42)


@pytest.fixture
def synthetic_activations(synthetic_model):
    """Random activations for output_mse computation.

    Returns dict: ``{"layers.0.q_proj.weight": tensor(128, 256), ...}``.
    """
    acts = {}
    for name in synthetic_model:
        if name.endswith(".weight"):
            acts[name] = make_realistic_flux_activation(
                tokens=128, in_features=synthetic_model[name].shape[1], seed=42
            )
    return acts


@pytest.fixture
def synthetic_ffn_model():
    """Model with FFN-like names for SmoothRot testing.

    Generates PAIRS of tensors per block (up + down) so that
    ``detect_ffn_pairs()`` can match them.
    """
    from converter.benchmark import make_synthetic_model
    return make_synthetic_model(num_layers=8, shape=(64, 256), ff_pairs=True, seed=42)


@pytest.fixture
def synthetic_ffn_calibration(synthetic_ffn_model):
    """Calibration for FFN model."""
    from converter.benchmark import make_synthetic_calibration
    names = [k for k in synthetic_ffn_model if k.endswith(".weight")]
    return make_synthetic_calibration(names, in_features=256, seed=42)


@pytest.fixture
def real_weights():
    """Load real weights if available, else skip."""
    path = Path("ig4_bf16.safetensors")
    if not path.exists():
        pytest.skip("Real weights not available")
    from safetensors.torch import load_file
    return load_file(str(path))


@pytest.fixture
def real_weights_subset():
    """Load first 4 quantizable 2D layers from real weights, else skip.

    Fast enough (~10s) for CI smoke tests.
    """
    path = Path("ig4_bf16.safetensors")
    if not path.exists():
        pytest.skip("Real weights not available")
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as f:
        keys = [k for k in f.keys() if "weight" in k and "bias" not in k][:4]
        return {k: f.get_tensor(k) for k in keys}


@pytest.fixture
def real_calibration():
    """Load real calibration if available, else skip."""
    for path in ["v2-calibration.pt", "v4-convrot-calibration-4096.pt"]:
        p = Path(path)
        if p.exists():
            from converter.calibration_io import load_calibration
            return load_calibration(str(p))
    pytest.skip("Real calibration not available")
