"""Shared test fixtures and helpers for INT-Crush converter tests.

Install the package in editable mode so tests can import ``converter``::

    pip install -e .[dev]

No sys.path manipulation needed.
"""

import torch
import pytest
from safetensors.torch import save_file


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
    seed: int = 42,
    num_samples: int = 128,
) -> torch.Tensor:
    """Create a realistic positive-definite Hessian matrix.

    Returns a 2-D ``[in_features, in_features]`` tensor.
    """
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
