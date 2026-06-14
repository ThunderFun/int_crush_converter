"""Tests for calibration data I/O, Hessian loading, and name mapping."""

import tempfile

import torch
import pytest

from converter.gptq import gptq_quantize_layer
from converter.calibration_io import (
    load_calibration,
    build_name_map,
    get_hessian,
    get_permutation,
)


def _make_mock_calibration(layer_names: list[str], in_features: int) -> dict:
    """Create a mock calibration dict matching ComfyUI-GPTQ-Calibration format."""
    hessians = {}
    shapes = {}
    layer_types = {}
    for name in layer_names:
        X = torch.randn(32, in_features)
        hessians[name] = X.T @ X
        shapes[name] = (64, in_features)
        layer_types[name] = "Linear"
    return {
        "metadata": {
            "model_name": "test_model",
            "num_samples": 32,
            "hessian_block_size": 0,
            "recommended_damping_ratio": 0.01,
        },
        "hessians": hessians,
        "amax": {name: 1.0 for name in layer_names},
        "shapes": shapes,
        "layer_types": layer_types,
    }


class TestCalibrationIO:
    """Tests for calibration data loading and name mapping."""

    def test_save_load_roundtrip(self):
        cal = _make_mock_calibration(["layer.0", "layer.1"], 64)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(cal, f.name)
            loaded = load_calibration(f.name)
        assert set(loaded["hessians"].keys()) == set(cal["hessians"].keys())
        for key in cal["hessians"]:
            assert torch.allclose(loaded["hessians"][key], cal["hessians"][key])

    def test_load_missing_keys_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"metadata": {}}, f.name)
            with pytest.raises(KeyError):
                load_calibration(f.name)

    def test_load_missing_file_raises(self):
        with pytest.raises(ValueError, match="not found|No such file"):
            load_calibration("/nonexistent/path.pt")

    def test_load_corrupted_file_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"\x00\x01\x02garbage data not a real pt file")
            f.flush()
            with pytest.raises(ValueError, match="corrupted"):
                load_calibration(f.name)

    def test_load_invalid_format_missing_keys(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"metadata": {}}, f.name)
            with pytest.raises(KeyError):
                load_calibration(f.name)

    def test_build_name_map_basic(self):
        state_keys = [
            "blocks.0.attn.q_proj.weight",
            "blocks.0.attn.k_proj.weight",
            "blocks.0.norm.weight",
        ]
        cal_keys = ["blocks.0.attn.q_proj", "blocks.0.attn.k_proj"]
        mapping = build_name_map(state_keys, cal_keys)
        assert mapping == {
            "blocks.0.attn.q_proj.weight": "blocks.0.attn.q_proj",
            "blocks.0.attn.k_proj.weight": "blocks.0.attn.k_proj",
        }

    def test_build_name_map_no_match(self):
        assert build_name_map(["model.linear.weight"], ["different_layer"]) == {}

    def test_build_name_map_skips_non_weight(self):
        state_keys = ["model.linear.weight", "model.linear.bias", "embedding"]
        cal_keys = ["model.linear"]
        assert build_name_map(state_keys, cal_keys) == {"model.linear.weight": "model.linear"}

    def test_get_hessian_full(self):
        cal = _make_mock_calibration(["layer.0"], 64)
        H = get_hessian(cal, "layer.0", torch.Size([32, 64]))
        assert H is not None
        assert H.shape == (64, 64)

    def test_get_hessian_not_found(self):
        cal = _make_mock_calibration(["layer.0"], 64)
        assert get_hessian(cal, "missing_layer", torch.Size([32, 64])) is None

    def test_get_hessian_block_diagonal(self):
        block_size = 32
        in_features = 64
        num_blocks = in_features // block_size
        blocks = []
        for _ in range(num_blocks):
            X = torch.randn(16, block_size)
            blocks.append(X.T @ X)
        H_block = torch.stack(blocks)

        cal = {
            "hessians": {"layer.0": H_block},
            "shapes": {"layer.0": (32, in_features)},
            "layer_types": {"layer.0": "Linear"},
        }
        H = get_hessian(cal, "layer.0", torch.Size([32, in_features]))
        assert H is not None
        assert H.shape == (num_blocks, block_size, block_size)
        assert torch.allclose(H, H_block)

    def test_get_hessian_list_of_blocks(self):
        block_size = 32
        in_features = 64
        num_blocks = in_features // block_size
        blocks = []
        for _ in range(num_blocks):
            X = torch.randn(16, block_size)
            blocks.append(X.T @ X)

        cal = {
            "hessians": {"layer.0": blocks},
            "shapes": {"layer.0": (32, in_features)},
            "layer_types": {"layer.0": "Linear"},
        }
        H = get_hessian(cal, "layer.0", torch.Size([32, in_features]))
        assert H is not None
        assert H.dim() == 3
        assert H.shape == (num_blocks, block_size, block_size)
        for i in range(num_blocks):
            assert torch.allclose(H[i], blocks[i])

    def test_get_hessian_list_non_divisible(self):
        block_size = 32
        in_features = 80
        num_blocks = 3
        blocks = []
        for _ in range(num_blocks - 1):
            X = torch.randn(16, block_size)
            blocks.append(X.T @ X)
        X = torch.randn(16, 16)
        blocks.append(X.T @ X)

        cal = {
            "hessians": {"layer.0": blocks},
            "shapes": {"layer.0": (32, in_features)},
            "layer_types": {"layer.0": "Linear"},
        }
        H = get_hessian(cal, "layer.0", torch.Size([32, in_features]))
        assert H is not None
        assert H.dim() == 3
        assert H.shape == (3, block_size, block_size)
        assert torch.allclose(H[0], blocks[0])
        assert torch.allclose(H[1], blocks[1])
        assert torch.allclose(H[2, :16, :16], blocks[2])
        assert torch.all(H[2, 16:, :] == 0)
        assert torch.all(H[2][:, 16:] == 0)

    def test_get_hessian_list_gptq_end_to_end(self):
        torch.manual_seed(42)
        block_size = 32
        in_features = 80
        out_features = 16

        W = torch.randn(out_features, in_features)
        blocks = []
        for _ in range(2):
            X = torch.randn(64, block_size)
            blocks.append(X.T @ X)
        X = torch.randn(64, 16)
        blocks.append(X.T @ X)

        cal = {
            "hessians": {"layer.0": blocks},
            "shapes": {"layer.0": (out_features, in_features)},
            "layer_types": {"layer.0": "Linear"},
        }
        H = get_hessian(cal, "layer.0", torch.Size([out_features, in_features]))
        assert H is not None
        _qr = gptq_quantize_layer(W, H)
        assert _qr.quantized_W.shape == (out_features, in_features)
        assert _qr.quantized_W.min() >= -8
        assert _qr.quantized_W.max() <= 7
        assert not torch.any(torch.isnan(_qr.quantized_W.float()))
        assert not torch.any(torch.isinf(_qr.quantized_W.float()))

    def test_get_hessian_nan_returns_none(self):
        H = torch.full((64, 64), float("nan"))
        cal = {
            "hessians": {"layer.0": H},
            "shapes": {"layer.0": (32, 64)},
            "layer_types": {"layer.0": "Linear"},
        }
        assert get_hessian(cal, "layer.0", torch.Size([32, 64])) is None

    def test_get_hessian_inf_returns_none(self):
        H = torch.zeros(64, 64)
        H[0, 0] = float("inf")
        cal = {
            "hessians": {"layer.0": H},
            "shapes": {"layer.0": (32, 64)},
            "layer_types": {"layer.0": "Linear"},
        }
        assert get_hessian(cal, "layer.0", torch.Size([32, 64])) is None

    def test_get_hessian_finite_returns_hessian(self):
        X = torch.randn(32, 64)
        H = X.T @ X
        cal = {
            "hessians": {"layer.0": H},
            "shapes": {"layer.0": (32, 64)},
            "layer_types": {"layer.0": "Linear"},
        }
        result = get_hessian(cal, "layer.0", torch.Size([32, 64]))
        assert result is not None
        assert torch.allclose(result, H)


# ── Calibration permutations ─────────────────────────────────────────────────


class TestCalibrationPermutation:
    """Tests for using PermuQuant permutations from calibration data."""

    def test_get_permutation_returns_correct_data(self):
        perm = torch.randperm(64, dtype=torch.int32)
        cal = {
            "hessians": {},
            "shapes": {},
            "layer_types": {},
            "permuquant": {"layer.0": perm},
        }
        result = get_permutation(cal, "layer.0")
        assert result is not None
        assert result.dtype == torch.int64
        assert torch.equal(result, perm.to(torch.int64))

    def test_get_permutation_missing_key(self):
        cal = {"hessians": {}, "shapes": {}, "layer_types": {}}
        assert get_permutation(cal, "layer.0") is None

    def test_get_permutation_wrong_layer(self):
        cal = {
            "hessians": {},
            "shapes": {},
            "layer_types": {},
            "permuquant": {"layer.0": torch.arange(64, dtype=torch.int32)},
        }
        assert get_permutation(cal, "layer.1") is None

    def test_calibration_perm_applied_to_weight_only(self):
        """When calibration provides a permutation, GPTQ should use the
        Hessian as-is (already in permuted space) — no double permutation."""
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        perm = torch.randperm(64, dtype=torch.int32)
        X = torch.randn(32, 64)
        X_perm = X[:, perm]
        H_perm = X_perm.T @ X_perm

        W_perm = W[:, perm]
        _qr = gptq_quantize_layer(W_perm, H_perm)
        assert _qr.quantized_W.shape == (16, 64)
        assert _qr.quantized_W.min() >= -8
        assert _qr.quantized_W.max() <= 7

    def test_calibration_perm_matches_manual_perm(self):
        torch.manual_seed(42)
        in_features = 128
        out_features = 16
        W = torch.randn(out_features, in_features)
        perm = torch.randperm(in_features, dtype=torch.int32)
        X = torch.randn(32, in_features)
        H = X.T @ X

        # Path A: manual permute both weight and Hessian
        W_a = W[:, perm]
        H_a = H[perm][:, perm]
        q_a = gptq_quantize_layer(W_a, H_a).quantized_W

        # Path B: "calibration" provides perm; Hessian already in permuted space
        X_perm = X[:, perm]
        H_b = X_perm.T @ X_perm
        W_b = W[:, perm]
        q_b = gptq_quantize_layer(W_b, H_b).quantized_W

        assert torch.equal(q_a, q_b)

    def test_calibration_perm_with_block_diagonal_hessian(self):
        """Calibration permutation should work with block-diagonal Hessians."""
        torch.manual_seed(42)
        in_features = 128
        out_features = 16
        block_size = 32
        W = torch.randn(out_features, in_features)
        perm = torch.randperm(in_features, dtype=torch.int32)

        X = torch.randn(32, in_features)
        X_perm = X[:, perm]
        blocks = []
        for i in range(in_features // block_size):
            start = i * block_size
            end = start + block_size
            xi = X_perm[:, start:end]
            blocks.append(xi.T @ xi)
        H_blocks = torch.stack(blocks)

        W_perm = W[:, perm]
        _qr = gptq_quantize_layer(W_perm, H_blocks)
        assert _qr.quantized_W.shape == (out_features, in_features)
        assert _qr.quantized_W.min() >= -8
        assert _qr.quantized_W.max() <= 7
