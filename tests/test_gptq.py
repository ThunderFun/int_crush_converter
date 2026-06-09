"""Tests for GPTQ quantization and calibration I/O."""

import tempfile

import torch
import pytest

from converter.gptq import gptq_quantize_layer, gptq_quantize_layer_rtn
from converter.calibration_io import (
    load_calibration,
    build_name_map,
    get_hessian,
    get_permutation,
)
from converter.scales import calculate_scales, quantize_weights, calculate_scales_int8, quantize_weights_int8


class TestGPTQQuantizeLayer:
    """Tests for the core GPTQ quantization algorithm."""

    def _make_hessian(self, in_features: int, rank_deficient: bool = False) -> torch.Tensor:
        """Create a realistic positive-definite Hessian matrix."""
        X = torch.randn(64, in_features)
        H = X.T @ X
        if rank_deficient:
            # Zero out last row/col to make it rank-deficient
            H[-1, :] = 0
            H[:, -1] = 0
        return H

    def test_output_range(self):
        """Quantized values must be in [-8, 7]."""
        W = torch.randn(16, 64)
        H = self._make_hessian(64)
        q_W, scales = gptq_quantize_layer(W, H)
        assert q_W.min() >= -8
        assert q_W.max() <= 7

    def test_output_dtype(self):
        """Quantized output should be int8."""
        W = torch.randn(8, 32)
        H = self._make_hessian(32)
        q_W, scales = gptq_quantize_layer(W, H)
        assert q_W.dtype == torch.int8

    def test_scales_dtype(self):
        """Scales should be float16."""
        W = torch.randn(8, 32)
        H = self._make_hessian(32)
        _, scales = gptq_quantize_layer(W, H)
        assert scales.dtype == torch.float16

    def test_scales_shape_per_row(self):
        """Scales should have shape [out_features, 1] (per-row)."""
        W = torch.randn(16, 64)
        H = self._make_hessian(64)
        _, scales = gptq_quantize_layer(W, H)
        assert scales.shape == (16, 1)

    def test_scales_positive(self):
        """All scales must be positive."""
        W = torch.randn(16, 64)
        H = self._make_hessian(64)
        _, scales = gptq_quantize_layer(W, H)
        assert torch.all(scales > 0)

    def test_no_nan_inf_in_quantized(self):
        """Quantized weights and scales should not contain NaN or Inf."""
        W = torch.randn(16, 64)
        H = self._make_hessian(64)
        q_W, scales = gptq_quantize_layer(W, H)
        assert not torch.any(torch.isnan(q_W.float()))
        assert not torch.any(torch.isinf(q_W.float()))
        assert not torch.any(torch.isnan(scales))
        assert not torch.any(torch.isinf(scales))

    def test_zero_weights(self):
        """Zero weights should quantize to zero."""
        W = torch.zeros(8, 32)
        H = self._make_hessian(32)
        q_W, scales = gptq_quantize_layer(W, H)
        assert torch.all(q_W == 0)

    def test_gptq_reduces_error_vs_rtn(self):
        """GPTQ should produce lower quantization error than RTN on a simple example."""
        torch.manual_seed(42)
        in_features = 64
        out_features = 16

        # Create a weight matrix and a realistic Hessian
        W = torch.randn(out_features, in_features)
        X = torch.randn(128, in_features)
        H = X.T @ X  # Hessian from calibration data

        # GPTQ quantization
        q_W_gptq, scales_gptq = gptq_quantize_layer(W, H)

        # RTN quantization (simple round-to-nearest)
        scales_rtn = calculate_scales(W, in_features)
        q_W_rtn = quantize_weights(W, scales_rtn, in_features)

        # Dequantize both
        W_deq_gptq = q_W_gptq.float() * scales_gptq.float()
        W_deq_rtn = q_W_rtn.float() * scales_rtn.float()

        # Compute output error: ||X @ W.T - X @ Q.T||^2
        err_gptq = (X @ (W - W_deq_gptq).T).pow(2).sum().item()
        err_rtn = (X @ (W - W_deq_rtn).T).pow(2).sum().item()

        # GPTQ should be better (or at least not much worse)
        # Allow some tolerance since this is a small example
        assert err_gptq <= err_rtn * 1.1, (
            f"GPTQ error {err_gptq:.2f} > RTN error {err_rtn:.2f}"
        )

    def test_block_size_smaller_than_in_features(self):
        """GPTQ should work when block_size < in_features."""
        W = torch.randn(8, 128)
        H = self._make_hessian(128)
        q_W, scales = gptq_quantize_layer(W, H, block_size=32)
        assert q_W.shape == (8, 128)
        assert q_W.min() >= -8
        assert q_W.max() <= 7

    def test_block_size_equals_in_features(self):
        """GPTQ should work when block_size == in_features (single block)."""
        W = torch.randn(8, 64)
        H = self._make_hessian(64)
        q_W, scales = gptq_quantize_layer(W, H, block_size=64)
        assert q_W.shape == (8, 64)

    def test_damping_prevents_singular_hessian(self):
        """Damping should handle near-singular Hessians gracefully."""
        W = torch.randn(8, 32)
        # Create a very ill-conditioned Hessian
        H = torch.zeros(32, 32)
        H[0, 0] = 1.0  # rank-1
        q_W, scales = gptq_quantize_layer(W, H, damping=0.1)
        assert q_W.shape == (8, 32)
        assert not torch.any(torch.isnan(q_W.float()))

    def test_gptq_with_block_diagonal_hessian(self):
        """GPTQ should work with block-diagonal Hessians (3D tensor)."""
        torch.manual_seed(42)
        in_features = 128
        out_features = 16
        block_size = 32

        W = torch.randn(out_features, in_features)

        # Create block-diagonal Hessian
        num_blocks = in_features // block_size
        blocks = []
        for _ in range(num_blocks):
            X = torch.randn(64, block_size)
            blocks.append(X.T @ X)
        H_block = torch.stack(blocks)  # [num_blocks, block_size, block_size]

        q_W, scales = gptq_quantize_layer(W, H_block)
        assert q_W.shape == (out_features, in_features)
        assert scales.shape == (out_features, 1)
        assert q_W.min() >= -8
        assert q_W.max() <= 7
        assert not torch.any(torch.isnan(q_W.float()))

    def test_shape_mismatch_raises(self):
        """Should raise if Hessian shape doesn't match weight in_features."""
        W = torch.randn(8, 64)
        H = torch.randn(32, 32)
        with pytest.raises(ValueError):
            gptq_quantize_layer(W, H)

    def test_non_2d_weight_raises(self):
        """Should raise for non-2D weight tensor."""
        W = torch.randn(8, 16, 16)
        H = torch.randn(16, 16)
        with pytest.raises(ValueError):
            gptq_quantize_layer(W, H)


class TestGPTQRTNFallback:
    """Tests for the RTN fallback when no calibration data is available."""

    def test_output_range(self):
        W = torch.randn(16, 64)
        q_W, scales = gptq_quantize_layer_rtn(W)
        assert q_W.min() >= -8
        assert q_W.max() <= 7

    def test_output_dtype(self):
        W = torch.randn(8, 32)
        q_W, _ = gptq_quantize_layer_rtn(W)
        assert q_W.dtype == torch.int8

    def test_scales_positive(self):
        W = torch.randn(16, 64)
        _, scales = gptq_quantize_layer_rtn(W)
        assert torch.all(scales > 0)


class TestCalibrationIO:
    """Tests for calibration data loading and name mapping."""

    def _make_mock_calibration(self, layer_names: list[str], in_features: int) -> dict:
        """Create a mock calibration dict matching ComfyUI-GPTQ-Calibration format."""
        hessians = {}
        shapes = {}
        layer_types = {}
        for name in layer_names:
            X = torch.randn(32, in_features)
            hessians[name] = X.T @ X
            shapes[name] = (64, in_features)  # arbitrary out_features
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

    def test_save_load_roundtrip(self):
        """Calibration data should survive a save/load roundtrip."""
        cal = self._make_mock_calibration(["layer.0", "layer.1"], 64)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(cal, f.name)
            loaded = load_calibration(f.name)

        assert set(loaded["hessians"].keys()) == set(cal["hessians"].keys())
        for key in cal["hessians"]:
            assert torch.allclose(loaded["hessians"][key], cal["hessians"][key])

    def test_load_missing_keys_raises(self):
        """Loading a file missing required keys should raise KeyError."""
        bad_data = {"metadata": {}}
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(bad_data, f.name)
            with pytest.raises(KeyError):
                load_calibration(f.name)

    def test_build_name_map_basic(self):
        """Should map state_dict keys to calibration keys by stripping .weight."""
        state_keys = [
            "blocks.0.attn.q_proj.weight",
            "blocks.0.attn.k_proj.weight",
            "blocks.0.norm.weight",  # no .weight suffix in calibration
        ]
        cal_keys = ["blocks.0.attn.q_proj", "blocks.0.attn.k_proj"]
        mapping = build_name_map(state_keys, cal_keys)
        assert mapping == {
            "blocks.0.attn.q_proj.weight": "blocks.0.attn.q_proj",
            "blocks.0.attn.k_proj.weight": "blocks.0.attn.k_proj",
        }

    def test_build_name_map_no_match(self):
        """Should return empty dict if no names match."""
        state_keys = ["model.linear.weight"]
        cal_keys = ["different_layer"]
        mapping = build_name_map(state_keys, cal_keys)
        assert mapping == {}

    def test_build_name_map_skips_non_weight(self):
        """Should skip state_dict keys that don't end with .weight."""
        state_keys = ["model.linear.weight", "model.linear.bias", "embedding"]
        cal_keys = ["model.linear"]
        mapping = build_name_map(state_keys, cal_keys)
        assert mapping == {"model.linear.weight": "model.linear"}

    def test_get_hessian_full(self):
        """Should return full Hessian for 2D tensor."""
        cal = self._make_mock_calibration(["layer.0"], 64)
        H = get_hessian(cal, "layer.0", torch.Size([32, 64]))
        assert H is not None
        assert H.shape == (64, 64)

    def test_get_hessian_not_found(self):
        """Should return None for missing layer."""
        cal = self._make_mock_calibration(["layer.0"], 64)
        H = get_hessian(cal, "missing_layer", torch.Size([32, 64]))
        assert H is None

    def test_get_hessian_block_diagonal(self):
        """Should return raw block-diagonal Hessian (3D tensor)."""
        block_size = 32
        in_features = 64
        num_blocks = in_features // block_size

        blocks = []
        for _ in range(num_blocks):
            X = torch.randn(16, block_size)
            blocks.append(X.T @ X)
        H_block = torch.stack(blocks)  # [num_blocks, block_size, block_size]

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
        """Should accept list-of-blocks format and stack into 3D tensor."""
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
        """List-of-blocks with a smaller last block should be zero-padded."""
        block_size = 32
        in_features = 80  # 2 full blocks + 1 block of 16
        num_blocks = 3

        blocks = []
        for i in range(num_blocks - 1):
            X = torch.randn(16, block_size)
            blocks.append(X.T @ X)
        # last block is smaller
        X = torch.randn(16, 16)
        blocks.append(X.T @ X)  # [16, 16]

        cal = {
            "hessians": {"layer.0": blocks},
            "shapes": {"layer.0": (32, in_features)},
            "layer_types": {"layer.0": "Linear"},
        }

        H = get_hessian(cal, "layer.0", torch.Size([32, in_features]))
        assert H is not None
        assert H.dim() == 3
        assert H.shape == (3, block_size, block_size)
        # first two blocks are exact
        assert torch.allclose(H[0], blocks[0])
        assert torch.allclose(H[1], blocks[1])
        # last block is padded with zeros
        assert torch.allclose(H[2, :16, :16], blocks[2])
        assert torch.all(H[2, 16:, :] == 0)
        assert torch.all(H[2][:, 16:] == 0)

    def test_get_hessian_list_gptq_end_to_end(self):
        """List-of-blocks Hessian should produce correct GPTQ quantization."""
        torch.manual_seed(42)
        block_size = 32
        in_features = 80  # non-divisible: 2x32 + 1x16
        out_features = 16

        W = torch.randn(out_features, in_features)

        blocks = []
        for i in range(2):
            X = torch.randn(64, block_size)
            blocks.append(X.T @ X)
        X = torch.randn(64, 16)
        blocks.append(X.T @ X)  # [16, 16]

        cal = {
            "hessians": {"layer.0": blocks},
            "shapes": {"layer.0": (out_features, in_features)},
            "layer_types": {"layer.0": "Linear"},
        }

        H = get_hessian(cal, "layer.0", torch.Size([out_features, in_features]))
        assert H is not None
        q_W, scales = gptq_quantize_layer(W, H)
        assert q_W.shape == (out_features, in_features)
        assert q_W.min() >= -8
        assert q_W.max() <= 7
        assert not torch.any(torch.isnan(q_W.float()))
        assert not torch.any(torch.isinf(q_W.float()))


class TestGPTQIntegration:
    """Integration test: mock calibration -> GPTQ -> pack -> unpack."""

    def test_end_to_end_with_mock_calibration(self):
        """Full pipeline with mock calibration data."""
        from converter.packing import pack_int4, unpack_int4
        from converter.rotation import rotate_weights

        torch.manual_seed(42)
        in_features = 128
        out_features = 32

        # Create weight matrix and rotate
        W = torch.randn(out_features, in_features)
        W_rot = rotate_weights(W, rot_size=16)

        # Create a realistic Hessian
        X = torch.randn(64, in_features)
        H = X.T @ X

        # GPTQ quantize
        q_W, scales = gptq_quantize_layer(W_rot, H)
        assert q_W.shape == (out_features, in_features)
        assert scales.shape == (out_features, 1)  # per-row scales

        # Pack
        packed = pack_int4(q_W)
        assert packed.dtype == torch.uint8
        assert packed.shape == (out_features, in_features // 2)

        # Unpack and verify roundtrip
        unpacked = unpack_int4(packed, in_features)
        assert torch.equal(unpacked, q_W)

    def test_gptq_with_permuquant_permutation(self):
        """GPTQ should work on permuted weights (PermuQuant compatibility)."""
        from converter.permuquant import find_permutation_weight

        torch.manual_seed(42)
        in_features = 64
        out_features = 16

        W = torch.randn(out_features, in_features)

        # Find permutation
        perm = find_permutation_weight(W, group_size=32)

        # Apply permutation to weights
        W_perm = W[:, perm]

        # Create Hessian and permute it too
        X = torch.randn(32, in_features)
        H = X.T @ X
        H_perm = H[perm][:, perm]

        # GPTQ on permuted data
        q_W, scales = gptq_quantize_layer(W_perm, H_perm)
        assert q_W.shape == (out_features, in_features)
        assert q_W.min() >= -8
        assert q_W.max() <= 7

    def test_gptq_with_pre_rotated_calibration(self):
        """GPTQ should work with calibration data already in rotated space.

        When hessian_rotated=True in metadata, the converter should skip
        re-rotating the Hessian and use it directly.
        """
        from converter.rotation import rotate_weights, rotate_activations, rotate_hessian

        torch.manual_seed(42)
        in_features = 128
        out_features = 16
        rot_size = 128

        W = torch.randn(out_features, in_features)
        W_rot = rotate_weights(W, rot_size=rot_size)

        # Simulate calibration in rotated space:
        # rotated activations → rotated Hessian
        X = torch.randn(64, in_features)
        X_rot = rotate_activations(X, rot_size)
        H_rot = X_rot.T @ X_rot  # already in rotated space

        # This is what the calibration module now produces
        cal = {
            "hessians": {"layer.0": H_rot},
            "shapes": {"layer.0": (out_features, in_features)},
            "layer_types": {"layer.0": "Linear"},
            "metadata": {"hessian_rotated": True, "rot_size": rot_size},
        }

        # Load and verify hessian is returned as-is
        H = get_hessian(cal, "layer.0", torch.Size([out_features, in_features]))
        assert H is not None
        assert torch.allclose(H, H_rot, atol=1e-5)

        # GPTQ on the pre-rotated Hessian with rotated weights
        q_W, scales = gptq_quantize_layer(W_rot, H)
        assert q_W.shape == (out_features, in_features)
        assert q_W.min() >= -8
        assert q_W.max() <= 7
        assert not torch.any(torch.isnan(q_W.float()))
        assert not torch.any(torch.isinf(q_W.float()))

    def test_pre_rotated_matches_load_time_rotation(self):
        """Pre-rotated calibration should produce same GPTQ result as
        loading unrotated data and rotating at load time."""
        from converter.rotation import rotate_weights, rotate_activations, rotate_hessian

        torch.manual_seed(42)
        in_features = 256
        out_features = 16
        rot_size = 256  # must be power of 4 for converter's Regular Hadamard

        W = torch.randn(out_features, in_features)
        W_rot = rotate_weights(W, rot_size=rot_size)

        X = torch.randn(64, in_features)

        # Path A: unrotated calibration → converter rotates at load time
        H_unrot = X.T @ X
        cal_unrot = {
            "hessians": {"layer.0": H_unrot},
            "shapes": {"layer.0": (out_features, in_features)},
            "layer_types": {"layer.0": "Linear"},
            "metadata": {"hessian_rotated": False},
        }
        H_a = get_hessian(cal_unrot, "layer.0", torch.Size([out_features, in_features]))
        H_a = rotate_hessian(H_a, rot_size)  # converter does this
        q_a, s_a = gptq_quantize_layer(W_rot, H_a)

        # Path B: pre-rotated calibration → converter skips rotation
        torch.manual_seed(42)
        W = torch.randn(out_features, in_features)
        W_rot = rotate_weights(W, rot_size=rot_size)
        X = torch.randn(64, in_features)
        X_rot = rotate_activations(X, rot_size)
        H_rot = X_rot.T @ X_rot
        cal_rot = {
            "hessians": {"layer.0": H_rot},
            "shapes": {"layer.0": (out_features, in_features)},
            "layer_types": {"layer.0": "Linear"},
            "metadata": {"hessian_rotated": True, "rot_size": rot_size},
        }
        H_b = get_hessian(cal_rot, "layer.0", torch.Size([out_features, in_features]))
        q_b, s_b = gptq_quantize_layer(W_rot, H_b)

        # Both paths should produce identical Hessians and quantized weights
        assert torch.allclose(H_a, H_b, atol=1e-4)
        assert torch.equal(q_a, q_b)


class TestCalibrationPermutation:
    """Tests for using PermuQuant permutations from calibration data."""

    def test_get_permutation_returns_correct_data(self):
        """get_permutation should return the permutation tensor from calibration."""
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
        """get_permutation should return None when no permuquant data exists."""
        cal = {
            "hessians": {},
            "shapes": {},
            "layer_types": {},
        }
        assert get_permutation(cal, "layer.0") is None

    def test_get_permutation_wrong_layer(self):
        """get_permutation should return None for a layer not in the permuquant dict."""
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
        in_features = 64
        out_features = 16

        W = torch.randn(out_features, in_features)

        # Simulate calibration: permutation + permuted Hessian
        perm = torch.randperm(in_features, dtype=torch.int32)
        X = torch.randn(32, in_features)
        X_perm = X[:, perm]  # reindex activations by permutation
        H_perm = X_perm.T @ X_perm  # Hessian in permuted space

        cal = {
            "hessians": {"layer.0": H_perm},
            "shapes": {"layer.0": (out_features, in_features)},
            "layer_types": {"layer.0": "Linear"},
            "metadata": {"hessian_rotated": False},
            "permuquant": {"layer.0": perm},
        }

        # Apply permutation to weight (what the converter does)
        W_perm = W[:, perm]

        # GPTQ with the already-permuted Hessian — should NOT double-permute
        q_W, scales = gptq_quantize_layer(W_perm, H_perm)
        assert q_W.shape == (out_features, in_features)
        assert q_W.min() >= -8
        assert q_W.max() <= 7

    def test_calibration_perm_matches_manual_perm(self):
        """Using a calibration permutation should produce the same result as
        manually permuting weight + Hessian together."""
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
        q_a, s_a = gptq_quantize_layer(W_a, H_a)

        # Path B: "calibration" provides perm; Hessian already in permuted space
        X_perm = X[:, perm]
        H_b = X_perm.T @ X_perm
        W_b = W[:, perm]
        q_b, s_b = gptq_quantize_layer(W_b, H_b)

        # Both should be identical
        assert torch.equal(q_a, q_b)

    def test_calibration_perm_with_block_diagonal_hessian(self):
        """Calibration permutation should work with block-diagonal Hessians
        (no ValueError — the Hessian is already in permuted space)."""
        torch.manual_seed(42)
        in_features = 128
        out_features = 16
        block_size = 32

        W = torch.randn(out_features, in_features)
        perm = torch.randperm(in_features, dtype=torch.int32)

        # Simulate block-diagonal Hessian in permuted space
        X = torch.randn(32, in_features)
        X_perm = X[:, perm]
        blocks = []
        for i in range(in_features // block_size):
            start = i * block_size
            end = start + block_size
            xi = X_perm[:, start:end]
            blocks.append(xi.T @ xi)
        H_blocks = torch.stack(blocks)  # [4, 32, 32]

        # GPTQ should accept this without raising ValueError
        W_perm = W[:, perm]
        q_W, scales = gptq_quantize_layer(W_perm, H_blocks)
        assert q_W.shape == (out_features, in_features)
        assert q_W.min() >= -8
        assert q_W.max() <= 7


class TestGPTQQuantizeLayerINT8:
    """Tests for the core GPTQ quantization algorithm (INT8)."""

    def _make_hessian(self, in_features: int, rank_deficient: bool = False) -> torch.Tensor:
        """Create a realistic positive-definite Hessian matrix."""
        X = torch.randn(64, in_features)
        H = X.T @ X
        if rank_deficient:
            H[-1, :] = 0
            H[:, -1] = 0
        return H

    def test_output_range(self):
        """Quantized values must be in [-128, 127]."""
        W = torch.randn(16, 64)
        H = self._make_hessian(64)
        q_W, scales = gptq_quantize_layer(W, H, int_bits=8)
        assert q_W.min() >= -128
        assert q_W.max() <= 127

    def test_output_dtype(self):
        """Quantized output should be int8."""
        W = torch.randn(8, 32)
        H = self._make_hessian(32)
        q_W, scales = gptq_quantize_layer(W, H, int_bits=8)
        assert q_W.dtype == torch.int8

    def test_scales_shape(self):
        """Scales should have shape [out_features, 1] (per-row)."""
        W = torch.randn(16, 64)
        H = self._make_hessian(64)
        _, scales = gptq_quantize_layer(W, H, int_bits=8)
        assert scales.shape == (16, 1)

    def test_scales_positive(self):
        """All scales must be positive."""
        W = torch.randn(16, 64)
        H = self._make_hessian(64)
        _, scales = gptq_quantize_layer(W, H, int_bits=8)
        assert torch.all(scales > 0)

    def test_no_nan_inf(self):
        """Quantized weights and scales should not contain NaN or Inf."""
        W = torch.randn(16, 64)
        H = self._make_hessian(64)
        q_W, scales = gptq_quantize_layer(W, H, int_bits=8)
        assert not torch.any(torch.isnan(q_W.float()))
        assert not torch.any(torch.isinf(q_W.float()))
        assert not torch.any(torch.isnan(scales))
        assert not torch.any(torch.isinf(scales))

    def test_zero_weights(self):
        """Zero weights should quantize to zero."""
        W = torch.zeros(8, 32)
        H = self._make_hessian(32)
        q_W, scales = gptq_quantize_layer(W, H, int_bits=8)
        assert torch.all(q_W == 0)

    def test_gptq_reduces_error_vs_rtn(self):
        """GPTQ should produce lower quantization error than RTN on a simple example."""
        torch.manual_seed(42)
        in_features = 64
        out_features = 16

        W = torch.randn(out_features, in_features)
        X = torch.randn(128, in_features)
        H = X.T @ X

        q_W_gptq, scales_gptq = gptq_quantize_layer(W, H, int_bits=8)

        scales_rtn = calculate_scales_int8(W)
        q_W_rtn = quantize_weights_int8(W, scales_rtn)

        W_deq_gptq = q_W_gptq.float() * scales_gptq.float()
        W_deq_rtn = q_W_rtn.float() * scales_rtn.float()

        err_gptq = (X @ (W - W_deq_gptq).T).pow(2).sum().item()
        err_rtn = (X @ (W - W_deq_rtn).T).pow(2).sum().item()

        assert err_gptq <= err_rtn * 1.1, (
            f"GPTQ error {err_gptq:.2f} > RTN error {err_rtn:.2f}"
        )

    def test_block_size_variants(self):
        """GPTQ should work with various block sizes."""
        W = torch.randn(8, 128)
        H = self._make_hessian(128)
        q_W, scales = gptq_quantize_layer(W, H, block_size=32, int_bits=8)
        assert q_W.shape == (8, 128)
        assert q_W.min() >= -128
        assert q_W.max() <= 127

    def test_damping_prevents_singular(self):
        """Damping should handle near-singular Hessians gracefully."""
        W = torch.randn(8, 32)
        H = torch.zeros(32, 32)
        H[0, 0] = 1.0
        q_W, scales = gptq_quantize_layer(W, H, damping=0.1, int_bits=8)
        assert q_W.shape == (8, 32)
        assert not torch.any(torch.isnan(q_W.float()))

    def test_invalid_int_bits_raises(self):
        """Should raise for unsupported int_bits values."""
        W = torch.randn(8, 32)
        H = self._make_hessian(32)
        with pytest.raises(ValueError):
            gptq_quantize_layer(W, H, int_bits=2)


class TestGPTQRTNFallbackINT8:
    """Tests for INT8 RTN fallback."""

    def test_output_range(self):
        W = torch.randn(16, 64)
        q_W, scales = gptq_quantize_layer_rtn(W, int_bits=8)
        assert q_W.min() >= -128
        assert q_W.max() <= 127

    def test_output_dtype(self):
        W = torch.randn(8, 32)
        q_W, _ = gptq_quantize_layer_rtn(W, int_bits=8)
        assert q_W.dtype == torch.int8

    def test_scales_positive(self):
        W = torch.randn(16, 64)
        _, scales = gptq_quantize_layer_rtn(W, int_bits=8)
        assert torch.all(scales > 0)

    def test_invalid_int_bits_raises(self):
        W = torch.randn(8, 32)
        with pytest.raises(ValueError):
            gptq_quantize_layer_rtn(W, int_bits=6)


class TestGPTQIntegrationINT8:
    """Integration tests for INT8 quantization pipeline."""

    def test_end_to_end_rtn(self):
        """Full pipeline with RTN: weights -> rotate -> quantize -> dequantize."""
        from converter.rotation import rotate_weights

        torch.manual_seed(42)
        in_features = 128
        out_features = 32

        W = torch.randn(out_features, in_features)
        W_rot = rotate_weights(W, rot_size=16)

        scales = calculate_scales_int8(W_rot)
        q_W = quantize_weights_int8(W_rot, scales)

        assert q_W.shape == (out_features, in_features)
        assert q_W.dtype == torch.int8
        assert q_W.min() >= -128
        assert q_W.max() <= 127

        W_deq = q_W.float() * scales.float()
        mse = (W_rot - W_deq).pow(2).mean().item()
        assert mse < 1.0

    def test_end_to_end_gptq(self):
        """Full pipeline with GPTQ: weights -> rotate -> quantize -> dequantize."""
        from converter.rotation import rotate_weights

        torch.manual_seed(42)
        in_features = 128
        out_features = 32

        W = torch.randn(out_features, in_features)
        W_rot = rotate_weights(W, rot_size=16)

        X = torch.randn(64, in_features)
        H = X.T @ X

        q_W, scales = gptq_quantize_layer(W_rot, H, int_bits=8)

        assert q_W.shape == (out_features, in_features)
        assert scales.shape == (out_features, 1)
        assert q_W.min() >= -128
        assert q_W.max() <= 127

        W_deq = q_W.float() * scales.float()
        mse = (W_rot - W_deq).pow(2).mean().item()
        assert mse < 1.0

    def test_gptq_with_rotation(self):
        """GPTQ should work on rotated weights."""
        from converter.rotation import rotate_weights

        torch.manual_seed(42)
        in_features = 256
        out_features = 32

        W = torch.randn(out_features, in_features)
        W_rot = rotate_weights(W, rot_size=64)

        X = torch.randn(64, in_features)
        H = X.T @ X

        q_W, scales = gptq_quantize_layer(W_rot, H, block_size=64, int_bits=8)

        assert q_W.shape[0] == out_features
        assert q_W.dtype == torch.int8
        assert not torch.any(torch.isnan(q_W.float()))
