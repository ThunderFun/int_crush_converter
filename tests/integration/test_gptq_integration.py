"""Integration tests: mock calibration -> GPTQ -> pack -> unpack."""

import pytest
import torch

from converter.gptq import gptq_quantize_layer, gptq_quantize_layer_rtn
from converter.calibration_io import get_hessian
from converter.scales import calculate_scales_int8, quantize_weights_int8


class TestGPTQIntegration:
    """Integration test: mock calibration -> GPTQ -> pack -> unpack."""

    def test_end_to_end_with_mock_calibration(self):
        from converter.packing import pack_int4, unpack_int4
        from converter.rotation import rotate_weights

        torch.manual_seed(42)
        in_features = 128
        out_features = 32

        W = torch.randn(out_features, in_features)
        W_rot = rotate_weights(W, rot_size=16)

        X = torch.randn(64, in_features)
        H = X.T @ X

        _qr = gptq_quantize_layer(W_rot, H)
        assert _qr.quantized_W.shape == (out_features, in_features)
        assert _qr.scales.shape == (out_features, 1)

        packed = pack_int4(_qr.quantized_W)
        assert packed.dtype == torch.uint8
        assert packed.shape == (out_features, in_features // 2)

        unpacked = unpack_int4(packed, in_features)
        assert torch.equal(unpacked, _qr.quantized_W)

    def test_gptq_with_permuquant_permutation(self):
        from converter.permuquant import find_permutation_weight

        torch.manual_seed(42)
        W = torch.randn(16, 64)
        perm = find_permutation_weight(W, group_size=32)
        W_perm = W[:, perm]

        X = torch.randn(32, 64)
        H = X.T @ X
        H_perm = H[perm][:, perm]

        _qr = gptq_quantize_layer(W_perm, H_perm)
        assert _qr.quantized_W.shape == (16, 64)
        assert _qr.quantized_W.min() >= -8
        assert _qr.quantized_W.max() <= 7

    def test_gptq_with_pre_rotated_calibration(self):
        from converter.rotation import rotate_weights, rotate_activations

        torch.manual_seed(42)
        in_features = 128
        out_features = 16
        rot_size = 128

        W = torch.randn(out_features, in_features)
        W_rot = rotate_weights(W, rot_size=rot_size)

        X = torch.randn(64, in_features)
        X_rot = rotate_activations(X, rot_size)
        H_rot = X_rot.T @ X_rot

        cal = {
            "hessians": {"layer.0": H_rot},
            "shapes": {"layer.0": (out_features, in_features)},
            "layer_types": {"layer.0": "Linear"},
            "metadata": {"hessian_rotated": True, "rot_size": rot_size},
        }

        H = get_hessian(cal, "layer.0", torch.Size([out_features, in_features]))
        assert H is not None
        assert torch.allclose(H, H_rot, atol=1e-5)

        _qr = gptq_quantize_layer(W_rot, H)
        assert _qr.quantized_W.shape == (out_features, in_features)
        assert _qr.quantized_W.min() >= -8
        assert _qr.quantized_W.max() <= 7
        assert not torch.any(torch.isnan(_qr.quantized_W.float()))
        assert not torch.any(torch.isinf(_qr.quantized_W.float()))

    def test_pre_rotated_matches_load_time_rotation(self):
        from converter.rotation import rotate_weights, rotate_activations, rotate_hessian

        torch.manual_seed(42)
        in_features = 256
        out_features = 16
        rot_size = 256

        W = torch.randn(out_features, in_features)
        W_rot = rotate_weights(W, rot_size=rot_size)
        X = torch.randn(64, in_features)

        # Path A: unrotated calibration -> converter rotates at load time
        H_unrot = X.T @ X
        cal_unrot = {
            "hessians": {"layer.0": H_unrot},
            "shapes": {"layer.0": (out_features, in_features)},
            "layer_types": {"layer.0": "Linear"},
            "metadata": {"hessian_rotated": False},
        }
        H_a = get_hessian(cal_unrot, "layer.0", torch.Size([out_features, in_features]))
        H_a = rotate_hessian(H_a, rot_size)
        q_a = gptq_quantize_layer(W_rot, H_a).quantized_W

        # Path B: pre-rotated calibration -> converter skips rotation
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
        q_b = gptq_quantize_layer(W_rot, H_b).quantized_W

        assert torch.allclose(H_a, H_b, atol=1e-4)
        assert torch.equal(q_a, q_b)


# ── INT8 integration ─────────────────────────────────────────────────────────


class TestGPTQIntegrationINT8:
    """Integration tests for INT8 quantization pipeline."""

    def test_end_to_end_rtn(self):
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
        from converter.rotation import rotate_weights

        torch.manual_seed(42)
        in_features = 128
        out_features = 32

        W = torch.randn(out_features, in_features)
        W_rot = rotate_weights(W, rot_size=16)

        X = torch.randn(64, in_features)
        H = X.T @ X

        _qr = gptq_quantize_layer(W_rot, H, int_bits=8)
        assert _qr.quantized_W.shape == (out_features, in_features)
        assert _qr.scales.shape == (out_features, 1)
        assert _qr.quantized_W.min() >= -128
        assert _qr.quantized_W.max() <= 127

        W_deq = _qr.quantized_W.float() * _qr.scales.float()
        mse = (W_rot - W_deq).pow(2).mean().item()
        assert mse < 1.0

    def test_gptq_with_rotation(self):
        from converter.rotation import rotate_weights

        torch.manual_seed(42)
        in_features = 256
        out_features = 32

        W = torch.randn(out_features, in_features)
        W_rot = rotate_weights(W, rot_size=64)

        X = torch.randn(64, in_features)
        H = X.T @ X

        _qr = gptq_quantize_layer(W_rot, H, block_size=64, int_bits=8)
        assert _qr.quantized_W.shape[0] == out_features
        assert _qr.quantized_W.dtype == torch.int8
        assert not torch.any(torch.isnan(_qr.quantized_W.float()))
