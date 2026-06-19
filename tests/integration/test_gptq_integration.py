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


# ── DLR Hessian integration ──────────────────────────────────────────────────


def _make_dlr_from_hessian(H: torch.Tensor, rank: int) -> dict:
    """Decompose a dense Hessian into a DLR dict via eigendecomposition."""
    from converter.dlr import make_dlr_dict

    n = H.shape[0]
    D = H.diagonal().clone()
    eigvals, eigvecs = torch.linalg.eigh(H)
    r = min(rank, n)
    top_vals = eigvals[-r:].clamp(min=0)
    top_vecs = eigvecs[:, -r:]
    U = top_vecs * top_vals.sqrt().unsqueeze(0)
    return make_dlr_dict(D, U)


class TestGPTQIntegrationDLR:
    """Integration tests for GPTQ with DLR (Diagonal + Low-Rank) Hessians."""

    def test_dlr_end_to_end_int4(self):
        """Full flow: DLR calibration dict → get_hessian → GPTQ → pack → unpack."""
        from converter.packing import pack_int4, unpack_int4

        torch.manual_seed(42)
        in_features = 128
        out_features = 32

        W = torch.randn(out_features, in_features)
        X = torch.randn(64, in_features)
        H_true = X.T @ X
        dlr = _make_dlr_from_hessian(H_true, rank=16)

        cal = {
            "hessians": {"layer.0": dlr},
            "shapes": {"layer.0": (out_features, in_features)},
            "layer_types": {"layer.0": "Linear"},
            "metadata": {"hessian_format": "dlr", "dlr_rank": 16},
        }

        H = get_hessian(cal, "layer.0", torch.Size([out_features, in_features]))
        assert H is not None

        qr = gptq_quantize_layer(W, H, int_bits=4)
        assert qr.quantized_W.shape == (out_features, in_features)
        assert qr.quantized_W.min() >= -8
        assert qr.quantized_W.max() <= 7
        assert not torch.any(torch.isnan(qr.quantized_W.float()))

        packed = pack_int4(qr.quantized_W)
        assert packed.dtype == torch.uint8
        unpacked = unpack_int4(packed, in_features)
        assert torch.equal(unpacked, qr.quantized_W)

    def test_dlr_end_to_end_int8(self):
        torch.manual_seed(42)
        in_features = 128
        out_features = 32

        W = torch.randn(out_features, in_features)
        X = torch.randn(64, in_features)
        H_true = X.T @ X
        dlr = _make_dlr_from_hessian(H_true, rank=16)

        qr = gptq_quantize_layer(W, dlr, int_bits=8)
        assert qr.quantized_W.shape == (out_features, in_features)
        assert qr.quantized_W.min() >= -128
        assert qr.quantized_W.max() <= 127

        W_deq = qr.quantized_W.float() * qr.scales.float()
        mse = (W - W_deq).pow(2).mean().item()
        assert mse < 1.0

    def test_dlr_with_rotation_pre_rotated(self):
        """DLR Hessian already in rotated space (calibration collected with rotation)."""
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
        dlr = _make_dlr_from_hessian(H_rot, rank=16)

        cal = {
            "hessians": {"layer.0": dlr},
            "shapes": {"layer.0": (out_features, in_features)},
            "layer_types": {"layer.0": "Linear"},
            "metadata": {"hessian_rotated": True, "rot_size": rot_size,
                         "hessian_format": "dlr", "dlr_rank": 16},
        }

        H = get_hessian(cal, "layer.0", torch.Size([out_features, in_features]))
        assert H is not None
        qr = gptq_quantize_layer(W_rot, H, int_bits=8)
        assert qr.quantized_W.shape == (out_features, in_features)
        assert not torch.any(torch.isnan(qr.quantized_W.float()))

    def test_dlr_with_load_time_rotation(self):
        """DLR Hessian rotated at load time (calibration collected without rotation)."""
        from converter.rotation import rotate_weights, rotate_hessian

        torch.manual_seed(42)
        in_features = 64
        out_features = 16
        rot_size = 64

        W = torch.randn(out_features, in_features)
        W_rot = rotate_weights(W, rot_size=rot_size)

        X = torch.randn(64, in_features)
        H_true = X.T @ X
        dlr = _make_dlr_from_hessian(H_true, rank=16)

        # Rotate DLR → becomes dense 2-D (rotation doesn't preserve DLR structure)
        H_rot = rotate_hessian(dlr, rot_size)
        assert H_rot.dim() == 2  # DLR was materialised to dense for rotation

        qr = gptq_quantize_layer(W_rot, H_rot, int_bits=8)
        assert qr.quantized_W.shape == (out_features, in_features)
        assert not torch.any(torch.isnan(qr.quantized_W.float()))

    def test_dlr_quality_comparable_to_full(self):
        """DLR GPTQ quality should be comparable to full-Hessian GPTQ."""
        torch.manual_seed(42)
        in_features = 128
        out_features = 16

        W = torch.randn(out_features, in_features)
        X = torch.randn(128, in_features)
        H_true = X.T @ X
        dlr = _make_dlr_from_hessian(H_true, rank=32)

        qr_full = gptq_quantize_layer(W, H_true, int_bits=8)
        qr_dlr = gptq_quantize_layer(W, dlr, int_bits=8)

        # DLR should be within 3× of full Hessian quality
        assert qr_dlr.mse <= qr_full.mse * 3.0 + 1e-8, (
            f"DLR MSE {qr_dlr.mse:.6f} >> full MSE {qr_full.mse:.6f}"
        )
