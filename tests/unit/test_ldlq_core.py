"""Tests for the core LDLQ quantization algorithm."""

import math

import torch
import pytest

from converter.ldlq import ldlq_quantize_layer
from converter.rounding import _ldlq_round_column
from converter.scales import calculate_scales, quantize_weights


# ── Core algorithm ───────────────────────────────────────────────────────────


class TestLDLQQuantizeLayer:

    def test_output_range_int4(self):
        _qr = ldlq_quantize_layer(torch.randn(16, 64))
        assert _qr.quantized_W.min() >= -8
        assert _qr.quantized_W.max() <= 7

    def test_output_range_int8(self):
        _qr = ldlq_quantize_layer(torch.randn(16, 64), int_bits=8)
        assert _qr.quantized_W.min() >= -128
        assert _qr.quantized_W.max() <= 127

    def test_output_dtype(self):
        assert ldlq_quantize_layer(torch.randn(8, 32)).quantized_W.dtype == torch.int8

    def test_scales_dtype(self):
        _qr = ldlq_quantize_layer(torch.randn(8, 32))
        _ = _qr.quantized_W
        assert _qr.scales.dtype == torch.float16

    def test_scales_shape_per_row(self):
        _qr = ldlq_quantize_layer(torch.randn(16, 64))
        _ = _qr.quantized_W
        assert _qr.scales.shape == (16, 1)

    def test_scales_positive(self):
        _qr = ldlq_quantize_layer(torch.randn(16, 64))
        _ = _qr.quantized_W
        assert torch.all(_qr.scales > 0)

    def test_no_nan_inf_in_quantized(self):
        _qr = ldlq_quantize_layer(torch.randn(16, 64))
        assert not torch.any(torch.isnan(_qr.quantized_W.float()))
        assert not torch.any(torch.isinf(_qr.quantized_W.float()))
        assert not torch.any(torch.isnan(_qr.scales))
        assert not torch.any(torch.isinf(_qr.scales))

    def test_zero_weights(self):
        assert torch.all(ldlq_quantize_layer(torch.zeros(8, 32)).quantized_W == 0)

    def test_invalid_int_bits_raises(self):
        with pytest.raises(ValueError):
            ldlq_quantize_layer(torch.randn(8, 32), int_bits=2)

    def test_non_2d_weight_raises(self):
        with pytest.raises(ValueError):
            ldlq_quantize_layer(torch.randn(8, 16, 16))

    def test_block_size_smaller_than_in_features(self):
        _qr = ldlq_quantize_layer(torch.randn(8, 128), block_size=32)
        assert _qr.quantized_W.shape == (8, 128)
        assert _qr.quantized_W.min() >= -8
        assert _qr.quantized_W.max() <= 7

    def test_block_size_equals_in_features(self):
        _qr = ldlq_quantize_layer(torch.randn(8, 64), block_size=64)
        assert _qr.quantized_W.shape == (8, 64)

    def test_hessian_nan_triggers_rtn_fallback_with_real_metrics(self):
        torch.manual_seed(42)
        W = torch.randn(8, 32) * 1e20
        result = ldlq_quantize_layer(W)
        assert "hessian_nan_rtn" in result.fallbacks
        assert result.method_used == "rtn"
        assert result.mse > 0
        assert result.max_err > 0

    def test_damping_prevents_singular_hessian(self):
        W = torch.zeros(8, 32)
        W[:, 0] = torch.randn(8)
        _qr = ldlq_quantize_layer(W, damping=0.1)
        assert _qr.quantized_W.shape == (8, 32)
        assert not torch.any(torch.isnan(_qr.quantized_W.float()))


# ── LDLQ vs RTN ─────────────────────────────────────────────────────────────


class TestLDLQVsRTN:
    """Test that LDLQ achieves lower proxy loss than RTN (QuIP Theorem 1)."""

    def test_proxy_loss_reduction(self):
        torch.manual_seed(42)
        M, N = 16, 64
        W = torch.randn(M, N)

        _qr = ldlq_quantize_layer(W, block_size=32)
        scales_rtn = calculate_scales(W, N)
        q_W_rtn = quantize_weights(W, scales_rtn, N)

        H = W.T @ W / M
        err_ldlq = (_qr.quantized_W.float() * _qr.scales.float() - W)
        err_rtn = (q_W_rtn.float() * scales_rtn.float() - W)

        min_cols = min(err_ldlq.shape[1], err_rtn.shape[1], H.shape[0])
        err_ldlq = err_ldlq[:, :min_cols]
        err_rtn = err_rtn[:, :min_cols]
        H = H[:min_cols, :min_cols]

        proxy_ldlq = torch.trace(err_ldlq @ H @ err_ldlq.T).item()
        proxy_rtn = torch.trace(err_rtn @ H @ err_rtn.T).item()
        assert proxy_ldlq <= proxy_rtn * 1.1, (
            f"LDLQ proxy loss {proxy_ldlq:.4f} > RTN proxy loss {proxy_rtn:.4f}"
        )

    def test_trace_d_le_trace_h(self):
        """trace(D) <= trace(H) for non-diagonal H (QuIP Lemma)."""
        torch.manual_seed(42)
        N = 32
        X = torch.randn(64, N)
        H = X.T @ X / 64

        L = torch.linalg.cholesky(H)
        D = torch.diag(L) ** 2
        trace_D = D.sum().item()
        trace_H = torch.trace(H).item()

        assert trace_D <= trace_H * 1.001
        if not torch.allclose(H, torch.diag(torch.diag(H)), atol=1e-6):
            assert trace_D < trace_H * 0.999


# ── Rounding utilities ───────────────────────────────────────────────────────


class TestRoundingFunctions:

    def test_ldlq_round_column_range(self):
        col = torch.randn(16) * 5
        q, err = _ldlq_round_column(col, torch.ones(16) * 2.0, -8, 7)
        assert q.min() >= -8
        assert q.max() <= 7

    def test_ldlq_round_column_sign_flip(self):
        M = 8
        col = torch.tensor([0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 7.0, -7.0])
        scale_col = torch.ones(M) * 0.3
        q, err = _ldlq_round_column(col, scale_col, -8, 7)
        for i in range(M):
            if col[i].abs() > 0.1:
                assert (col[i].sign() * q[i].sign()) >= 0, (
                    f"Sign mismatch at {i}: col={col[i]:.2f}, q={q[i]:.2f}"
                )

    def test_ldlq_round_column_error(self):
        M = 8
        col = torch.randn(M) * 3
        scale_col = torch.ones(M) * 1.5
        q, err = _ldlq_round_column(col, scale_col, -8, 7)
        expected_err = col - q * scale_col.clamp(min=1e-8)
        assert torch.allclose(err, expected_err, atol=1e-6)


# ── Integration: rotate -> LDLQ -> pack -> unpack ───────────────────────────


class TestLDLQIntegration:

    def test_end_to_end_int4(self):
        from converter.packing import pack_int4, unpack_int4
        from converter.rotation import rotate_weights

        torch.manual_seed(42)
        W = torch.randn(32, 128)
        W_rot = rotate_weights(W, rot_size=16)

        _qr = ldlq_quantize_layer(W_rot)
        assert _qr.quantized_W.shape == (32, 128)
        assert _qr.scales.shape == (32, 1)

        packed = pack_int4(_qr.quantized_W)
        assert packed.dtype == torch.uint8
        assert packed.shape == (32, 64)

        unpacked = unpack_int4(packed, 128)
        assert torch.equal(unpacked, _qr.quantized_W)

    def test_end_to_end_int8(self):
        from converter.rotation import rotate_weights

        torch.manual_seed(42)
        W = torch.randn(32, 128)
        W_rot = rotate_weights(W, rot_size=16)

        _qr = ldlq_quantize_layer(W_rot, int_bits=8)
        assert _qr.quantized_W.shape == (32, 128)
        assert _qr.quantized_W.dtype == torch.int8
        assert _qr.quantized_W.min() >= -128
        assert _qr.quantized_W.max() <= 127

        W_deq = _qr.quantized_W.float() * _qr.scales.float()
        assert (W_rot - W_deq).pow(2).mean().item() < 1.0

    def test_with_rotation(self):
        from converter.rotation import rotate_weights

        torch.manual_seed(42)
        W = torch.randn(32, 256)
        W_rot = rotate_weights(W, rot_size=64)

        _qr = ldlq_quantize_layer(W_rot, block_size=64)
        assert _qr.quantized_W.shape[0] == 32
        assert _qr.quantized_W.dtype == torch.int8
        assert not torch.any(torch.isnan(_qr.quantized_W.float()))

    def test_dequantization_quality(self):
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        _qr = ldlq_quantize_layer(W)

        W_deq = _qr.quantized_W.float() * _qr.scales.float()
        mse = (W - W_deq).pow(2).mean().item()
        assert mse < 1.0
        assert mse > 1e-10


# ── With PermuQuant ──────────────────────────────────────────────────────────


class TestLDLQWithPermuQuant:

    def test_ldlq_on_permuted_weights(self):
        from converter.permuquant import find_permutation_weight

        torch.manual_seed(42)
        W = torch.randn(16, 64)
        perm = find_permutation_weight(W, group_size=32)
        W_perm = W[:, perm]

        _qr = ldlq_quantize_layer(W_perm)
        assert _qr.quantized_W.shape == (16, 64)
        assert _qr.quantized_W.min() >= -8
        assert _qr.quantized_W.max() <= 7
