"""Tests for the core GPTQ quantization algorithm (INT4 and INT8)."""

import torch
import pytest

from converter.gptq import gptq_quantize_layer
from converter.scales import calculate_scales, quantize_weights, calculate_scales_int8, quantize_weights_int8


def _make_hessian(in_features: int, rank_deficient: bool = False) -> torch.Tensor:
    """Create a realistic positive-definite Hessian matrix."""
    X = torch.randn(64, in_features)
    H = X.T @ X
    if rank_deficient:
        H[-1, :] = 0
        H[:, -1] = 0
    return H


# ── INT4 core ────────────────────────────────────────────────────────────────


class TestGPTQQuantizeLayer:
    """Tests for the core GPTQ quantization algorithm."""

    def test_output_range(self):
        """Quantized values must be in [-8, 7]."""
        W = torch.randn(16, 64)
        H = _make_hessian(64)
        _qr = gptq_quantize_layer(W, H)
        assert _qr.quantized_W.min() >= -8
        assert _qr.quantized_W.max() <= 7

    def test_output_dtype(self):
        W = torch.randn(8, 32)
        H = _make_hessian(32)
        assert gptq_quantize_layer(W, H).quantized_W.dtype == torch.int8

    def test_scales_dtype(self):
        W = torch.randn(8, 32)
        H = _make_hessian(32)
        _qr = gptq_quantize_layer(W, H)
        _ = _qr.quantized_W
        assert _qr.scales.dtype == torch.float16

    def test_scales_shape_per_row(self):
        W = torch.randn(16, 64)
        H = _make_hessian(64)
        _qr = gptq_quantize_layer(W, H)
        _ = _qr.quantized_W
        assert _qr.scales.shape == (16, 1)

    def test_scales_positive(self):
        W = torch.randn(16, 64)
        H = _make_hessian(64)
        _qr = gptq_quantize_layer(W, H)
        _ = _qr.quantized_W
        assert torch.all(_qr.scales > 0)

    def test_no_nan_inf_in_quantized(self):
        W = torch.randn(16, 64)
        H = _make_hessian(64)
        _qr = gptq_quantize_layer(W, H)
        assert not torch.any(torch.isnan(_qr.quantized_W.float()))
        assert not torch.any(torch.isinf(_qr.quantized_W.float()))
        assert not torch.any(torch.isnan(_qr.scales))
        assert not torch.any(torch.isinf(_qr.scales))

    def test_zero_weights(self):
        W = torch.zeros(8, 32)
        H = _make_hessian(32)
        assert torch.all(gptq_quantize_layer(W, H).quantized_W == 0)

    def test_gptq_reduces_error_vs_rtn(self):
        """GPTQ should produce lower quantization error than RTN."""
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        X = torch.randn(128, 64)
        H = X.T @ X

        _qr = gptq_quantize_layer(W, H)
        scales_rtn = calculate_scales(W, 64)
        q_W_rtn = quantize_weights(W, scales_rtn, 64)

        if _qr.zero_points is not None:
            W_deq_gptq = (_qr.quantized_W.float() - _qr.zero_points.float()) * _qr.scales.float()
        else:
            W_deq_gptq = _qr.quantized_W.float() * _qr.scales.float()
        W_deq_rtn = q_W_rtn.float() * scales_rtn.float()

        err_gptq = (X @ (W - W_deq_gptq).T).pow(2).sum().item()
        err_rtn = (X @ (W - W_deq_rtn).T).pow(2).sum().item()
        assert err_gptq <= err_rtn * 1.1, f"GPTQ error {err_gptq:.2f} > RTN error {err_rtn:.2f}"

    def test_block_size_smaller_than_in_features(self):
        W = torch.randn(8, 128)
        H = _make_hessian(128)
        _qr = gptq_quantize_layer(W, H, block_size=32)
        assert _qr.quantized_W.shape == (8, 128)
        assert _qr.quantized_W.min() >= -8
        assert _qr.quantized_W.max() <= 7

    def test_block_size_equals_in_features(self):
        W = torch.randn(8, 64)
        H = _make_hessian(64)
        assert gptq_quantize_layer(W, H, block_size=64).quantized_W.shape == (8, 64)

    def test_damping_prevents_singular_hessian(self):
        W = torch.randn(8, 32)
        H = torch.zeros(32, 32)
        H[0, 0] = 1.0
        _qr = gptq_quantize_layer(W, H, damping=0.1)
        assert _qr.quantized_W.shape == (8, 32)
        assert not torch.any(torch.isnan(_qr.quantized_W.float()))

    def test_gptq_with_block_diagonal_hessian(self):
        """GPTQ should work with block-diagonal Hessians (3D tensor)."""
        torch.manual_seed(42)
        W = torch.randn(16, 128)
        blocks = []
        for _ in range(4):
            X = torch.randn(64, 32)
            blocks.append(X.T @ X)
        H_block = torch.stack(blocks)

        _qr = gptq_quantize_layer(W, H_block)
        assert _qr.quantized_W.shape == (16, 128)
        assert _qr.scales.shape == (16, 1)
        assert _qr.quantized_W.min() >= -8
        assert _qr.quantized_W.max() <= 7
        assert not torch.any(torch.isnan(_qr.quantized_W.float()))

    def test_shape_mismatch_raises(self):
        W = torch.randn(8, 64)
        H = torch.randn(32, 32)
        with pytest.raises(ValueError):
            gptq_quantize_layer(W, H)

    def test_non_2d_weight_raises(self):
        W = torch.randn(8, 16, 16)
        H = torch.randn(16, 16)
        with pytest.raises(ValueError):
            gptq_quantize_layer(W, H)


# ── INT8 core ────────────────────────────────────────────────────────────────


class TestGPTQQuantizeLayerINT8:
    """Tests for the core GPTQ quantization algorithm (INT8)."""

    def test_output_range(self):
        W = torch.randn(16, 64)
        H = _make_hessian(64)
        _qr = gptq_quantize_layer(W, H, int_bits=8)
        assert _qr.quantized_W.min() >= -128
        assert _qr.quantized_W.max() <= 127

    def test_output_dtype(self):
        W = torch.randn(8, 32)
        H = _make_hessian(32)
        assert gptq_quantize_layer(W, H, int_bits=8).quantized_W.dtype == torch.int8

    def test_scales_shape(self):
        W = torch.randn(16, 64)
        H = _make_hessian(64)
        _qr = gptq_quantize_layer(W, H, int_bits=8)
        _ = _qr.quantized_W
        assert _qr.scales.shape == (16, 1)

    def test_scales_positive(self):
        W = torch.randn(16, 64)
        H = _make_hessian(64)
        _qr = gptq_quantize_layer(W, H, int_bits=8)
        _ = _qr.quantized_W
        assert torch.all(_qr.scales > 0)

    def test_no_nan_inf(self):
        W = torch.randn(16, 64)
        H = _make_hessian(64)
        _qr = gptq_quantize_layer(W, H, int_bits=8)
        assert not torch.any(torch.isnan(_qr.quantized_W.float()))
        assert not torch.any(torch.isinf(_qr.quantized_W.float()))
        assert not torch.any(torch.isnan(_qr.scales))
        assert not torch.any(torch.isinf(_qr.scales))

    def test_zero_weights(self):
        W = torch.zeros(8, 32)
        H = _make_hessian(32)
        assert torch.all(gptq_quantize_layer(W, H, int_bits=8).quantized_W == 0)

    def test_gptq_reduces_error_vs_rtn(self):
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        X = torch.randn(128, 64)
        H = X.T @ X

        _qr = gptq_quantize_layer(W, H, int_bits=8)
        scales_rtn = calculate_scales_int8(W)
        q_W_rtn = quantize_weights_int8(W, scales_rtn)

        W_deq_gptq = _qr.quantized_W.float() * _qr.scales.float()
        W_deq_rtn = q_W_rtn.float() * scales_rtn.float()

        err_gptq = (X @ (W - W_deq_gptq).T).pow(2).sum().item()
        err_rtn = (X @ (W - W_deq_rtn).T).pow(2).sum().item()
        assert err_gptq <= err_rtn * 1.1, f"GPTQ error {err_gptq:.2f} > RTN error {err_rtn:.2f}"

    def test_block_size_variants(self):
        W = torch.randn(8, 128)
        H = _make_hessian(128)
        _qr = gptq_quantize_layer(W, H, block_size=32, int_bits=8)
        assert _qr.quantized_W.shape == (8, 128)
        assert _qr.quantized_W.min() >= -128
        assert _qr.quantized_W.max() <= 127

    def test_damping_prevents_singular(self):
        W = torch.randn(8, 32)
        H = torch.zeros(32, 32)
        H[0, 0] = 1.0
        _qr = gptq_quantize_layer(W, H, damping=0.1, int_bits=8)
        assert _qr.quantized_W.shape == (8, 32)
        assert not torch.any(torch.isnan(_qr.quantized_W.float()))

    def test_invalid_int_bits_raises(self):
        W = torch.randn(8, 32)
        H = _make_hessian(32)
        with pytest.raises(ValueError):
            gptq_quantize_layer(W, H, int_bits=2)
