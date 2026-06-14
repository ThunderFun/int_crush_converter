"""Tests for the GPTQ RTN fallback (no Hessian, INT4 and INT8)."""

import torch
import pytest

from converter.gptq import gptq_quantize_layer_rtn


class TestGPTQRTNFallback:
    """Tests for the RTN fallback when no calibration data is available."""

    def test_output_range(self):
        _qr = gptq_quantize_layer_rtn(torch.randn(16, 64))
        assert _qr.quantized_W.min() >= -8
        assert _qr.quantized_W.max() <= 7

    def test_output_dtype(self):
        assert gptq_quantize_layer_rtn(torch.randn(8, 32)).quantized_W.dtype == torch.int8

    def test_scales_positive(self):
        _qr = gptq_quantize_layer_rtn(torch.randn(16, 64))
        _ = _qr.quantized_W
        assert torch.all(_qr.scales > 0)


class TestGPTQRTNFallbackINT8:
    """Tests for INT8 RTN fallback."""

    def test_output_range(self):
        _qr = gptq_quantize_layer_rtn(torch.randn(16, 64), int_bits=8)
        assert _qr.quantized_W.min() >= -128
        assert _qr.quantized_W.max() <= 127

    def test_output_dtype(self):
        assert gptq_quantize_layer_rtn(torch.randn(8, 32), int_bits=8).quantized_W.dtype == torch.int8

    def test_scales_positive(self):
        _qr = gptq_quantize_layer_rtn(torch.randn(16, 64), int_bits=8)
        _ = _qr.quantized_W
        assert torch.all(_qr.scales > 0)

    def test_invalid_int_bits_raises(self):
        with pytest.raises(ValueError):
            gptq_quantize_layer_rtn(torch.randn(8, 32), int_bits=6)
