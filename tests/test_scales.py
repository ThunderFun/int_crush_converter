"""Tests for quant.scales — per-row scale calculation."""

import torch

from converter.scales import calculate_scales, quantize_weights


class TestCalculateScales:
    def test_shape(self):
        W = torch.randn(32, 64)
        scales = calculate_scales(W)
        assert scales.shape == (32, 1)

    def test_dtype(self):
        W = torch.randn(16, 64)
        scales = calculate_scales(W)
        assert scales.dtype == torch.float16

    def test_no_nan_inf(self):
        W = torch.randn(16, 64)
        scales = calculate_scales(W)
        assert not torch.any(torch.isnan(scales))
        assert not torch.any(torch.isinf(scales))

    def test_positive(self):
        W = torch.randn(16, 64)
        scales = calculate_scales(W)
        assert torch.all(scales > 0)

    def test_scale_value(self):
        """Scale should be max(|W|) / 7.0 per row."""
        W = torch.tensor([[1.0, -2.0, 3.0], [-7.0, 1.0, 0.5]], dtype=torch.float32)
        scales = calculate_scales(W)
        expected_row0 = 3.0 / 7.0
        expected_row1 = 7.0 / 7.0
        assert torch.allclose(scales[0].float(), torch.tensor(expected_row0), atol=1e-3)
        assert torch.allclose(scales[1].float(), torch.tensor(expected_row1), atol=1e-3)


class TestQuantizeWeights:
    def test_output_range(self):
        W = torch.randn(16, 64, dtype=torch.float32)
        scales = calculate_scales(W)
        quantized = quantize_weights(W, scales)
        assert quantized.min() >= -8
        assert quantized.max() <= 7

    def test_dtype(self):
        W = torch.randn(16, 64)
        scales = calculate_scales(W)
        quantized = quantize_weights(W, scales)
        assert quantized.dtype == torch.int8

    def test_zero_weight(self):
        W = torch.zeros(4, 16)
        scales = calculate_scales(W)
        quantized = quantize_weights(W, scales)
        assert torch.all(quantized == 0)
