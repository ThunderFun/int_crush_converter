"""Tests for quant.scales — per-row scale calculation and scale clipping."""

import pytest
import torch

from converter.config import FP16_SCALE_FLOOR, MAX_FP16_SCALE, INT4_SCALE_DIVISOR, INT8_SCALE_DIVISOR
from converter.gptq import gptq_quantize_layer, gptq_quantize_layer_rtn
from converter.scales import (
    _search_clipping_ratio,
    calculate_scales,
    quantize_weights,
    calculate_scales_int8,
    quantize_weights_int8,
    calculate_scales_asymmetric,
    quantize_weights_asymmetric,
    calculate_scales_int8_asymmetric,
    quantize_weights_int8_asymmetric,
)


def _all_finite_fp16(t: torch.Tensor) -> bool:
    """Return True if tensor is float16 with no Inf/NaN."""
    return t.dtype == torch.float16 and t.isfinite().all().item()


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

    def test_group_wise(self):
        """With group_size=32, a 64-wide tensor should produce 2 groups."""
        W = torch.randn(8, 64)
        scales = calculate_scales(W, group_size=32)
        assert scales.shape == (8, 2)

    def test_clipping_ratios_no_worse(self):
        """Clipping ratio search should produce MSE <= no-clipping."""
        torch.manual_seed(42)
        W = torch.randn(16, 128)
        scales_no_clip = calculate_scales(W)
        scales_clip = calculate_scales(W, clipping_ratios=[0.85, 0.9, 0.95, 1.0])

        # Both should produce valid scales
        assert scales_clip.shape == scales_no_clip.shape
        assert torch.all(scales_clip > 0)

        # Quantize with both and compare MSE
        q_no_clip = quantize_weights(W, scales_no_clip)
        q_clip = quantize_weights(W, scales_clip)
        dequant_no = q_no_clip.float() * scales_no_clip.float()
        dequant_clip = q_clip.float() * scales_clip.float()
        mse_no = (W - dequant_no).pow(2).mean()
        mse_clip = (W - dequant_clip).pow(2).mean()
        # Clipping should be <= no-clipping (per-group selection)
        assert mse_clip <= mse_no + 1e-6

    def test_clipping_ratios_group_wise(self):
        """Clipping ratio search should work with group-wise scales."""
        torch.manual_seed(42)
        W = torch.randn(8, 64)
        scales = calculate_scales(W, group_size=32, clipping_ratios=[0.9, 1.0])
        assert scales.shape == (8, 2)


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


class TestINT8:
    def test_shape(self):
        W = torch.randn(16, 64)
        scales = calculate_scales_int8(W)
        assert scales.shape == (16, 1)

    def test_output_range(self):
        W = torch.randn(16, 64)
        scales = calculate_scales_int8(W)
        quantized = quantize_weights_int8(W, scales)
        assert quantized.min() >= -128
        assert quantized.max() <= 127

    def test_int8_clipping_ratios(self):
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        scales_no = calculate_scales_int8(W)
        scales_clip = calculate_scales_int8(W, clipping_ratios=[0.85, 0.9, 0.95, 1.0])
        assert scales_clip.shape == scales_no.shape
        assert torch.all(scales_clip > 0)


class TestAsymmetricINT4:
    def test_shape(self):
        W = torch.randn(16, 64)
        scales, zps = calculate_scales_asymmetric(W)
        assert scales.shape == (16, 1)
        assert zps.shape == (16, 1)

    def test_dtype(self):
        W = torch.randn(16, 64)
        scales, zps = calculate_scales_asymmetric(W)
        assert scales.dtype == torch.float16
        assert zps.dtype == torch.int8

    def test_zero_point_range(self):
        W = torch.randn(16, 64)
        _, zps = calculate_scales_asymmetric(W)
        assert zps.min() >= -8
        assert zps.max() <= 7

    def test_quantize_range(self):
        W = torch.randn(16, 64, dtype=torch.float32)
        scales, zps = calculate_scales_asymmetric(W)
        quantized = quantize_weights_asymmetric(W, scales, zps)
        assert quantized.min() >= -8
        assert quantized.max() <= 7
        assert quantized.dtype == torch.int8

    def test_roundtrip(self):
        """Dequantized values should approximate original weights."""
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        scales, zps = calculate_scales_asymmetric(W)
        quantized = quantize_weights_asymmetric(W, scales, zps)
        # Slice to original in_features (quantize pads to group_size)
        dequant = (quantized.float() - zps.float()) * scales.float()
        dequant = dequant[:, :W.shape[1]]
        mse = (W - dequant).pow(2).mean().item()
        assert mse < 1.0  # loose sanity check

    def test_asymmetric_vs_symmetric_skewed(self):
        """Asymmetric should do better on skewed distributions."""
        torch.manual_seed(42)
        # Create strongly positive-skewed weights
        W = torch.randn(16, 64).abs() * 3.0 + 2.0  # mostly positive [2, 5+]
        scales_s, zps_s = calculate_scales_asymmetric(W)
        scales_sym = calculate_scales(W)

        q_asym = quantize_weights_asymmetric(W, scales_s, zps_s)
        q_sym = quantize_weights(W, scales_sym)

        dequant_asym = (q_asym.float() - zps_s.float()) * scales_s.float()
        dequant_asym = dequant_asym[:, :W.shape[1]]
        dequant_sym = q_sym.float() * scales_sym.float()
        dequant_sym = dequant_sym[:, :W.shape[1]]

        mse_asym = (W - dequant_asym).pow(2).mean().item()
        mse_sym = (W - dequant_sym).pow(2).mean().item()
        # Asymmetric should be better or equal on skewed data
        assert mse_asym <= mse_sym + 1e-6

    def test_asymmetric_clipping_ratios(self):
        """Asymmetric + clipping ratio search should work together."""
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        scales, zps = calculate_scales_asymmetric(W, clipping_ratios=[0.85, 0.9, 0.95, 1.0])
        assert scales.shape == (16, 1)
        assert zps.shape == (16, 1)
        assert torch.all(scales > 0)

    def test_asymmetric_group_wise(self):
        W = torch.randn(8, 64)
        scales, zps = calculate_scales_asymmetric(W, group_size=32)
        assert scales.shape == (8, 2)
        assert zps.shape == (8, 2)

    def test_asymmetric_constant_row(self):
        """A row with all-same values should produce valid scales/zps."""
        W = torch.ones(4, 32) * 5.0
        scales, zps = calculate_scales_asymmetric(W)
        assert not torch.any(torch.isnan(scales))
        assert not torch.any(torch.isinf(scales))
        # All quantized values should be the same
        quantized = quantize_weights_asymmetric(W, scales, zps)
        assert torch.all(quantized[:, 0] == quantized[:, 1])


class TestAsymmetricINT8:
    def test_shape(self):
        W = torch.randn(16, 64)
        scales, zps = calculate_scales_int8_asymmetric(W)
        assert scales.shape == (16, 1)
        assert zps.shape == (16, 1)

    def test_dtype(self):
        W = torch.randn(16, 64)
        scales, zps = calculate_scales_int8_asymmetric(W)
        assert scales.dtype == torch.float16
        assert zps.dtype == torch.int8

    def test_quantize_range(self):
        W = torch.randn(16, 64, dtype=torch.float32)
        scales, zps = calculate_scales_int8_asymmetric(W)
        quantized = quantize_weights_int8_asymmetric(W, scales, zps)
        assert quantized.min() >= -128
        assert quantized.max() <= 127
        assert quantized.dtype == torch.int8

    def test_roundtrip(self):
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        scales, zps = calculate_scales_int8_asymmetric(W)
        quantized = quantize_weights_int8_asymmetric(W, scales, zps)
        dequant = (quantized.float() - zps.float()) * scales.float()
        mse = (W - dequant).pow(2).mean().item()
        assert mse < 0.1  # INT8 should be very precise

    def test_int8_asymmetric_clipping(self):
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        scales, zps = calculate_scales_int8_asymmetric(W, clipping_ratios=[0.9, 0.95, 1.0])
        assert scales.shape == (16, 1)
        assert torch.all(scales > 0)


# ---------------------------------------------------------------------------
# _search_clipping_ratio (shared helper)
# ---------------------------------------------------------------------------


class TestSearchClippingRatio:
    def test_single_ratio_returns_scales(self):
        """With a single ratio, the returned scales should be from that ratio."""
        torch.manual_seed(42)
        W = torch.randn(4, 16)
        max_vals = W.abs().amax(dim=1, keepdim=True)

        def _compute(ratio):
            scales = (max_vals.float() * ratio / 7.0).clamp(min=1e-8, max=65504.0)
            q = (W.unsqueeze(1) / scales.unsqueeze(2)).round().clamp(-8, 7)
            dequant = q * scales.unsqueeze(2)
            mse = (W.unsqueeze(1) - dequant).pow(2).mean(dim=2)
            return scales, None, mse

        best_scales, best_zp = _search_clipping_ratio([1.0], _compute)
        assert best_zp is None
        assert best_scales.shape == (4, 1)

    def test_picks_best_ratio(self):
        """With multiple ratios, the helper should pick the best ratio per row."""
        torch.manual_seed(42)
        W = torch.randn(4, 32)
        max_vals = W.abs().amax(dim=1, keepdim=True)

        # Compute scales with each ratio individually
        ratios = [0.8, 0.9, 1.0]
        per_ratio_mse = []
        for ratio in ratios:
            scales = (max_vals.float() * ratio / 7.0).clamp(min=1e-8, max=65504.0)
            q = (W.unsqueeze(1) / scales.unsqueeze(2)).round().clamp(-8, 7)
            dequant = q * scales.unsqueeze(2)
            mse = (W.unsqueeze(1) - dequant).pow(2).mean(dim=2)
            per_ratio_mse.append(mse)

        def _compute(ratio):
            scales = (max_vals.float() * ratio / 7.0).clamp(min=1e-8, max=65504.0)
            q = (W.unsqueeze(1) / scales.unsqueeze(2)).round().clamp(-8, 7)
            dequant = q * scales.unsqueeze(2)
            mse = (W.unsqueeze(1) - dequant).pow(2).mean(dim=2)
            return scales, None, mse

        best_scales, _ = _search_clipping_ratio(ratios, _compute)

        # The best MSE per-row from the helper should match the minimum across ratios
        W_3d = W.unsqueeze(1)
        best_q = (W_3d / best_scales.unsqueeze(2)).round().clamp(-8, 7)
        best_dequant = best_q * best_scales.unsqueeze(2)
        best_mse = (W_3d - best_dequant).pow(2).mean(dim=2)

        # For each row, the MSE should be <= the MSE from any single ratio
        for i in range(W.shape[0]):
            for r_mse in per_ratio_mse:
                assert best_mse[i] <= r_mse[i] + 1e-6

    def test_with_zero_points(self):
        """Asymmetric variant: zp should be tracked and returned."""
        torch.manual_seed(42)
        W = torch.randn(4, 32)
        w_min = W.amin(dim=1, keepdim=True)
        w_max = W.amax(dim=1, keepdim=True)

        def _compute(ratio):
            range_vals = (w_max - w_min).float() * ratio
            scales = (range_vals / 15.0).clamp(min=1e-8, max=65504.0)
            zp = (-8 - torch.round(w_min / scales)).clamp(-8, 7)
            q = (W.unsqueeze(1) / scales.unsqueeze(2) + zp.unsqueeze(2)).round().clamp(-8, 7)
            dequant = (q - zp.unsqueeze(2)) * scales.unsqueeze(2)
            mse = (W.unsqueeze(1) - dequant).pow(2).mean(dim=2)
            return scales, zp, mse

        best_scales, best_zp = _search_clipping_ratio([0.9, 1.0], _compute)
        assert best_zp is not None
        assert best_scales.shape == (4, 1)
        assert best_zp.shape == (4, 1)

    def test_mse_never_increases_with_more_ratios(self):
        """Adding more ratios should never increase best MSE."""
        torch.manual_seed(42)
        W = torch.randn(4, 32)
        max_vals = W.abs().amax(dim=1, keepdim=True)

        def _compute(ratio):
            scales = (max_vals.float() * ratio / 7.0).clamp(min=1e-8, max=65504.0)
            q = (W.unsqueeze(1) / scales.unsqueeze(2)).round().clamp(-8, 7)
            dequant = q * scales.unsqueeze(2)
            mse = (W.unsqueeze(1) - dequant).pow(2).mean(dim=2)
            return scales, None, mse

        scales_1, _ = _search_clipping_ratio([1.0], _compute)
        scales_3, _ = _search_clipping_ratio([0.8, 0.9, 1.0], _compute)

        q1 = (W.unsqueeze(1) / scales_1.unsqueeze(2)).round().clamp(-8, 7)
        mse1 = (W.unsqueeze(1) - q1 * scales_1.unsqueeze(2)).pow(2).mean()

        q3 = (W.unsqueeze(1) / scales_3.unsqueeze(2)).round().clamp(-8, 7)
        mse3 = (W.unsqueeze(1) - q3 * scales_3.unsqueeze(2)).pow(2).mean()

        assert mse3 <= mse1 + 1e-6


# ---------------------------------------------------------------------------
# Scale clipping — extreme weights and clipping ratios
# ---------------------------------------------------------------------------


class TestCalculateScalesClipping:
    """INT4 symmetric: calculate_scales() with clipping ratios and extreme weights."""

    def test_normal_weights_no_inf(self):
        torch.manual_seed(0)
        W = torch.randn(32, 128)
        scales = calculate_scales(W)
        assert _all_finite_fp16(scales)
        assert scales.min() > 0

    def test_extreme_weights_clamped(self):
        W = torch.full((4, 64), 1e8)
        scales = calculate_scales(W)
        assert _all_finite_fp16(scales)
        assert scales.max() <= MAX_FP16_SCALE

    def test_mixed_extreme_rows(self):
        W = torch.randn(8, 64)
        W[3] = 1e10
        scales = calculate_scales(W)
        assert _all_finite_fp16(scales)
        assert scales[0] < 1.0
        assert scales[3] <= MAX_FP16_SCALE

    def test_clipping_ratios_with_extreme_weights(self):
        W = torch.full((4, 64), 1e8)
        scales = calculate_scales(W, clipping_ratios=[0.8, 0.9, 1.0])
        assert _all_finite_fp16(scales)
        assert scales.max() <= MAX_FP16_SCALE

    def test_fp16_roundtrip_never_inf(self):
        W = torch.randn(16, 64) * 1000
        scales = calculate_scales(W)
        roundtripped = scales.float()
        assert roundtripped.isfinite().all()

    def test_zero_weight_still_works(self):
        W = torch.zeros(4, 32)
        scales = calculate_scales(W)
        assert _all_finite_fp16(scales)
        assert scales.min() > 0

    def test_quantize_with_clamped_scales(self):
        W = torch.randn(8, 64) * 100
        scales = calculate_scales(W)
        q = quantize_weights(W, scales)
        assert q.min() >= -8
        assert q.max() <= 7


class TestCalculateScalesInt8Clipping:
    """INT8 symmetric: calculate_scales_int8() with extreme weights."""

    def test_normal_weights_no_inf(self):
        W = torch.randn(32, 128)
        scales = calculate_scales_int8(W)
        assert _all_finite_fp16(scales)

    def test_extreme_weights_clamped(self):
        W = torch.full((4, 64), 1e8)
        scales = calculate_scales_int8(W)
        assert _all_finite_fp16(scales)
        assert scales.max() <= MAX_FP16_SCALE

    def test_clipping_ratios_extreme(self):
        W = torch.full((4, 64), 1e10)
        scales = calculate_scales_int8(W, clipping_ratios=[0.85, 0.9, 1.0])
        assert _all_finite_fp16(scales)

    def test_quantize_with_clamped_scales(self):
        W = torch.randn(16, 64) * 1000
        scales = calculate_scales_int8(W)
        q = quantize_weights_int8(W, scales)
        assert q.min() >= -128
        assert q.max() <= 127

    def test_fp16_roundtrip(self):
        W = torch.randn(16, 64) * 500
        scales = calculate_scales_int8(W)
        assert scales.float().isfinite().all()


class TestCalculateScalesAsymmetricClipping:
    """INT4 asymmetric: calculate_scales_asymmetric() with extreme weights."""

    def test_normal_weights_no_inf(self):
        W = torch.randn(16, 64)
        scales, zps = calculate_scales_asymmetric(W)
        assert _all_finite_fp16(scales)

    def test_extreme_weights_clamped(self):
        W = torch.full((4, 64), 1e8)
        scales, zps = calculate_scales_asymmetric(W)
        assert _all_finite_fp16(scales)
        assert scales.max() <= MAX_FP16_SCALE

    def test_clipping_ratios_extreme(self):
        W = torch.randn(8, 64) * 1e6
        scales, zps = calculate_scales_asymmetric(W, clipping_ratios=[0.9, 1.0])
        assert _all_finite_fp16(scales)

    def test_quantize_with_clamped_scales(self):
        W = torch.randn(8, 64) * 100
        scales, zps = calculate_scales_asymmetric(W)
        q = quantize_weights_asymmetric(W, scales, zps)
        assert q.min() >= -8
        assert q.max() <= 7


class TestCalculateScalesInt8AsymmetricClipping:
    """INT8 asymmetric: calculate_scales_int8_asymmetric() with extreme weights."""

    def test_normal_weights_no_inf(self):
        W = torch.randn(16, 64)
        scales, zps = calculate_scales_int8_asymmetric(W)
        assert _all_finite_fp16(scales)

    def test_extreme_weights_clamped(self):
        W = torch.full((4, 64), 1e8)
        scales, zps = calculate_scales_int8_asymmetric(W)
        assert _all_finite_fp16(scales)
        assert scales.max() <= MAX_FP16_SCALE

    def test_clipping_ratios_extreme(self):
        W = torch.randn(8, 64) * 1e6
        scales, zps = calculate_scales_int8_asymmetric(W, clipping_ratios=[0.9, 1.0])
        assert _all_finite_fp16(scales)


# ---------------------------------------------------------------------------
# Scale validation fallback (simulates CLI behavior)
# ---------------------------------------------------------------------------


class TestScaleValidationFallback:
    """Verify that the CLI's validation + recompute fallback works."""

    @staticmethod
    def _simulate_validation(W_work, scales, int_bits=8):
        divisor = INT8_SCALE_DIVISOR if int_bits == 8 else INT4_SCALE_DIVISOR
        bad = scales.isinf() | scales.isnan() | (scales == 0)
        if bad.any():
            fix = (W_work.float().abs().amax(dim=1, keepdim=True) / divisor
                   ).clamp(min=FP16_SCALE_FLOOR, max=MAX_FP16_SCALE).to(torch.float16)
            scales = torch.where(bad, fix, scales)
        return scales

    def test_inf_scale_replaced(self):
        torch.manual_seed(0)
        W_work = torch.randn(4, 64) * 30.0
        scales = torch.tensor([[0.001], [float('inf')], [0.002], [0.003]],
                              dtype=torch.float16)
        fixed = self._simulate_validation(W_work, scales)
        assert _all_finite_fp16(fixed)
        assert fixed[0] == scales[0]
        assert fixed[1] != float('inf')

    def test_nan_scale_replaced(self):
        torch.manual_seed(0)
        W_work = torch.randn(4, 64) * 30.0
        scales = torch.tensor([[0.001], [float('nan')], [0.002], [0.003]],
                              dtype=torch.float16)
        fixed = self._simulate_validation(W_work, scales)
        assert _all_finite_fp16(fixed)

    def test_zero_scale_replaced(self):
        torch.manual_seed(0)
        W_work = torch.randn(4, 64) * 30.0
        scales = torch.tensor([[0.001], [0.0], [0.002], [0.003]],
                              dtype=torch.float16)
        fixed = self._simulate_validation(W_work, scales)
        assert _all_finite_fp16(fixed)
        assert fixed[1] > 0

    def test_all_good_scales_unchanged(self):
        torch.manual_seed(0)
        W_work = torch.randn(4, 64) * 30.0
        scales = torch.tensor([[0.001], [0.002], [0.003], [0.004]],
                              dtype=torch.float16)
        fixed = self._simulate_validation(W_work, scales)
        assert torch.equal(fixed, scales)

    def test_multiple_bad_scales(self):
        torch.manual_seed(0)
        W_work = torch.randn(4, 64) * 30.0
        scales = torch.tensor([[float('inf')], [float('nan')], [0.0], [0.003]],
                              dtype=torch.float16)
        fixed = self._simulate_validation(W_work, scales)
        assert _all_finite_fp16(fixed)
        assert fixed[3] == scales[3]

    def test_int4_repair_scale_magnitude(self):
        torch.manual_seed(0)
        W_work = torch.randn(4, 64) * 30.0
        scales = torch.tensor([[float('inf')], [1.0], [0.0], [0.5]], dtype=torch.float16)
        fixed = self._simulate_validation(W_work, scales, int_bits=4)
        assert fixed[0].item() > 2.0
        assert fixed[2].item() > 2.0
        assert fixed[1].item() == 1.0
        assert fixed[3].item() == 0.5

    def test_int8_repair_scale_magnitude(self):
        torch.manual_seed(0)
        W_work = torch.randn(4, 64) * 500.0
        scales = torch.tensor([[float('inf')], [1.0], [0.0], [0.5]], dtype=torch.float16)
        fixed = self._simulate_validation(W_work, scales, int_bits=8)
        assert fixed[0].item() > 2.0
        assert fixed[1].item() == 1.0
        assert fixed[3].item() == 0.5
