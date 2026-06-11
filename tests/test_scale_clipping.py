"""Tests for scale clipping — prevents Inf/NaN in float16 scales.

These tests verify:
  1. All scale functions produce finite float16 output for normal inputs
  2. Extreme weight values are clamped, not overflowed
  3. Clipping ratios interact correctly with the upper-bound clamp
  4. GPTQ scales are finite even for pathological Hessians
  5. The CLI validation fallback recomputes corrupted scales
  6. The float16 roundtrip never produces Inf
"""

import torch
import pytest

from converter.scales import (
    MAX_FP16_SCALE,
    calculate_scales,
    quantize_weights,
    calculate_scales_int8,
    quantize_weights_int8,
    calculate_scales_asymmetric,
    quantize_weights_asymmetric,
    calculate_scales_int8_asymmetric,
    quantize_weights_int8_asymmetric,
)
from converter.gptq import gptq_quantize_layer, gptq_quantize_layer_rtn


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_hessian(in_features: int) -> torch.Tensor:
    X = torch.randn(64, in_features)
    return X.T @ X


def _all_finite_fp16(t: torch.Tensor) -> bool:
    """Return True if tensor is float16 with no Inf/NaN."""
    return t.dtype == torch.float16 and t.isfinite().all().item()


# ---------------------------------------------------------------------------
# INT4 symmetric scales
# ---------------------------------------------------------------------------

class TestCalculateScalesClipping:
    """INT4 symmetric: calculate_scales() + quantize_weights()."""

    def test_normal_weights_no_inf(self):
        torch.manual_seed(0)
        W = torch.randn(32, 128)
        scales = calculate_scales(W)
        assert _all_finite_fp16(scales)
        assert scales.min() > 0

    def test_extreme_weights_clamped(self):
        """Weights so large that scale would overflow fp16 must be clamped."""
        W = torch.full((4, 64), 1e8)
        scales = calculate_scales(W)
        assert _all_finite_fp16(scales)
        assert scales.max() <= MAX_FP16_SCALE

    def test_mixed_extreme_rows(self):
        """One extreme row among normal rows — only that row should be clamped."""
        W = torch.randn(8, 64)
        W[3] = 1e10  # extreme row
        scales = calculate_scales(W)
        assert _all_finite_fp16(scales)
        # Normal rows should have much smaller scales
        assert scales[0] < 1.0
        # Extreme row should be clamped
        assert scales[3] <= MAX_FP16_SCALE

    def test_clipping_ratios_with_extreme_weights(self):
        """Clipping ratios should not produce Inf even with extreme weights."""
        W = torch.full((4, 64), 1e8)
        scales = calculate_scales(W, clipping_ratios=[0.8, 0.9, 1.0])
        assert _all_finite_fp16(scales)
        assert scales.max() <= MAX_FP16_SCALE

    def test_fp16_roundtrip_never_inf(self):
        """Scale -> float16 -> float32 should never produce Inf."""
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
        """Quantization with clamped scales should still produce valid int8."""
        W = torch.randn(8, 64) * 100
        scales = calculate_scales(W)
        q = quantize_weights(W, scales)
        assert q.min() >= -8
        assert q.max() <= 7


# ---------------------------------------------------------------------------
# INT8 symmetric scales
# ---------------------------------------------------------------------------

class TestCalculateScalesInt8Clipping:
    """INT8 symmetric: calculate_scales_int8() + quantize_weights_int8()."""

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


# ---------------------------------------------------------------------------
# INT4 asymmetric scales
# ---------------------------------------------------------------------------

class TestCalculateScalesAsymmetricClipping:
    """INT4 asymmetric: calculate_scales_asymmetric()."""

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


# ---------------------------------------------------------------------------
# INT8 asymmetric scales
# ---------------------------------------------------------------------------

class TestCalculateScalesInt8AsymmetricClipping:
    """INT8 asymmetric: calculate_scales_int8_asymmetric()."""

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
# GPTQ scales
# ---------------------------------------------------------------------------

class TestGPTQScaleClipping:
    """GPTQ quantization: gptq_quantize_layer() and gptq_quantize_layer_rtn()."""

    def test_gptq_int4_scales_finite(self):
        W = torch.randn(16, 64)
        H = _make_hessian(64)
        _, scales, _ = gptq_quantize_layer(W, H, int_bits=4)
        assert _all_finite_fp16(scales)

    def test_gptq_int8_scales_finite(self):
        W = torch.randn(16, 64)
        H = _make_hessian(64)
        _, scales, _ = gptq_quantize_layer(W, H, int_bits=8)
        assert _all_finite_fp16(scales)

    def test_gptq_block_diagonal_scales_finite(self):
        """Block-diagonal Hessians should not produce Inf scales."""
        torch.manual_seed(42)
        W = torch.randn(16, 128)
        blocks = []
        for _ in range(4):
            X = torch.randn(64, 32)
            blocks.append(X.T @ X)
        H = torch.stack(blocks)
        _, scales, _ = gptq_quantize_layer(W, H, int_bits=8)
        assert _all_finite_fp16(scales)

    def test_gptq_ill_conditioned_hessian_scales_finite(self):
        """Near-singular Hessian should not produce Inf scales."""
        W = torch.randn(8, 32)
        H = torch.zeros(32, 32)
        H[0, 0] = 1.0
        _, scales, _ = gptq_quantize_layer(W, H, int_bits=8, damping=0.1)
        assert _all_finite_fp16(scales)

    def test_gptq_zero_weights_scales_finite(self):
        W = torch.zeros(8, 32)
        H = _make_hessian(32)
        _, scales, _ = gptq_quantize_layer(W, H, int_bits=8)
        assert _all_finite_fp16(scales)

    def test_rtn_int4_scales_finite(self):
        W = torch.randn(16, 64)
        _, scales = gptq_quantize_layer_rtn(W, int_bits=4)
        assert _all_finite_fp16(scales)

    def test_rtn_int8_scales_finite(self):
        W = torch.randn(16, 64)
        _, scales = gptq_quantize_layer_rtn(W, int_bits=8)
        assert _all_finite_fp16(scales)

    def test_rtn_with_clipping_ratios_scales_finite(self):
        W = torch.randn(16, 64)
        _, scales = gptq_quantize_layer_rtn(
            W, int_bits=8, clipping_ratios=[0.8, 0.85, 0.9, 0.95, 1.0]
        )
        assert _all_finite_fp16(scales)

    def test_gptq_scales_upper_bound(self):
        """GPTQ scales should never exceed MAX_FP16_SCALE."""
        W = torch.randn(16, 64)
        H = _make_hessian(64)
        _, scales, _ = gptq_quantize_layer(W, H, int_bits=8)
        assert scales.max() <= MAX_FP16_SCALE

    def test_gptq_scales_lower_bound(self):
        """GPTQ scales should never be zero or negative."""
        W = torch.randn(16, 64)
        H = _make_hessian(64)
        _, scales, _ = gptq_quantize_layer(W, H, int_bits=8)
        assert scales.min() > 0


# ---------------------------------------------------------------------------
# Scale validation fallback (simulates CLI behavior)
# ---------------------------------------------------------------------------

class TestScaleValidationFallback:
    """Verify that the CLI's validation + recompute fallback works."""

    @staticmethod
    def _simulate_validation(quantized_W, scales):
        """Reproduce the CLI's validation logic."""
        bad = scales.isinf() | scales.isnan() | (scales == 0)
        if bad.any():
            fix = (quantized_W.float().abs().amax(dim=1, keepdim=True) / 127.0
                   ).clamp(min=1e-6, max=65000.0).to(torch.float16)
            scales = torch.where(bad, fix, scales)
        return scales

    def test_inf_scale_replaced(self):
        """An Inf scale should be replaced by the fallback."""
        quantized_W = torch.randint(-128, 127, (4, 64), dtype=torch.int8)
        scales = torch.tensor([[0.001], [float('inf')], [0.002], [0.003]],
                              dtype=torch.float16)
        fixed = self._simulate_validation(quantized_W, scales)
        assert _all_finite_fp16(fixed)
        assert fixed[0] == scales[0]  # untouched
        assert fixed[1] != float('inf')  # replaced

    def test_nan_scale_replaced(self):
        quantized_W = torch.randint(-128, 127, (4, 64), dtype=torch.int8)
        scales = torch.tensor([[0.001], [float('nan')], [0.002], [0.003]],
                              dtype=torch.float16)
        fixed = self._simulate_validation(quantized_W, scales)
        assert _all_finite_fp16(fixed)

    def test_zero_scale_replaced(self):
        quantized_W = torch.randint(-128, 127, (4, 64), dtype=torch.int8)
        scales = torch.tensor([[0.001], [0.0], [0.002], [0.003]],
                              dtype=torch.float16)
        fixed = self._simulate_validation(quantized_W, scales)
        assert _all_finite_fp16(fixed)
        assert fixed[1] > 0

    def test_all_good_scales_unchanged(self):
        """If all scales are valid, the fallback should not modify them."""
        quantized_W = torch.randint(-128, 127, (4, 64), dtype=torch.int8)
        scales = torch.tensor([[0.001], [0.002], [0.003], [0.004]],
                              dtype=torch.float16)
        fixed = self._simulate_validation(quantized_W, scales)
        assert torch.equal(fixed, scales)

    def test_multiple_bad_scales(self):
        """Multiple bad scales should all be replaced."""
        quantized_W = torch.randint(-128, 127, (4, 64), dtype=torch.int8)
        scales = torch.tensor([[float('inf')], [float('nan')], [0.0], [0.003]],
                              dtype=torch.float16)
        fixed = self._simulate_validation(quantized_W, scales)
        assert _all_finite_fp16(fixed)
        assert fixed[3] == scales[3]  # only good one untouched


# ---------------------------------------------------------------------------
# Stress test: repeated quantization (simulates non-deterministic bug)
# ---------------------------------------------------------------------------

class TestRepeatedQuantization:
    """Run quantization many times to catch rare non-deterministic failures."""

    def test_int8_gptq_scales_always_finite(self):
        """100 runs of GPTQ INT8 — scales must always be finite."""
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        X = torch.randn(64, 64)
        H = X.T @ X
        for i in range(100):
            _, scales, _ = gptq_quantize_layer(W, H, int_bits=8)
            assert _all_finite_fp16(scales), f"Inf/NaN in run {i}"

    def test_int8_rtn_scales_always_finite(self):
        """100 runs of RTN INT8 — scales must always be finite."""
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        for i in range(100):
            _, scales = gptq_quantize_layer_rtn(W, int_bits=8)
            assert _all_finite_fp16(scales), f"Inf/NaN in run {i}"

    def test_int8_rtn_with_clipping_always_finite(self):
        """100 runs of RTN INT8 + clipping — scales must always be finite."""
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        for i in range(100):
            _, scales = gptq_quantize_layer_rtn(
                W, int_bits=8, clipping_ratios=[0.8, 0.85, 0.9, 0.95, 1.0]
            )
            assert _all_finite_fp16(scales), f"Inf/NaN in run {i}"


# ---------------------------------------------------------------------------
# MAX_FP16_SCALE constant sanity
# ---------------------------------------------------------------------------

class TestMaxFP16ScaleConstant:
    def test_below_fp16_max(self):
        """MAX_FP16_SCALE must be below float16 max to prevent overflow."""
        assert MAX_FP16_SCALE < torch.finfo(torch.float16).max

    def test_above_typical_scales(self):
        """MAX_FP16_SCALE must be far above typical model scales (~0.001)."""
        assert MAX_FP16_SCALE > 1.0
