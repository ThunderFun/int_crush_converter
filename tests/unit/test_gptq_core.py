"""Tests for the core GPTQ quantization algorithm (INT4 and INT8)."""

import torch
import pytest

from converter.gptq import gptq_quantize_layer
from converter.config import MAX_FP16_SCALE, FP16_SCALE_FLOOR
from converter.scales import calculate_scales, quantize_weights, calculate_scales_int8, quantize_weights_int8
from tests.conftest import make_hessian


def _make_hessian(in_features: int) -> torch.Tensor:
    """Create a realistic Hessian using current RNG state (no re-seed)."""
    return make_hessian(in_features, seed=None, num_samples=64)


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


# ── INT8 asymmetric ──────────────────────────────────────────────────────────


class TestGPTQQuantizeLayerINT8Asymmetric:
    """Tests for GPTQ INT8 asymmetric quantization."""

    def test_output_range(self):
        """Asymmetric INT8 output must be in [-128, 127]."""
        W = torch.randn(16, 64)
        H = _make_hessian(64)
        _qr = gptq_quantize_layer(W, H, int_bits=8, asymmetric=True)
        assert _qr.quantized_W.min() >= -128
        assert _qr.quantized_W.max() <= 127

    def test_zero_points_present(self):
        """asymmetric=True must produce zero-points."""
        W = torch.randn(16, 64)
        H = _make_hessian(64)
        _qr = gptq_quantize_layer(W, H, int_bits=8, asymmetric=True)
        assert _qr.zero_points is not None
        assert _qr.zero_points.shape == (16, 1)
        assert _qr.zero_points.dtype == torch.int8

    def test_zero_points_range(self):
        """Zero-points must be in [-128, 127] for INT8."""
        W = torch.randn(16, 64)
        H = _make_hessian(64)
        _qr = gptq_quantize_layer(W, H, int_bits=8, asymmetric=True)
        assert _qr.zero_points.min() >= -128
        assert _qr.zero_points.max() <= 127

    def test_symmetric_has_no_zero_points(self):
        """Default (symmetric) INT8 must have zero_points=None."""
        W = torch.randn(16, 64)
        H = _make_hessian(64)
        _qr = gptq_quantize_layer(W, H, int_bits=8)
        assert _qr.zero_points is None

    def test_zero_weights(self):
        """Zero weights must quantize to zero with asymmetric INT8."""
        W = torch.zeros(8, 32)
        H = _make_hessian(32)
        _qr = gptq_quantize_layer(W, H, int_bits=8, asymmetric=True)
        assert torch.all(_qr.quantized_W == 0)

    def test_scales_positive(self):
        W = torch.randn(16, 64)
        H = _make_hessian(64)
        _qr = gptq_quantize_layer(W, H, int_bits=8, asymmetric=True)
        assert torch.all(_qr.scales > 0)

    def test_no_nan_inf(self):
        W = torch.randn(16, 64)
        H = _make_hessian(64)
        _qr = gptq_quantize_layer(W, H, int_bits=8, asymmetric=True)
        assert not torch.any(torch.isnan(_qr.quantized_W.float()))
        assert not torch.any(torch.isinf(_qr.quantized_W.float()))
        assert not torch.any(torch.isnan(_qr.scales))
        assert not torch.any(torch.isinf(_qr.scales))

    def test_asymmetric_better_for_skewed(self):
        """Asymmetric INT8 should have lower MSE than symmetric for skewed weights."""
        torch.manual_seed(42)
        # Skewed distribution (e.g., ReLU-like): all positive
        W = torch.randn(32, 64).abs()
        X = torch.randn(128, 64)
        H = X.T @ X

        qr_sym = gptq_quantize_layer(W, H, int_bits=8, asymmetric=False)
        qr_asym = gptq_quantize_layer(W, H, int_bits=8, asymmetric=True)

        # Asymmetric should use all 256 levels for the positive range
        assert qr_asym.zero_points is not None
        # Asymmetric should have lower or equal MSE for skewed distributions
        assert qr_asym.mse <= qr_sym.mse * 1.05, (
            f"Asymmetric MSE {qr_asym.mse:.6f} should be <= symmetric {qr_sym.mse:.6f}"
        )

    def test_asymmetric_large_mean_no_inf_mse(self):
        """Regression: asymmetric INT8 with large-mean weights must not produce inf MSE.

        When weights have a large mean and small variance (e.g., 5.0 ± 0.001),
        the range-based scale is tiny and the zero-point gets clamped from a
        huge negative value to -128.  Without the scale fix, all Q values
        saturate at 127, dequant ≈ 0, and the massive systematic error causes
        GPTQ error propagation to blow up to inf.
        """
        torch.manual_seed(42)
        # Large positive mean, small variance
        W = 5.0 + torch.randn(16, 64) * 0.001
        X = torch.randn(128, 64)
        H = X.T @ X

        result = gptq_quantize_layer(W, H, int_bits=8, asymmetric=True)
        assert torch.isfinite(torch.tensor(result.mse)), (
            f"Asymmetric MSE must be finite for large-mean weights, got {result.mse}"
        )
        assert result.mse < 1.0, (
            f"Asymmetric MSE should be reasonable for large-mean weights, got {result.mse}"
        )

    def test_asymmetric_all_negative_no_inf_mse(self):
        """Regression: asymmetric INT8 with all-negative weights must not produce inf MSE."""
        torch.manual_seed(42)
        W = -5.0 + torch.randn(16, 64) * 0.001
        X = torch.randn(128, 64)
        H = X.T @ X

        result = gptq_quantize_layer(W, H, int_bits=8, asymmetric=True)
        assert torch.isfinite(torch.tensor(result.mse)), (
            f"Asymmetric MSE must be finite for all-negative weights, got {result.mse}"
        )
        assert result.mse < 1.0

    def test_asymmetric_very_large_weights_finite(self):
        """Regression: asymmetric INT8 with very large weights must produce finite MSE."""
        torch.manual_seed(42)
        W = 1000.0 + torch.randn(16, 64) * 10
        X = torch.randn(128, 64)
        H = X.T @ X

        result = gptq_quantize_layer(W, H, int_bits=8, asymmetric=True)
        assert torch.isfinite(torch.tensor(result.mse)), (
            f"Asymmetric MSE must be finite for very large weights, got {result.mse}"
        )


# ── Scale clamping regression ────────────────────────────────────────────────


class TestGPTQScaleClampConsistency:
    """Scales must be clamped BEFORE the GPTQ loop so (Q, scale) is consistent."""

    def test_int8_scales_within_fp16_range(self):
        """All INT8 scales must survive fp16 round-trip."""
        torch.manual_seed(42)
        # Weights large enough that raw scale would exceed fp16 range
        W = torch.randn(8, 32) * 1e7
        X = torch.randn(64, 32)
        H = X.T @ X

        result = gptq_quantize_layer(W, H, int_bits=8)
        assert result.scales.dtype == torch.float16
        assert result.scales.isfinite().all(), "Scales must be finite"
        assert (result.scales > 0).all(), "Scales must be positive"
        assert (result.scales <= MAX_FP16_SCALE).all(), (
            f"Scales must be <= MAX_FP16_SCALE ({MAX_FP16_SCALE}); "
            f"max={result.scales.max().item()}"
        )

    def test_int8_q_scale_consistency(self):
        """Q * stored_scale must stay within the representable range.

        For INT8 symmetric, q in [-128, 127], so |dequant| <= 128 * scale.
        """
        torch.manual_seed(42)
        W = torch.randn(8, 32) * 1e7
        X = torch.randn(64, 32)
        H = X.T @ X

        result = gptq_quantize_layer(W, H, int_bits=8)
        scales_f32 = result.scales.float()
        dequant = result.quantized_W.float() * scales_f32
        # INT8 symmetric: max |q| = 128 (from q = -128)
        max_bound = 128.0 * scales_f32.max().item()
        assert dequant.abs().max().item() <= max_bound * 1.001, (
            f"Dequant max {dequant.abs().max():.2f} exceeds "
            f"128 * stored_scale_max = {max_bound:.2f}"
        )

    def test_int8_mse_isfinite_with_huge_weights(self):
        """MSE must be finite even for very large weights."""
        torch.manual_seed(42)
        W = torch.randn(8, 32) * 1e7
        X = torch.randn(64, 32)
        H = X.T @ X

        result = gptq_quantize_layer(W, H, int_bits=8)
        assert torch.isfinite(torch.tensor(result.mse)), (
            f"MSE must be finite for huge weights, got {result.mse}"
        )
        assert torch.isfinite(torch.tensor(result.max_err)), (
            f"max_err must be finite for huge weights, got {result.max_err}"
        )

    def test_int4_scales_within_fp16_range(self):
        """All INT4 scales must survive fp16 round-trip."""
        torch.manual_seed(42)
        W = torch.randn(8, 32) * 1e6
        X = torch.randn(64, 32)
        H = X.T @ X

        result = gptq_quantize_layer(W, H, int_bits=4)
        assert result.scales.dtype == torch.float16
        assert result.scales.isfinite().all(), "Scales must be finite"
        assert (result.scales > 0).all(), "Scales must be positive"
        assert (result.scales <= MAX_FP16_SCALE).all(), (
            f"INT4 scales must be <= MAX_FP16_SCALE ({MAX_FP16_SCALE}); "
            f"max={result.scales.max().item()}"
        )

    def test_int4_q_scale_consistency(self):
        """Q * stored_scale must stay within the representable range.

        For INT4 asymmetric, |dequant| = |q - zp| * scale,
        where |q - zp| <= max(|-8 - zp|, |7 - zp|).
        """
        torch.manual_seed(42)
        W = torch.randn(8, 32) * 1e6
        X = torch.randn(64, 32)
        H = X.T @ X

        result = gptq_quantize_layer(W, H, int_bits=4)
        scales_f32 = result.scales.float()
        zp_f32 = result.zero_points.float()
        dequant = (result.quantized_W.float() - zp_f32) * scales_f32
        # Per-row max |q - zp|: depends on the row's zp
        max_zp_offset = torch.max((-8 - zp_f32).abs(), (7 - zp_f32).abs())
        max_bound = (max_zp_offset * scales_f32).max().item()
        assert dequant.abs().max().item() <= max_bound * 1.001, (
            f"INT4 dequant max {dequant.abs().max():.2f} exceeds "
            f"max(|q-zp|)*scale = {max_bound:.2f}"
        )

    def test_int8_symmetric_normal_weights_unchanged(self):
        """Normal-weight GPTQ INT8 (scale << MAX_FP16_SCALE) must not change."""
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        X = torch.randn(128, 64)
        H = X.T @ X

        result = gptq_quantize_layer(W, H, int_bits=8)
        assert result.scales.max().item() < MAX_FP16_SCALE / 100
        assert result.quantized_W.min() >= -128
        assert result.quantized_W.max() <= 127
        assert result.zero_points is None  # symmetric

    def test_int4_asymmetric_normal_weights_unchanged(self):
        """Normal-weight GPTQ INT4 (always asymmetric) must not change."""
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        X = torch.randn(128, 64)
        H = X.T @ X

        result = gptq_quantize_layer(W, H, int_bits=4)
        assert result.scales.max().item() < MAX_FP16_SCALE / 100
        assert result.quantized_W.min() >= -8
        assert result.quantized_W.max() <= 7
        assert result.zero_points is not None
