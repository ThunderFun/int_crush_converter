"""Tests for SmoothQuant per-channel smoothing functions."""

import pytest
import torch

from converter.smoothquant import (
    compute_smoothing_factors,
    apply_smoothing_to_weight,
    compute_smoothing_from_hessian_diag,
    compute_smoothing_weight_only,
)


class TestComputeSmoothingFactors:
    """Core SmoothQuant formula tests."""

    def test_output_shape(self):
        act_amax = torch.rand(64)
        W = torch.randn(16, 64)
        s = compute_smoothing_factors(act_amax, W)
        assert s.shape == (64,)

    def test_output_dtype(self):
        act_amax = torch.rand(64, dtype=torch.float16)
        W = torch.randn(16, 64, dtype=torch.float16)
        s = compute_smoothing_factors(act_amax, W)
        assert s.dtype == torch.float32

    def test_formula_alpha_half(self):
        """s_i = max|X_i|^0.5 / max|W_i|^0.5"""
        act_amax = torch.tensor([4.0, 9.0])
        W = torch.tensor([[1.0, 3.0], [2.0, 6.0]])
        s = compute_smoothing_factors(act_amax, W, alpha=0.5)
        # max|W[:,0]| = 2, max|W[:,1]| = 6
        expected = torch.tensor([4.0**0.5 / 2.0**0.5, 9.0**0.5 / 6.0**0.5])
        assert torch.allclose(s, expected, atol=1e-5)

    def test_formula_alpha_arbitrary(self):
        """s_i = max|X_i|^alpha / max|W_i|^(1-alpha)"""
        act_amax = torch.tensor([100.0, 1.0])
        W = torch.tensor([[2.0, 4.0], [3.0, 8.0]])
        alpha = 0.7
        s = compute_smoothing_factors(act_amax, W, alpha=alpha)
        w_scale = W.abs().amax(dim=0).float()
        expected = torch.tensor([100.0**alpha / w_scale[0]**(1-alpha),
                                  1.0**alpha / w_scale[1]**(1-alpha)])
        assert torch.allclose(s, expected, atol=1e-5)

    def test_alpha_zero_uses_only_weights(self):
        """alpha=0: s_i = 1 / max|W_i|"""
        W = torch.tensor([[1.0, 3.0], [2.0, 6.0]])
        s = compute_smoothing_factors(torch.ones(2), W, alpha=0.0)
        expected = 1.0 / W.abs().amax(dim=0).float()
        assert torch.allclose(s, expected, atol=1e-5)

    def test_alpha_one_uses_only_activations(self):
        """alpha=1: s_i = max|X_i|"""
        act_amax = torch.tensor([4.0, 9.0])
        W = torch.randn(4, 2)
        s = compute_smoothing_factors(act_amax, W, alpha=1.0)
        assert torch.allclose(s, act_amax.float(), atol=1e-5)

    def test_clamp_prevents_division_by_zero(self):
        """Zero activation/weight should not cause inf/nan."""
        act_amax = torch.tensor([0.0, 5.0])
        W = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        s = compute_smoothing_factors(act_amax, W)
        assert torch.isfinite(s).all()

    def test_mathematical_equivalence(self):
        """X @ W == (X / s) @ (W * s) for all X, W."""
        torch.manual_seed(42)
        X = torch.randn(8, 16)
        W = torch.randn(4, 16)
        act_amax = X.abs().amax(dim=0)
        s = compute_smoothing_factors(act_amax, W, alpha=0.5)
        Y_orig = X @ W.T
        Y_smooth = (X / s.unsqueeze(0)) @ (W * s.unsqueeze(0)).T
        assert torch.allclose(Y_orig, Y_smooth, atol=1e-3)

    def test_higher_alpha_shifts_more_to_weights(self):
        """Larger alpha → larger smoothing factors for outlier channels."""
        act_amax = torch.tensor([100.0, 1.0])
        W = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        s_low = compute_smoothing_factors(act_amax, W, alpha=0.3)
        s_high = compute_smoothing_factors(act_amax, W, alpha=0.7)
        # High alpha should produce larger factors for the outlier channel
        assert s_high[0] > s_low[0]

    def test_act_amax_wrong_dim_raises(self):
        with pytest.raises(ValueError, match="1D"):
            compute_smoothing_factors(torch.rand(4, 16), torch.randn(8, 16))

    def test_weight_wrong_dim_raises(self):
        with pytest.raises(ValueError, match="2D"):
            compute_smoothing_factors(torch.rand(16), torch.randn(16))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="in_features"):
            compute_smoothing_factors(torch.rand(32), torch.randn(8, 16))


class TestApplySmoothingToWeight:
    """Weight column scaling tests."""

    def test_shape_preserved(self):
        W = torch.randn(8, 16)
        s = torch.rand(16)
        W_smooth = apply_smoothing_to_weight(W, s)
        assert W_smooth.shape == W.shape

    def test_dtype_preserved(self):
        W = torch.randn(8, 16, dtype=torch.float16)
        s = torch.rand(16, dtype=torch.float32)
        W_smooth = apply_smoothing_to_weight(W, s)
        assert W_smooth.dtype == torch.float16

    def test_column_scaling(self):
        W = torch.ones(4, 3)
        s = torch.tensor([2.0, 0.5, 1.0])
        W_smooth = apply_smoothing_to_weight(W, s)
        expected = torch.tensor([[2.0, 0.5, 1.0]] * 4)
        assert torch.allclose(W_smooth, expected)

    def test_inverse_recovers_original(self):
        torch.manual_seed(42)
        W = torch.randn(8, 16)
        s = torch.rand(16).clamp(min=0.1)
        W_smooth = apply_smoothing_to_weight(W, s)
        W_recovered = W_smooth / s.unsqueeze(0)
        assert torch.allclose(W_recovered.float(), W.float(), atol=1e-5)

    def test_identity_smoothing(self):
        """s = all ones should be identity."""
        W = torch.randn(8, 16)
        s = torch.ones(16)
        W_smooth = apply_smoothing_to_weight(W, s)
        assert torch.equal(W_smooth, W)


class TestComputeSmoothingFromHessianDiag:
    """Hessian-diagonal approximation tests."""

    def test_output_shape(self):
        h_diag = torch.rand(64)
        W = torch.randn(16, 64)
        s = compute_smoothing_from_hessian_diag(h_diag, W)
        assert s.shape == (64,)

    def test_formula(self):
        """s_i = (H_diag[i] / n)^0.5^alpha / w_scale^(1-alpha)"""
        h_diag = torch.tensor([16.0, 64.0])
        W = torch.tensor([[2.0, 4.0], [3.0, 8.0]])
        alpha = 0.5
        n = 4
        s = compute_smoothing_from_hessian_diag(h_diag, W, alpha=alpha, num_calibration_samples=n)
        act_rms = torch.tensor([16.0 / 4, 64.0 / 4]).sqrt()
        w_scale = W.abs().amax(dim=0).float()
        expected = (act_rms ** alpha) / (w_scale ** (1 - alpha))
        assert torch.allclose(s, expected, atol=1e-5)

    def test_zero_hessian_diag_handled(self):
        h_diag = torch.zeros(16)
        W = torch.randn(8, 16)
        s = compute_smoothing_from_hessian_diag(h_diag, W)
        assert torch.isfinite(s).all()
        assert (s > 0).all()

    def test_approximates_amax_for_normal_distribution(self):
        """For normal activations, RMS ≈ amax/sqrt(2/pi), factors should be similar."""
        torch.manual_seed(42)
        X = torch.randn(2048, 32)
        h_diag = (X.T @ X).diagonal()
        act_amax = X.abs().amax(dim=0)
        W = torch.randn(8, 32)
        s_rms = compute_smoothing_from_hessian_diag(h_diag, W, alpha=0.5, num_calibration_samples=2048)
        s_amax = compute_smoothing_factors(act_amax, W, alpha=0.5)
        # Same order of magnitude (within 3x)
        ratio = (s_rms / s_amax).clamp(0.2, 5.0)
        assert ratio.mean() > 0.5

    def test_num_calibration_samples_zero_clamped(self):
        """n=0 should not cause division by zero."""
        h_diag = torch.rand(16)
        W = torch.randn(8, 16)
        s = compute_smoothing_from_hessian_diag(h_diag, W, num_calibration_samples=0)
        assert torch.isfinite(s).all()


class TestComputeSmoothingWeightOnly:
    """Weight-only smoothing tests."""

    def test_output_shape(self):
        W = torch.randn(16, 64)
        s = compute_smoothing_weight_only(W)
        assert s.shape == (64,)

    def test_normalizes_columns(self):
        W = torch.tensor([[1.0, 3.0], [2.0, 6.0]])
        s = compute_smoothing_weight_only(W)
        W_smooth = apply_smoothing_to_weight(W, s)
        # All columns should have max = 1.0
        assert torch.allclose(W_smooth.abs().amax(dim=0), torch.ones(2), atol=1e-6)

    def test_reduces_per_row_dynamic_range(self):
        """Weight-only smoothing should reduce the ratio max(row)/min(|nonzero row|)."""
        torch.manual_seed(42)
        W = torch.randn(8, 64)
        W[:, 0] *= 100  # Add an outlier column
        s = compute_smoothing_weight_only(W)
        W_smooth = apply_smoothing_to_weight(W, s)
        # Dynamic range per row should be smaller after smoothing
        orig_range = W.abs().amax(dim=1) / W.abs().amin(dim=1).clamp(min=1e-8)
        smooth_range = W_smooth.abs().amax(dim=1) / W_smooth.abs().amin(dim=1).clamp(min=1e-8)
        assert smooth_range.mean() < orig_range.mean()

    def test_zero_columns_handled(self):
        W = torch.zeros(4, 8)
        s = compute_smoothing_weight_only(W)
        assert torch.isfinite(s).all()
        assert (s > 0).all()

    def test_single_row(self):
        W = torch.tensor([[2.0, 4.0, 8.0]])
        s = compute_smoothing_weight_only(W)
        expected = 1.0 / torch.tensor([2.0, 4.0, 8.0])
        assert torch.allclose(s, expected, atol=1e-6)
