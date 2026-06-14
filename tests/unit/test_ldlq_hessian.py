"""Tests for the weight-only Hessian H = W^T @ W / M."""

import torch

from converter.rounding import _invert_hessian


class TestLDLQHessian:

    def test_hessian_positive_semidefinite(self):
        torch.manual_seed(42)
        W = torch.randn(32, 64)
        H = W.T @ W / 32

        eigenvalues = torch.linalg.eigvalsh(H)
        assert torch.all(eigenvalues >= -1e-6), (
            f"H has negative eigenvalues: {eigenvalues.min():.6f}"
        )

    def test_hessian_symmetric(self):
        torch.manual_seed(42)
        W = torch.randn(32, 64)
        H = W.T @ W / 32
        assert torch.allclose(H, H.T, atol=1e-6)

    def test_hessian_shape(self):
        W = torch.randn(16, 48)
        H = W.T @ W / 16
        assert H.shape == (48, 48)


class TestInvertHessian:

    def test_round_half_away_from_zero_positive(self):
        from converter.rounding import _round_half_away_from_zero
        x = torch.tensor([0.5, 1.5, 2.5, 3.5])
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert torch.allclose(_round_half_away_from_zero(x), expected)

    def test_round_half_away_from_zero_negative(self):
        from converter.rounding import _round_half_away_from_zero
        x = torch.tensor([-0.5, -1.5, -2.5, -3.5])
        expected = torch.tensor([-1.0, -2.0, -3.0, -4.0])
        assert torch.allclose(_round_half_away_from_zero(x), expected)

    def test_round_half_away_from_zero_exact(self):
        from converter.rounding import _round_half_away_from_zero
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert torch.allclose(_round_half_away_from_zero(x), x)

    def test_invert_hessian_pinv_fallback(self):
        """Pseudoinverse fallback should handle truly singular Hessians."""
        import pytest
        N = 8
        H = torch.zeros(N, N)
        H[0, 0] = 1.0

        with pytest.warns(UserWarning, match="singular/ill-conditioned"):
            H_inv = _invert_hessian(H)

        assert H_inv.shape == (N, N)
        assert torch.all(torch.isfinite(H_inv))
        assert torch.allclose(H @ H_inv @ H, H, atol=1e-5)
