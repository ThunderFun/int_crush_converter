"""Tests for the Triton PiSO (Piecewise Scale Optimization) kernels."""

import pytest
import torch

from converter.piso_triton import (
    compute_piso_scales_int8_triton,
    compute_piso_scales_int8_asymmetric_triton,
    compute_piso_scales_int4_triton,
    compute_piso_scales_int4_asymmetric_triton,
    _HAS_TRITON,
)
from converter.piso import (
    _compute_piso_scales_pytorch,
    _compute_piso_scales_asymmetric_pytorch,
)


# ── Symmetric kernel ─────────────────────────────────────────────────────────


class TestPisoSymmetric:
    """Tests for the symmetric INT8 PiSO Triton kernel."""

    @pytest.mark.gpu
    def test_matches_pytorch_small(self):
        """Triton scales match PyTorch scales for a small matrix."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        torch.manual_seed(42)
        M, D = 32, 512
        W = torch.randn(M, D, device="cuda")
        h_diag = torch.rand(D, device="cuda").clamp(min=1e-6)

        triton_result = compute_piso_scales_int8_triton(W, h_diag, num_coarse=32, num_fine=8)
        assert triton_result is not None
        assert triton_result.shape == (M, 1)

        pytorch_result = _compute_piso_scales_pytorch(
            W.cpu(), h_diag.cpu(), num_coarse=32, num_fine=8,
            qmin=-128, qmax=127, divisor=127.0,
        )

        # Tolerance: grid search with 32+8 candidates may disagree by at most
        # one grid step.  The Triton kernel uses round-half-away-from-zero
        # while PyTorch's .round() uses banker's rounding; these differ at
        # exact half-integers, causing adjacent candidates to be selected.
        # Use 5% relative tolerance (grid spacing is ~3% for 32 coarse
        # candidates on [0.5, 2.0]).
        torch.testing.assert_close(
            triton_result.cpu(),
            pytorch_result,
            rtol=0.05,
            atol=1e-8,
        )

    @pytest.mark.gpu
    def test_matches_pytorch_medium(self):
        """Triton scales match PyTorch for a typical model dimension."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        torch.manual_seed(123)
        M, D = 64, 2048
        W = torch.randn(M, D, device="cuda")
        h_diag = torch.rand(D, device="cuda").clamp(min=1e-6)

        triton_result = compute_piso_scales_int8_triton(W, h_diag, num_coarse=64, num_fine=16)
        assert triton_result is not None

        pytorch_result = _compute_piso_scales_pytorch(
            W.cpu(), h_diag.cpu(), num_coarse=64, num_fine=16,
            qmin=-128, qmax=127, divisor=127.0,
        )

        # 5% tolerance — see test_matches_pytorch_small for rationale.
        torch.testing.assert_close(
            triton_result.cpu(),
            pytorch_result,
            rtol=0.05,
            atol=1e-8,
        )

    @pytest.mark.gpu
    def test_output_shape_and_dtype(self):
        """Output is [M, 1] float32."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        M, D = 16, 256
        W = torch.randn(M, D, device="cuda")
        h_diag = torch.ones(D, device="cuda")

        result = compute_piso_scales_int8_triton(W, h_diag, num_coarse=16, num_fine=4)
        assert result is not None
        assert result.shape == (M, 1)
        assert result.dtype == torch.float32

    @pytest.mark.gpu
    def test_scales_are_positive(self):
        """All returned scales must be positive."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        torch.manual_seed(7)
        M, D = 32, 512
        W = torch.randn(M, D, device="cuda")
        h_diag = torch.rand(D, device="cuda").clamp(min=1e-6)

        result = compute_piso_scales_int8_triton(W, h_diag)
        assert result is not None
        assert (result > 0).all()

    @pytest.mark.gpu
    def test_all_zero_row(self):
        """All-zero rows should produce a valid (clamped) scale."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        M, D = 8, 256
        W = torch.zeros(M, D, device="cuda")
        h_diag = torch.ones(D, device="cuda")

        result = compute_piso_scales_int8_triton(W, h_diag, num_coarse=16, num_fine=4)
        assert result is not None
        # For all-zero rows, s_absmax is clamped to 1e-10, then the coarse
        # factor (0.5) produces 5e-11.  Just check positive and finite.
        assert torch.isfinite(result).all()
        assert (result > 0).all()

    @pytest.mark.gpu
    def test_large_weights(self):
        """Large-magnitude weights should produce proportionally large scales."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        torch.manual_seed(99)
        M, D = 16, 256
        W = torch.randn(M, D, device="cuda") * 100.0
        h_diag = torch.ones(D, device="cuda")

        result = compute_piso_scales_int8_triton(W, h_diag, num_coarse=16, num_fine=4)
        assert result is not None
        # Scale should be roughly |w_max| / 127 ≈ 100/127 ≈ 0.79
        assert result.min() > 0.1
        assert result.max() < 10.0

    @pytest.mark.gpu
    @pytest.mark.parametrize("D", [64, 128, 256, 512, 1024, 2048, 4096])
    def test_various_d_sizes(self, D):
        """Kernel compiles and runs for various D values (powers of 2)."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        torch.manual_seed(0)
        M = 16
        W = torch.randn(M, D, device="cuda")
        h_diag = torch.rand(D, device="cuda").clamp(min=1e-6)

        result = compute_piso_scales_int8_triton(W, h_diag, num_coarse=16, num_fine=4)
        assert result is not None
        assert result.shape == (M, 1)
        assert torch.isfinite(result).all()

    @pytest.mark.gpu
    def test_non_power_of_two_d(self):
        """Kernel handles D that is not a power of 2."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        torch.manual_seed(5)
        M, D = 16, 777
        W = torch.randn(M, D, device="cuda")
        h_diag = torch.rand(D, device="cuda").clamp(min=1e-6)

        result = compute_piso_scales_int8_triton(W, h_diag, num_coarse=16, num_fine=4)
        assert result is not None
        assert result.shape == (M, 1)
        assert torch.isfinite(result).all()


# ── Asymmetric kernel ────────────────────────────────────────────────────────


class TestPisoAsymmetric:
    """Tests for the asymmetric INT8 PiSO Triton kernel."""

    @pytest.mark.gpu
    def test_matches_pytorch_small(self):
        """Triton asymmetric scales match PyTorch for a small matrix."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        torch.manual_seed(42)
        M, D = 32, 512
        W = torch.randn(M, D, device="cuda")
        h_diag = torch.rand(D, device="cuda").clamp(min=1e-6)

        triton_result = compute_piso_scales_int8_asymmetric_triton(
            W, h_diag, num_coarse=32, num_fine=8
        )
        assert triton_result is not None
        scales_t, zp_t = triton_result
        assert scales_t.shape == (M, 1)
        assert zp_t.shape == (M, 1)

        pytorch_result = _compute_piso_scales_asymmetric_pytorch(
            W.cpu(), h_diag.cpu(), num_coarse=32, num_fine=8,
            qmin=-128, qmax=127, coarse_low=0.8, coarse_high=1.2,
        )
        scales_p, zp_p = pytorch_result

        # Scales should be close (within grid resolution).
        # 5% tolerance — see symmetric test for rounding-mode rationale.
        torch.testing.assert_close(
            scales_t.cpu(), scales_p, rtol=0.05, atol=1e-8,
        )

        # Zero-points are integers; they should match exactly (or differ by
        # at most 1 due to rounding differences near grid boundaries).
        assert (zp_t.cpu() - zp_p).abs().max() <= 1

    @pytest.mark.gpu
    def test_output_shapes(self):
        """Output shapes are [M, 1] for both scales and zero-points."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        M, D = 16, 256
        W = torch.randn(M, D, device="cuda")
        h_diag = torch.ones(D, device="cuda")

        result = compute_piso_scales_int8_asymmetric_triton(
            W, h_diag, num_coarse=16, num_fine=4
        )
        assert result is not None
        scales, zp = result
        assert scales.shape == (M, 1)
        assert scales.dtype == torch.float32
        assert zp.shape == (M, 1)
        assert zp.dtype == torch.int8

    @pytest.mark.gpu
    def test_zp_range(self):
        """Zero-points must be in [-128, 127]."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        torch.manual_seed(77)
        M, D = 32, 512
        W = torch.randn(M, D, device="cuda")
        h_diag = torch.rand(D, device="cuda").clamp(min=1e-6)

        result = compute_piso_scales_int8_asymmetric_triton(W, h_diag, num_coarse=32, num_fine=8)
        assert result is not None
        _, zp = result
        assert zp.min() >= -128
        assert zp.max() <= 127

    @pytest.mark.gpu
    def test_scales_are_positive(self):
        """All returned scales must be positive."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        torch.manual_seed(11)
        M, D = 16, 256
        W = torch.randn(M, D, device="cuda")
        h_diag = torch.rand(D, device="cuda").clamp(min=1e-6)

        result = compute_piso_scales_int8_asymmetric_triton(W, h_diag, num_coarse=16, num_fine=4)
        assert result is not None
        scales, _ = result
        assert (scales > 0).all()

    @pytest.mark.gpu
    def test_skewed_weights(self):
        """Asymmetric handles weights with skewed distributions."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        torch.manual_seed(33)
        M, D = 16, 256
        # Skewed: mostly positive with some negative
        W = torch.randn(M, D, device="cuda") + 2.0
        h_diag = torch.rand(D, device="cuda").clamp(min=1e-6)

        result = compute_piso_scales_int8_asymmetric_triton(W, h_diag, num_coarse=16, num_fine=4)
        assert result is not None
        scales, zp = result
        assert torch.isfinite(scales).all()
        assert (scales > 0).all()


# ── INT4 Symmetric kernel ────────────────────────────────────────────────────


class TestPisoInt4Symmetric:
    """Tests for the symmetric INT4 PiSO Triton kernel."""

    @pytest.mark.gpu
    def test_matches_pytorch_small(self):
        """Triton INT4 symmetric scales match PyTorch for a small matrix."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        torch.manual_seed(42)
        M, D = 32, 512
        W = torch.randn(M, D, device="cuda")
        h_diag = torch.rand(D, device="cuda").clamp(min=1e-6)

        triton_result = compute_piso_scales_int4_triton(W, h_diag, num_coarse=32, num_fine=8)
        assert triton_result is not None
        assert triton_result.shape == (M, 1)

        pytorch_result = _compute_piso_scales_pytorch(
            W.cpu(), h_diag.cpu(), num_coarse=32, num_fine=8,
            qmin=-8, qmax=7, divisor=7.0,
        )

        torch.testing.assert_close(
            triton_result.cpu(),
            pytorch_result,
            rtol=0.05,
            atol=1e-8,
        )

    @pytest.mark.gpu
    def test_output_shape_and_dtype(self):
        """Output is [M, 1] float32."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        M, D = 16, 256
        W = torch.randn(M, D, device="cuda")
        h_diag = torch.ones(D, device="cuda")

        result = compute_piso_scales_int4_triton(W, h_diag, num_coarse=16, num_fine=4)
        assert result is not None
        assert result.shape == (M, 1)
        assert result.dtype == torch.float32

    @pytest.mark.gpu
    def test_scales_are_positive(self):
        """All returned scales must be positive."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        torch.manual_seed(7)
        M, D = 32, 512
        W = torch.randn(M, D, device="cuda")
        h_diag = torch.rand(D, device="cuda").clamp(min=1e-6)

        result = compute_piso_scales_int4_triton(W, h_diag)
        assert result is not None
        assert (result > 0).all()


# ── INT4 Asymmetric kernel ───────────────────────────────────────────────────


class TestPisoInt4Asymmetric:
    """Tests for the asymmetric INT4 PiSO Triton kernel."""

    @pytest.mark.gpu
    def test_matches_pytorch_small(self):
        """Triton INT4 asymmetric scales match PyTorch for a small matrix."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        torch.manual_seed(42)
        M, D = 32, 512
        W = torch.randn(M, D, device="cuda")
        h_diag = torch.rand(D, device="cuda").clamp(min=1e-6)

        triton_result = compute_piso_scales_int4_asymmetric_triton(
            W, h_diag, num_coarse=32, num_fine=8
        )
        assert triton_result is not None
        scales_t, zp_t = triton_result
        assert scales_t.shape == (M, 1)
        assert zp_t.shape == (M, 1)

        pytorch_result = _compute_piso_scales_asymmetric_pytorch(
            W.cpu(), h_diag.cpu(), num_coarse=32, num_fine=8,
            qmin=-8, qmax=7, coarse_low=0.5, coarse_high=2.0,
        )
        scales_p, zp_p = pytorch_result

        torch.testing.assert_close(
            scales_t.cpu(), scales_p, rtol=0.05, atol=1e-8,
        )

        # Zero-points are integers; may differ by at most 1 due to rounding.
        assert (zp_t.cpu() - zp_p).abs().max() <= 1

    @pytest.mark.gpu
    def test_output_shapes(self):
        """Output shapes are [M, 1] for both scales and zero-points."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        M, D = 16, 256
        W = torch.randn(M, D, device="cuda")
        h_diag = torch.ones(D, device="cuda")

        result = compute_piso_scales_int4_asymmetric_triton(
            W, h_diag, num_coarse=16, num_fine=4
        )
        assert result is not None
        scales, zp = result
        assert scales.shape == (M, 1)
        assert scales.dtype == torch.float32
        assert zp.shape == (M, 1)
        assert zp.dtype == torch.int8

    @pytest.mark.gpu
    def test_zp_range(self):
        """Zero-points must be in [-8, 7]."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        torch.manual_seed(77)
        M, D = 32, 512
        W = torch.randn(M, D, device="cuda")
        h_diag = torch.rand(D, device="cuda").clamp(min=1e-6)

        result = compute_piso_scales_int4_asymmetric_triton(W, h_diag, num_coarse=32, num_fine=8)
        assert result is not None
        _, zp = result
        assert zp.min() >= -8
        assert zp.max() <= 7

    @pytest.mark.gpu
    def test_scales_are_positive(self):
        """All returned scales must be positive."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        torch.manual_seed(11)
        M, D = 16, 256
        W = torch.randn(M, D, device="cuda")
        h_diag = torch.rand(D, device="cuda").clamp(min=1e-6)

        result = compute_piso_scales_int4_asymmetric_triton(W, h_diag, num_coarse=16, num_fine=4)
        assert result is not None
        scales, _ = result
        assert (scales > 0).all()

    @pytest.mark.gpu
    def test_skewed_weights(self):
        """Asymmetric INT4 handles weights with skewed distributions."""
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        torch.manual_seed(33)
        M, D = 16, 256
        W = torch.randn(M, D, device="cuda") + 2.0
        h_diag = torch.rand(D, device="cuda").clamp(min=1e-6)

        result = compute_piso_scales_int4_asymmetric_triton(W, h_diag, num_coarse=16, num_fine=4)
        assert result is not None
        scales, zp = result
        assert torch.isfinite(scales).all()
        assert (scales > 0).all()


# ── Dispatch tests (no GPU required) ─────────────────────────────────────────


class TestPisoDispatch:
    """Test the dispatch logic in piso.py (CPU fallback path)."""

    def test_symmetric_cpu_fallback(self):
        """CPU fallback produces valid scales without crashing."""
        torch.manual_seed(42)
        M, D = 8, 128
        W = torch.randn(M, D)
        h_diag = torch.rand(D).clamp(min=1e-6)

        from converter.piso import compute_piso_scales_int8
        result = compute_piso_scales_int8(W, h_diag, num_coarse=16, num_fine=4)
        assert result.shape == (M, 1)
        assert result.dtype == torch.float32
        assert torch.isfinite(result).all()
        assert (result > 0).all()

    def test_asymmetric_cpu_fallback(self):
        """CPU fallback produces valid scales and zero-points."""
        torch.manual_seed(42)
        M, D = 8, 128
        W = torch.randn(M, D)
        h_diag = torch.rand(D).clamp(min=1e-6)

        from converter.piso import compute_piso_scales_int8_asymmetric
        scales, zp = compute_piso_scales_int8_asymmetric(
            W, h_diag, num_coarse=16, num_fine=4
        )
        assert scales.shape == (M, 1)
        assert zp.shape == (M, 1)
        assert (scales > 0).all()
        assert zp.min() >= -128
        assert zp.max() <= 127

    def test_invalid_input_raises(self):
        """Non-2D input raises ValueError."""
        from converter.piso import compute_piso_scales_int8
        with pytest.raises(ValueError, match="2D"):
            compute_piso_scales_int8(torch.randn(8), torch.ones(64))

    def test_hessian_diag_length_mismatch_raises(self):
        """Mismatched hessian_diag length raises ValueError."""
        from converter.piso import compute_piso_scales_int8
        with pytest.raises(ValueError, match="in_features"):
            compute_piso_scales_int8(torch.randn(8, 64), torch.ones(32))
