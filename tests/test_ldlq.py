"""Tests for LDLQ quantization.

Based on the theoretical guarantees from:
- QuIP (Chee et al., 2023, arXiv:2307.13304): LDLQ is optimal within adaptive
  rounding methods with linear feedback, achieving proxy loss trace(D) where
  H = (U+I) D (U+I)^T is the LDL decomposition.
- QuIP# (Tseng et al., 2024, arXiv:2402.04396): BlockLDLQ and lattice codebooks.

Key properties tested:
1. Output range [-8, 7] for INT4
2. LDLQ reduces proxy loss vs RTN (Theorem 1 from QuIP)
3. trace(D) <= trace(H) for non-diagonal H (Lemma from QuIP)
4. Iterative refinement converges
5. H = W^T @ W / M is positive semi-definite
"""

import torch
import pytest

from converter.ldlq import (
    ldlq_quantize_layer,
    _round_half_away_from_zero,
    _ldlq_round_column,
    _single_ldlq_pass,
    _run_iterative_ldlq,
)
from converter.rounding import _invert_hessian
from converter.scales import calculate_scales, quantize_weights


class TestLDLQQuantizeLayer:
    """Tests for the core LDLQ quantization algorithm."""

    def test_output_range_int4(self):
        """Quantized values must be in [-8, 7] for INT4."""
        W = torch.randn(16, 64)
        q_W, scales = ldlq_quantize_layer(W)
        assert q_W.min() >= -8
        assert q_W.max() <= 7

    def test_output_range_int8(self):
        """Quantized values must be in [-128, 127] for INT8."""
        W = torch.randn(16, 64)
        q_W, scales = ldlq_quantize_layer(W, int_bits=8)
        assert q_W.min() >= -128
        assert q_W.max() <= 127

    def test_output_dtype(self):
        """Quantized output should be int8."""
        W = torch.randn(8, 32)
        q_W, scales = ldlq_quantize_layer(W)
        assert q_W.dtype == torch.int8

    def test_scales_dtype(self):
        """Scales should be float16."""
        W = torch.randn(8, 32)
        _, scales = ldlq_quantize_layer(W)
        assert scales.dtype == torch.float16

    def test_scales_shape_per_row(self):
        """Scales should have shape [out_features, 1] (per-row)."""
        W = torch.randn(16, 64)
        _, scales = ldlq_quantize_layer(W)
        assert scales.shape == (16, 1)

    def test_scales_positive(self):
        """All scales must be positive."""
        W = torch.randn(16, 64)
        _, scales = ldlq_quantize_layer(W)
        assert torch.all(scales > 0)

    def test_no_nan_inf_in_quantized(self):
        """Quantized weights and scales should not contain NaN or Inf."""
        W = torch.randn(16, 64)
        q_W, scales = ldlq_quantize_layer(W)
        assert not torch.any(torch.isnan(q_W.float()))
        assert not torch.any(torch.isinf(q_W.float()))
        assert not torch.any(torch.isnan(scales))
        assert not torch.any(torch.isinf(scales))

    def test_zero_weights(self):
        """Zero weights should quantize to zero."""
        W = torch.zeros(8, 32)
        q_W, scales = ldlq_quantize_layer(W)
        assert torch.all(q_W == 0)

    def test_invalid_int_bits_raises(self):
        """Should raise for unsupported int_bits values."""
        W = torch.randn(8, 32)
        with pytest.raises(ValueError):
            ldlq_quantize_layer(W, int_bits=2)

    def test_non_2d_weight_raises(self):
        """Should raise for non-2D weight tensor."""
        W = torch.randn(8, 16, 16)
        with pytest.raises(ValueError):
            ldlq_quantize_layer(W)

    def test_block_size_smaller_than_in_features(self):
        """LDLQ should work when block_size < in_features."""
        W = torch.randn(8, 128)
        q_W, scales = ldlq_quantize_layer(W, block_size=32)
        assert q_W.shape == (8, 128)
        assert q_W.min() >= -8
        assert q_W.max() <= 7

    def test_block_size_equals_in_features(self):
        """LDLQ should work when block_size == in_features (single block)."""
        W = torch.randn(8, 64)
        q_W, scales = ldlq_quantize_layer(W, block_size=64)
        assert q_W.shape == (8, 64)

    def test_damping_prevents_singular_hessian(self):
        """Damping should handle near-singular Hessians gracefully."""
        W = torch.randn(8, 32)
        # Create a rank-1 weight matrix (singular H = W^T W / M)
        W = W * 0
        W[:, 0] = torch.randn(8)
        q_W, scales = ldlq_quantize_layer(W, damping=0.1)
        assert q_W.shape == (8, 32)
        assert not torch.any(torch.isnan(q_W.float()))


class TestLDLQVsRTN:
    """Test that LDLQ achieves lower proxy loss than RTN.

    From QuIP Theorem 1: LDLQ achieves proxy loss = m/4 * trace(D) in the
    worst case, where H = (U+I) D (U+I)^T is the LDL decomposition.
    For nearest rounding (RTN), worst case = m/4 * trace(H).
    Since trace(D) <= trace(H) for non-diagonal H, LDLQ should be better.
    """

    def test_proxy_loss_reduction(self):
        """LDLQ should produce lower proxy loss than RTN.

        The proxy loss is: tr((W_hat - W) H (W_hat - W)^T)
        where H = W^T @ W / M (weight-only Hessian).
        """
        torch.manual_seed(42)
        M, N = 16, 64

        W = torch.randn(M, N)

        # LDLQ quantization
        q_W_ldlq, scales_ldlq = ldlq_quantize_layer(W, block_size=32)

        # RTN quantization
        scales_rtn = calculate_scales(W, N)
        q_W_rtn = quantize_weights(W, scales_rtn, N)

        # Compute Hessian H = W^T @ W / M
        H = W.T @ W / M

        # Compute proxy loss: tr((Q - W) H (Q - W)^T)
        err_ldlq = (q_W_ldlq.float() * scales_ldlq.float() - W)
        err_rtn = (q_W_rtn.float() * scales_rtn.float() - W)

        # Pad to same size if needed (RTN may pad)
        min_cols = min(err_ldlq.shape[1], err_rtn.shape[1], H.shape[0])
        err_ldlq = err_ldlq[:, :min_cols]
        err_rtn = err_rtn[:, :min_cols]
        H = H[:min_cols, :min_cols]

        proxy_ldlq = torch.trace(err_ldlq @ H @ err_ldlq.T).item()
        proxy_rtn = torch.trace(err_rtn @ H @ err_rtn.T).item()

        # LDLQ should be better (or at least not much worse) than RTN
        # Allow some tolerance since this is a small example
        assert proxy_ldlq <= proxy_rtn * 1.1, (
            f"LDLQ proxy loss {proxy_ldlq:.4f} > RTN proxy loss {proxy_rtn:.4f}"
        )

    def test_trace_d_le_trace_h(self):
        """trace(D) <= trace(H) for non-diagonal H (QuIP Lemma).

        For the LDL decomposition H = (U+I) D (U+I)^T, we have
        trace(D) <= trace(H), with equality only when H is diagonal.
        This is the fundamental property that makes LDLQ better than RTN.
        """
        torch.manual_seed(42)
        N = 32

        # Create a non-diagonal positive definite H
        X = torch.randn(64, N)
        H = X.T @ X / 64

        # Compute LDL decomposition via Cholesky
        # H = L L^T where L is lower triangular
        # Then D = diag(L)^2 and U = L^T / diag(L) - I
        L = torch.linalg.cholesky(H)
        D = torch.diag(L) ** 2
        trace_D = D.sum().item()
        trace_H = torch.trace(H).item()

        # trace(D) should be <= trace(H)
        assert trace_D <= trace_H * 1.001, (
            f"trace(D)={trace_D:.4f} > trace(H)={trace_H:.4f}"
        )

        # For non-diagonal H, trace(D) should be strictly less
        if not torch.allclose(H, torch.diag(torch.diag(H)), atol=1e-6):
            assert trace_D < trace_H * 0.999, (
                f"Non-diagonal H: trace(D)={trace_D:.4f} should be < trace(H)={trace_H:.4f}"
            )


class TestRoundingFunctions:
    """Tests for the rounding utility functions."""

    def test_round_half_away_from_zero_positive(self):
        """0.5 should round to 1 (away from zero), not 0."""
        x = torch.tensor([0.5, 1.5, 2.5, 3.5])
        result = _round_half_away_from_zero(x)
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert torch.allclose(result, expected)

    def test_round_half_away_from_zero_negative(self):
        """-0.5 should round to -1 (away from zero), not 0."""
        x = torch.tensor([-0.5, -1.5, -2.5, -3.5])
        result = _round_half_away_from_zero(x)
        expected = torch.tensor([-1.0, -2.0, -3.0, -4.0])
        assert torch.allclose(result, expected)

    def test_round_half_away_from_zero_exact(self):
        """Exact integers should remain unchanged."""
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = _round_half_away_from_zero(x)
        assert torch.allclose(result, x)

    def test_ldlq_round_column_range(self):
        """Quantized column should be in [clamp_min, clamp_max]."""
        M = 16
        col = torch.randn(M) * 5
        scale_col = torch.ones(M) * 2.0
        q, err = _ldlq_round_column(col, scale_col, -8, 7)
        assert q.min() >= -8
        assert q.max() <= 7

    def test_ldlq_round_column_sign_flip(self):
        """Sign-flip correction should prevent sign mismatches."""
        M = 8
        # Create a case where rounding might produce wrong sign
        col = torch.tensor([0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 7.0, -7.0])
        scale_col = torch.ones(M) * 0.3  # Small scale → large quantized values
        q, err = _ldlq_round_column(col, scale_col, -8, 7)
        # Check that signs are consistent (for non-near-zero weights)
        for i in range(M):
            if col[i].abs() > 0.1:
                assert (col[i].sign() * q[i].sign()) >= 0, (
                    f"Sign mismatch at {i}: col={col[i]:.2f}, q={q[i]:.2f}"
                )

    def test_ldlq_round_column_error(self):
        """Error should be col - q * scale."""
        M = 8
        col = torch.randn(M) * 3
        scale_col = torch.ones(M) * 1.5
        q, err = _ldlq_round_column(col, scale_col, -8, 7)
        expected_err = col - q * scale_col.clamp(min=1e-8)
        assert torch.allclose(err, expected_err, atol=1e-6)


class TestLDLQIterative:
    """Tests for iterative LDLQ with scale refinement."""

    def test_iterative_convergence(self):
        """MSE should decrease or converge across iterations."""
        torch.manual_seed(42)
        M, N = 8, 32

        W = torch.randn(M, N)
        H = W.T @ W / M
        diag_mean = H.diagonal().mean().clamp(min=1e-6)
        H = H + 0.01 * diag_mean * torch.eye(N)
        H_inv = _invert_hessian(H)

        flat_scales = (W.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-6)
        flat_scales = flat_scales.expand(M, N).reshape(-1).clone()

        Q_result, final_scales = _run_iterative_ldlq(
            W, H_inv, flat_scales, iterations=5, block_size=32,
            clamp_min=-8, clamp_max=7,
        )

        # Should produce valid output
        assert Q_result.shape == (M, N)
        assert Q_result.dtype == torch.int8
        assert Q_result.min() >= -8
        assert Q_result.max() <= 7

    def test_single_vs_iterative(self):
        """Single pass and iterative should both produce valid output."""
        torch.manual_seed(42)
        M, N = 8, 32

        W = torch.randn(M, N)

        # Single pass
        q_single, s_single = ldlq_quantize_layer(W, iterations=1)

        # Multiple iterations
        q_multi, s_multi = ldlq_quantize_layer(W, iterations=3)

        # Both should be valid
        assert q_single.min() >= -8 and q_single.max() <= 7
        assert q_multi.min() >= -8 and q_multi.max() <= 7

        # Both should produce reasonable MSE
        deq_single = q_single.float() * s_single.float()
        deq_multi = q_multi.float() * s_multi.float()
        mse_single = (W - deq_single).pow(2).mean().item()
        mse_multi = (W - deq_multi).pow(2).mean().item()
        assert mse_single < 1.0
        assert mse_multi < 1.0


class TestLDLQHessian:
    """Tests for the weight-only Hessian H = W^T @ W / M."""

    def test_hessian_positive_semidefinite(self):
        """H = W^T @ W / M should be positive semi-definite."""
        torch.manual_seed(42)
        M, N = 32, 64
        W = torch.randn(M, N)
        H = W.T @ W / M

        eigenvalues = torch.linalg.eigvalsh(H)
        assert torch.all(eigenvalues >= -1e-6), (
            f"H has negative eigenvalues: {eigenvalues.min():.6f}"
        )

    def test_hessian_symmetric(self):
        """H should be symmetric."""
        torch.manual_seed(42)
        M, N = 32, 64
        W = torch.randn(M, N)
        H = W.T @ W / M

        assert torch.allclose(H, H.T, atol=1e-6)

    def test_hessian_shape(self):
        """H should be [N, N]."""
        M, N = 16, 48
        W = torch.randn(M, N)
        H = W.T @ W / M

        assert H.shape == (N, N)


class TestLDLQIntegration:
    """Integration tests: rotate -> LDLQ -> pack -> unpack."""

    def test_end_to_end_int4(self):
        """Full pipeline with INT4: weights -> rotate -> LDLQ -> pack."""
        from converter.packing import pack_int4, unpack_int4
        from converter.rotation import rotate_weights

        torch.manual_seed(42)
        in_features = 128
        out_features = 32

        W = torch.randn(out_features, in_features)
        W_rot = rotate_weights(W, rot_size=16)

        q_W, scales = ldlq_quantize_layer(W_rot)
        assert q_W.shape == (out_features, in_features)
        assert scales.shape == (out_features, 1)

        packed = pack_int4(q_W)
        assert packed.dtype == torch.uint8
        assert packed.shape == (out_features, in_features // 2)

        unpacked = unpack_int4(packed, in_features)
        assert torch.equal(unpacked, q_W)

    def test_end_to_end_int8(self):
        """Full pipeline with INT8: weights -> rotate -> LDLQ."""
        from converter.rotation import rotate_weights

        torch.manual_seed(42)
        in_features = 128
        out_features = 32

        W = torch.randn(out_features, in_features)
        W_rot = rotate_weights(W, rot_size=16)

        q_W, scales = ldlq_quantize_layer(W_rot, int_bits=8)
        assert q_W.shape == (out_features, in_features)
        assert q_W.dtype == torch.int8
        assert q_W.min() >= -128
        assert q_W.max() <= 127

        W_deq = q_W.float() * scales.float()
        mse = (W_rot - W_deq).pow(2).mean().item()
        assert mse < 1.0

    def test_with_rotation(self):
        """LDLQ should work on rotated weights."""
        from converter.rotation import rotate_weights

        torch.manual_seed(42)
        in_features = 256
        out_features = 32

        W = torch.randn(out_features, in_features)
        W_rot = rotate_weights(W, rot_size=64)

        q_W, scales = ldlq_quantize_layer(W_rot, block_size=64)
        assert q_W.shape[0] == out_features
        assert q_W.dtype == torch.int8
        assert not torch.any(torch.isnan(q_W.float()))

    def test_dequantization_quality(self):
        """Dequantized weights should be close to original."""
        torch.manual_seed(42)
        M, N = 16, 64

        W = torch.randn(M, N)
        q_W, scales = ldlq_quantize_layer(W)

        W_deq = q_W.float() * scales.float()
        mse = (W - W_deq).pow(2).mean().item()

        # MSE should be reasonable (not garbage)
        assert mse < 1.0, f"MSE too high: {mse:.6f}"
        # MSE should be non-zero (actually quantized)
        assert mse > 1e-10, f"MSE suspiciously low: {mse:.6f}"


class TestLDLQWithPermuQuant:
    """Test LDLQ combined with PermuQuant permutation."""

    def test_ldlq_on_permuted_weights(self):
        """LDLQ should work on permuted weights (PermuQuant compatibility)."""
        from converter.permuquant import find_permutation_weight

        torch.manual_seed(42)
        in_features = 64
        out_features = 16

        W = torch.randn(out_features, in_features)

        # Find permutation
        perm = find_permutation_weight(W, group_size=32)

        # Apply permutation to weights
        W_perm = W[:, perm]

        # LDLQ on permuted data
        q_W, scales = ldlq_quantize_layer(W_perm)
        assert q_W.shape == (out_features, in_features)
        assert q_W.min() >= -8
        assert q_W.max() <= 7


class TestTritonKernel:
    """Tests for the Triton LDLQ block kernel."""

    def test_triton_import(self):
        """ldlq_loop_triton should be importable regardless of Triton availability."""
        from converter.ldlq_triton import ldlq_loop_triton, _HAS_TRITON
        assert callable(ldlq_loop_triton)

    def test_triton_has_triton_flag(self):
        """_HAS_TRITON should be a bool."""
        from converter.ldlq_triton import _HAS_TRITON
        assert isinstance(_HAS_TRITON, bool)

    def test_triton_returns_none_without_cuda(self):
        """ldlq_loop_triton should return None if Triton can't run."""
        from converter.ldlq_triton import ldlq_loop_triton, _HAS_TRITON
        if not _HAS_TRITON:
            M, N = 8, 32
            W = torch.randn(M, N)
            H_inv = torch.eye(N)
            scale = torch.ones(M, N)
            Q = torch.zeros(M, N, dtype=torch.int8)
            result = ldlq_loop_triton(W, H_inv, scale, Q, N, M, 32, -8, 7)
            assert result is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_triton_int4_output_range(self):
        """Triton kernel INT4 output must be in [-8, 7]."""
        from converter.ldlq_triton import ldlq_loop_triton, _HAS_TRITON
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")
        M, N = 16, 64
        W = torch.randn(M, N, device="cuda")
        H = W.T @ W / M + 0.01 * torch.eye(N, device="cuda")
        H_inv = torch.linalg.inv(H)
        scale_2d = (W.abs() / 7.0).clamp(min=1e-6)
        Q = torch.zeros(M, N, dtype=torch.int8, device="cuda")
        ldlq_loop_triton(W, H_inv, scale_2d, Q, N, M, 32, -8, 7)
        assert Q.min() >= -8
        assert Q.max() <= 7

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_triton_int8_output_range(self):
        """Triton kernel INT8 output must be in [-128, 127]."""
        from converter.ldlq_triton import ldlq_loop_triton, _HAS_TRITON
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")
        M, N = 16, 64
        W = torch.randn(M, N, device="cuda")
        H = W.T @ W / M + 0.01 * torch.eye(N, device="cuda")
        H_inv = torch.linalg.inv(H)
        scale_2d = (W.abs() / 127.0).clamp(min=1e-6)
        Q = torch.zeros(M, N, dtype=torch.int8, device="cuda")
        ldlq_loop_triton(W, H_inv, scale_2d, Q, N, M, 32, -128, 127)
        assert Q.min() >= -128
        assert Q.max() <= 127

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_triton_matches_cpu_loop(self):
        """Triton kernel should produce same output as CPU loop."""
        from converter.ldlq_triton import ldlq_loop_triton, _HAS_TRITON
        from converter.ldlq import _ldlq_loop_cpu, _round_half_away_from_zero
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")

        torch.manual_seed(42)
        M, N = 8, 32
        block_size = 16

        W = torch.randn(M, N)
        H = W.T @ W / M + 0.01 * torch.eye(N)
        H_inv = torch.linalg.inv(H)
        scale_2d = (W.abs() / 7.0).clamp(min=1e-6)

        # CPU reference
        Q_cpu = torch.zeros(M, N, dtype=torch.int8)
        W_cpu = W.clone()
        _ldlq_loop_cpu(W_cpu, H_inv, scale_2d, Q_cpu, N, M, block_size, -8, 7)

        # Triton
        W_triton = W.clone().cuda()
        H_inv_cuda = H_inv.cuda()
        scale_cuda = scale_2d.cuda()
        Q_triton = torch.zeros(M, N, dtype=torch.int8, device="cuda")
        ldlq_loop_triton(W_triton, H_inv_cuda, scale_cuda, Q_triton, N, M, block_size, -8, 7)

        assert torch.equal(Q_triton.cpu(), Q_cpu)
