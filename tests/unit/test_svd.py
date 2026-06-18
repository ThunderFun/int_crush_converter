"""Tests for SVD-absorbed low-rank decomposition."""

import math

import pytest
import torch

from converter.svd import decompose_weight, SVDResult
from tests.conftest import quantize_int8_per_row, dequantize_int8_per_row


# ── decompose_weight ────────────────────────────────────────────────────────


class TestDecomposeWeight:
    """Core SVD decomposition tests."""

    def test_output_types(self):
        W = torch.randn(64, 128)
        result = decompose_weight(W, rank=8)
        assert isinstance(result, SVDResult)
        assert result.L1.dtype == torch.float16
        assert result.L2.dtype == torch.float16
        assert result.residual.dtype == torch.float32

    def test_shapes(self):
        out_f, in_f, rank = 64, 128, 8
        W = torch.randn(out_f, in_f)
        result = decompose_weight(W, rank=rank)
        assert result.L1.shape == (out_f, rank)
        assert result.L2.shape == (rank, in_f)
        assert result.residual.shape == (out_f, in_f)

    def test_reconstruction_identity(self):
        """W == L1.float() @ L2.float() + residual (machine precision)."""
        torch.manual_seed(42)
        W = torch.randn(64, 128)
        result = decompose_weight(W, rank=16)
        W_recon = result.L1.float() @ result.L2.float() + result.residual
        assert torch.allclose(W, W_recon, atol=1e-5)

    def test_rank_clamped_to_min_dim(self):
        """rank > min(out, in) is clamped gracefully."""
        W = torch.randn(16, 32)
        result = decompose_weight(W, rank=100)
        # Clamped to min(100, 16) = 16
        assert result.L1.shape[1] == 16
        assert result.L2.shape[0] == 16

    def test_rank_1(self):
        """rank=1 gives the best rank-1 approximation."""
        torch.manual_seed(42)
        W = torch.randn(32, 64)
        result = decompose_weight(W, rank=1)
        assert result.L1.shape == (32, 1)
        assert result.L2.shape == (1, 64)
        # Residual should capture everything except the top singular vector
        W_recon = result.L1.float() @ result.L2.float() + result.residual
        assert torch.allclose(W, W_recon, atol=1e-5)

    def test_rank_equals_min_dim(self):
        """rank == min(out, in) should produce near-zero residual."""
        torch.manual_seed(42)
        W = torch.randn(16, 32)
        result = decompose_weight(W, rank=16)
        # With full rank, residual should be just FP16 rounding error
        assert result.residual.abs().max() < 0.1

    def test_residual_smaller_than_original(self):
        """The residual should have lower Frobenius norm than W."""
        torch.manual_seed(42)
        W = torch.randn(64, 128)
        # Make W have a dominant singular value
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        S[0] *= 10  # amplify top singular value
        W_dominant = U @ torch.diag(S) @ Vh

        result = decompose_weight(W_dominant, rank=8)
        assert result.residual.norm() < W_dominant.norm()

    def test_wrong_dim_raises(self):
        with pytest.raises(ValueError, match="2D"):
            decompose_weight(torch.randn(4, 8, 16), rank=2)

    def test_rank_zero_raises(self):
        with pytest.raises(ValueError, match="> 0"):
            decompose_weight(torch.randn(8, 16), rank=0)

    def test_square_matrix(self):
        W = torch.randn(64, 64)
        result = decompose_weight(W, rank=16)
        W_recon = result.L1.float() @ result.L2.float() + result.residual
        assert torch.allclose(W, W_recon, atol=1e-5)

    def test_small_matrix(self):
        W = torch.randn(4, 8)
        result = decompose_weight(W, rank=2)
        W_recon = result.L1.float() @ result.L2.float() + result.residual
        assert torch.allclose(W, W_recon, atol=1e-5)


# ── SVD + INT8 composition ─────────────────────────────────────────────────


class TestSVDQuantizationComposition:
    """SVD-absorbed quantization: low-rank FP16 + quantized INT8 residual."""

    def test_output_mse_better_than_rtn(self):
        """SVD + INT8 residual should beat plain INT8 RTN on output MSE.

        This is the core value proposition of SVDQuant: the low-rank branch
        absorbs dominant singular values, leaving a cleaner residual.
        """
        torch.manual_seed(42)
        # Create a weight with dominant singular values (realistic for transformers)
        out_f, in_f = 64, 128
        W = torch.randn(out_f, in_f) / math.sqrt(in_f)
        # Amplify top singular values to simulate outlier structure
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        S[:5] *= 20
        W = U @ torch.diag(S) @ Vh

        X = torch.randn(32, in_f)  # activation

        # Plain RTN
        q_rtn, s_rtn = quantize_int8_per_row(W)
        W_rtn = dequantize_int8_per_row(q_rtn, s_rtn)
        Y_orig = X @ W.T
        Y_rtn = X @ W_rtn.T
        mse_rtn = ((Y_orig - Y_rtn) ** 2).mean().item()

        # SVD-absorbed: rank-8 FP16 + INT8 residual
        svd = decompose_weight(W, rank=8)
        q_res, s_res = quantize_int8_per_row(svd.residual)
        residual_rec = dequantize_int8_per_row(q_res, s_res)
        W_svd = svd.L1.float() @ svd.L2.float() + residual_rec
        Y_svd = X @ W_svd.T
        mse_svd = ((Y_orig - Y_svd) ** 2).mean().item()

        assert mse_svd < mse_rtn, (
            f"SVD+INT8 ({mse_svd:.4e}) should beat plain RTN ({mse_rtn:.4e})"
        )

    def test_svd_with_convrot_composition(self):
        """SVD + ConvRot: Y = X@L2^T@L1^T + X_rot@R_rot^T == X@W^T.

        The low-rank branch operates in the original basis; the residual
        is rotated.  Mathematically exact before quantization.
        """
        from converter.rotation import get_hadamard
        import torch.nn.functional as F

        torch.manual_seed(42)
        rot_size = 64
        out_f, in_f = 32, 128
        W = torch.randn(out_f, in_f)
        X = torch.randn(16, in_f)

        svd = decompose_weight(W, rank=8)

        # Low-rank branch (original basis)
        lr_out = X @ svd.L2.float().T @ svd.L1.float().T

        # Residual branch (rotated)
        R = svd.residual
        if in_f % rot_size != 0:
            R = F.pad(R, (0, rot_size - in_f % rot_size))
            X_pad = F.pad(X, (0, rot_size - in_f % rot_size))
        else:
            X_pad = X
        H = get_hadamard(rot_size, dtype=torch.float32, device=str(R.device))
        R_rot = R.reshape(out_f, -1, rot_size) @ H.T
        R_rot = R_rot.reshape(out_f, -1)
        X_rot = X_pad.reshape(-1, in_f_padded := X_pad.shape[-1] // rot_size, rot_size) @ H.T
        X_rot = X_rot.reshape(X.shape[0], -1)
        res_out = X_rot @ R_rot.T

        Y_approx = lr_out + res_out
        Y_orig = X @ W.T

        # Should be exact up to padding (which adds zero-energy columns)
        # Crop to original in_f for comparison
        assert torch.allclose(Y_orig, Y_approx[:, :out_f], atol=1e-4)

    def test_fp16_rounding_absorbed_into_residual(self):
        """The residual absorbs FP16 rounding of L1/L2, so at inference
        the stored FP16 factors reconstruct correctly."""
        torch.manual_seed(42)
        W = torch.randn(32, 64)
        svd = decompose_weight(W, rank=8)

        # Simulate inference: use stored FP16 L1, L2 + quantized residual
        q_res, s_res = quantize_int8_per_row(svd.residual)
        residual_rec = dequantize_int8_per_row(q_res, s_res)
        W_inference = svd.L1.float() @ svd.L2.float() + residual_rec

        # Error should be bounded by INT8 quantization error of residual
        err = (W - W_inference).abs()
        # Per-row INT8 step size for the residual
        row_scale = svd.residual.abs().amax(dim=1) / 127.0
        # Error should be at most ~1 scale step per element
        max_expected_err = row_scale.max().item()
        assert err.max() < max_expected_err * 2  # generous margin
