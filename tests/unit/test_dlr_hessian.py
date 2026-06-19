"""Tests for Diagonal + Low-Rank (DLR) Hessian support.

DLR stores ``H ≈ diag(D) + UUᵀ`` where:
  - ``D ∈ ℝⁿ`` — the exact per-channel second moments (diagonal of H)
  - ``U ∈ ℝⁿˣʳ`` — the top-``r`` correlation directions

These tests verify:
  - **Woodbury identity** for efficient H⁻¹ computation
  - **DLR helpers** (validation, materialisation, smoothquant, permutation, rotation)
  - **GPTQ** with DLR Hessians (INT4 + INT8, asymmetric)
  - **LDLQ** with DLR Hessians
  - **Calibration I/O** (get_hessian, get_hessian_diag with DLR dicts)
  - **Quality** vs full Hessian and block-diagonal
"""

from __future__ import annotations

import math

import torch
import pytest

from converter.dlr import (
    is_dlr,
    validate_dlr,
    woodbury_inverse,
    dlr_to_dense,
    damped_diag_mean,
    transform_dlr_for_smoothquant,
    permute_dlr,
    rotate_dlr_to_dense,
    make_dlr_dict,
)
from converter.gptq import gptq_quantize_layer
from converter.ldlq import ldlq_quantize_layer
from converter.calibration_io import get_hessian, get_hessian_diag
from converter.rotation import rotate_hessian
from converter.config import DIAG_MEAN_FLOOR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dlr_from_hessian(H: torch.Tensor, rank: int) -> dict:
    """Decompose a dense Hessian into a DLR dict via eigendecomposition.

    ``D = diag(H) − diag(UUᵀ)`` (residual diagonal), ``U = top-r eigenvectors × √λ``.
    This matches the convention produced by ComfyUI-GPTQ-Calibration's
    Frequent Directions sketch, where ``diag(D + UUᵀ) = diag(H)`` exactly.
    """
    n = H.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(H)
    r = min(rank, n)
    top_vals = eigvals[-r:].clamp(min=0)
    top_vecs = eigvecs[:, -r:]
    U = top_vecs * top_vals.sqrt().unsqueeze(0)
    D = (H.diagonal() - (U ** 2).sum(dim=1)).clamp(min=0)
    return make_dlr_dict(D, U)


def _make_dlr_hessian(n: int, rank: int, num_samples: int = 128,
                       seed: int = 42) -> tuple[torch.Tensor, dict, torch.Tensor]:
    """Create a true Hessian, its DLR approximation, and weights.

    Returns ``(H_true, dlr_dict, W)``.
    """
    torch.manual_seed(seed)
    X = torch.randn(num_samples, n)
    H_true = X.T @ X
    dlr_dict = _make_dlr_from_hessian(H_true, rank)
    W = torch.randn(16, n)
    return H_true, dlr_dict, W


# ---------------------------------------------------------------------------
# is_dlr / validate_dlr
# ---------------------------------------------------------------------------


class TestDLRDetection:
    """Tests for DLR format detection and validation."""

    def test_is_dlr_true_for_valid_dict(self):
        d = make_dlr_dict(torch.rand(8), torch.randn(8, 4))
        assert is_dlr(d)

    def test_is_dlr_false_for_tensor(self):
        assert not is_dlr(torch.randn(8, 8))
        assert not is_dlr(torch.randn(2, 8, 8))

    def test_is_dlr_false_for_list(self):
        assert not is_dlr([torch.randn(8, 8)])

    def test_is_dlr_false_for_plain_dict(self):
        assert not is_dlr({"foo": 1})
        assert not is_dlr({"D": torch.rand(8), "U": torch.randn(8, 4)})  # no "format"

    def test_is_dlr_false_for_wrong_format(self):
        d = {"format": "block", "D": torch.rand(8), "U": torch.randn(8, 4)}
        assert not is_dlr(d)

    def test_validate_dlr_valid(self):
        d = make_dlr_dict(torch.rand(64) + 0.1, torch.randn(64, 8) * 0.3)
        assert validate_dlr(d)

    def test_validate_dlr_with_expected_features(self):
        d = make_dlr_dict(torch.rand(64) + 0.1, torch.randn(64, 8) * 0.3)
        assert validate_dlr(d, in_features=64)
        assert not validate_dlr(d, in_features=128)

    def test_validate_dlr_rejects_nan(self):
        D = torch.rand(8) + 0.1
        D[0] = float("nan")
        d = make_dlr_dict(D, torch.randn(8, 4))
        assert not validate_dlr(d)

    def test_validate_dlr_rejects_negative_diagonal(self):
        D = torch.rand(8) + 0.1
        D[0] = -1.0
        d = make_dlr_dict(D, torch.randn(8, 4))
        assert not validate_dlr(d)

    def test_validate_dlr_rejects_shape_mismatch(self):
        d = make_dlr_dict(torch.rand(8), torch.randn(16, 4))  # D(8) vs U(16)
        assert not validate_dlr(d)

    def test_validate_dlr_rejects_non_tensor(self):
        d = {"format": "dlr", "D": [1, 2, 3], "U": torch.randn(3, 2)}
        assert not validate_dlr(d)


# ---------------------------------------------------------------------------
# Woodbury identity
# ---------------------------------------------------------------------------


class TestWoodburyInverse:
    """Tests for computing H⁻¹ = (diag(D) + UUᵀ)⁻¹ via Woodbury."""

    def test_woodbury_matches_direct_inverse(self):
        torch.manual_seed(42)
        n, rank = 16, 4
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3

        H = torch.diag(D) + U @ U.T
        H_inv_direct = torch.linalg.inv(H)
        H_inv_woodbury = woodbury_inverse(D, U)

        assert torch.allclose(H_inv_woodbury, H_inv_direct, atol=1e-5)

    def test_woodbury_with_damping(self):
        """Woodbury with damping λ should compute (H + λ·mean(diag(H))·I)⁻¹."""
        torch.manual_seed(42)
        n, rank = 16, 4
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3
        damping = 0.01

        H = torch.diag(D) + U @ U.T
        diag_mean = (D + (U * U).sum(dim=1)).mean()
        H_damped = H + damping * diag_mean * torch.eye(n)
        H_inv_direct = torch.linalg.inv(H_damped)
        H_inv_woodbury = woodbury_inverse(D, U, damping=damping)

        assert torch.allclose(H_inv_woodbury, H_inv_direct, atol=1e-5)

    def test_woodbury_identity_reconstruction(self):
        """H @ H⁻¹ should be the identity (to numerical precision)."""
        torch.manual_seed(42)
        n, rank = 16, 4
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3

        H = torch.diag(D) + U @ U.T
        H_inv = woodbury_inverse(D, U)

        product = H @ H_inv
        assert torch.allclose(product, torch.eye(n), atol=1e-4)

    def test_woodbury_purely_diagonal(self):
        """When U=0, Woodbury should reduce to diag(1/D)."""
        torch.manual_seed(42)
        n = 16
        D = torch.rand(n) + 0.1
        U = torch.zeros(n, 1)

        H_inv = woodbury_inverse(D, U)
        expected = torch.diag(1.0 / D)

        assert torch.allclose(H_inv, expected, atol=1e-6)

    def test_woodbury_rank_one(self):
        """Rank-1 case: H = D + uuᵀ. Inverse via Sherman-Morrison."""
        torch.manual_seed(42)
        n = 16
        D = torch.rand(n) + 0.1
        u = torch.randn(n, 1) * 0.3

        H_inv_woodbury = woodbury_inverse(D, u)

        # Sherman-Morrison: (D + uuᵀ)⁻¹ = D⁻¹ - D⁻¹u uᵀD⁻¹ / (1 + uᵀD⁻¹u)
        D_inv = 1.0 / D
        D_inv_u = D_inv * u.squeeze()
        denom = 1 + (u.squeeze() * D_inv_u).sum()
        H_inv_sm = torch.diag(D_inv) - torch.outer(D_inv_u, D_inv_u) / denom

        assert torch.allclose(H_inv_woodbury, H_inv_sm, atol=1e-6)

    def test_woodbury_zero_damping_matches_no_damping(self):
        """damping=0.0 should produce the same result as no damping."""
        torch.manual_seed(42)
        n, rank = 16, 4
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3

        H_inv_0 = woodbury_inverse(D, U, damping=0.0)
        H_inv_none = woodbury_inverse(D, U)

        assert torch.allclose(H_inv_0, H_inv_none)

    def test_woodbury_handles_zero_diagonal_entries(self):
        """Zero diagonal entries (padding columns) should not cause NaN."""
        torch.manual_seed(42)
        n, rank = 16, 4
        D = torch.rand(n) + 0.1
        D[0] = 0.0  # padding column
        U = torch.randn(n, rank) * 0.3

        H_inv = woodbury_inverse(D, U, damping=0.01)
        assert torch.all(torch.isfinite(H_inv))

    def test_woodbury_output_shape(self):
        n, rank = 32, 8
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3
        H_inv = woodbury_inverse(D, U)
        assert H_inv.shape == (n, n)

    def test_woodbury_symmetric(self):
        """The inverse of a symmetric matrix should be symmetric."""
        torch.manual_seed(42)
        n, rank = 16, 4
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3
        H_inv = woodbury_inverse(D, U)
        assert torch.allclose(H_inv, H_inv.T, atol=1e-6)


# ---------------------------------------------------------------------------
# DLR helpers: damped_diag_mean, dlr_to_dense
# ---------------------------------------------------------------------------


class TestDLRHelpers:
    """Tests for DLR utility functions."""

    def test_damped_diag_mean_matches_full_hessian(self):
        torch.manual_seed(42)
        n, rank = 16, 4
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3
        H = torch.diag(D) + U @ U.T

        assert math.isclose(damped_diag_mean(D, U).item(), H.diagonal().mean().item(),
                             rel_tol=1e-5)

    def test_damped_diag_mean_floored(self):
        """All-zero DLR should produce DIAG_MEAN_FLOOR."""
        D = torch.zeros(8)
        U = torch.zeros(8, 4)
        assert math.isclose(damped_diag_mean(D, U).item(), DIAG_MEAN_FLOOR, rel_tol=1e-5)

    def test_dlr_to_dense_matches_reconstruction(self):
        torch.manual_seed(42)
        n, rank = 16, 4
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3

        H = dlr_to_dense(D, U)
        assert torch.allclose(H, torch.diag(D) + U @ U.T, atol=1e-6)

    def test_dlr_to_dense_shape(self):
        D = torch.rand(32)
        U = torch.randn(32, 8)
        H = dlr_to_dense(D, U)
        assert H.shape == (32, 32)

    def test_make_dlr_dict_structure(self):
        D = torch.rand(16)
        U = torch.randn(16, 4)
        d = make_dlr_dict(D, U)
        assert d["format"] == "dlr"
        assert torch.equal(d["D"], D)
        assert torch.equal(d["U"], U)
        assert d["rank"] == 4
        assert d["n"] == 16


# ---------------------------------------------------------------------------
# SmoothQuant transform
# ---------------------------------------------------------------------------


class TestDLRSmoothQuant:
    """Tests for SmoothQuant transform on DLR factors."""

    def test_transform_preserves_dlr_structure(self):
        """D_new/s² and U_new/s should reconstruct the transformed Hessian."""
        torch.manual_seed(42)
        n, rank = 16, 4
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3
        s = torch.rand(n) + 0.5

        D_new, U_new = transform_dlr_for_smoothquant(D, U, s)

        # Reconstruct transformed Hessian both ways
        H_orig = torch.diag(D) + U @ U.T
        H_expected = H_orig / (s.unsqueeze(0) * s.unsqueeze(1))
        H_dlr_new = torch.diag(D_new) + U_new @ U_new.T

        assert torch.allclose(H_dlr_new, H_expected, atol=1e-5)

    def test_transform_diagonal(self):
        """D_new[i] should equal D[i] / s[i]²."""
        torch.manual_seed(42)
        n = 16
        D = torch.rand(n) + 0.1
        U = torch.randn(n, 4) * 0.3
        s = torch.rand(n) + 0.5

        D_new, _ = transform_dlr_for_smoothquant(D, U, s)
        expected = D / s ** 2
        assert torch.allclose(D_new, expected, atol=1e-6)

    def test_transform_handles_zero_smoothing(self):
        """s=1 should be identity."""
        torch.manual_seed(42)
        n = 16
        D = torch.rand(n) + 0.1
        U = torch.randn(n, 4) * 0.3
        s = torch.ones(n)

        D_new, U_new = transform_dlr_for_smoothquant(D, U, s)
        assert torch.allclose(D_new, D, atol=1e-6)
        assert torch.allclose(U_new, U, atol=1e-6)

    def test_transform_clamps_zero_s(self):
        """s=0 should not produce NaN (clamped to small epsilon)."""
        torch.manual_seed(42)
        n = 16
        D = torch.rand(n) + 0.1
        U = torch.randn(n, 4) * 0.3
        s = torch.ones(n)
        s[0] = 0.0

        D_new, U_new = transform_dlr_for_smoothquant(D, U, s)
        assert torch.all(torch.isfinite(D_new))
        assert torch.all(torch.isfinite(U_new))


# ---------------------------------------------------------------------------
# Permutation
# ---------------------------------------------------------------------------


class TestDLRPermutation:
    """Tests for permuting DLR factors."""

    def test_permute_matches_dense_permutation(self):
        torch.manual_seed(42)
        n, rank = 16, 4
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3
        perm = torch.randperm(n)

        D_new, U_new = permute_dlr(D, U, perm)

        # Compare with dense permutation
        H = torch.diag(D) + U @ U.T
        H_perm = H[perm][:, perm]
        H_dlr_perm = torch.diag(D_new) + U_new @ U_new.T

        assert torch.allclose(H_dlr_perm, H_perm, atol=1e-5)

    def test_permute_preserves_values(self):
        """Permutation should reorder, not change values."""
        torch.manual_seed(42)
        n = 16
        D = torch.rand(n) + 0.1
        U = torch.randn(n, 4) * 0.3
        perm = torch.randperm(n)

        D_new, U_new = permute_dlr(D, U, perm)
        assert torch.equal(D_new, D[perm])
        assert torch.equal(U_new, U[perm])

    def test_permute_identity(self):
        """Identity permutation should be a no-op."""
        torch.manual_seed(42)
        n = 16
        D = torch.rand(n) + 0.1
        U = torch.randn(n, 4) * 0.3
        perm = torch.arange(n)

        D_new, U_new = permute_dlr(D, U, perm)
        assert torch.allclose(D_new, D)
        assert torch.allclose(U_new, U)

    def test_permute_partial(self):
        """Permutation on a subset of channels should work."""
        torch.manual_seed(42)
        n = 16
        D = torch.rand(n) + 0.1
        U = torch.randn(n, 4) * 0.3
        perm = torch.randperm(n)[:8]

        D_new, U_new = permute_dlr(D, U, perm)
        assert D_new.shape == (8,)
        assert U_new.shape == (8, 4)
        assert torch.equal(D_new, D[perm])


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


class TestDLRRotation:
    """Tests for rotating DLR Hessians."""

    def test_rotate_dlr_matches_dense_rotation(self):
        """rotate_dlr_to_dense should match rotate_hessian on the dense Hessian."""
        torch.manual_seed(42)
        n, rank = 64, 8
        rot_size = 64
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3

        H_dense = dlr_to_dense(D, U)
        H_rot_dense = rotate_hessian(H_dense, rot_size)
        H_rot_dlr = rotate_dlr_to_dense(D, U, rot_size)

        assert torch.allclose(H_rot_dlr, H_rot_dense, atol=1e-5)

    def test_rotate_hessian_accepts_dlr_dict(self):
        """rotate_hessian should accept a DLR dict and return a 2-D tensor."""
        torch.manual_seed(42)
        n, rank = 64, 8
        rot_size = 64
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3
        dlr = make_dlr_dict(D, U)

        H_rot = rotate_hessian(dlr, rot_size)
        assert isinstance(H_rot, torch.Tensor)
        assert H_rot.dim() == 2
        assert H_rot.shape[0] >= n

    def test_rotate_dlr_output_is_finite(self):
        torch.manual_seed(42)
        n, rank = 64, 8
        rot_size = 64
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3

        H_rot = rotate_dlr_to_dense(D, U, rot_size)
        assert torch.all(torch.isfinite(H_rot))


# ---------------------------------------------------------------------------
# GPTQ with DLR Hessian
# ---------------------------------------------------------------------------


class TestGPTQWithDLR:
    """Tests for GPTQ quantization with DLR Hessians."""

    def test_int4_output_range(self):
        H_true, dlr, W = _make_dlr_hessian(64, 8)
        qr = gptq_quantize_layer(W, dlr, int_bits=4)
        assert qr.quantized_W.min() >= -8
        assert qr.quantized_W.max() <= 7

    def test_int4_output_dtype(self):
        H_true, dlr, W = _make_dlr_hessian(32, 4)
        qr = gptq_quantize_layer(W, dlr, int_bits=4)
        assert qr.quantized_W.dtype == torch.int8

    def test_int4_scales_shape(self):
        H_true, dlr, W = _make_dlr_hessian(64, 8)
        qr = gptq_quantize_layer(W, dlr, int_bits=4)
        assert qr.scales.shape == (16, 1)

    def test_int4_scales_positive(self):
        H_true, dlr, W = _make_dlr_hessian(64, 8)
        qr = gptq_quantize_layer(W, dlr, int_bits=4)
        assert torch.all(qr.scales > 0)

    def test_int4_no_nan_inf(self):
        H_true, dlr, W = _make_dlr_hessian(64, 8)
        qr = gptq_quantize_layer(W, dlr, int_bits=4)
        assert not torch.any(torch.isnan(qr.quantized_W.float()))
        assert not torch.any(torch.isinf(qr.quantized_W.float()))
        assert not torch.any(torch.isnan(qr.scales))
        assert not torch.any(torch.isinf(qr.scales))

    def test_int4_has_zero_points(self):
        """INT4 always uses asymmetric → zero_points should be present."""
        H_true, dlr, W = _make_dlr_hessian(64, 8)
        qr = gptq_quantize_layer(W, dlr, int_bits=4)
        assert qr.zero_points is not None

    def test_int8_output_range(self):
        H_true, dlr, W = _make_dlr_hessian(64, 8)
        qr = gptq_quantize_layer(W, dlr, int_bits=8)
        assert qr.quantized_W.min() >= -128
        assert qr.quantized_W.max() <= 127

    def test_int8_no_zero_points_symmetric(self):
        H_true, dlr, W = _make_dlr_hessian(64, 8)
        qr = gptq_quantize_layer(W, dlr, int_bits=8, asymmetric=False)
        assert qr.zero_points is None

    def test_int8_asymmetric_has_zero_points(self):
        H_true, dlr, W = _make_dlr_hessian(64, 8)
        qr = gptq_quantize_layer(W, dlr, int_bits=8, asymmetric=True)
        assert qr.zero_points is not None

    def test_int8_no_nan_inf(self):
        H_true, dlr, W = _make_dlr_hessian(64, 8)
        qr = gptq_quantize_layer(W, dlr, int_bits=8)
        assert not torch.any(torch.isnan(qr.quantized_W.float()))
        assert not torch.any(torch.isinf(qr.quantized_W.float()))

    def test_dlr_gptq_reduces_error_vs_rtn(self):
        """GPTQ with DLR should produce lower error than RTN."""
        torch.manual_seed(42)
        n = 64
        X = torch.randn(128, n)
        H_true = X.T @ X
        dlr = _make_dlr_from_hessian(H_true, rank=8)
        W = torch.randn(16, n)

        qr_dlr = gptq_quantize_layer(W, dlr, int_bits=8)

        # RTN
        from converter.scales import calculate_scales_int8, quantize_weights_int8
        scales_rtn = calculate_scales_int8(W)
        q_rtn = quantize_weights_int8(W, scales_rtn)

        W_deq_gptq = qr_dlr.quantized_W.float() * qr_dlr.scales.float()
        W_deq_rtn = q_rtn.float() * scales_rtn.float()

        err_gptq = (X @ (W - W_deq_gptq).T).pow(2).sum().item()
        err_rtn = (X @ (W - W_deq_rtn).T).pow(2).sum().item()
        assert err_gptq <= err_rtn * 1.1, (
            f"GPTQ-DLR error {err_gptq:.2f} > RTN error {err_rtn:.2f}"
        )

    def test_dlr_vs_full_hessian_quality(self):
        """DLR GPTQ should be comparable to full-Hessian GPTQ."""
        torch.manual_seed(42)
        n = 64
        X = torch.randn(256, n)
        H_true = X.T @ X
        dlr = _make_dlr_from_hessian(H_true, rank=16)
        W = torch.randn(16, n)

        qr_full = gptq_quantize_layer(W, H_true, int_bits=8)
        qr_dlr = gptq_quantize_layer(W, dlr, int_bits=8)

        # DLR should be within 2× of full Hessian quality
        assert qr_dlr.mse <= qr_full.mse * 2.0 + 1e-8, (
            f"DLR MSE {qr_dlr.mse:.6f} >> full Hessian MSE {qr_full.mse:.6f}"
        )

    def test_dlr_vs_block_diagonal_cross_block(self):
        """DLR should outperform block-diagonal for cross-block correlations."""
        torch.manual_seed(42)
        n = 128
        rank = 16
        block_size = 32

        # Create activations with strong cross-block correlation
        X = torch.randn(500, n)
        X[:, 0] += X[:, 127] * 0.8
        H_true = X.T @ X

        D = H_true.diagonal().clone()
        eigvals, eigvecs = torch.linalg.eigh(H_true)
        U = eigvecs[:, -rank:] * eigvals[-rank:].clamp(min=0).sqrt().unsqueeze(0)
        dlr = make_dlr_dict(D, U)

        # Block-diagonal Hessian
        num_blocks = n // block_size
        blocks = []
        for i in range(num_blocks):
            s = i * block_size
            e = s + block_size
            blocks.append(H_true[s:e, s:e])
        H_block = torch.stack(blocks)

        W = torch.randn(8, n)
        qr_dlr = gptq_quantize_layer(W, dlr, int_bits=8, damping=0.01)
        qr_block = gptq_quantize_layer(W, H_block, int_bits=8, damping=0.01)

        # Compare output error in H-norm
        def _h_norm_error(W_orig, qr):
            if qr.zero_points is not None:
                W_deq = (qr.quantized_W.float() - qr.zero_points.float()) * qr.scales.float()
            else:
                W_deq = qr.quantized_W.float() * qr.scales.float()
            diff = W_orig - W_deq
            return (diff @ H_true @ diff.T).trace().item()

        err_dlr = _h_norm_error(W, qr_dlr)
        err_block = _h_norm_error(W, qr_block)

        # DLR should be at least as good (captures cross-block correlations)
        assert err_dlr <= err_block * 1.5, (
            f"DLR error ({err_dlr:.4f}) should not be much worse than "
            f"block-diagonal ({err_block:.4f})"
        )

    def test_damping_prevents_singular(self):
        """Damping should prevent NaN from near-singular DLR Hessian."""
        n = 32
        D = torch.zeros(n)
        D[0] = 1.0
        U = torch.zeros(n, 4)
        dlr = make_dlr_dict(D, U)
        W = torch.randn(8, n)

        qr = gptq_quantize_layer(W, dlr, int_bits=4, damping=0.1)
        assert not torch.any(torch.isnan(qr.quantized_W.float()))

    def test_block_size_variants(self):
        H_true, dlr, W = _make_dlr_hessian(128, 8)
        for bs in [32, 64, 128]:
            qr = gptq_quantize_layer(W, dlr, block_size=bs, int_bits=4)
            assert qr.quantized_W.shape == (16, 128)

    def test_dlr_dim_mismatch_raises(self):
        dlr = make_dlr_dict(torch.rand(32), torch.randn(32, 4))
        W = torch.randn(8, 64)  # in_features mismatch
        with pytest.raises(ValueError, match="dim.*in_features"):
            gptq_quantize_layer(W, dlr, int_bits=4)

    def test_zero_weights(self):
        H_true, dlr, W = _make_dlr_hessian(32, 4)
        W_zero = torch.zeros(8, 32)
        qr = gptq_quantize_layer(W_zero, dlr, int_bits=4)
        assert torch.all(qr.quantized_W == 0)


# ---------------------------------------------------------------------------
# LDLQ with DLR Hessian
# ---------------------------------------------------------------------------


class TestLDLQWithDLR:
    """Tests for LDLQ quantization with DLR Hessians."""

    def test_int4_output_range(self):
        H_true, dlr, W = _make_dlr_hessian(64, 8)
        qr = ldlq_quantize_layer(W, dlr, int_bits=4)
        assert qr.quantized_W.min() >= -8
        assert qr.quantized_W.max() <= 7

    def test_int8_output_range(self):
        H_true, dlr, W = _make_dlr_hessian(64, 8)
        qr = ldlq_quantize_layer(W, dlr, int_bits=8)
        assert qr.quantized_W.min() >= -128
        assert qr.quantized_W.max() <= 127

    def test_output_dtype(self):
        H_true, dlr, W = _make_dlr_hessian(32, 4)
        qr = ldlq_quantize_layer(W, dlr, int_bits=4)
        assert qr.quantized_W.dtype == torch.int8

    def test_no_nan_inf(self):
        H_true, dlr, W = _make_dlr_hessian(64, 8)
        qr = ldlq_quantize_layer(W, dlr, int_bits=4)
        assert not torch.any(torch.isnan(qr.quantized_W.float()))
        assert not torch.any(torch.isinf(qr.quantized_W.float()))

    def test_scales_positive(self):
        H_true, dlr, W = _make_dlr_hessian(64, 8)
        qr = ldlq_quantize_layer(W, dlr, int_bits=4)
        assert torch.all(qr.scales > 0)

    def test_ldlq_dlr_reduces_error_vs_weight_only(self):
        """LDLQ with DLR Hessian should beat weight-only LDLQ."""
        torch.manual_seed(42)
        n = 64
        X = torch.randn(128, n)
        H_true = X.T @ X
        dlr = _make_dlr_from_hessian(H_true, rank=8)
        W = torch.randn(16, n)

        qr_dlr = ldlq_quantize_layer(W, dlr, int_bits=8, damping=0.01)
        qr_wo = ldlq_quantize_layer(W, hessian=None, int_bits=8, damping=0.01)

        W_deq_dlr = qr_dlr.quantized_W.float() * qr_dlr.scales.float()
        W_deq_wo = qr_wo.quantized_W.float() * qr_wo.scales.float()

        err_dlr = (X @ (W - W_deq_dlr).T).pow(2).sum().item()
        err_wo = (X @ (W - W_deq_wo).T).pow(2).sum().item()
        # DLR should generally be better (or at least not much worse)
        assert err_dlr <= err_wo * 1.5, (
            f"LDLQ-DLR error {err_dlr:.2f} >> weight-only error {err_wo:.2f}"
        )

    def test_ldlq_dlr_dim_mismatch_raises(self):
        dlr = make_dlr_dict(torch.rand(32), torch.randn(32, 4))
        W = torch.randn(8, 64)
        with pytest.raises(ValueError, match="dim.*in_features"):
            ldlq_quantize_layer(W, dlr, int_bits=4)


# ---------------------------------------------------------------------------
# Calibration I/O with DLR
# ---------------------------------------------------------------------------


class TestCalibrationIODLR:
    """Tests for loading DLR Hessians from calibration data."""

    def _make_dlr_calibration(self, in_features: int, rank: int) -> dict:
        """Create a mock calibration dict with DLR Hessians."""
        torch.manual_seed(42)
        X = torch.randn(32, in_features)
        H = X.T @ X
        dlr = _make_dlr_from_hessian(H, rank)
        return {
            "hessians": {"layer.0": dlr},
            "shapes": {"layer.0": (64, in_features)},
            "layer_types": {"layer.0": "Linear"},
            "metadata": {
                "hessian_format": "dlr",
                "dlr_rank": rank,
            },
        }

    def test_get_hessian_returns_dlr_dict(self):
        cal = self._make_dlr_calibration(64, 8)
        H = get_hessian(cal, "layer.0", torch.Size([32, 64]))
        assert H is not None
        assert is_dlr(H)
        assert H["D"].shape == (64,)
        assert H["U"].shape == (64, 8)

    def test_get_hessian_dlr_not_found(self):
        cal = self._make_dlr_calibration(64, 8)
        assert get_hessian(cal, "missing", torch.Size([32, 64])) is None

    def test_get_hessian_dlr_dim_mismatch_returns_none(self):
        cal = self._make_dlr_calibration(64, 8)
        # weight_shape says 128 in_features but DLR is 64
        result = get_hessian(cal, "layer.0", torch.Size([32, 128]))
        assert result is None  # validation fails on dim mismatch

    def test_get_hessian_dlr_nan_returns_none(self):
        cal = self._make_dlr_calibration(64, 8)
        cal["hessians"]["layer.0"]["D"][0] = float("nan")
        result = get_hessian(cal, "layer.0", torch.Size([32, 64]))
        assert result is None

    def test_get_hessian_diag_returns_full_diagonal(self):
        """get_hessian_diag should return the full diagonal D + diag(UUᵀ), not just D."""
        cal = self._make_dlr_calibration(64, 8)
        diag = get_hessian_diag(cal, "layer.0", 64)
        assert diag is not None
        assert diag.shape == (64,)
        # D is the residual diagonal; full diagonal is D + diag(UUᵀ)
        H_raw = cal["hessians"]["layer.0"]
        expected = H_raw["D"].float() + (H_raw["U"].float() ** 2).sum(dim=1)
        assert torch.allclose(diag, expected, atol=1e-5)

    def test_get_hessian_diag_truncates_to_in_features(self):
        cal = self._make_dlr_calibration(64, 8)
        diag = get_hessian_diag(cal, "layer.0", 32)
        assert diag is not None
        assert diag.shape == (32,)

    def test_get_hessian_diag_missing_layer(self):
        cal = self._make_dlr_calibration(64, 8)
        assert get_hessian_diag(cal, "missing", 64) is None

    def test_dlr_calibration_roundtrip(self, tmp_path):
        """DLR calibration data should survive a save/load roundtrip."""
        import torch
        cal = self._make_dlr_calibration(64, 8)
        path = tmp_path / "dlr_cal.pt"
        torch.save(cal, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        H = get_hessian(loaded, "layer.0", torch.Size([32, 64]))
        assert H is not None
        assert is_dlr(H)
        assert torch.equal(H["D"], cal["hessians"]["layer.0"]["D"])
        assert torch.equal(H["U"], cal["hessians"]["layer.0"]["U"])

    def test_dlr_gptq_end_to_end_from_calibration(self):
        """Full flow: calibration dict → get_hessian → gptq_quantize_layer."""
        cal = self._make_dlr_calibration(64, 8)
        W = torch.randn(16, 64)
        H = get_hessian(cal, "layer.0", torch.Size([16, 64]))
        assert H is not None
        qr = gptq_quantize_layer(W, H, int_bits=4)
        assert qr.quantized_W.shape == (16, 64)
        assert qr.quantized_W.min() >= -8
        assert qr.quantized_W.max() <= 7
        assert not torch.any(torch.isnan(qr.quantized_W.float()))


# ---------------------------------------------------------------------------
# Mixed format: DLR coexists with other Hessian formats
# ---------------------------------------------------------------------------


class TestMixedHessianFormats:
    """Tests that DLR and tensor Hessians coexist in the same calibration."""

    def test_mixed_calibration(self):
        """A calibration dict can have both DLR and full Hessians."""
        torch.manual_seed(42)
        n = 64

        # DLR layer
        X1 = torch.randn(32, n)
        H1 = X1.T @ X1
        dlr = _make_dlr_from_hessian(H1, rank=8)

        # Full Hessian layer
        X2 = torch.randn(32, n)
        H2 = X2.T @ X2

        cal = {
            "hessians": {"dlr_layer": dlr, "full_layer": H2},
            "shapes": {"dlr_layer": (16, n), "full_layer": (16, n)},
            "layer_types": {"dlr_layer": "Linear", "full_layer": "Linear"},
        }

        H_dlr = get_hessian(cal, "dlr_layer", torch.Size([16, n]))
        H_full = get_hessian(cal, "full_layer", torch.Size([16, n]))
        assert is_dlr(H_dlr)
        assert isinstance(H_full, torch.Tensor)
        assert H_full.dim() == 2

        W = torch.randn(16, n)
        qr_dlr = gptq_quantize_layer(W, H_dlr, int_bits=4)
        qr_full = gptq_quantize_layer(W, H_full, int_bits=4)
        assert qr_dlr.quantized_W.shape == (16, n)
        assert qr_full.quantized_W.shape == (16, n)
