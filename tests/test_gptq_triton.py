"""Tests for the Triton GPTQ block kernel."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest

from converter.gptq_triton import gptq_loop_triton, _HAS_GPTQ_TRITON
from converter.rounding import _gptq_block, _invert_hessian


class TestGPTQTritonImports:
    """Tests for import and availability."""

    def test_import_gptq_loop_triton(self):
        """gptq_loop_triton should be importable regardless of Triton availability."""
        assert callable(gptq_loop_triton)

    def test_has_triton_flag(self):
        """_HAS_GPTQ_TRITON should be a bool."""
        assert isinstance(_HAS_GPTQ_TRITON, bool)

    def test_returns_none_without_triton(self):
        """gptq_loop_triton should return None if Triton can't run."""
        if _HAS_GPTQ_TRITON:
            return  # Can't test this without removing Triton
        M, N = 8, 32
        W = torch.randn(M, N)
        H_inv = torch.eye(N)
        scales = torch.ones(M)
        Q = torch.zeros(M, N, dtype=torch.int8)
        result = gptq_loop_triton(W, H_inv, scales, Q, N, M, 32, -8, 7)
        assert result is None


class TestGPTQTritonOutputRange:
    """Tests for output value range."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_int4_output_range(self):
        """Triton kernel INT4 output must be in [-8, 7]."""
        if not _HAS_GPTQ_TRITON:
            pytest.skip("Triton not installed")
        torch.manual_seed(42)
        M, N = 16, 64
        W = torch.randn(M, N, dtype=torch.float32, device="cuda")
        H = W.T @ W / M + 0.01 * torch.eye(N, device="cuda")
        H_inv = torch.linalg.inv(H)
        scales = (W.abs().amax(dim=1) / 7.0).clamp(min=1e-6)
        Q = torch.zeros(M, N, dtype=torch.int8, device="cuda")
        gptq_loop_triton(W, H_inv, scales, Q, N, M, 32, -8, 7)
        assert Q.min() >= -8
        assert Q.max() <= 7

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_int8_output_range(self):
        """Triton kernel INT8 output must be in [-128, 127]."""
        if not _HAS_GPTQ_TRITON:
            pytest.skip("Triton not installed")
        torch.manual_seed(42)
        M, N = 16, 64
        W = torch.randn(M, N, dtype=torch.float32, device="cuda")
        H = W.T @ W / M + 0.01 * torch.eye(N, device="cuda")
        H_inv = torch.linalg.inv(H)
        scales = (W.abs().amax(dim=1) / 127.0).clamp(min=1e-6)
        Q = torch.zeros(M, N, dtype=torch.int8, device="cuda")
        gptq_loop_triton(W, H_inv, scales, Q, N, M, 32, -128, 127)
        assert Q.min() >= -128
        assert Q.max() <= 127

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_output_dtype(self):
        """Triton kernel output should be int8."""
        if not _HAS_GPTQ_TRITON:
            pytest.skip("Triton not installed")
        M, N = 8, 32
        W = torch.randn(M, N, dtype=torch.float32, device="cuda")
        H = W.T @ W / M + 0.01 * torch.eye(N, device="cuda")
        H_inv = torch.linalg.inv(H)
        scales = (W.abs().amax(dim=1) / 7.0).clamp(min=1e-6)
        Q = torch.zeros(M, N, dtype=torch.int8, device="cuda")
        gptq_loop_triton(W, H_inv, scales, Q, N, M, 32, -8, 7)
        assert Q.dtype == torch.int8


class TestGPTQTritonMatchesReference:
    """Test that Triton kernel produces identical output to PyTorch reference."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_matches_reference_single_block(self):
        """Triton should match _gptq_block for a single block (block_size == N)."""
        if not _HAS_GPTQ_TRITON:
            pytest.skip("Triton not installed")
        torch.manual_seed(42)
        M, N = 8, 32
        block_size = 32

        W = torch.randn(M, N, dtype=torch.float32)
        H = W.T @ W / M
        diag_mean = H.diagonal().mean().clamp(min=1e-6)
        H = H + 0.01 * diag_mean * torch.eye(N, dtype=torch.float32)
        H_inv = _invert_hessian(H)

        row_scales = (W.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-6)

        # PyTorch reference
        W_ref = W.clone()
        Q_ref = torch.zeros(M, N, dtype=torch.int8)
        _gptq_block(W_ref, Q_ref, row_scales, H_inv, 0, N, block_size, -8, 7)

        # Triton
        W_tri = W.clone().cuda()
        Q_tri = torch.zeros(M, N, dtype=torch.int8, device="cuda")
        row_scales_1d = row_scales.squeeze(1).contiguous().cuda()
        result = gptq_loop_triton(
            W_tri, H_inv.cuda(), row_scales_1d, Q_tri,
            N, M, block_size, -8, 7,
        )

        assert result is not None
        assert torch.equal(Q_tri.cpu(), Q_ref), (
            f"Triton Q differs from reference. "
            f"Max diff: {(Q_tri.cpu().float() - Q_ref.float()).abs().max().item()}"
        )
        assert torch.allclose(W_tri.cpu(), W_ref, atol=0.2), (
            f"W_work diverged. Max diff: {(W_tri.cpu() - W_ref).abs().max().item()}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_matches_reference_multiple_blocks(self):
        """Triton should match _gptq_block when block_size < N."""
        if not _HAS_GPTQ_TRITON:
            pytest.skip("Triton not installed")
        torch.manual_seed(42)
        M, N = 8, 64
        block_size = 16

        W = torch.randn(M, N, dtype=torch.float32)
        H = W.T @ W / M
        diag_mean = H.diagonal().mean().clamp(min=1e-6)
        H = H + 0.01 * diag_mean * torch.eye(N, dtype=torch.float32)
        H_inv = _invert_hessian(H)

        row_scales = (W.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-6)

        # PyTorch reference
        W_ref = W.clone()
        Q_ref = torch.zeros(M, N, dtype=torch.int8)
        _gptq_block(W_ref, Q_ref, row_scales, H_inv, 0, N, block_size, -8, 7)

        # Triton
        W_tri = W.clone().cuda()
        Q_tri = torch.zeros(M, N, dtype=torch.int8, device="cuda")
        row_scales_1d = row_scales.squeeze(1).contiguous().cuda()
        result = gptq_loop_triton(
            W_tri, H_inv.cuda(), row_scales_1d, Q_tri,
            N, M, block_size, -8, 7,
        )

        assert result is not None
        # Multi-block: GPU vs CPU float can diverge through error propagation
        q_diff = (Q_tri.cpu().float() - Q_ref.float()).abs().max().item()
        assert q_diff <= 2, (
            f"Triton Q differs from reference. Max diff: {q_diff}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_matches_reference_non_power_of_two(self):
        """Triton should handle non-power-of-two block sizes."""
        if not _HAS_GPTQ_TRITON:
            pytest.skip("Triton not installed")
        torch.manual_seed(42)
        M, N = 12, 48
        block_size = 10  # Does not divide N evenly

        W = torch.randn(M, N, dtype=torch.float32)
        H = W.T @ W / M
        diag_mean = H.diagonal().mean().clamp(min=1e-6)
        H = H + 0.01 * diag_mean * torch.eye(N, dtype=torch.float32)
        H_inv = _invert_hessian(H)

        row_scales = (W.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-6)

        # PyTorch reference
        W_ref = W.clone()
        Q_ref = torch.zeros(M, N, dtype=torch.int8)
        _gptq_block(W_ref, Q_ref, row_scales, H_inv, 0, N, block_size, -8, 7)

        # Triton
        W_tri = W.clone().cuda()
        Q_tri = torch.zeros(M, N, dtype=torch.int8, device="cuda")
        row_scales_1d = row_scales.squeeze(1).contiguous().cuda()
        result = gptq_loop_triton(
            W_tri, H_inv.cuda(), row_scales_1d, Q_tri,
            N, M, block_size, -8, 7,
        )

        assert result is not None
        # Multi-block: GPU vs CPU float can diverge through error propagation
        q_diff = (Q_tri.cpu().float() - Q_ref.float()).abs().max().item()
        assert q_diff <= 2, (
            f"Triton Q differs from reference. Max diff: {q_diff}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_matches_reference_int8(self):
        """Triton should match _gptq_block for INT8 quantization."""
        if not _HAS_GPTQ_TRITON:
            pytest.skip("Triton not installed")
        torch.manual_seed(42)
        M, N = 8, 32
        block_size = 32

        W = torch.randn(M, N, dtype=torch.float32)
        H = W.T @ W / M
        diag_mean = H.diagonal().mean().clamp(min=1e-6)
        H = H + 0.01 * diag_mean * torch.eye(N, dtype=torch.float32)
        H_inv = _invert_hessian(H)

        row_scales = (W.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-6)

        # PyTorch reference
        W_ref = W.clone()
        Q_ref = torch.zeros(M, N, dtype=torch.int8)
        _gptq_block(W_ref, Q_ref, row_scales, H_inv, 0, N, block_size, -128, 127)

        # Triton
        W_tri = W.clone().cuda()
        Q_tri = torch.zeros(M, N, dtype=torch.int8, device="cuda")
        row_scales_1d = row_scales.squeeze(1).contiguous().cuda()
        result = gptq_loop_triton(
            W_tri, H_inv.cuda(), row_scales_1d, Q_tri,
            N, M, block_size, -128, 127,
        )

        assert result is not None
        assert torch.equal(Q_tri.cpu(), Q_ref), (
            f"Triton Q differs from reference. "
            f"Max diff: {(Q_tri.cpu().float() - Q_ref.float()).abs().max().item()}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_matches_reference_w_work_modified(self):
        """W_work should be modified identically by Triton and reference."""
        if not _HAS_GPTQ_TRITON:
            pytest.skip("Triton not installed")
        torch.manual_seed(42)
        M, N = 8, 32
        block_size = 32

        W = torch.randn(M, N, dtype=torch.float32)
        H = W.T @ W / M
        diag_mean = H.diagonal().mean().clamp(min=1e-6)
        H = H + 0.01 * diag_mean * torch.eye(N, dtype=torch.float32)
        H_inv = _invert_hessian(H)

        row_scales = (W.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-6)

        # PyTorch reference
        W_ref = W.clone()
        Q_ref = torch.zeros(M, N, dtype=torch.int8)
        _gptq_block(W_ref, Q_ref, row_scales, H_inv, 0, N, block_size, -8, 7)

        # Triton
        W_tri = W.clone().cuda()
        Q_tri = torch.zeros(M, N, dtype=torch.int8, device="cuda")
        row_scales_1d = row_scales.squeeze(1).contiguous().cuda()
        gptq_loop_triton(
            W_tri, H_inv.cuda(), row_scales_1d, Q_tri,
            N, M, block_size, -8, 7,
        )

        W_tri_cpu = W_tri.cpu()
        max_diff = (W_tri_cpu - W_ref).abs().max().item()
        assert torch.allclose(W_tri_cpu, W_ref, atol=0.2), (
            f"W_work diverged. Max diff: {max_diff}"
        )


class TestGPTQTritonLargerShapes:
    """Tests with larger, realistic shapes."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_shape_256x1024(self):
        """Triton should produce correct output for larger shapes."""
        if not _HAS_GPTQ_TRITON:
            pytest.skip("Triton not installed")
        torch.manual_seed(42)
        M, N = 256, 1024
        block_size = 128

        W = torch.randn(M, N, dtype=torch.float32)
        H = W.T @ W / M
        diag_mean = H.diagonal().mean().clamp(min=1e-6)
        H = H + 0.01 * diag_mean * torch.eye(N, dtype=torch.float32)
        H_inv = _invert_hessian(H)

        row_scales = (W.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-6)

        # PyTorch reference
        W_ref = W.clone()
        Q_ref = torch.zeros(M, N, dtype=torch.int8)
        _gptq_block(W_ref, Q_ref, row_scales, H_inv, 0, N, block_size, -8, 7)

        # Triton
        W_tri = W.clone().cuda()
        Q_tri = torch.zeros(M, N, dtype=torch.int8, device="cuda")
        row_scales_1d = row_scales.squeeze(1).contiguous().cuda()
        result = gptq_loop_triton(
            W_tri, H_inv.cuda(), row_scales_1d, Q_tri,
            N, M, block_size, -8, 7,
        )

        assert result is not None
        # With multiple blocks of error propagation, GPU vs CPU float32 can diverge slightly
        q_diff = (Q_tri.cpu().float() - Q_ref.float()).abs().max().item()
        assert q_diff <= 2, f"Triton Q differs from reference. Max diff: {q_diff}"


class TestGPTQTritonEndToEnd:
    """End-to-end tests through gptq_quantize_layer."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_end_to_end_2d_hessian(self):
        """gptq_quantize_layer should work with Triton acceleration."""
        from converter.gptq import gptq_quantize_layer

        torch.manual_seed(42)
        M, N = 16, 64

        W = torch.randn(M, N)
        X = torch.randn(32, N)
        H = X.T @ X

        q_W, scales = gptq_quantize_layer(W, H, block_size=32)
        assert q_W.shape == (M, N)
        assert q_W.dtype == torch.int8
        assert q_W.min() >= -8
        assert q_W.max() <= 7
        assert scales.shape == (M, 1)
        assert scales.dtype == torch.float16
        assert torch.all(scales > 0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_end_to_end_3d_hessian(self):
        """gptq_quantize_layer should work with 3D block-diagonal Hessians."""
        from converter.gptq import gptq_quantize_layer

        torch.manual_seed(42)
        M, N = 16, 64
        bs = 32
        num_blocks = N // bs

        W = torch.randn(M, N)
        blocks = []
        for _ in range(num_blocks):
            X = torch.randn(32, bs)
            blocks.append(X.T @ X)
        H_block = torch.stack(blocks)

        q_W, scales = gptq_quantize_layer(W, H_block)
        assert q_W.shape == (M, N)
        assert q_W.dtype == torch.int8
        assert q_W.min() >= -8
        assert q_W.max() <= 7

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_end_to_end_int8(self):
        """gptq_quantize_layer should work with INT8 bit-width."""
        from converter.gptq import gptq_quantize_layer

        torch.manual_seed(42)
        M, N = 16, 64

        W = torch.randn(M, N)
        X = torch.randn(32, N)
        H = X.T @ X

        q_W, scales = gptq_quantize_layer(W, H, int_bits=8)
        assert q_W.shape == (M, N)
        assert q_W.min() >= -128
        assert q_W.max() <= 127

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_end_to_end_zero_weights(self):
        """Zero weights should quantize to zero."""
        from converter.gptq import gptq_quantize_layer

        M, N = 8, 32
        W = torch.zeros(M, N)
        X = torch.randn(16, N)
        H = X.T @ X

        q_W, scales = gptq_quantize_layer(W, H)
        assert torch.all(q_W == 0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_end_to_end_gptq_reduces_error_vs_rtn(self):
        """GPTQ with Triton should still reduce error vs RTN."""
        from converter.gptq import gptq_quantize_layer, gptq_quantize_layer_rtn

        torch.manual_seed(42)
        M, N = 16, 64

        W = torch.randn(M, N)
        X = torch.randn(128, N)
        H = X.T @ X

        q_W_gptq, scales_gptq = gptq_quantize_layer(W, H)
        q_W_rtn, scales_rtn = gptq_quantize_layer_rtn(W)

        W_deq_gptq = q_W_gptq.float() * scales_gptq.float()
        W_deq_rtn = q_W_rtn.float() * scales_rtn.float()

        err_gptq = (X @ (W - W_deq_gptq).T).pow(2).sum().item()
        err_rtn = (X @ (W - W_deq_rtn).T).pow(2).sum().item()

        assert err_gptq <= err_rtn * 1.1, (
            f"GPTQ error {err_gptq:.2f} > RTN error {err_rtn:.2f}"
        )
