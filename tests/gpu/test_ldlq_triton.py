"""Tests for the Triton LDLQ block kernel."""

import pytest
import torch

from converter.ldlq_triton import ldlq_loop_triton, _HAS_TRITON


class TestTritonKernel:

    def test_triton_import(self):
        assert callable(ldlq_loop_triton)

    def test_triton_has_triton_flag(self):
        assert isinstance(_HAS_TRITON, bool)

    def test_triton_returns_none_without_cuda(self):
        if not _HAS_TRITON:
            M, N = 8, 32
            W = torch.randn(M, N)
            H_inv = torch.eye(N)
            scale = torch.ones(M, N)
            Q = torch.zeros(M, N, dtype=torch.int8)
            assert ldlq_loop_triton(W, H_inv, scale, Q, N, M, 32, -8, 7) is None

    @pytest.mark.gpu
    def test_triton_int4_output_range(self):
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

    @pytest.mark.gpu
    def test_triton_int8_output_range(self):
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

    @pytest.mark.gpu
    def test_triton_matches_cpu_loop(self):
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")
        from converter.ldlq import _ldlq_loop_cpu

        torch.manual_seed(42)
        M, N = 8, 32
        block_size = 16

        W = torch.randn(M, N)
        H = W.T @ W / M + 0.01 * torch.eye(N)
        H_inv = torch.linalg.inv(H)
        scale_2d = (W.abs() / 7.0).clamp(min=1e-6)

        Q_cpu = torch.zeros(M, N, dtype=torch.int8)
        W_cpu = W.clone()
        _ldlq_loop_cpu(W_cpu, H_inv, scale_2d, Q_cpu, N, M, block_size, -8, 7)

        W_triton = W.clone().cuda()
        Q_triton = torch.zeros(M, N, dtype=torch.int8, device="cuda")
        ldlq_loop_triton(W_triton, H_inv.cuda(), scale_2d.cuda(), Q_triton, N, M, block_size, -8, 7)

        assert torch.equal(Q_triton.cpu(), Q_cpu)
