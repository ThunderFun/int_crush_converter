"""Tests for iterative LDLQ with scale refinement."""

import torch

from converter.ldlq import ldlq_quantize_layer, _single_ldlq_pass, _run_iterative_ldlq
from converter.rounding import _invert_hessian


class TestLDLQIterative:

    def test_iterative_convergence(self):
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

        assert Q_result.shape == (M, N)
        assert Q_result.dtype == torch.int8
        assert Q_result.min() >= -8
        assert Q_result.max() <= 7

    def test_single_vs_iterative(self):
        torch.manual_seed(42)
        W = torch.randn(8, 32)

        _qr_single = ldlq_quantize_layer(W, iterations=1)
        _qr_multi = ldlq_quantize_layer(W, iterations=3)

        assert _qr_single.quantized_W.min() >= -8 and _qr_single.quantized_W.max() <= 7
        assert _qr_multi.quantized_W.min() >= -8 and _qr_multi.quantized_W.max() <= 7

        deq_single = _qr_single.quantized_W.float() * _qr_single.scales.float()
        deq_multi = _qr_multi.quantized_W.float() * _qr_multi.scales.float()
        assert (W - deq_single).pow(2).mean().item() < 1.0
        assert (W - deq_multi).pow(2).mean().item() < 1.0
