"""Test: Iterative LDLQ reduces MSE across iterations."""

import torch

from converter.ldlq import _single_ldlq_pass, _run_iterative_ldlq, _update_ldlq_scales
from converter.rounding import _invert_hessian


class TestIterativeLDLQConvergence:

    def test_mse_decreases_across_iterations(self):
        torch.manual_seed(42)
        M, N = 16, 128
        W = torch.randn(M, N)
        H = W.T @ W / M
        diag_mean = H.diagonal().mean().clamp(min=1e-6)
        H_inv = _invert_hessian(H + 0.01 * diag_mean * torch.eye(N))

        row_scales = (W.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-6)
        flat_scales = row_scales.expand(M, N).reshape(-1).clone()

        current_scales = flat_scales.clone()
        mse_values = []
        for i in range(3):
            Q = _single_ldlq_pass(W, H_inv, current_scales, block_size=64,
                                  clamp_min=-8, clamp_max=7)
            scale_2d = current_scales.reshape(M, N)
            W_deq = Q.float() * scale_2d
            mse_values.append((W - W_deq).pow(2).mean().item())

            if i < 2:
                current_scales = _update_ldlq_scales(
                    W, Q, current_scales, flat_scales,
                    momentum=0.3, nonzero_threshold=1e-8,
                )

        assert mse_values[1] <= mse_values[0] * 1.01
        assert mse_values[2] <= mse_values[1] * 1.01

    def test_q_values_change_between_iterations(self):
        torch.manual_seed(42)
        M, N = 8, 32
        W = torch.randn(M, N)
        H = W.T @ W / M
        diag_mean = H.diagonal().mean().clamp(min=1e-6)
        H_inv = _invert_hessian(H + 0.01 * diag_mean * torch.eye(N))

        row_scales = (W.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-6)
        flat_scales = row_scales.expand(M, N).reshape(-1).clone()

        Q1 = _single_ldlq_pass(W, H_inv, flat_scales, block_size=32,
                               clamp_min=-8, clamp_max=7)
        new_scales = _update_ldlq_scales(
            W, Q1, flat_scales, flat_scales,
            momentum=0.3, nonzero_threshold=1e-8,
        )
        Q2 = _single_ldlq_pass(W, H_inv, new_scales, block_size=32,
                               clamp_min=-8, clamp_max=7)

        assert (Q1 != Q2).sum().item() > 0, "Q did not change between iterations"
