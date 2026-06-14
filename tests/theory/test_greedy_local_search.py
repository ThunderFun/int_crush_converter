"""Test: Greedy local search reduces proxy loss."""

import torch

from converter.ldlq import _single_ldlq_pass
from converter.rotation import rotate_weights
from converter.rounding import _invert_hessian


def _compute_proxy_loss(W_orig, W_dequant, H):
    err = W_dequant - W_orig
    return torch.trace(err @ H @ err.T).item()


class TestGreedyLocalSearch:

    def _greedy_single_pass(self, W, Q, scales, H):
        M, N = W.shape
        Q = Q.clone().float()
        scale_2d = scales.reshape(M, N)
        grid = torch.arange(-8, 8, dtype=torch.float32)

        for col in range(N):
            for row in range(M):
                s = scale_2d[row, col]
                q_old = Q[row, col]
                err_row = (Q[row, :] * scale_2d[row, :]) - W[row, :]
                h_col = H[col, :]
                cross = err_row @ h_col - err_row[col] * H[col, col]

                best_q = q_old
                best_loss = float("inf")
                for q_new in grid:
                    new_err_col = err_row[col] + (q_new - q_old) * s
                    new_loss = new_err_col ** 2 * H[col, col] + 2 * new_err_col * cross
                    if new_loss < best_loss:
                        best_loss = new_loss
                        best_q = q_new
                Q[row, col] = best_q
        return Q.to(torch.int8)

    def test_greedy_reduces_proxy_loss(self):
        torch.manual_seed(42)
        M, N = 8, 32
        W = torch.randn(M, N)
        H = W.T @ W / M
        diag_mean = H.diagonal().mean().clamp(min=1e-6)
        H_inv = _invert_hessian(H + 0.01 * diag_mean * torch.eye(N))

        row_scales = (W.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-6)
        flat_scales = row_scales.expand(M, N).reshape(-1).clone()

        Q_ldlq = _single_ldlq_pass(W, H_inv, flat_scales, block_size=32,
                                    clamp_min=-8, clamp_max=7)
        scale_2d = flat_scales.reshape(M, N)
        proxy_ldlq = _compute_proxy_loss(W, Q_ldlq.float() * scale_2d, H)

        Q_greedy = self._greedy_single_pass(W, Q_ldlq, scale_2d, H)
        proxy_greedy = _compute_proxy_loss(W, Q_greedy.float() * scale_2d, H)

        assert proxy_greedy <= proxy_ldlq * 1.01, (
            f"Greedy did not reduce proxy loss: ldlq={proxy_ldlq:.6f}, greedy={proxy_greedy:.6f}"
        )

    def test_greedy_with_convrot(self):
        torch.manual_seed(42)
        M, N = 8, 64
        rot_size = 64

        W = torch.randn(M, N)
        W_rot = rotate_weights(W, rot_size=rot_size)
        H = W_rot.T @ W_rot / M
        diag_mean = H.diagonal().mean().clamp(min=1e-6)
        H_inv = _invert_hessian(H + 0.01 * diag_mean * torch.eye(N))

        row_scales = (W_rot.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-6)
        flat_scales = row_scales.expand(M, N).reshape(-1).clone()

        Q_ldlq = _single_ldlq_pass(W_rot, H_inv, flat_scales, block_size=32,
                                    clamp_min=-8, clamp_max=7)
        scale_2d = flat_scales.reshape(M, N)
        proxy_ldlq = _compute_proxy_loss(W_rot, Q_ldlq.float() * scale_2d, H)

        Q_greedy = self._greedy_single_pass(W_rot, Q_ldlq, scale_2d, H)
        proxy_greedy = _compute_proxy_loss(W_rot, Q_greedy.float() * scale_2d, H)

        assert proxy_greedy <= proxy_ldlq * 1.01
