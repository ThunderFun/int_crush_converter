"""Test: Block-diagonal Hessian equivalence under group-wise rotation."""

import math
import torch

from converter.ldlq import _single_ldlq_pass
from converter.rotation import rotate_weights
from converter.rounding import _invert_hessian


class TestBlockDiagonalHessianEquivalence:

    def test_block_ldlq_matches_full_ldlq_group_wise(self):
        torch.manual_seed(42)
        M, N = 16, 128
        rot_size = 64

        W = torch.randn(M, N)
        W_rot = rotate_weights(W, rot_size=rot_size)
        H_rot = W_rot.T @ W_rot / M

        # Full LDLQ
        diag_mean = H_rot.diagonal().mean().clamp(min=1e-6)
        H_inv_full = _invert_hessian(H_rot + 0.01 * diag_mean * torch.eye(N))
        row_scales = (W_rot.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-6)
        flat_scales = row_scales.expand(M, N).reshape(-1).clone()
        Q_full, _ = _single_ldlq_pass(W_rot, H_inv_full, flat_scales, block_size=16,
                                   clamp_min=-8, clamp_max=7)

        # Block-wise LDLQ
        num_groups = N // rot_size
        Q_blocks = torch.zeros_like(Q_full)
        for g in range(num_groups):
            start = g * rot_size
            end = start + rot_size
            W_g = W_rot[:, start:end]
            H_g = H_rot[start:end, start:end]
            diag_g = H_g.diagonal().mean().clamp(min=1e-6)
            H_g_inv = _invert_hessian(H_g + 0.01 * diag_g * torch.eye(rot_size))
            scales_g = flat_scales.reshape(M, N)[:, start:end].reshape(-1).clone()
            Q_g, _ = _single_ldlq_pass(W_g, H_g_inv, scales_g, block_size=16,
                                    clamp_min=-8, clamp_max=7)
            Q_blocks[:, start:end] = Q_g

        scale_2d = flat_scales.reshape(M, N)
        deq_full = Q_full.float() * scale_2d
        deq_blocks = Q_blocks.float() * scale_2d
        mse_full = (W_rot - deq_full).pow(2).mean().item()
        mse_blocks = (W_rot - deq_blocks).pow(2).mean().item()

        assert mse_blocks <= mse_full * 1.15, (
            f"Block-wise MSE ({mse_blocks:.6f}) much worse than full MSE ({mse_full:.6f})"
        )

    def test_rotate_hessian_block_form(self):
        from converter.rotation import rotate_hessian

        torch.manual_seed(42)
        num_blocks = 4
        bs = 32
        rot_size = 16

        H = torch.zeros(num_blocks, bs, bs)
        for i in range(num_blocks):
            block = torch.randn(bs, bs)
            block = block @ block.T + torch.eye(bs)
            H[i] = block

        H_rot = rotate_hessian(H, rot_size=rot_size)
        assert H_rot.shape == H.shape

        n_sub = bs // rot_size
        for i in range(num_blocks):
            H_rot_4d = H_rot[i].reshape(n_sub, rot_size, n_sub, rot_size)
            H_rot_diag_sum = sum(H_rot_4d[s, :, s, :].trace().item() for s in range(n_sub))
            H_orig_4d = H[i].reshape(n_sub, rot_size, n_sub, rot_size)
            H_orig_diag_sum = sum(H_orig_4d[s, :, s, :].trace().item() for s in range(n_sub))
            assert abs(H_rot_diag_sum - H_orig_diag_sum) < 1e-3
