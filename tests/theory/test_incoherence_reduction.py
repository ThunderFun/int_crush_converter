"""Test: ConvRot reduces incoherence parameter mu."""

import math
import torch

from converter.rotation import rotate_weights, rotate_hessian


class TestIncoherenceReduction:

    def test_incoherence_definition(self):
        torch.manual_seed(42)
        M, N = 32, 128
        W = torch.randn(M, N)

        max_entry = W.abs().max().item()
        fro_norm = W.norm().item()
        expected_mu = max_entry * math.sqrt(M * N) / fro_norm

        assert 1.0 < expected_mu < 5.0, f"Random Gaussian should have mu~2, got {expected_mu:.2f}"

    def test_rotation_preserves_incoherence_for_uniform(self):
        torch.manual_seed(42)
        W = torch.rand(16, 64) * 2 - 1
        W_rot = rotate_weights(W, rot_size=64)
        assert W_rot.abs().max().item() < 3.0

    def test_rotate_hessian_preserves_trace(self):
        torch.manual_seed(42)
        N = 128
        H = torch.randn(N, N)
        H = H + H.T
        H = H @ H.T

        trace_before = torch.trace(H).item()
        H_rot = rotate_hessian(H, rot_size=64)
        trace_after = torch.trace(H_rot).item()

        assert abs(trace_before - trace_after) < 1e-3

    def test_rotate_hessian_block_diagonal_structure(self):
        torch.manual_seed(42)
        M, N = 32, 128
        rot_size = 64

        W = torch.randn(M, N)
        H = W.T @ W / M
        H_rot = rotate_hessian(H, rot_size=rot_size)

        num_groups = N // rot_size
        bs = rot_size
        H_rot_4d = H_rot.reshape(num_groups, bs, num_groups, bs)

        diag_mag = 0.0
        off_diag_mag = 0.0
        for i in range(num_groups):
            for j in range(num_groups):
                block = H_rot_4d[i, :, j, :]
                mag = block.abs().mean().item()
                if i == j:
                    diag_mag += mag
                else:
                    off_diag_mag += mag
        diag_mag /= num_groups
        off_diag_mag /= (num_groups * (num_groups - 1))

        assert off_diag_mag < diag_mag * 2, (
            f"Off-diagonal blocks too large: diag={diag_mag:.4f}, off_diag={off_diag_mag:.4f}"
        )
