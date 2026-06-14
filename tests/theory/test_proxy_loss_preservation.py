"""Test: Orthogonal rotation preserves proxy loss (QuIP theory)."""

import torch

from converter.scales import calculate_scales, quantize_weights


def _dequantize_per_row(q_int, scales):
    return q_int.float() * scales.float()


class TestProxyLossPreservation:

    def test_proxy_loss_preserved_global_rotation(self):
        torch.manual_seed(42)
        M, N = 16, 64
        W = torch.randn(M, N)
        H = W.T @ W / M

        A = torch.randn(N, N)
        Q, _ = torch.linalg.qr(A)
        R = Q

        W_rot = W @ R
        H_rot = R.T @ H @ R

        scales = W_rot.abs().amax(dim=1, keepdim=True) / 7.0
        q_W = quantize_weights(W_rot, scales, N)
        W_deq_rot = _dequantize_per_row(q_W, scales)
        W_deq = W_deq_rot @ R.T

        err_orig = W_deq - W
        proxy_orig = torch.trace(err_orig @ H @ err_orig.T).item()
        err_rot = W_deq_rot - W_rot
        proxy_rot = torch.trace(err_rot @ H_rot @ err_rot.T).item()

        assert abs(proxy_orig - proxy_rot) < 1e-4 * max(proxy_orig, 1e-8), (
            f"Proxy loss not preserved: orig={proxy_orig:.6f}, rot={proxy_rot:.6f}"
        )

    def test_proxy_loss_preserved_group_wise_convrot(self):
        from converter.rotation import rotate_weights, make_hadamard_regular

        torch.manual_seed(42)
        M, N = 16, 256
        rot_size = 64

        W = torch.randn(M, N)
        H = W.T @ W / M

        W_rot = rotate_weights(W, rot_size=rot_size)

        R = torch.zeros(N, N)
        for g in range(N // rot_size):
            H_block = make_hadamard_regular(rot_size, dtype=torch.float32)
            R[g * rot_size:(g + 1) * rot_size, g * rot_size:(g + 1) * rot_size] = H_block

        H_rot = R.T @ H @ R
        assert torch.allclose(R @ R.T, torch.eye(N), atol=1e-5)

        scales = W_rot.abs().amax(dim=1, keepdim=True) / 7.0
        q_W = quantize_weights(W_rot, scales, N)
        W_deq_rot = _dequantize_per_row(q_W, scales)
        W_deq = W_deq_rot @ R.T

        err_orig = W_deq - W
        proxy_orig = torch.trace(err_orig @ H @ err_orig.T).item()
        err_rot = W_deq_rot - W_rot
        proxy_rot = torch.trace(err_rot @ H_rot @ err_rot.T).item()

        assert abs(proxy_orig - proxy_rot) < 1e-3 * max(proxy_orig, 1e-8), (
            f"Group-wise ConvRot does not preserve proxy loss: "
            f"orig={proxy_orig:.6f}, rot={proxy_rot:.6f}"
        )
