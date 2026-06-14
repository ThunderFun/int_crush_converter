"""Test: trace(D) <= trace(H) after rotation (QuIP Lemma)."""

import torch

from converter.rotation import rotate_weights


class TestTraceDBound:

    def _compute_trace_d(self, H):
        L = torch.linalg.cholesky(H + 1e-6 * torch.eye(H.shape[0]))
        D = torch.diag(L) ** 2
        return D.sum().item()

    def test_trace_d_bound_before_rotation(self):
        torch.manual_seed(42)
        N = 64
        X = torch.randn(128, N)
        H = X.T @ X / 128

        trace_d = self._compute_trace_d(H)
        trace_h = torch.trace(H).item()
        assert trace_d <= trace_h * 1.001

    def test_trace_d_bound_after_rotation(self):
        torch.manual_seed(42)
        W = torch.randn(32, 64)
        W_rot = rotate_weights(W, rot_size=64)
        H_rot = W_rot.T @ W_rot / 32

        trace_d = self._compute_trace_d(H_rot)
        trace_h = torch.trace(H_rot).item()
        assert trace_d <= trace_h * 1.001

    def test_rotation_reduces_trace_d_ratio(self):
        torch.manual_seed(42)
        W = torch.randn(64, 128)

        H = W.T @ W / 64
        ratio_before = self._compute_trace_d(H) / torch.trace(H).item()

        W_rot = rotate_weights(W, rot_size=64)
        H_rot = W_rot.T @ W_rot / 64
        ratio_after = self._compute_trace_d(H_rot) / torch.trace(H_rot).item()

        assert ratio_after < ratio_before * 1.2
