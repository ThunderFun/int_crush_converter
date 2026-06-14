"""Test script for Triton kernel implementation of greedy local search.

Three implementations compared:
  - PyTorch: per-column matmul for cross terms (baseline)
  - Triton v2: precomputed cross terms + rank-1 update + Triton kernel
    (corresponds to greedy_local_search_triton in converter.ldlq_triton)
  - Triton low-rank: fused kernel with low-rank Hessian approximation
    (corresponds to greedy_lowrank_triton in converter.ldlq_triton)

Usage:
    python tests/test_greedy_triton.py              # correctness + benchmark
    python tests/test_greedy_triton.py --correctness # correctness only
    python tests/test_greedy_triton.py --benchmark   # benchmark only
"""

import math
import torch
import pytest

from converter.greedy import (
    greedy_local_search_triton,
    greedy_lowrank_triton,
    greedy_local_search_pytorch,
    _HAS_TRITON,
)


# ---------------------------------------------------------------------------
# PyTorch reference (same as ldlq._greedy_local_search)
# ---------------------------------------------------------------------------

def greedy_pytorch(
    W: torch.Tensor,
    Q: torch.Tensor,
    scale_2d: torch.Tensor,
    H: torch.Tensor,
    num_passes: int,
    clamp_min: int,
    clamp_max: int,
) -> torch.Tensor:
    """PyTorch reference implementation."""
    M, N = W.shape
    device = W.device
    Q = Q.clone().float()
    grid = torch.arange(clamp_min, clamp_max + 1, dtype=torch.float32, device=device)
    H_diag = torch.diag(H)
    err = Q * scale_2d - W

    for pass_idx in range(num_passes):
        for j in range(N):
            q_old = Q[:, j]
            s_j = scale_2d[:, j]
            cross = err @ H[j, :] - err[:, j] * H[j, j]
            delta = (grid.unsqueeze(0) - q_old.unsqueeze(1)) * s_j.unsqueeze(1)
            new_err = err[:, j].unsqueeze(1) + delta
            cost = new_err.pow(2) * H_diag[j] + 2 * new_err * cross.unsqueeze(1)
            best_idx = cost.argmin(dim=1)
            q_new = grid[best_idx]
            delta_actual = (q_new - q_old) * s_j
            Q[:, j] = q_new
            err[:, j] += delta_actual

    return Q.clamp(clamp_min, clamp_max).to(torch.int8)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestGreedyTritonCorrectness:
    """Triton kernels must produce Q values of comparable quality to PyTorch reference."""

    @pytest.fixture(autouse=True)
    def skip_no_triton(self):
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def _make_data(self, M, N, clamp_min=-8, clamp_max=7, seed=42):
        torch.manual_seed(seed)
        W = torch.randn(M, N, device="cuda")
        H = W.T @ W / M
        H = H + 0.01 * H.diagonal().mean() * torch.eye(N, device="cuda")
        row_scales = (W.abs().amax(dim=1, keepdim=True) / float(clamp_max)).clamp(min=1e-6)
        scale_2d = row_scales.expand(M, N).contiguous()
        Q_init = (W / scale_2d).round().clamp(clamp_min, clamp_max).to(torch.int8)
        return W, H, scale_2d, Q_init

    def _check_quality(self, W, H, scale_2d, Q_ref, Q_tri, max_diff_frac=0.05):
        """Verify Triton produces results of comparable quality to reference."""
        q_min, q_max = int(Q_ref.min()), int(Q_ref.max())
        assert Q_tri.min() >= min(q_min, -8) and Q_tri.max() <= max(q_max, 7)
        n_diff = (Q_ref != Q_tri).sum().item()
        n_total = Q_ref.numel()
        diff_frac = n_diff / n_total
        assert diff_frac < max_diff_frac, (
            f"Too many differences: {n_diff}/{n_total} ({diff_frac:.1%} > {max_diff_frac:.1%})"
        )
        err_ref = Q_ref.float() * scale_2d - W
        err_tri = Q_tri.float() * scale_2d - W
        proxy_ref = (err_ref @ H * err_ref).sum().item()
        proxy_tri = (err_tri @ H * err_tri).sum().item()
        assert proxy_tri <= proxy_ref * 1.1, (
            f"Triton proxy loss much worse: ref={proxy_ref:.4f}, tri={proxy_tri:.4f}"
        )

    # --- v2 tests (batched cross terms, via greedy_local_search_triton) ---

    def test_v2_small(self):
        W, H, s, Q = self._make_data(64, 128)
        Q_ref = greedy_pytorch(W, Q, s, H, 1, -8, 7)
        Q_tri = greedy_local_search_triton(W, Q, s, H, 1, -8, 7)
        assert torch.equal(Q_ref, Q_tri)

    def test_v2_medium(self):
        W, H, s, Q = self._make_data(256, 512)
        Q_ref = greedy_pytorch(W, Q, s, H, 1, -8, 7)
        Q_tri = greedy_local_search_triton(W, Q, s, H, 1, -8, 7)
        self._check_quality(W, H, s, Q_ref, Q_tri, max_diff_frac=0.02)

    def test_v2_large(self):
        W, H, s, Q = self._make_data(1024, 1024)
        Q_ref = greedy_pytorch(W, Q, s, H, 1, -8, 7)
        Q_tri = greedy_local_search_triton(W, Q, s, H, 1, -8, 7)
        self._check_quality(W, H, s, Q_ref, Q_tri, max_diff_frac=0.02)

    def test_v2_wide(self):
        W, H, s, Q = self._make_data(128, 4096)
        Q_ref = greedy_pytorch(W, Q, s, H, 1, -8, 7)
        Q_tri = greedy_local_search_triton(W, Q, s, H, 1, -8, 7)
        self._check_quality(W, H, s, Q_ref, Q_tri, max_diff_frac=0.05)

    def test_v2_tall(self):
        W, H, s, Q = self._make_data(4096, 1024)
        Q_ref = greedy_pytorch(W, Q, s, H, 1, -8, 7)
        Q_tri = greedy_local_search_triton(W, Q, s, H, 1, -8, 7)
        self._check_quality(W, H, s, Q_ref, Q_tri, max_diff_frac=0.02)

    def test_v2_two_passes(self):
        W, H, s, Q = self._make_data(128, 256)
        Q_ref = greedy_pytorch(W, Q, s, H, 2, -8, 7)
        Q_tri = greedy_local_search_triton(W, Q, s, H, 2, -8, 7)
        self._check_quality(W, H, s, Q_ref, Q_tri, max_diff_frac=0.05)

    def test_v2_int8(self):
        W, H, s, Q = self._make_data(128, 256, clamp_min=-128, clamp_max=127)
        Q_ref = greedy_pytorch(W, Q, s, H, 1, -128, 127)
        Q_tri = greedy_local_search_triton(W, Q, s, H, 1, -128, 127)
        assert Q_tri.min() >= -128 and Q_tri.max() <= 127
        self._check_quality(W, H, s, Q_ref, Q_tri, max_diff_frac=0.05)

    def test_v2_non_block_aligned(self):
        W, H, s, Q = self._make_data(300, 256)
        Q_ref = greedy_pytorch(W, Q, s, H, 1, -8, 7)
        Q_tri = greedy_local_search_triton(W, Q, s, H, 1, -8, 7)
        self._check_quality(W, H, s, Q_ref, Q_tri, max_diff_frac=0.02)

    # --- Quality tests ---

    def test_v2_proxy_loss_decreases(self):
        W, H, s, Q = self._make_data(256, 512)
        err_init = Q.float() * s - W
        proxy_init = (err_init @ H * err_init).sum().item()
        Q_greedy = greedy_local_search_triton(W, Q, s, H, 3, -8, 7)
        err_greedy = Q_greedy.float() * s - W
        proxy_greedy = (err_greedy @ H * err_greedy).sum().item()
        assert proxy_greedy <= proxy_init


# ---------------------------------------------------------------------------
# Low-rank Triton tests (production greedy_lowrank_triton)
# ---------------------------------------------------------------------------


def _make_lowrank_data(M, N, true_rank=50, noise_scale=0.01, clamp_min=-8,
                       clamp_max=7, seed=42, device="cuda"):
    """Create weight matrix with approximately low-rank Hessian."""
    torch.manual_seed(seed)
    U = torch.randn(M, true_rank, device=device)
    V = torch.randn(N, true_rank, device=device)
    W = (U @ V.T) / math.sqrt(true_rank) + noise_scale * torch.randn(M, N, device=device)
    H = W.T @ W / M
    H = H + 0.01 * H.diagonal().mean() * torch.eye(N, device=device)
    row_scales = (W.abs().amax(dim=1, keepdim=True) / float(clamp_max)).clamp(min=1e-6)
    scale_2d = row_scales.expand(M, N).contiguous()
    Q_init = (W / scale_2d).round().clamp(clamp_min, clamp_max).to(torch.int8)
    return W, H, scale_2d, Q_init


@pytest.mark.gpu
class TestLowrankTriton:
    """Low-rank Triton kernel: correctness and fallback behavior."""

    @pytest.fixture(autouse=True)
    def skip_no_triton(self):
        if not _HAS_TRITON:
            pytest.skip("Triton not installed")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def _check_quality(self, W, H, s, Q_ref, Q_tri, max_proxy_ratio=1.05,
                       max_diff_frac=0.10):
        q_min, q_max = int(Q_ref.min()), int(Q_ref.max())
        assert Q_tri.min() >= min(q_min, -8) and Q_tri.max() <= max(q_max, 7)
        n_diff = (Q_ref != Q_tri).sum().item()
        assert n_diff / Q_ref.numel() < max_diff_frac
        err_ref = Q_ref.float() * s - W
        err_tri = Q_tri.float() * s - W
        proxy_ref = (err_ref @ H * err_ref).sum().item()
        proxy_tri = (err_tri @ H * err_tri).sum().item()
        assert proxy_tri <= proxy_ref * max_proxy_ratio

    def test_lowrank_small(self):
        W, H, s, Q = _make_lowrank_data(64, 128, true_rank=10)
        Q_ref = greedy_pytorch(W, Q, s, H, 1, -8, 7)
        result = greedy_lowrank_triton(W, Q, s, H, 1, -8, 7)
        assert result is not None
        Q_tri, k, used = result
        self._check_quality(W, H, s, Q_ref, Q_tri, max_diff_frac=0.05)

    def test_lowrank_medium(self):
        W, H, s, Q = _make_lowrank_data(256, 512, true_rank=20)
        Q_ref = greedy_pytorch(W, Q, s, H, 1, -8, 7)
        result = greedy_lowrank_triton(W, Q, s, H, 1, -8, 7)
        assert result is not None
        Q_tri, k, used = result
        self._check_quality(W, H, s, Q_ref, Q_tri, max_diff_frac=0.05)

    def test_lowrank_wide(self):
        W, H, s, Q = _make_lowrank_data(128, 4096, true_rank=20)
        Q_ref = greedy_pytorch(W, Q, s, H, 1, -8, 7)
        result = greedy_lowrank_triton(W, Q, s, H, 1, -8, 7)
        assert result is not None
        Q_tri, k, used = result
        self._check_quality(W, H, s, Q_ref, Q_tri, max_diff_frac=0.05)

    def test_lowrank_int8(self):
        W, H, s, Q = _make_lowrank_data(128, 256, true_rank=15,
                                         clamp_min=-128, clamp_max=127)
        Q_ref = greedy_pytorch(W, Q, s, H, 1, -128, 127)
        result = greedy_lowrank_triton(W, Q, s, H, 1, -128, 127)
        assert result is not None
        Q_tri, k, used = result
        assert Q_tri.min() >= -128 and Q_tri.max() <= 127
        self._check_quality(W, H, s, Q_ref, Q_tri, max_proxy_ratio=1.03,
                            max_diff_frac=0.05)

    def test_lowrank_two_passes(self):
        W, H, s, Q = _make_lowrank_data(128, 256, true_rank=15)
        Q_ref = greedy_pytorch(W, Q, s, H, 2, -8, 7)
        result = greedy_lowrank_triton(W, Q, s, H, 2, -8, 7)
        assert result is not None
        Q_tri, k, used = result
        self._check_quality(W, H, s, Q_ref, Q_tri, max_proxy_ratio=1.05,
                            max_diff_frac=0.10)

    def test_fallback_on_high_rank(self):
        """Random Gaussian triggers fallback (high effective rank)."""
        W, H, s, Q = _make_lowrank_data(128, 256, true_rank=120, noise_scale=1.0)
        result = greedy_lowrank_triton(W, Q, s, H, 1, -8, 7)
        # Should either return a result (fallback to v2) or None
        if result is not None:
            Q_tri, k, used = result
            if used:
                Q_ref = greedy_pytorch(W, Q, s, H, 1, -8, 7)
                self._check_quality(W, H, s, Q_ref, Q_tri, max_proxy_ratio=1.1,
                                    max_diff_frac=0.15)

    def test_doesnt_worsen_proxy_loss(self):
        """Low-rank greedy should not worsen proxy loss from initial Q."""
        W, H, s, Q = _make_lowrank_data(256, 512, true_rank=20)
        err_init = Q.float() * s - W
        proxy_init = (err_init @ H * err_init).sum().item()
        result = greedy_lowrank_triton(W, Q, s, H, 1, -8, 7)
        if result is None:
            pytest.skip("Triton not available")
        Q_tri, k, used = result
        if Q_tri is None:
            pytest.skip("Low-rank not applicable")
        err_tri = Q_tri.float() * s - W
        proxy_tri = (err_tri @ H * err_tri).sum().item()
        assert proxy_tri <= proxy_init * 1.01

