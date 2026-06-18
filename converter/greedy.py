"""Greedy local search for quantization grid optimization.

Coordinate descent on the quantization grid: tries all representable INT
values at each position, picks the one minimizing proxy loss
tr((W_hat - W) H (W_hat - W)^T).

Three backends (fastest first):
1. Low-rank Triton — fused kernel with low-rank Hessian approximation (B=64 cols/launch).
2. Full-rank Triton (v2) — batched cross terms + rank-1 corrections.
3. PyTorch — per-column matmul (fallback, always available).

Ref: QuIP (Chee et al., 2023, arXiv:2307.13304) — greedy as post-processing after LDLQ.
"""

import torch

from .config import (
    EIGENVALUE_FLOOR, LOWRANK_MAX_RANK_FRAC, LOWRANK_MAX_K, STALL_THRESHOLD,
    _HAS_TRITON, TRITON_BLOCK_ROWS_GREEDY, GREEDY_COL_BLOCK, GREEDY_RECOMPUTE_EVERY,
)
from .log import logger

if _HAS_TRITON:
    import triton
    import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _greedy_column_kernel(
        err_ptr,
        Q_ptr,
        scale_ptr,
        H_diag_ptr,
        grid_ptr,
        cross_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        G: tl.constexpr,
        j,
        CLAMP_MIN: tl.constexpr,
        CLAMP_MAX: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        """Evaluate all INT candidates for column j, pick best per row.

        For each row, computes the proxy-loss cost for every representable
        INT value on the grid and selects the one that minimises
        ``new_err² · H_diag[j] + 2 · new_err · cross``.

        Args:
            err_ptr: [M, N] error matrix (W - Q·scale), read/write.
            Q_ptr: [M, N] quantized values, read/write.
            scale_ptr: [M, N] per-element scales.
            H_diag_ptr: [N] diagonal of the Hessian.
            grid_ptr: [G] candidate INT values (e.g. -8..7 for INT4).
            M: number of rows.
            N: number of columns.
            G: number of grid candidates.
            j: column index to optimise.
            CLAMP_MIN: minimum representable INT value.
            CLAMP_MAX: maximum representable INT value.
            BLOCK_ROWS: rows per Triton program (compile-time constant).
        """
        pid = tl.program_id(0)
        row_start = pid * BLOCK_ROWS
        row_offs = row_start + tl.arange(0, BLOCK_ROWS)
        row_mask = row_offs < M

        cross = tl.load(cross_ptr + row_offs, mask=row_mask, other=0.0).to(tl.float32)
        err_j = tl.load(err_ptr + row_offs * N + j, mask=row_mask, other=0.0).to(tl.float32)
        s_j = tl.load(scale_ptr + row_offs * N + j, mask=row_mask, other=1.0).to(tl.float32)
        s_j = tl.maximum(s_j, 1e-8)
        q_old = tl.load(Q_ptr + row_offs * N + j, mask=row_mask, other=0.0).to(tl.float32)
        h_diag_j = tl.load(H_diag_ptr + j).to(tl.float32)

        best_cost = tl.full([BLOCK_ROWS], value=1e30, dtype=tl.float32)
        best_q = q_old

        for gi in range(G):
            q_cand = tl.load(grid_ptr + gi).to(tl.float32)
            delta = (q_cand - q_old) * s_j
            new_err = err_j + delta
            cost = new_err * new_err * h_diag_j + 2.0 * new_err * cross
            is_better = cost < best_cost
            best_cost = tl.where(is_better, cost, best_cost)
            best_q = tl.where(is_better, q_cand, best_q)

        best_q = tl.minimum(tl.maximum(best_q, CLAMP_MIN), CLAMP_MAX)
        delta_actual = (best_q - q_old) * s_j
        tl.store(Q_ptr + row_offs * N + j, best_q, mask=row_mask)
        tl.store(err_ptr + row_offs * N + j, err_j + delta_actual, mask=row_mask)

    @triton.jit
    def _greedy_lowrank_kernel(
        err_ptr,            # [M, N] error matrix, modified in-place
        Q_ptr,              # [M, N] quantized values, modified in-place
        scale_ptr,          # [M, N] per-element scales
        H_diag_ptr,         # [N] diagonal of Hessian
        grid_ptr,           # [G] candidate grid values
        err_lr_t_ptr,       # [K, M] low-rank error projection (transposed)
        U_k_ptr,            # [N, K] eigenvectors
        s_k_ptr,            # [K] eigenvalues
        M,
        N,
        G,
        K,
        j_start,
        B,
        CLAMP_MIN: tl.constexpr,
        CLAMP_MAX: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        """Fused low-rank greedy kernel: processes B columns per launch.

        Computes cross terms from the low-rank projection ``err_lr_t``,
        evaluates all grid candidates, and performs rank-1 updates to
        ``err_lr_t`` — all in a single kernel launch per column block.

        Args:
            err_ptr: [M, N] error matrix, read/write.
            Q_ptr: [M, N] quantized values, read/write.
            scale_ptr: [M, N] per-element scales.
            H_diag_ptr: [N] diagonal of Hessian.
            grid_ptr: [G] candidate INT values.
            err_lr_t_ptr: [K, M] low-rank error projection (transposed).
            U_k_ptr: [N, K] eigenvectors (low-rank basis).
            s_k_ptr: [K] eigenvalues.
            M: number of rows.
            N: number of columns.
            G: number of grid candidates.
            K: low-rank dimension.
            j_start: first column index in this block.
            B: number of columns in this block.
            CLAMP_MIN: minimum representable INT value.
            CLAMP_MAX: maximum representable INT value.
            BLOCK_ROWS: rows per Triton program (compile-time constant).
        """
        pid = tl.program_id(0)
        row_start = pid * BLOCK_ROWS
        row_offs = row_start + tl.arange(0, BLOCK_ROWS)
        row_mask = row_offs < M

        for j_local in range(B):
            j = j_local + j_start

            # Cross term: cross = err_lr @ (s_k * U_k[j,:]) - err[:,j] * H_diag[j]
            cross = tl.zeros([BLOCK_ROWS], dtype=tl.float32)
            for ki in range(K):
                err_lr_val = tl.load(
                    err_lr_t_ptr + ki * M + row_offs,
                    mask=row_mask, other=0.0
                ).to(tl.float32)
                u_val = tl.load(U_k_ptr + j * K + ki).to(tl.float32)
                s_val = tl.load(s_k_ptr + ki).to(tl.float32)
                cross += err_lr_val * (s_val * u_val)

            err_j = tl.load(err_ptr + row_offs * N + j, mask=row_mask, other=0.0).to(tl.float32)
            h_diag_j = tl.load(H_diag_ptr + j).to(tl.float32)
            cross -= err_j * h_diag_j

            # Evaluate all G candidates
            s_j = tl.load(scale_ptr + row_offs * N + j, mask=row_mask, other=1.0).to(tl.float32)
            s_j = tl.maximum(s_j, 1e-8)
            q_old = tl.load(Q_ptr + row_offs * N + j, mask=row_mask, other=0.0).to(tl.float32)

            best_cost = tl.full([BLOCK_ROWS], value=1e30, dtype=tl.float32)
            best_q = q_old

            for gi in range(G):
                q_cand = tl.load(grid_ptr + gi).to(tl.float32)
                delta = (q_cand - q_old) * s_j
                new_err = err_j + delta
                cost = new_err * new_err * h_diag_j + 2.0 * new_err * cross
                is_better = cost < best_cost
                best_cost = tl.where(is_better, cost, best_cost)
                best_q = tl.where(is_better, q_cand, best_q)

            best_q = tl.minimum(tl.maximum(best_q, CLAMP_MIN), CLAMP_MAX)
            delta_actual = (best_q - q_old) * s_j

            # Update err[:,j] and Q[:,j]
            tl.store(Q_ptr + row_offs * N + j, best_q, mask=row_mask)
            tl.store(err_ptr + row_offs * N + j, err_j + delta_actual, mask=row_mask)

            # Rank-1 update: err_lr_t[ki, :] += delta_actual * U_k[j, ki]
            for ki in range(K):
                err_lr_val = tl.load(
                    err_lr_t_ptr + ki * M + row_offs,
                    mask=row_mask, other=0.0
                ).to(tl.float32)
                u_val = tl.load(U_k_ptr + j * K + ki).to(tl.float32)
                new_val = err_lr_val + delta_actual * u_val
                tl.store(err_lr_t_ptr + ki * M + row_offs, new_val, mask=row_mask)


# ---------------------------------------------------------------------------
# Full-rank Triton (v2)
# ---------------------------------------------------------------------------


def greedy_local_search_triton(
    W: torch.Tensor,
    Q: torch.Tensor,
    scale_2d: torch.Tensor,
    H: torch.Tensor,
    num_passes: int,
    clamp_min: int,
    clamp_max: int,
    col_block: int = GREEDY_COL_BLOCK,
) -> torch.Tensor:
    """Greedy local search with Triton kernel + batched cross terms.

    Precomputes cross terms for blocks of columns via matmul, applies
    rank-1 corrections within each block, and uses a Triton kernel for
    the per-row candidate evaluation.

    Returns None if Triton is not available (caller should fall back to
    the PyTorch implementation).
    """
    if not _HAS_TRITON:
        return None

    M, N = W.shape
    device = W.device

    if not W.is_contiguous():
        W = W.contiguous()
    if not scale_2d.is_contiguous():
        scale_2d = scale_2d.contiguous()

    # Q is already float32 contiguous from _greedy_local_search — no clone needed.
    err = (Q * scale_2d - W).contiguous()
    H_contig = H.contiguous()
    H_diag = torch.diag(H).contiguous()
    grid_vals = torch.arange(clamp_min, clamp_max + 1, dtype=torch.float32, device=device).contiguous()
    G = grid_vals.shape[0]
    cross_buf = torch.empty(M, device=device, dtype=torch.float32).contiguous()

    BLOCK_ROWS = TRITON_BLOCK_ROWS_GREEDY
    grid = (triton.cdiv(M, BLOCK_ROWS),)
    recompute_every = GREEDY_RECOMPUTE_EVERY

    for pass_idx in range(num_passes):
        for j_start in range(0, N, col_block):
            j_end = min(j_start + col_block, N)
            B = j_end - j_start

            cross_block = (err @ H_contig[j_start:j_end, :].T).contiguous()

            for j_local in range(B):
                j = j_start + j_local

                if j_local > 0 and j_local % recompute_every == 0:
                    cross_block = (err @ H_contig[j_start:j_end, :].T).contiguous()

                err_old_j = err[:, j].clone()
                cross_buf.copy_(cross_block[:, j_local] - err_old_j * H_diag[j])

                _greedy_column_kernel[grid](
                    err, Q, scale_2d, H_diag, grid_vals, cross_buf,
                    M=M, N=N, G=G, j=j,
                    CLAMP_MIN=float(clamp_min), CLAMP_MAX=float(clamp_max),
                    BLOCK_ROWS=BLOCK_ROWS,
                )

                if j_local + 1 < B:
                    delta_j = err[:, j] - err_old_j
                    h_slice = H_contig[j, j_start + j_local + 1:j_end]
                    cross_block[:, j_local + 1:] += delta_j.unsqueeze(1) * h_slice.unsqueeze(0)

        if num_passes > 1:
            try:
                proxy = (err @ H_contig * err).sum().item()
                logger.debug("Greedy[Triton] pass %d/%d: proxy_loss=%.6f", pass_idx + 1, num_passes, proxy)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning("Greedy[Triton] pass %d/%d: proxy_loss=OOM (skipped)", pass_idx + 1, num_passes)

    return Q.clamp(clamp_min, clamp_max).to(torch.int8)


# ---------------------------------------------------------------------------
# Low-rank Triton
# ---------------------------------------------------------------------------


def _lowrank_decompose(H, rank_threshold=0.01, max_rank_frac=0.3):
    """Low-rank decomposition of H using iterative SVD.

    Uses ``torch.svd_lowrank`` (O(N²k) Lanczos iteration) which is 2-3x
    faster than ``torch.linalg.eigh`` (O(N³)) for large matrices.  Falls
    back to ``eigh`` if ``svd_lowrank`` fails (e.g. cusolver issues on
    some GPU/driver combinations).

    Args:
        H: [N, N] symmetric PSD Hessian
        rank_threshold: keep eigenvalues > threshold * max_eigenvalue
        max_rank_frac: if k > frac * N, fall back to full-rank

    Returns:
        (s_k, U_k, k, used_lowrank).
        If used_lowrank=False, s_k and U_k are None and caller should
        fall back to full-rank greedy.
    """
    N = H.shape[0]

    # Try svd_lowrank first (faster for large N)
    s_k, U_k, k, ok = _lowrank_decompose_svd(H, N, rank_threshold, max_rank_frac)
    if ok is not None:
        return s_k, U_k, k, ok

    # Fallback to eigh (reliable but O(N³))
    return _lowrank_decompose_eigh(H, N, rank_threshold, max_rank_frac)


def _lowrank_decompose_svd(H, N, rank_threshold, max_rank_frac):
    """Try low-rank decomposition via ``svd_lowrank``.

    Returns:
        ``(s_k, U_k, k, ok)`` where *ok* is one of:
        - ``True``  — low-rank succeeded; ``s_k`` and ``U_k`` are valid.
        - ``False`` — ``svd_lowrank`` ran but the effective rank is too
          high (> ``max_rank_frac * N`` or > ``LOWRANK_MAX_K``), or the
          Hessian is near-zero.  Caller should fall back to full-rank.
        - ``None``  — ``svd_lowrank`` raised (e.g. cusolver failure);
          caller should retry with ``eigh``.
    """
    q = min(max(N // 2, 64), N)
    try:
        U, s, V = torch.svd_lowrank(H, q=q)
    except (RuntimeError, torch._C._LinAlgError):
        return None, None, 0, None  # signal fallback

    max_eigenvalue = s[0].item()
    if max_eigenvalue < EIGENVALUE_FLOOR:
        return None, None, 0, False

    mask = s > max_eigenvalue * rank_threshold
    k = int(mask.sum().item())
    if k > max_rank_frac * N or k > LOWRANK_MAX_K:
        return None, None, k, False

    return s[mask], U[:, mask], k, True


def _lowrank_decompose_eigh(H, N, rank_threshold, max_rank_frac):
    """Full eigendecomposition fallback for low-rank Hessian extraction.

    Used when ``svd_lowrank`` fails (e.g. cusolver issues on some
    GPU/driver combinations).  O(N³) but numerically reliable.

    Returns:
        ``(s_k, U_k, k, ok)`` with the same semantics as
        :func:`_lowrank_decompose_svd`, except *ok* is never ``None``
        (no fallback left to try).
    """
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(H)
    except (torch._C._LinAlgError, RuntimeError):
        return None, None, 0, False

    # eigh returns ascending order; flip to descending
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    max_eigenvalue = eigenvalues[0].item()
    if max_eigenvalue < EIGENVALUE_FLOOR:
        return None, None, 0, False

    mask = eigenvalues > max_eigenvalue * rank_threshold
    k = int(mask.sum().item())
    if k > max_rank_frac * N or k > LOWRANK_MAX_K:
        return None, None, k, False

    return eigenvalues[mask], eigenvectors[:, mask], k, True


def greedy_lowrank_triton(
    W: torch.Tensor,
    Q: torch.Tensor,
    scale_2d: torch.Tensor,
    H: torch.Tensor,
    num_passes: int,
    clamp_min: int,
    clamp_max: int,
    rank_threshold: float = 0.01,
    max_rank_frac: float = 0.3,
    block_rows: int = TRITON_BLOCK_ROWS_GREEDY,
    col_block: int = GREEDY_COL_BLOCK,
) -> tuple | None:
    """Greedy local search with low-rank Hessian, fused Triton kernel.

    Uses eigendecomposition to decompose H ≈ U_k diag(s_k) U_k^T, then runs
    a fused kernel that computes cross terms, evaluates candidates, and
    updates err_lr in-place.

    Falls back to greedy_local_search_triton (v2) if effective rank is too
    high, and returns None if Triton is not available.

    Returns:
        (quantized_Q, effective_rank, used_lowrank) or None if Triton unavailable.
    """
    if not _HAS_TRITON:
        return None

    M, N = W.shape
    device = W.device

    if not W.is_contiguous():
        W = W.contiguous()
    if not scale_2d.is_contiguous():
        scale_2d = scale_2d.contiguous()

    # Low-rank decomposition
    s_k, U_k, k, used_lowrank = _lowrank_decompose(
        H, rank_threshold=rank_threshold, max_rank_frac=max_rank_frac
    )
    if not used_lowrank:
        del s_k, U_k
        torch.cuda.empty_cache()
        result = greedy_local_search_triton(
            W, Q, scale_2d, H, num_passes, clamp_min, clamp_max,
        )
        if result is not None:
            return result, k, False
        return None

    # Prepare data
    # Q is already float32 contiguous from _greedy_local_search — no clone needed.
    err = (Q * scale_2d - W).contiguous()
    H_diag = torch.diag(H).contiguous()
    grid_vals = torch.arange(clamp_min, clamp_max + 1, dtype=torch.float32, device=device).contiguous()
    G = grid_vals.shape[0]

    # Low-rank error projection — transposed for coalesced kernel access
    err_lr = (err @ U_k).contiguous()    # [M, K]
    err_lr_t = err_lr.T.contiguous()     # [K, M]
    s_k = s_k.contiguous()
    U_k = U_k.contiguous()

    # Launch kernel for each block of columns
    prev_proxy = float("inf")
    for pass_idx in range(num_passes):
        for j_start in range(0, N, col_block):
            j_end = min(j_start + col_block, N)
            B = j_end - j_start

            grid = (triton.cdiv(M, block_rows),)
            _greedy_lowrank_kernel[grid](
                err, Q, scale_2d, H_diag, grid_vals,
                err_lr_t, U_k, s_k,
                M, N, G, k,
                j_start, B,
                CLAMP_MIN=float(clamp_min), CLAMP_MAX=float(clamp_max),
                BLOCK_ROWS=block_rows,
            )

        try:
            proxy = (err @ H * err).sum().item()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            proxy = float("inf")
        if num_passes > 1:
            logger.debug("Greedy[lowrank] pass %d/%d: proxy_loss=%.6f (k=%d)",
                         pass_idx + 1, num_passes, proxy, k)

        if pass_idx > 0 and proxy >= prev_proxy * STALL_THRESHOLD:
            if num_passes > 1:
                logger.warning("Greedy[lowrank] stalled at pass %d", pass_idx + 1)
            break
        prev_proxy = proxy

    return Q.clamp(clamp_min, clamp_max).to(torch.int8), k, True


# ---------------------------------------------------------------------------
# PyTorch fallback
# ---------------------------------------------------------------------------


def greedy_local_search_pytorch(
    W: torch.Tensor,
    Q: torch.Tensor,
    scale_2d: torch.Tensor,
    H: torch.Tensor,
    num_passes: int,
    clamp_min: int,
    clamp_max: int,
) -> torch.Tensor:
    """Greedy local search: coordinate descent on the quantization grid.

    For each position, tries all representable INT values and picks the one
    that minimizes proxy loss tr((W_hat - W) H (W_hat - W)^T).  Vectorized
    across rows; column loop is sequential since each update changes the
    error vector for subsequent columns.

    Caller must provide Q as float32 contiguous (caller owns the clone).
    """
    M, N = W.shape
    device = W.device
    # Q is already float32 contiguous from the caller — no clone needed.
    grid = torch.arange(clamp_min, clamp_max + 1, dtype=torch.float32, device=device)

    H_diag = torch.diag(H)
    err = Q * scale_2d - W

    for pass_idx in range(num_passes):
        for j in range(N):
            q_old = Q[:, j]
            s_j = scale_2d[:, j]

            # Cross term: full dot product minus the j-th diagonal contribution
            cross = err @ H[j, :] - err[:, j] * H[j, j]

            # Evaluate all grid candidates across all rows: [M, G]
            delta = (grid.unsqueeze(0) - q_old.unsqueeze(1)) * s_j.unsqueeze(1)
            new_err = err[:, j].unsqueeze(1) + delta
            cost = new_err.pow(2) * H_diag[j] + 2 * new_err * cross.unsqueeze(1)

            best_idx = cost.argmin(dim=1)
            q_new = grid[best_idx]

            delta_actual = (q_new - q_old) * s_j
            Q[:, j] = q_new
            err[:, j] += delta_actual

        if num_passes > 1:
            try:
                proxy = (err @ H * err).sum().item()
                logger.debug("Greedy pass %d/%d: proxy_loss=%.6f", pass_idx + 1, num_passes, proxy)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning("Greedy pass %d/%d: proxy_loss=OOM (skipped)", pass_idx + 1, num_passes)

    return Q.clamp(clamp_min, clamp_max).to(torch.int8)
