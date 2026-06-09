"""Triton-accelerated LDLQ block kernel.

Processes one full block of columns in a single kernel launch, keeping
Hinv_block in shared memory for the entire block. Supports INT4 and INT8
via constexpr parameters.
"""

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _ldlq_block_kernel(
        W_work_ptr,
        scale_ptr,
        Hinv_block_ptr,
        Q_prime_ptr,
        err_accum_ptr,
        M,
        N,
        i1,
        CLAMP_MIN: tl.constexpr,
        CLAMP_MAX: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        """Triton kernel for full LDLQ block processing."""
        pid = tl.program_id(0)
        row_start = pid * BLOCK_ROWS
        row_offs = row_start + tl.arange(0, BLOCK_ROWS)
        row_mask = row_offs < M

        for j_local in range(BLOCK_COLS):
            j = i1 + j_local
            w_idx = row_offs * N + j

            w_work = tl.load(W_work_ptr + w_idx, mask=row_mask, other=0.0).to(tl.float32)
            scale = tl.load(scale_ptr + w_idx, mask=row_mask, other=1.0).to(tl.float32)
            scale = tl.maximum(scale, 1e-8)

            y = w_work / scale
            q = tl.where(y >= 0.0, tl.floor(y + 0.5), tl.ceil(y - 0.5))
            q = tl.minimum(tl.maximum(q, CLAMP_MIN), CLAMP_MAX)

            work_s = tl.where(w_work > 0.0, 1.0, tl.where(w_work < 0.0, -1.0, 0.0))
            q_s = tl.where(q > 0.0, 1.0, tl.where(q < 0.0, -1.0, 0.0))
            w_s = tl.where(tl.abs(w_work) > 1e-8, work_s, q_s)
            should_flip = (w_s * q_s) < 0.0
            # can_flip MUST be computed BEFORE the nudge (q=CLAMP_MIN -> CLAMP_MAX),
            # otherwise the nudge changes the value and allows an incorrect flip.
            can_flip = should_flip & (q > CLAMP_MIN)
            nudge_min = should_flip & (q == CLAMP_MIN)
            q = tl.where(nudge_min, CLAMP_MAX, q)
            q = tl.where(can_flip, -q, q)
            q = tl.minimum(tl.maximum(q, CLAMP_MIN), CLAMP_MAX)

            tl.store(Q_prime_ptr + w_idx, q.to(tl.int8), mask=row_mask)

            err = w_work - q * scale
            hinv_d = tl.load(Hinv_block_ptr + j_local * BLOCK_COLS + j_local)
            hinv_d = tl.maximum(hinv_d, 1e-8)
            err_norm = err / hinv_d

            tl.store(err_accum_ptr + row_offs * BLOCK_COLS + j_local, err_norm, mask=row_mask)

            for k_local in range(BLOCK_COLS):
                if k_local > j_local:
                    hinv_val = tl.load(Hinv_block_ptr + j_local * BLOCK_COLS + k_local)
                    col_k = i1 + k_local
                    w_k_idx = row_offs * N + col_k
                    w_k = tl.load(W_work_ptr + w_k_idx, mask=row_mask, other=0.0).to(tl.float32)
                    w_k = w_k - err_norm * hinv_val
                    tl.store(W_work_ptr + w_k_idx, w_k, mask=row_mask)


def ldlq_loop_triton(
    W_work: torch.Tensor,
    H_inv: torch.Tensor,
    scale_2d: torch.Tensor,
    Q_prime: torch.Tensor,
    N: int,
    M: int,
    block_size: int,
    clamp_min: int,
    clamp_max: int,
) -> torch.Tensor:
    """GPU LDLQ loop using the Triton block kernel.

    Args:
        W_work: [M, N] working weights, modified in-place
        H_inv: [N, N] inverse Hessian
        scale_2d: [M, N] per-element scales
        Q_prime: [M, N] output int8 tensor
        N: number of columns
        M: number of rows
        block_size: column block size
        clamp_min: minimum quantized value (-8 for INT4, -128 for INT8)
        clamp_max: maximum quantized value (7 for INT4, 127 for INT8)

    Returns:
        Q_prime (same tensor, modified in-place)
    """
    if not _HAS_TRITON:
        return None

    if not W_work.is_contiguous():
        W_work = W_work.contiguous()
    if not scale_2d.is_contiguous():
        scale_2d = scale_2d.contiguous()

    BLOCK_ROWS = 64

    for i1 in range(0, N, block_size):
        i2 = min(i1 + block_size, N)
        block_count = i2 - i1

        Hinv_block = H_inv[i1:i2, i1:i2].contiguous()
        err_accum = torch.zeros(M, block_count, device=W_work.device, dtype=torch.float32)

        grid = (triton.cdiv(M, BLOCK_ROWS),)
        _ldlq_block_kernel[grid](
            W_work, scale_2d, Hinv_block, Q_prime, err_accum,
            M, N, i1,
            CLAMP_MIN=float(clamp_min),
            CLAMP_MAX=float(clamp_max),
            BLOCK_COLS=block_count,
            BLOCK_ROWS=BLOCK_ROWS,
        )

        # Inter-block batch update (still done in PyTorch — matmul is fast)
        if i2 < N:
            update_slice = H_inv[i1:i2, i2:]
            W_work[:, i2:] -= err_accum @ update_slice

        del Hinv_block, err_accum

    return Q_prime
