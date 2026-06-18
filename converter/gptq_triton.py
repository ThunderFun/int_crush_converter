"""Triton-accelerated GPTQ block kernel.

Fuses full block column processing into one kernel launch, keeping Hinv_block
in registers. Supports INT4/INT8 via constexpr params and asymmetric via optional zero-point.
"""

import torch

from .config import _HAS_TRITON, TRITON_BLOCK_ROWS_GPTQ

# Backward-compatible alias used by gptq.py.
_HAS_GPTQ_TRITON = _HAS_TRITON

if _HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _gptq_block_kernel(
        W_work_ptr,
        scale_ptr,
        zp_ptr,
        Hinv_block_ptr,
        Q_out_ptr,
        err_accum_ptr,
        M,
        W_N,
        Q_N,
        col_start,
        block_count,
        hinv_stride,
        CLAMP_MIN: tl.constexpr,
        CLAMP_MAX: tl.constexpr,
        HAS_ZP: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        """Triton kernel for full GPTQ block processing.

        When HAS_ZP is True, uses asymmetric quantization:
            q = round(W / scale + zp)
            err = W - (q - zp) * scale
        When HAS_ZP is False, uses symmetric quantization:
            q = round(W / scale)
            err = W - q * scale
        """
        pid = tl.program_id(0)
        row_start = pid * BLOCK_ROWS
        row_offs = row_start + tl.arange(0, BLOCK_ROWS)
        row_mask = row_offs < M

        zp_vals = tl.zeros([BLOCK_ROWS], dtype=tl.float32)
        if HAS_ZP:
            zp_vals = tl.load(zp_ptr + row_offs, mask=row_mask, other=0.0).to(tl.float32)

        for j_local in range(block_count):
            j = col_start + j_local
            w_idx = row_offs * W_N + j
            q_idx = row_offs * Q_N + j

            w_work = tl.load(W_work_ptr + w_idx, mask=row_mask, other=0.0).to(tl.float32)

            scale_vals = tl.load(scale_ptr + row_offs, mask=row_mask, other=1.0).to(tl.float32)
            scale_vals = tl.maximum(scale_vals, 1e-8)

            if HAS_ZP:
                y = w_work / scale_vals + zp_vals
            else:
                y = w_work / scale_vals

            # Round half away from zero (consistent with PyTorch path)
            q = tl.where(y >= 0, tl.floor(y + 0.5), -tl.floor(-y + 0.5))

            q = tl.minimum(tl.maximum(q, CLAMP_MIN), CLAMP_MAX)
            tl.store(Q_out_ptr + q_idx, q.to(tl.int8), mask=row_mask)

            if HAS_ZP:
                err = w_work - (q - zp_vals) * scale_vals
            else:
                err = w_work - q * scale_vals

            hinv_diag = tl.load(Hinv_block_ptr + j_local * hinv_stride + j_local)
            hinv_diag = tl.maximum(hinv_diag, 1e-8)
            err_norm = err / hinv_diag

            err_norm = tl.minimum(tl.maximum(err_norm, -100.0), 100.0)
            err_norm = tl.where(err_norm != err_norm, 0.0, err_norm)

            tl.store(err_accum_ptr + row_offs * block_count + j_local, err_norm, mask=row_mask)

            for k_local in range(block_count):
                if k_local > j_local:
                    col_k = col_start + k_local
                    w_k_idx = row_offs * W_N + col_k
                    w_k = tl.load(W_work_ptr + w_k_idx, mask=row_mask, other=0.0).to(tl.float32)
                    hinv_val = tl.load(Hinv_block_ptr + j_local * hinv_stride + k_local)
                    w_k = w_k - err_norm * hinv_val
                    tl.store(W_work_ptr + w_k_idx, w_k, mask=row_mask)


def gptq_loop_triton(
    W_work: torch.Tensor,
    H_inv: torch.Tensor,
    row_scales: torch.Tensor,
    Q_out: torch.Tensor,
    N: int,
    M: int,
    block_size: int,
    clamp_min: int,
    clamp_max: int,
    row_zp: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """GPU GPTQ loop using the Triton block kernel.

    Args:
        W_work: [M, N] working weights, modified in-place
        H_inv: [N, N] inverse Hessian
        row_scales: [M] per-row scales (1D)
        Q_out: [M, N] output int8 tensor
        N: number of columns
        M: number of rows
        block_size: column block size
        clamp_min: minimum quantized value (-8 for INT4, -128 for INT8)
        clamp_max: maximum quantized value (7 for INT4, 127 for INT8)
        row_zp: [M] per-row zero-points (1D, optional). When provided,
                uses asymmetric quantization: q = round(W/scale + zp).

    Returns:
        Q_out (same tensor, modified in-place), or None if Triton unavailable.
    """
    if not _HAS_GPTQ_TRITON:
        return None

    if not torch.cuda.is_available():
        return None

    if not W_work.is_contiguous():
        W_work = W_work.contiguous()
    if not row_scales.is_contiguous():
        row_scales = row_scales.contiguous()

    has_zp = row_zp is not None
    if has_zp:
        zp_1d = row_zp.squeeze().contiguous().to(torch.float32)
        zp_gpu = zp_1d.to(W_work.device)
    else:
        zp_gpu = None

    BLOCK_ROWS = TRITON_BLOCK_ROWS_GPTQ

    # When HAS_ZP is False the kernel never reads zp_ptr, but passing
    # the same tensor for both scale_ptr and zp_ptr can cause
    # speculative reads on some GPUs that corrupt the scale data.
    # Always pass a distinct dummy buffer for zp when it is unused.
    if not has_zp:
        zp_dummy = torch.zeros(M, device=W_work.device, dtype=torch.float32)
    else:
        zp_dummy = None

    for i1 in range(0, N, block_size):
        i2 = min(i1 + block_size, N)
        block_count = i2 - i1

        Hinv_block = H_inv[i1:i2, i1:i2].contiguous()
        err_accum = torch.zeros(M, block_count, device=W_work.device, dtype=torch.float32)

        grid = (triton.cdiv(M, BLOCK_ROWS),)
        _gptq_block_kernel[grid](
            W_work, row_scales, zp_gpu if has_zp else zp_dummy,
            Hinv_block, Q_out, err_accum,
            M, W_work.shape[1], Q_out.shape[1], i1,
            block_count, Hinv_block.shape[1],
            CLAMP_MIN=float(clamp_min),
            CLAMP_MAX=float(clamp_max),
            HAS_ZP=has_zp,
            BLOCK_ROWS=BLOCK_ROWS,
        )

        if i2 < N:
            update_slice = H_inv[i1:i2, i2:]
            W_work[:, i2:] -= err_accum @ update_slice

        del Hinv_block, err_accum

    return Q_out
