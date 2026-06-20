"""Triton-accelerated PiSO kernels (arXiv:2606.10890).

Fuses candidate-scale evaluation into one kernel launch, eliminating
O(B×C×D) intermediates of the PyTorch path.

Two kernels:
- Symmetric INT8: evaluates coarse+fine candidate scales, picks Hessian-weighted optimum per row.
- Asymmetric INT8: same, also optimizes zero-point per candidate.

Candidate-major, D-minor loop order keeps W in L2 cache across evaluations.
Only 3 registers per row (running_cost, best_cost, best_scale) vs O(C×D) broadcast.
"""

import torch

from .log import logger
from .config import _HAS_TRITON, TRITON_BLOCK_ROWS_PISO, PISO_D_BLOCK

if _HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _piso_symmetric_kernel(
        W_ptr,
        h_diag_ptr,
        s_absmax_ptr,
        scales_ptr,
        M,
        D,
        num_d_blocks,
        NUM_COARSE: tl.constexpr,
        NUM_FINE: tl.constexpr,
        D_BLOCK: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        CLAMP_MIN: tl.constexpr,
        CLAMP_MAX: tl.constexpr,
    ):
        """Triton kernel for symmetric PiSO grid search.

        Each program evaluates ``NUM_COARSE + NUM_FINE`` candidate scales
        for ``BLOCK_ROWS`` rows and stores the best scale per row.

        The candidate-major loop order means W stays in L2 cache across
        candidate evaluations — each D-block is loaded from DRAM once and
        re-read from L2 for all ``NUM_COARSE + NUM_FINE`` candidates.

        Args:
            W_ptr:          [M, D] float32 weight matrix.
            h_diag_ptr:     [D] float32 Hessian diagonal.
            s_absmax_ptr:   [M] float32 absmax scale per row.
            scales_ptr:     [M] float32 output optimal scales.
            M:              number of rows.
            D:              number of columns.
            num_d_blocks:   ``ceil(D / D_BLOCK)``, runtime.
            NUM_COARSE:     number of coarse candidates (constexpr).
            NUM_FINE:       number of fine candidates (constexpr).
            D_BLOCK:        column block size (constexpr).
            BLOCK_ROWS:     rows per program (constexpr).
            CLAMP_MIN:      minimum quantized value (-128.0).
            CLAMP_MAX:      maximum quantized value (127.0).
        """
        pid = tl.program_id(0)
        row_start = pid * BLOCK_ROWS
        row_offs = row_start + tl.arange(0, BLOCK_ROWS)
        row_mask = row_offs < M

        # Load absmax scale for this program's rows.
        s_am = tl.load(s_absmax_ptr + row_offs, mask=row_mask, other=1e-10).to(tl.float32)
        s_am = tl.maximum(s_am, 1e-10)

        best_cost = tl.full([BLOCK_ROWS], value=1e30, dtype=tl.float32)
        best_s = s_am

        coarse_step = 1.5 / (NUM_COARSE - 1)  # 0.5 → 2.0

        # ── Coarse search ──
        for ci in range(NUM_COARSE):
            factor = 0.5 + ci.to(tl.float32) * coarse_step
            s = s_am * factor
            s_col = s[:, None]  # [BLOCK_ROWS, 1] for broadcast

            running = tl.zeros([BLOCK_ROWS], dtype=tl.float32)

            for db in range(num_d_blocks):
                d_start = db * D_BLOCK
                d_offs = d_start + tl.arange(0, D_BLOCK)
                d_mask = d_offs < D

                w = tl.load(
                    W_ptr + row_offs[:, None] * D + d_offs[None, :],
                    mask=row_mask[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)

                h = tl.load(h_diag_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)

                y = w / s_col
                q = tl.where(y >= 0.0, tl.floor(y + 0.5), -tl.floor(-y + 0.5))
                q = tl.minimum(tl.maximum(q, CLAMP_MIN), CLAMP_MAX)

                err = w - q * s_col
                running += tl.sum(err * err * h[None, :], axis=1)

            is_better = running < best_cost
            best_cost = tl.where(is_better, running, best_cost)
            best_s = tl.where(is_better, s, best_s)

        fine_step = 0.1 / (NUM_FINE - 1)  # 0.95 → 1.05

        # ── Fine search ──
        for fi in range(NUM_FINE):
            factor = 0.95 + fi.to(tl.float32) * fine_step
            s = best_s * factor
            s_col = s[:, None]

            running = tl.zeros([BLOCK_ROWS], dtype=tl.float32)

            for db in range(num_d_blocks):
                d_start = db * D_BLOCK
                d_offs = d_start + tl.arange(0, D_BLOCK)
                d_mask = d_offs < D

                w = tl.load(
                    W_ptr + row_offs[:, None] * D + d_offs[None, :],
                    mask=row_mask[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)

                h = tl.load(h_diag_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)

                y = w / s_col
                q = tl.where(y >= 0.0, tl.floor(y + 0.5), -tl.floor(-y + 0.5))
                q = tl.minimum(tl.maximum(q, CLAMP_MIN), CLAMP_MAX)

                err = w - q * s_col
                running += tl.sum(err * err * h[None, :], axis=1)

            is_better = running < best_cost
            best_cost = tl.where(is_better, running, best_cost)
            best_s = tl.where(is_better, s, best_s)

        tl.store(scales_ptr + row_offs, best_s, mask=row_mask)

    @triton.jit
    def _piso_asymmetric_kernel(
        W_ptr,
        h_diag_ptr,
        w_min_ptr,
        w_max_ptr,
        scales_ptr,
        zp_ptr,
        M,
        D,
        num_d_blocks,
        NUM_COARSE: tl.constexpr,
        NUM_FINE: tl.constexpr,
        D_BLOCK: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        CLAMP_MIN: tl.constexpr,
        CLAMP_MAX: tl.constexpr,
    ):
        """Triton kernel for asymmetric PiSO grid search.

        Like the symmetric kernel but also optimizes the zero-point per
        candidate.  For each candidate scale ``s``, the zero-point is
        ``zp = clamp(round(CLAMP_MIN - w_min / s), CLAMP_MIN, CLAMP_MAX)``
        and quantization is ``q = round(w / s + zp)``.

        Args:
            W_ptr:          [M, D] float32 weight matrix.
            h_diag_ptr:     [D] float32 Hessian diagonal.
            w_min_ptr:      [M] float32 per-row weight minimum.
            w_max_ptr:      [M] float32 per-row weight maximum.
            scales_ptr:     [M] float32 output optimal scales.
            zp_ptr:         [M] int16 output optimal zero-points.
            M:              number of rows.
            D:              number of columns.
            num_d_blocks:   ``ceil(D / D_BLOCK)``, runtime.
            NUM_COARSE:     number of coarse candidates (constexpr).
            NUM_FINE:       number of fine candidates (constexpr).
            D_BLOCK:        column block size (constexpr).
            BLOCK_ROWS:     rows per program (constexpr).
            CLAMP_MIN:      minimum quantized value (-128.0).
            CLAMP_MAX:      maximum quantized value (127.0).
        """
        pid = tl.program_id(0)
        row_start = pid * BLOCK_ROWS
        row_offs = row_start + tl.arange(0, BLOCK_ROWS)
        row_mask = row_offs < M

        w_min = tl.load(w_min_ptr + row_offs, mask=row_mask, other=0.0).to(tl.float32)
        w_max = tl.load(w_max_ptr + row_offs, mask=row_mask, other=0.0).to(tl.float32)
        range_val = w_max - w_min
        s_ref = range_val / 255.0
        s_ref = tl.maximum(s_ref, 1e-10)

        best_cost = tl.full([BLOCK_ROWS], value=1e30, dtype=tl.float32)
        best_s = s_ref
        best_zp = tl.zeros([BLOCK_ROWS], dtype=tl.float32)

        coarse_step = 0.4 / (NUM_COARSE - 1)  # 0.8 → 1.2

        # ── Coarse search ──
        for ci in range(NUM_COARSE):
            factor = 0.8 + ci.to(tl.float32) * coarse_step
            s = s_ref * factor
            s = tl.maximum(s, 1e-10)
            s_col = s[:, None]

            # Compute zero-point per row for this candidate scale
            zp_raw = CLAMP_MIN - w_min / s
            # Round half away from zero
            zp_rounded = tl.where(zp_raw >= 0.0, tl.floor(zp_raw + 0.5), -tl.floor(-zp_raw + 0.5))
            zp = tl.minimum(tl.maximum(zp_rounded, CLAMP_MIN), CLAMP_MAX)

            running = tl.zeros([BLOCK_ROWS], dtype=tl.float32)

            for db in range(num_d_blocks):
                d_start = db * D_BLOCK
                d_offs = d_start + tl.arange(0, D_BLOCK)
                d_mask = d_offs < D

                w = tl.load(
                    W_ptr + row_offs[:, None] * D + d_offs[None, :],
                    mask=row_mask[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)

                h = tl.load(h_diag_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)

                y = w / s_col + zp[:, None]
                q = tl.where(y >= 0.0, tl.floor(y + 0.5), -tl.floor(-y + 0.5))
                q = tl.minimum(tl.maximum(q, CLAMP_MIN), CLAMP_MAX)

                dequant = (q - zp[:, None]) * s_col
                err = w - dequant
                running += tl.sum(err * err * h[None, :], axis=1)

            is_better = running < best_cost
            best_cost = tl.where(is_better, running, best_cost)
            best_s = tl.where(is_better, s, best_s)
            best_zp = tl.where(is_better, zp, best_zp)

        fine_step = 0.1 / (NUM_FINE - 1)  # 0.95 → 1.05

        # ── Fine search ──
        for fi in range(NUM_FINE):
            factor = 0.95 + fi.to(tl.float32) * fine_step
            s = best_s * factor
            s = tl.maximum(s, 1e-10)
            s_col = s[:, None]

            zp_raw = CLAMP_MIN - w_min / s
            zp_rounded = tl.where(zp_raw >= 0.0, tl.floor(zp_raw + 0.5), -tl.floor(-zp_raw + 0.5))
            zp = tl.minimum(tl.maximum(zp_rounded, CLAMP_MIN), CLAMP_MAX)

            running = tl.zeros([BLOCK_ROWS], dtype=tl.float32)

            for db in range(num_d_blocks):
                d_start = db * D_BLOCK
                d_offs = d_start + tl.arange(0, D_BLOCK)
                d_mask = d_offs < D

                w = tl.load(
                    W_ptr + row_offs[:, None] * D + d_offs[None, :],
                    mask=row_mask[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)

                h = tl.load(h_diag_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)

                y = w / s_col + zp[:, None]
                q = tl.where(y >= 0.0, tl.floor(y + 0.5), -tl.floor(-y + 0.5))
                q = tl.minimum(tl.maximum(q, CLAMP_MIN), CLAMP_MAX)

                dequant = (q - zp[:, None]) * s_col
                err = w - dequant
                running += tl.sum(err * err * h[None, :], axis=1)

            is_better = running < best_cost
            best_cost = tl.where(is_better, running, best_cost)
            best_s = tl.where(is_better, s, best_s)
            best_zp = tl.where(is_better, zp, best_zp)

        tl.store(scales_ptr + row_offs, best_s, mask=row_mask)
        tl.store(zp_ptr + row_offs, best_zp.to(tl.int16), mask=row_mask)


def compute_piso_scales_int8_triton(
    W: torch.Tensor,
    hessian_diag: torch.Tensor,
    num_coarse: int = 128,
    num_fine: int = 16,
) -> torch.Tensor | None:
    """Triton-accelerated PiSO grid search for symmetric INT8.

    Args:
        W:              [M, D] float32 weight matrix on CUDA.
        hessian_diag:   [D] float32 Hessian diagonal.
        num_coarse:     number of coarse candidates (default 128).
        num_fine:       number of fine candidates (default 16).

    Returns:
        [M, 1] float32 optimal per-row scales, or None if Triton is
        unavailable or the kernel fails.
    """
    if not _HAS_TRITON:
        return None
    if not torch.cuda.is_available():
        return None
    if num_coarse < 2 or num_fine < 2:
        logger.warning("PiSO: num_coarse and num_fine must be >= 2, got %d and %d", num_coarse, num_fine)
        return None

    M, D = W.shape
    if not W.is_contiguous():
        W = W.contiguous()

    h_diag = hessian_diag.float()[:D]
    if not h_diag.is_contiguous():
        h_diag = h_diag.contiguous()

    # Precompute absmax scale per row on GPU.
    s_absmax = W.abs().amax(dim=1) / 127.0
    s_absmax = s_absmax.clamp(min=1e-10).contiguous()

    scales = torch.empty(M, dtype=torch.float32, device=W.device)
    num_d_blocks = triton.cdiv(D, PISO_D_BLOCK)

    BLOCK_ROWS = TRITON_BLOCK_ROWS_PISO
    grid = (triton.cdiv(M, BLOCK_ROWS),)

    try:
        _piso_symmetric_kernel[grid](
            W, h_diag, s_absmax, scales,
            M, D, num_d_blocks,
            NUM_COARSE=num_coarse,
            NUM_FINE=num_fine,
            D_BLOCK=PISO_D_BLOCK,
            BLOCK_ROWS=BLOCK_ROWS,
            CLAMP_MIN=-128.0,
            CLAMP_MAX=127.0,
        )
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        logger.debug("PiSO Triton symmetric fallback: %s", e)
        return None

    return scales.unsqueeze(1)  # [M, 1]


def compute_piso_scales_int8_asymmetric_triton(
    W: torch.Tensor,
    hessian_diag: torch.Tensor,
    num_coarse: int = 128,
    num_fine: int = 16,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Triton-accelerated PiSO grid search for asymmetric INT8.

    Args:
        W:              [M, D] float32 weight matrix on CUDA.
        hessian_diag:   [D] float32 Hessian diagonal.
        num_coarse:     number of coarse candidates (default 128).
        num_fine:       number of fine candidates (default 16).

    Returns:
        ``(scales, zero_points)`` where scales is [M, 1] float32 and
        zero_points is [M, 1] int16, or None if Triton is unavailable.
    """
    if not _HAS_TRITON:
        return None
    if not torch.cuda.is_available():
        return None
    if num_coarse < 2 or num_fine < 2:
        logger.warning("PiSO: num_coarse and num_fine must be >= 2, got %d and %d", num_coarse, num_fine)
        return None

    M, D = W.shape
    if not W.is_contiguous():
        W = W.contiguous()

    h_diag = hessian_diag.float()[:D]
    if not h_diag.is_contiguous():
        h_diag = h_diag.contiguous()

    w_min = W.amin(dim=1).contiguous()
    w_max = W.amax(dim=1).contiguous()

    scales = torch.empty(M, dtype=torch.float32, device=W.device)
    zp = torch.empty(M, dtype=torch.int16, device=W.device)
    num_d_blocks = triton.cdiv(D, PISO_D_BLOCK)

    BLOCK_ROWS = TRITON_BLOCK_ROWS_PISO
    grid = (triton.cdiv(M, BLOCK_ROWS),)

    try:
        _piso_asymmetric_kernel[grid](
            W, h_diag, w_min, w_max, scales, zp,
            M, D, num_d_blocks,
            NUM_COARSE=num_coarse,
            NUM_FINE=num_fine,
            D_BLOCK=PISO_D_BLOCK,
            BLOCK_ROWS=BLOCK_ROWS,
            CLAMP_MIN=-128.0,
            CLAMP_MAX=127.0,
        )
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        logger.debug("PiSO Triton asymmetric fallback: %s", e)
        return None

    return scales.unsqueeze(1), zp.unsqueeze(1)  # [M, 1] each
