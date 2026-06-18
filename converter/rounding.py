"""Shared rounding and Hessian utilities for GPTQ and LDLQ.

Core column-by-column quantization loop used by both GPTQ (calibration
Hessian) and LDLQ (weight-only H = W^T W).

Two rounding modes:
- GPTQ: per-row scales, banker's rounding, lazy batch error propagation
- LDLQ: per-element scales, round-half-away-from-zero, sign-flip correction,
        immediate intra-block error propagation
"""

import warnings

import torch

from .config import ERR_CLAMP_RANGE, SCALE_FLOOR, SIGN_FLIP_THRESHOLD


def _invert_hessian(H: torch.Tensor) -> torch.Tensor:
    """Compute inverse Hessian. Try Cholesky first (numerically stabler), fall back to inv, then pinv."""
    try:
        L = torch.linalg.cholesky(H)
        return torch.cholesky_inverse(L)
    except torch.linalg.LinAlgError:
        pass
    try:
        return torch.linalg.inv(H)
    except torch.linalg.LinAlgError:
        warnings.warn(
            "Hessian is singular/ill-conditioned; falling back to pseudoinverse. "
            "This may produce inaccurate quantization scales.",
            stacklevel=3,
        )
        return torch.linalg.pinv(H)


def _gptq_block(
    W_work: torch.Tensor,
    quantized_W: torch.Tensor,
    row_scales: torch.Tensor,
    H_inv: torch.Tensor,
    col_start: int,
    col_end: int,
    block_size: int,
    clamp_min: int,
    clamp_max: int,
    row_zp: torch.Tensor | None = None,
) -> None:
    """Run GPTQ on a range of columns using the given inverse Hessian.

    Processes columns in sub-blocks of ``block_size``.  Within each sub-block,
    error propagation is eager (each column updates all later columns in the
    same sub-block immediately).  Across sub-blocks, error propagation is lazy:
    accumulated errors are applied as a single matmul after the sub-block
    finishes.  This two-phase strategy (Frantar et al., 2022) keeps the
    H_inv slice in cache for the inner loop while amortising the outer-loop
    update into a single GEMV.

    Modifies W_work and quantized_W in-place.
    """
    out_features = W_work.shape[0]
    n_cols = col_end - col_start

    for blk_start in range(0, n_cols, block_size):
        blk_end = min(blk_start + block_size, n_cols)
        actual_blk = blk_end - blk_start

        # Accumulate normalised errors for the lazy inter-block update
        E_block = torch.zeros(out_features, actual_blk, dtype=torch.float32)

        for j_offset in range(actual_blk):
            j_local = blk_start + j_offset
            j_global = col_start + j_local

            # Quantize column using per-row scales and optional zero-point
            col = W_work[:, j_global]
            if row_zp is not None:
                raw = col / row_scales.squeeze(1) + row_zp.squeeze(1)
                q_col = _round_half_away_from_zero(raw).clamp(clamp_min, clamp_max)
            else:
                raw = col / row_scales.squeeze(1)
                q_col = _round_half_away_from_zero(raw).clamp(clamp_min, clamp_max)
            quantized_W[:, j_global] = q_col.to(torch.int8)

            # Normalised quantization error: (w - dequant(q)) / H_inv[j, j]
            if row_zp is not None:
                dequant = (q_col - row_zp.squeeze(1)) * row_scales.squeeze(1)
            else:
                dequant = q_col * row_scales.squeeze(1)
            err = (col - dequant) / H_inv[j_local, j_local]
            err = err.clamp(-ERR_CLAMP_RANGE, ERR_CLAMP_RANGE)  # prevent numerical explosion
            err = torch.nan_to_num(err, nan=0.0, posinf=ERR_CLAMP_RANGE, neginf=-ERR_CLAMP_RANGE)
            E_block[:, j_offset] = err

            # Eager intra-block update: propagate error to remaining columns
            # in this sub-block using the corresponding row of H_inv.
            if j_offset < actual_blk - 1:
                W_work[:, j_global:col_start + blk_end] -= err.unsqueeze(1) * H_inv[j_local, j_local:blk_end].unsqueeze(0)

        # Lazy inter-block update: apply accumulated errors to all columns
        # after this sub-block in one matmul.
        if blk_end < n_cols:
            W_work[:, col_start + blk_end:col_end] -= E_block @ H_inv[blk_start:blk_end, blk_end:n_cols]


def _round_half_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    """Round half away from zero (symmetric rounding).

    Unlike banker's rounding (.round()), 0.5 always rounds up (away from zero).
    This is the standard rounding mode for QuIP/LDLQ quantization.
    """
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))


def _ldlq_round_column(
    col: torch.Tensor,
    scale_col: torch.Tensor,
    clamp_min: int,
    clamp_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LDLQ rounding with sign-flip correction.

    Per-element scales, round-half-away-from-zero. Corrects sign mismatches
    between weight and quantized value (important with iterative scale refinement).

    Args:
        col: [M] weight column
        scale_col: [M] per-element scales for this column
        clamp_min: minimum quantized value (e.g. -8 for INT4)
        clamp_max: maximum quantized value (e.g. 7 for INT4)

    Returns:
        q_j: [M] quantized values (float, not yet cast to int8)
        err_j_norm: [M] error / scale (ready for Hessian propagation)
    """
    scale_safe = scale_col.clamp(min=SCALE_FLOOR)
    y = col / scale_safe
    q = _round_half_away_from_zero(y)
    q = q.clamp(float(clamp_min), float(clamp_max))

    # Sign-flip correction: iterative scale refinement can make round(W/s)
    # land on the opposite side of zero for near-zero weights. The subsequent
    # Hessian-weighted error propagation amplifies this bias. Clamping the
    # quantized value to match the original sign eliminates the drift.
    W_sign = torch.where(col.abs() > SIGN_FLIP_THRESHOLD, col.sign(), q.sign())
    should_flip = (W_sign * q.sign()) < 0
    # Prevent -8 -> +8 overflow: nudge stuck -8 values to +7
    can_flip = should_flip & (q > float(clamp_min))
    q = torch.where(should_flip & (q == float(clamp_min)),
                    torch.full_like(q, float(clamp_max)), q)
    q = torch.where(can_flip, -q, q)
    q = q.clamp(float(clamp_min), float(clamp_max))

    err = col - q * scale_safe
    return q, err
