"""SVD-absorbed low-rank decomposition (arXiv:2411.05007).

Decomposes W into low-rank FP16 branch (L1 @ L2) and residual R = W - L1 @ L2
that flows through the quantization pipeline. The low-rank branch absorbs
dominant singular values, leaving a cleaner residual for INT8/INT4 quantization.

Usage:
    result = decompose_weight(W, rank=32)
    # result.L1 [out, r] FP16, result.L2 [r, in] FP16
    # result.residual [out, in] float32 (quantize this instead of W)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SVDResult:
    """Result of SVD decomposition of a weight matrix."""

    L1: torch.Tensor
    """[out, r] FP16 — left low-rank factor (U * S)."""

    L2: torch.Tensor
    """[r, in] FP16 — right low-rank factor (V^T)."""

    residual: torch.Tensor
    """[out, in] float32 — residual W - L1.float() @ L2.float(),
    ready for downstream quantization."""


def decompose_weight(W: torch.Tensor, rank: int = 32) -> SVDResult:
    """SVD decomposition: W ≈ L1 @ L2 + residual.

    Uses :func:`torch.svd_lowrank` (randomized SVD) for efficiency.
    The low-rank factors are stored in FP16; the residual is kept in
    float32 so the downstream quantizer sees the full-precision remainder.

    Args:
        W: [out, in] weight matrix, float32.
        rank: Number of singular values to keep.  Clamped to
            ``min(rank, min(W.shape))``.  Typical: 16 (INT8), 32 (INT4).

    Returns:
        :class:`SVDResult` with L1, L2 (FP16) and residual (float32).
    """
    if W.dim() != 2:
        raise ValueError(f"decompose_weight requires a 2D tensor, got {W.dim()}D")

    out_features, in_features = W.shape
    r = min(rank, min(out_features, in_features))

    if r <= 0:
        raise ValueError(f"rank must be > 0, got {rank} (clamped to {r})")

    # Use randomized SVD for speed on large matrices.
    # svd_lowrank returns U, S, Vh such that W ≈ U @ diag(S) @ Vh^T.
    # We need L1 = U[:, :r] @ diag(S[:r])  and  L2 = Vh[:r, :].
    W_f32 = W.float()
    U, S, Vh = torch.svd_lowrank(W_f32, q=r, niter=2)

    # L1 absorbs the singular values: L1 = U * S (broadcast along columns)
    L1 = (U[:, :r] * S[:r].unsqueeze(0)).to(torch.float16)  # [out, r]
    L2 = Vh[:, :r].T.to(torch.float16)                       # [r, in]

    # Residual computed in float32 so downstream quantizer sees full precision.
    # The FP16 rounding of L1/L2 is absorbed into the residual, meaning
    # at inference the stored FP16 factors exactly reconstruct the low-rank
    # branch and Q(residual) fills in the remainder.
    residual = W_f32 - L1.float() @ L2.float()

    return SVDResult(L1=L1, L2=L2, residual=residual)
