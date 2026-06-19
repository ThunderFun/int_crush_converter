"""Diagonal + Low-Rank (DLR) Hessian support.

Represents ``H ≈ diag(D) + UUᵀ`` where ``D`` is the exact diagonal and
``U`` is a rank-``r`` factor.  ``O(n + nr)`` storage, comparable to
block-diagonal but capturing cross-block correlations.

Key operation: :func:`woodbury_inverse` computes ``H⁻¹`` in ``O(nr²)``
via the Woodbury identity (vs ``O(n³)`` direct inverse).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import DIAG_MEAN_FLOOR
from .log import logger

# Minimum damped diagonal value.  Guards against division-by-zero in the
# Woodbury formula when a channel has zero variance (e.g. padding columns
# introduced by rotation).
_DLR_DIAG_FLOOR = 1e-12


def is_dlr(hessian) -> bool:
    """Return True if *hessian* is a DLR-format dict."""
    if not isinstance(hessian, dict):
        return False
    if hessian.get("format") != "dlr":
        return False
    if "D" not in hessian or "U" not in hessian:
        return False
    return True


def validate_dlr(hessian: dict, in_features: int | None = None) -> bool:
    """Validate a DLR dict for finiteness and shape consistency.

    Returns True if valid, False otherwise (logs a warning, never raises).
    """
    if not is_dlr(hessian):
        return False

    D = hessian["D"]
    U = hessian["U"]

    if not isinstance(D, torch.Tensor) or not isinstance(U, torch.Tensor):
        logger.warning("DLR: D or U is not a tensor")
        return False
    if D.dim() != 1 or U.dim() != 2:
        logger.warning("DLR: expected D (n,) and U (n, r), got %sD, %sD",
                        D.dim(), U.dim())
        return False
    if D.shape[0] != U.shape[0]:
        logger.warning("DLR: D rows (%d) != U rows (%d)", D.shape[0], U.shape[0])
        return False
    if in_features is not None and D.shape[0] != in_features:
        logger.warning("DLR: feature dim %d != expected %d", D.shape[0], in_features)
        return False
    if not torch.isfinite(D).all() or not torch.isfinite(U).all():
        logger.warning("DLR: non-finite values in D or U")
        return False
    if D.min().item() < 0:
        logger.warning("DLR: negative diagonal entry (min=%.6e)", D.min().item())
        return False
    return True


def damped_diag_mean(D: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Mean of the true Hessian diagonal ``D + diag(UUᵀ)``, for damping."""
    diag_full = D + (U * U).sum(dim=1)
    return diag_full.mean().clamp(min=DIAG_MEAN_FLOOR)


def woodbury_inverse(
    D: torch.Tensor,
    U: torch.Tensor,
    damping: float = 0.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute ``(diag(D) + UUᵀ + λ·mean(diag(H))·I)⁻¹`` via Woodbury.

    ``O(nr² + r³)`` — far cheaper than ``O(n³)`` direct inverse for ``r ≪ n``.
    """
    D = D.float()
    U = U.float()
    n = D.shape[0]
    r = U.shape[1]

    # Absorb damping into the diagonal: D' = D + λ·mean(diag(H)).
    if damping > 0.0:
        diag_mean = damped_diag_mean(D, U)
        D_damped = D + damping * diag_mean
    else:
        D_damped = D

    # Guard against zero/negative diagonal entries (padding columns, etc.)
    D_safe = D_damped.clamp(min=_DLR_DIAG_FLOOR)
    D_inv = 1.0 / D_safe  # (n,)

    # Z = diag(1/D') U  — (n, r)
    Z = D_inv.unsqueeze(1) * U

    # M = I + Uᵀ diag(1/D') U = I + Uᵀ Z  — (r, r)
    eye_r = torch.eye(r, dtype=torch.float32)
    M = eye_r + U.T @ Z

    # M⁻¹ — (r, r).  Use Cholesky when possible (M is SPD for well-formed DLR),
    # fall back to direct inverse.
    try:
        L_m = torch.linalg.cholesky(M)
        M_inv = torch.cholesky_inverse(L_m)
    except torch.linalg.LinAlgError:
        try:
            M_inv = torch.linalg.inv(M)
        except torch.linalg.LinAlgError:
            M_inv = torch.linalg.pinv(M)

    # H⁻¹ = diag(D_inv) − Z M⁻¹ Zᵀ  — (n, n)
    H_inv = torch.diag(D_inv) - Z @ M_inv @ Z.T

    if device is not None:
        H_inv = H_inv.to(device)
    return H_inv


def dlr_to_dense(D: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Materialise ``diag(D) + UUᵀ`` as a dense 2-D tensor."""
    D = D.float()
    U = U.float()
    n = D.shape[0]
    return torch.diag(D) + U @ U.T


def transform_dlr_for_smoothquant(
    D: torch.Tensor,
    U: torch.Tensor,
    smoothing_factors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply SmoothQuant column scaling: ``D → D/s²``, ``U → U/s``.

    Preserves DLR structure exactly.
    """
    s = smoothing_factors.float()
    # Clamp to prevent division by zero (matches SMOOTHQUANT_HESSIAN_FLOOR logic)
    s_safe = s.clamp(min=1e-8)
    s_inv_sq = 1.0 / (s_safe * s_safe)
    D_new = D.float() * s_inv_sq
    U_new = U.float() / s_safe.unsqueeze(1)
    return D_new, U_new


def permute_dlr(
    D: torch.Tensor,
    U: torch.Tensor,
    perm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder DLR factors by a channel permutation: ``D[perm], U[perm]``."""
    perm = perm.to(torch.int64)
    return D[perm].clone(), U[perm].clone()


def rotate_dlr_to_dense(
    D: torch.Tensor,
    U: torch.Tensor,
    rot_size: int,
) -> torch.Tensor:
    """Materialise DLR to dense and apply Hadamard rotation.

    Rotation destroys the DLR structure (``Rᵀ diag(D) R`` is block-dense),
    so this is an edge case — when calibration was collected *with* rotation,
    the DLR factors are already in rotated space.
    """
    # Import here to avoid a circular dependency at module load time.
    from .rotation import rotate_hessian

    H_dense = dlr_to_dense(D, U)
    return rotate_hessian(H_dense, rot_size)


def make_dlr_dict(D: torch.Tensor, U: torch.Tensor) -> dict:
    """Build a DLR dict from ``D`` and ``U`` tensors."""
    return {
        "format": "dlr",
        "D": D,
        "U": U,
        "rank": int(U.shape[1]),
        "n": int(D.shape[0]),
    }
