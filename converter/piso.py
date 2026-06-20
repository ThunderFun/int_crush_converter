"""PiSO: Piecewise-Optimal Scale Optimization (arXiv:2606.10890).

Computes data-aware per-row scales minimizing output reconstruction error
||Xw - X * quant(w/s)||² using the Hessian diagonal H = X^T X.
Unlike absmax scales, PiSO accounts for how quantization error propagates
through the Hessian.

Algorithm: coarse-to-fine grid search. The proxy objective E(s) = ||diag(h) * (w - s*q(s))||²
is piecewise quadratic in s, so a dense grid reliably finds the global minimum.
Diagonal approximation H ≈ diag(diag(H)) for O(D|G|) per-channel complexity.

Backend: Triton GPU (fastest) → PyTorch (fallback).
"""

import torch

from .log import logger
from .config import _HAS_TRITON
from .rounding import _round_half_away_from_zero


def compute_piso_scales_int8(
    W: torch.Tensor,
    hessian_diag: torch.Tensor,
    num_coarse: int = 128,
    num_fine: int = 16,
) -> torch.Tensor:
    """Compute PiSO-optimal per-row scales for symmetric INT8 quantization.

    Uses a coarse-to-fine grid search: evaluate the Hessian-weighted
    quantization error at candidate scale values around the absmax scale,
    then refine around the best candidate.

    Dispatches to Triton GPU when available, falling back to PyTorch.

    Args:
        W: [out_features, in_features] weight matrix (float32)
        hessian_diag: [in_features] diagonal of H = X^T X
        num_coarse: number of coarse candidates (0.5x–2.0x absmax)
        num_fine: number of fine candidates (0.95x–1.05x coarse winner)

    Returns:
        scales: [out_features, 1] float32 optimal per-row scales
    """
    if W.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {W.dim()}D")
    out_features, in_features = W.shape
    h_diag = hessian_diag.float()[:in_features]
    if h_diag.shape[0] != in_features:
        raise ValueError(
            f"hessian_diag length {h_diag.shape[0]} != in_features {in_features}"
        )

    # ── Try Triton GPU path ──
    if _HAS_TRITON and torch.cuda.is_available():
        try:
            W_gpu = W.float().to("cuda", non_blocking=True)
            h_gpu = h_diag.to("cuda", non_blocking=True)
            from .piso_triton import compute_piso_scales_int8_triton
            result = compute_piso_scales_int8_triton(W_gpu, h_gpu, num_coarse, num_fine)
            if result is not None:
                logger.debug("PiSO: Triton symmetric, %dx%d", out_features, in_features)
                return result.to(W.device)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug("PiSO: OOM on GPU, falling back to CPU")
        except Exception as exc:
            logger.debug("PiSO: Triton failed (%s), falling back to CPU", exc)

    # ── PyTorch fallback ──
    return _compute_piso_scales_int8_pytorch(W, h_diag, num_coarse, num_fine)


def _compute_piso_scales_int8_pytorch(
    W: torch.Tensor,
    h_diag: torch.Tensor,
    num_coarse: int,
    num_fine: int,
) -> torch.Tensor:
    """PyTorch implementation of PiSO grid search (fallback path).

    Processes rows in batches to bound peak memory at ~512 MB.
    """
    out_features, in_features = W.shape

    scales = torch.empty(out_features, 1, dtype=torch.float32, device=W.device)
    coarse_factors = torch.linspace(0.5, 2.0, num_coarse, device=W.device, dtype=torch.float32)
    fine_factors = torch.linspace(0.95, 1.05, num_fine, device=W.device, dtype=torch.float32)
    num_candidates = num_coarse + num_fine

    # Batch size: process as many rows as fit in ~512MB of intermediates.
    # Each row needs num_candidates * D * 4 bytes for the q/err tensors.
    # At D=4096, num_candidates=144: 144*4096*4 = 2.4MB/row → batch=200.
    # At D=15360, num_candidates=144: 144*15360*4 = 8.8MB/row → batch=56.
    bytes_per_row = num_candidates * in_features * 4
    batch_size = max(1, min(out_features, (512 * 1024 * 1024) // max(1, bytes_per_row)))

    for start in range(0, out_features, batch_size):
        end = min(start + batch_size, out_features)
        W_batch = W[start:end]       # [B, D]
        B = end - start

        # Absmax scale per row
        s_absmax = W_batch.abs().amax(dim=1, keepdim=True) / 127.0  # [B, 1]
        s_absmax = s_absmax.clamp(min=1e-10)

        # Coarse search
        s_coarse = s_absmax * coarse_factors.unsqueeze(0)  # [B, C]
        W_3d = W_batch.unsqueeze(1)         # [B, 1, D]
        s_3d = s_coarse.unsqueeze(2)        # [B, C, 1]
        q = _round_half_away_from_zero(W_3d / s_3d).clamp(-128, 127)
        cost = ((W_3d - q * s_3d) ** 2 * h_diag).sum(dim=2)  # [B, C]
        best_idx = cost.argmin(dim=1)
        best_scale = s_coarse[torch.arange(B), best_idx]

        # Fine search
        s_fine = best_scale.unsqueeze(1) * fine_factors.unsqueeze(0)  # [B, F]
        s_3d_fine = s_fine.unsqueeze(2)
        q_fine = _round_half_away_from_zero(W_3d / s_3d_fine).clamp(-128, 127)
        cost_fine = ((W_3d - q_fine * s_3d_fine) ** 2 * h_diag).sum(dim=2)
        best_fine_idx = cost_fine.argmin(dim=1)
        scales[start:end, 0] = s_fine[torch.arange(B), best_fine_idx]

    return scales


def compute_piso_scales_int8_asymmetric(
    W: torch.Tensor,
    hessian_diag: torch.Tensor,
    num_coarse: int = 128,
    num_fine: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PiSO-optimal per-row scales and zero-points for asymmetric INT8.

    Uses a grid search over candidate scales around the range-based scale.
    For asymmetric quantization, the zero-point is determined by the scale.

    Dispatches to Triton GPU when available, falling back to PyTorch.

    Args:
        W: [out_features, in_features] weight matrix (float32)
        hessian_diag: [in_features] diagonal of H = X^T X
        num_coarse: number of coarse candidates (0.8x–1.2x range-based)
        num_fine: number of fine candidates (0.95x–1.05x coarse winner)

    Returns:
        (scales, zero_points):
            scales: [out_features, 1] float32 optimal per-row scales
            zero_points: [out_features, 1] int16 per-row zero-points
    """
    if W.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {W.dim()}D")

    out_features, in_features = W.shape
    h_diag = hessian_diag.float()[:in_features]

    # ── Try Triton GPU path ──
    if _HAS_TRITON and torch.cuda.is_available():
        try:
            W_gpu = W.float().to("cuda", non_blocking=True)
            h_gpu = h_diag.to("cuda", non_blocking=True)
            from .piso_triton import compute_piso_scales_int8_asymmetric_triton
            result = compute_piso_scales_int8_asymmetric_triton(W_gpu, h_gpu, num_coarse, num_fine)
            if result is not None:
                scales_gpu, zp_gpu = result
                logger.debug("PiSO: Triton asymmetric, %dx%d", out_features, in_features)
                return scales_gpu.to(W.device), zp_gpu.to(W.device)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug("PiSO: OOM on GPU (asymmetric), falling back to CPU")
        except Exception as exc:
            logger.debug("PiSO: Triton asymmetric failed (%s), falling back to CPU", exc)

    # ── PyTorch fallback ──
    return _compute_piso_scales_int8_asymmetric_pytorch(W, h_diag, num_coarse, num_fine)


def _compute_piso_scales_int8_asymmetric_pytorch(
    W: torch.Tensor,
    h_diag: torch.Tensor,
    num_coarse: int,
    num_fine: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch implementation of asymmetric PiSO grid search (fallback path).

    Uses a vectorized grid search over candidate scales.  For each
    candidate, the zero-point is ``zp = round(-128 - w_min / s)``,
    then ``q = round(w / s + zp)``.
    """
    out_features, in_features = W.shape

    scales = torch.empty(out_features, 1, dtype=torch.float32, device=W.device)
    zero_points = torch.empty(out_features, 1, dtype=torch.int16, device=W.device)

    coarse_factors = torch.linspace(0.8, 1.2, num_coarse, device=W.device, dtype=torch.float32)
    fine_factors = torch.linspace(0.95, 1.05, num_fine, device=W.device, dtype=torch.float32)

    # Vectorized row processing (batches to bound memory).
    num_candidates = num_coarse + num_fine
    bytes_per_row = num_candidates * in_features * 4
    batch_size = max(1, min(out_features, (512 * 1024 * 1024) // max(1, bytes_per_row)))

    for start in range(0, out_features, batch_size):
        end = min(start + batch_size, out_features)
        W_batch = W[start:end].float()  # [B, D]
        B = end - start

        w_min = W_batch.amin(dim=1, keepdim=True)  # [B, 1]
        w_max = W_batch.amax(dim=1, keepdim=True)  # [B, 1]
        range_val = w_max - w_min
        s_ref = (range_val / 255.0).clamp(min=1e-10)  # [B, 1]

        # Coarse search
        s_coarse = s_ref * coarse_factors.unsqueeze(0)  # [B, C]
        s_coarse = s_coarse.clamp(min=1e-10)
        W_3d = W_batch.unsqueeze(1)  # [B, 1, D]
        s_3d = s_coarse.unsqueeze(2)  # [B, C, 1]
        zp_3d = _round_half_away_from_zero(-128 - w_min.unsqueeze(2) / s_3d).clamp(-128, 127)  # [B, C, 1]
        q = _round_half_away_from_zero(W_3d / s_3d + zp_3d).clamp(-128, 127)
        dequant = (q - zp_3d) * s_3d
        cost = ((W_3d - dequant) ** 2 * h_diag).sum(dim=2)  # [B, C]
        best_idx = cost.argmin(dim=1)  # [B]
        best_scale = s_coarse[torch.arange(B), best_idx]
        best_zp = zp_3d.squeeze(2)[torch.arange(B), best_idx]

        # Fine search
        s_fine = best_scale.unsqueeze(1) * fine_factors.unsqueeze(0)  # [B, F]
        s_fine = s_fine.clamp(min=1e-10)
        s_3d_fine = s_fine.unsqueeze(2)
        zp_3d_fine = _round_half_away_from_zero(-128 - w_min.unsqueeze(2) / s_3d_fine).clamp(-128, 127)
        q_fine = _round_half_away_from_zero(W_3d / s_3d_fine + zp_3d_fine).clamp(-128, 127)
        dequant_fine = (q_fine - zp_3d_fine) * s_3d_fine
        cost_fine = ((W_3d - dequant_fine) ** 2 * h_diag).sum(dim=2)
        best_fine_idx = cost_fine.argmin(dim=1)

        scales[start:end, 0] = s_fine[torch.arange(B), best_fine_idx]
        zero_points[start:end, 0] = zp_3d_fine.squeeze(2)[torch.arange(B), best_fine_idx].to(torch.int16)

    return scales, zero_points
