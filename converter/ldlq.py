"""LDLQ: Weight-only quantization using W^T W as the Hessian.

Implements the LDLQ algorithm from QuIP (Chee et al., 2023) for INT4/INT8
quantization without calibration data. The Hessian is computed as
H = W^T @ W / M, making this a weight-only method.

The column loop is identical to GPTQ (Frantar et al., 2022) but uses
per-element scales with sign-flip correction and round-half-away-from-zero.
"""

import math
import warnings

import torch

from .scales import INT4_SCALE_DIVISOR, INT8_SCALE_DIVISOR
from .rounding import _invert_hessian, _round_half_away_from_zero, _ldlq_round_column
from .ldlq_triton import ldlq_loop_triton, _HAS_TRITON


_ldlq_compiled_fn = None
_HAS_TORCH_COMPILE = hasattr(torch, "compile")


def _get_ldlq_compiled_fn():
    """Lazily create a torch.compile'd LDLQ column body function.

    Fuses ~10 GPU kernels per column into 1-2 compiled kernels.
    """
    global _ldlq_compiled_fn
    if _ldlq_compiled_fn is not None:
        return _ldlq_compiled_fn
    if not _HAS_TORCH_COMPILE:
        return None
    try:
        @torch.compile(mode="default")
        def _compiled_ldlq_column(W_col, W_work_col, scale_col, hinv_diag_val,
                                  clamp_min, clamp_max):
            scale_safe = scale_col.clamp(min=1e-8)
            y = W_col / scale_safe
            q = torch.where(y >= 0, torch.floor(y + 0.5), torch.ceil(y - 0.5))
            q = q.clamp(clamp_min, clamp_max)
            W_sign = torch.where(W_work_col.abs() > 1e-8, W_work_col.sign(), q.sign())
            should_flip = (W_sign * q.sign()) < 0
            can_flip = should_flip & (q > clamp_min)
            q = torch.where(should_flip & (q == clamp_min),
                            torch.full_like(q, clamp_max), q)
            q = torch.where(can_flip, -q, q)
            q = q.clamp(clamp_min, clamp_max)
            err = W_col - q * scale_safe
            err_norm = err / hinv_diag_val
            return q.to(torch.int8), err_norm
        _ldlq_compiled_fn = _compiled_ldlq_column
    except Exception:
        _ldlq_compiled_fn = None
    return _ldlq_compiled_fn


def ldlq_quantize_layer(
    W: torch.Tensor,
    block_size: int = 128,
    damping: float = 0.01,
    int_bits: int = 4,
    iterations: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight matrix using LDLQ (weight-only, no calibration).

    Computes H = W^T @ W / M as the Hessian, then runs the GPTQ column loop
    with per-element scales and sign-flip correction.

    Args:
        W: [out_features, in_features] weight tensor
        block_size: block size for column processing
        damping: damping ratio as fraction of mean diagonal (default: 0.01)
        int_bits: quantization bit-width (4 or 8)
        iterations: number of LDLQ iterations with scale refinement (1 = single pass)

    Returns:
        (quantized_W, scales):
            quantized_W: [out_features, in_features] quantized values as int8
            scales: [out_features, 1] float16 per-row scales
    """
    if int_bits not in (4, 8):
        raise ValueError(f"int_bits must be 4 or 8, got {int_bits}")
    if W.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor, got {W.dim()}D")

    W = W.float()
    M, N = W.shape

    scale_divisor = INT8_SCALE_DIVISOR if int_bits == 8 else INT4_SCALE_DIVISOR
    clamp_min = -128 if int_bits == 8 else -8
    clamp_max = 127 if int_bits == 8 else 7

    # Compute Hessian from weights: H = W^T @ W / M
    H = W.T @ W / M
    diag_mean = H.diagonal().mean().clamp(min=1e-6)
    H = H + damping * diag_mean * torch.eye(N, dtype=torch.float32)
    H_inv = _invert_hessian(H)

    # Initial per-row scales
    row_scales = (W.abs().amax(dim=1, keepdim=True) / scale_divisor).clamp(min=1e-6)

    # Expand to per-element scales for LDLQ (all elements in a row share
    # the same scale, so this is equivalent to per-row quantization with
    # column-by-column error propagation).
    flat_scales = row_scales.expand(M, N).reshape(-1).clone()

    if iterations > 1:
        warnings.warn(
            "LDLQ scale refinement (iterations > 1) is incompatible with "
            "per-row scale storage. Using a single pass instead.",
            stacklevel=2,
        )

    quantized_W = _single_ldlq_pass(W, H_inv, flat_scales, block_size, clamp_min, clamp_max)

    # Compute the per-row scale that best fits the quantized integers to the
    # original weights.  For each row i the least-squares optimum is:
    #   s_i = (Q · W).sum(dim=1) / (Q · Q).sum(dim=1)
    # This keeps Q unchanged while finding the best dequantization scale.
    Q_float = quantized_W.float()
    numerator = (Q_float * W).sum(dim=1, keepdim=True)
    denominator = (Q_float * Q_float).sum(dim=1, keepdim=True).clamp(min=1e-8)
    final_scales = (numerator / denominator).clamp(min=1e-6)
    return quantized_W, final_scales.to(torch.float16)


def _single_ldlq_pass(
    W: torch.Tensor,
    H_inv: torch.Tensor,
    flat_scales: torch.Tensor,
    block_size: int,
    clamp_min: int,
    clamp_max: int,
) -> torch.Tensor:
    """Single pass of LDLQ quantization with OOM fallback.

    Allocates work tensors on CUDA when available, falling back to CPU on
    out-of-memory.  On GPU, selects the fastest available backend:

    1. **Triton** — fuses the per-column quantize + sign-flip + error-propagate
       loop into a single kernel launch per block (fastest).
    2. **torch.compile** — fuses ~10 PyTorch ops per column into 1-2 compiled
       kernels (good, but still Python-level column loop).
    3. **Eager** — plain PyTorch (slowest, always available).

    Args:
        W: [M, N] weight tensor (float32)
        H_inv: [N, N] inverse Hessian
        flat_scales: [M*N] per-element scales
        block_size: column block size
        clamp_min, clamp_max: quantization range

    Returns:
        Q_prime: [M, N] quantized values as int8
    """
    M, N = W.shape
    original_device = W.device

    # Prefer CUDA if available, fall back to CPU on OOM
    target_device = torch.device("cuda") if torch.cuda.is_available() else original_device
    try:
        Q_prime = torch.zeros(M, N, dtype=torch.int8, device=target_device)
        W_work = W.to(target_device).clone()
        H_inv = H_inv.to(target_device)
        flat_scales = flat_scales.to(target_device)
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        target_device = torch.device("cpu")
        W_work = W.cpu().clone()
        H_inv = H_inv.cpu()
        flat_scales = flat_scales.cpu()
        Q_prime = torch.zeros(M, N, dtype=torch.int8, device="cpu")

    # Reshape scales to [M, N]
    scale_2d = flat_scales.reshape(M, N)

    # Sanitize non-finite scales
    if not torch.isfinite(scale_2d).all():
        scale_2d = torch.nan_to_num(scale_2d, nan=1e-8, posinf=1e30, neginf=1e-8)
        scale_2d = scale_2d.clamp(min=1e-8)

    # Choose processing strategy: Triton > torch.compile > eager
    if target_device.type == "cpu":
        print(f"    LDLQ: CPU path, {M}x{N}, block_size={block_size}")
        Q_prime = _ldlq_loop_cpu(W_work, H_inv, scale_2d, Q_prime, N, M, block_size, clamp_min, clamp_max)
    else:
        # Try Triton block kernel first (fastest — fuses entire block into one launch)
        if _HAS_TRITON and block_size <= N:
            result = ldlq_loop_triton(W_work, H_inv, scale_2d, Q_prime, N, M, block_size, clamp_min, clamp_max)
            if result is not None:
                print(f"    LDLQ: GPU path (Triton), {M}x{N}, block_size={block_size}")
                Q_prime = result
            else:
                compiled_fn = _get_ldlq_compiled_fn()
                accel = "torch.compile" if compiled_fn is not None else "eager"
                print(f"    LDLQ: GPU path ({accel}), {M}x{N}, block_size={block_size}")
                Q_prime = _ldlq_loop_gpu(W_work, H_inv, scale_2d, Q_prime, N, M, block_size, clamp_min, clamp_max,
                                         compiled_fn=compiled_fn)
        else:
            compiled_fn = _get_ldlq_compiled_fn()
            accel = "torch.compile" if compiled_fn is not None else "eager"
            print(f"    LDLQ: GPU path ({accel}), {M}x{N}, block_size={block_size}")
            Q_prime = _ldlq_loop_gpu(W_work, H_inv, scale_2d, Q_prime, N, M, block_size, clamp_min, clamp_max,
                                     compiled_fn=compiled_fn)

    # Move result back to original device
    if Q_prime.device != original_device:
        Q_prime = Q_prime.to(original_device)

    # Clamp to valid range
    Q_prime = Q_prime.clamp(clamp_min, clamp_max)
    return Q_prime


def _ldlq_loop_cpu(
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
    """CPU LDLQ loop with immediate intra-block error propagation.

    Processes columns one at a time within each block, then applies a lazy
    batch update to columns outside the block. This is the standard GPTQ
    algorithm (Frantar et al., 2022) with lazy batch updates.
    """
    hinv_diag_inv_cache = {}

    for i1 in range(0, N, block_size):
        i2 = min(i1 + block_size, N)
        block_count = i2 - i1

        Hinv_block = H_inv[i1:i2, i1:i2]
        hinv_diag = torch.diag(Hinv_block).clamp(min=1e-8)
        hinv_diag_inv = (1.0 / hinv_diag).to(torch.float32)

        err_accum = torch.zeros(M, block_count, dtype=torch.float32)

        for j_local in range(block_count):
            j = i1 + j_local

            scale_col = scale_2d[:, j].clamp(min=1e-8)
            q_j = _round_half_away_from_zero(W_work[:, j] / scale_col)
            q_j = q_j.clamp(float(clamp_min), float(clamp_max))

            # Sign-flip correction
            W_sign = torch.where(W_work[:, j].abs() > 1e-8, W_work[:, j].sign(), q_j.sign())
            should_flip = (W_sign * q_j.sign()) < 0
            can_flip = should_flip & (q_j > float(clamp_min))
            q_j = torch.where(should_flip & (q_j == float(clamp_min)),
                              torch.full_like(q_j, float(clamp_max)), q_j)
            q_j = torch.where(can_flip, -q_j, q_j)
            q_j = q_j.clamp(float(clamp_min), float(clamp_max))

            Q_prime[:, j] = q_j.to(torch.int8)

            err_j = W_work[:, j] - q_j * scale_col
            err_j_norm = err_j * hinv_diag_inv[j_local]
            err_accum[:, j_local] = err_j_norm

            # Intra-block error propagation
            if j_local + 1 < block_count:
                hinv_row = Hinv_block[j_local, j_local + 1:block_count]
                W_work[:, j + 1:i2] -= err_j_norm.unsqueeze(1) * hinv_row.unsqueeze(0)

        # Inter-block batch update
        if i2 < N:
            update_slice = H_inv[i1:i2, i2:]
            W_work[:, i2:] -= err_accum @ update_slice

        del Hinv_block, hinv_diag, hinv_diag_inv, err_accum

    return Q_prime


def _ldlq_loop_gpu(
    W_work: torch.Tensor,
    H_inv: torch.Tensor,
    scale_2d: torch.Tensor,
    Q_prime: torch.Tensor,
    N: int,
    M: int,
    block_size: int,
    clamp_min: int,
    clamp_max: int,
    compiled_fn=None,
) -> torch.Tensor:
    """GPU LDLQ loop with optional torch.compile acceleration.

    When compiled_fn is provided, uses torch.compile to fuse the per-column
    quantize+sign-flip+error-compute operations, eliminating ~10 kernel
    launches per column.
    """
    for i1 in range(0, N, block_size):
        i2 = min(i1 + block_size, N)
        block_count = i2 - i1

        Hinv_block = H_inv[i1:i2, i1:i2]
        hinv_diag = torch.diag(Hinv_block).clamp(min=1e-8)

        err_accum = torch.zeros(M, block_count, device=W_work.device, dtype=torch.float32)

        for j_local in range(block_count):
            j = i1 + j_local

            if compiled_fn is not None:
                q_int8, err_j_norm = compiled_fn(
                    W_work[:, j], W_work[:, j], scale_2d[:, j], hinv_diag[j_local],
                    float(clamp_min), float(clamp_max),
                )
                Q_prime[:, j] = q_int8
                err_accum[:, j_local] = err_j_norm
            else:
                scale_col = scale_2d[:, j].clamp(min=1e-8)
                q_j = _round_half_away_from_zero(W_work[:, j] / scale_col)
                q_j = q_j.clamp(float(clamp_min), float(clamp_max))

                # Sign-flip correction
                W_sign = torch.where(W_work[:, j].abs() > 1e-8, W_work[:, j].sign(), q_j.sign())
                should_flip = (W_sign * q_j.sign()) < 0
                can_flip = should_flip & (q_j > float(clamp_min))
                q_j = torch.where(should_flip & (q_j == float(clamp_min)),
                                  torch.full_like(q_j, float(clamp_max)), q_j)
                q_j = torch.where(can_flip, -q_j, q_j)
                q_j = q_j.clamp(float(clamp_min), float(clamp_max))

                Q_prime[:, j] = q_j.to(torch.int8)

                err_j = W_work[:, j] - q_j * scale_col
                err_j_norm = err_j / hinv_diag[j_local]
                err_accum[:, j_local] = err_j_norm

            # Intra-block error propagation
            if j_local + 1 < block_count:
                hinv_row = Hinv_block[j_local, j_local + 1:block_count]
                W_work[:, j + 1:i2] -= err_j_norm.unsqueeze(1) * hinv_row.unsqueeze(0)

        # Inter-block batch update
        if i2 < N:
            update_slice = H_inv[i1:i2, i2:]
            W_work[:, i2:] -= err_accum @ update_slice

        del Hinv_block, hinv_diag, err_accum

    return Q_prime


def _run_iterative_ldlq(
    W: torch.Tensor,
    H_inv: torch.Tensor,
    flat_scales: torch.Tensor,
    iterations: int,
    block_size: int,
    clamp_min: int,
    clamp_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run iterative LDLQ with scale refinement.

    Each iteration re-estimates per-element scales from the quantized result,
    then re-runs LDLQ. Convergence is detected when MSE improvement drops
    below 0.01%. Divergence detection reverts to previous scales if MSE
    increases by more than 5%.
    """
    M, N = W.shape
    current_scales = flat_scales.clone()
    Q_result = None
    prev_mse = float("inf")

    # Sanitize non-finite scales
    if not torch.isfinite(current_scales).all():
        current_scales = torch.nan_to_num(current_scales, nan=1e-8, posinf=1e30, neginf=1e-8)
        current_scales = current_scales.clamp(min=1e-8)

    SCALE_MOMENTUM = 0.3
    MSE_DIVERGENCE_THRESHOLD = 1.05
    W_NONZERO_THRESHOLD = 1e-8

    mse_values = []
    prev_scales = None

    for iter_idx in range(iterations):
        Q_iter = _single_ldlq_pass(W, H_inv, current_scales, block_size, clamp_min, clamp_max)
        Q_result = Q_iter

        # Compute MSE
        mse = (W - Q_iter.float() * current_scales.reshape(M, N)).abs().pow(2).mean().item()
        mse_values.append(mse)
        print(f"    LDLQ iteration {iter_idx + 1}/{iterations}: MSE={mse:.6f}")

        if not math.isfinite(mse):
            print(f"    LDLQ: non-finite MSE at iteration {iter_idx + 1}, reverting")
            if prev_scales is not None:
                current_scales = prev_scales
            break

        if iter_idx > 0 and mse > prev_mse * MSE_DIVERGENCE_THRESHOLD:
            print(f"    LDLQ: diverged at iteration {iter_idx + 1} (MSE increased), reverting")
            if prev_scales is not None:
                current_scales = prev_scales
            break

        improvement = (prev_mse - mse) / (prev_mse + 1e-10)
        if iter_idx > 0 and improvement < 1e-4:
            print(f"    LDLQ: converged at iteration {iter_idx + 1} (improvement={improvement:.6f})")
            break

        prev_mse = mse

        if iter_idx < iterations - 1:
            prev_scales = current_scales.clone()
            current_scales = _update_ldlq_scales(
                W, Q_iter, current_scales, flat_scales, SCALE_MOMENTUM, W_NONZERO_THRESHOLD
            )

        # Free intermediates between iterations
        if iter_idx < iterations - 1 and Q_iter is not Q_result:
            del Q_iter
            if iter_idx % 2 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    if len(mse_values) > 1:
        mse_str = " ".join([f"{m:.6f}" for m in mse_values])

    return Q_result, current_scales


def _update_ldlq_scales(
    W: torch.Tensor,
    Q_iter: torch.Tensor,
    current_scales: torch.Tensor,
    flat_scales: torch.Tensor,
    momentum: float,
    nonzero_threshold: float,
) -> torch.Tensor:
    """Update scales for next LDLQ iteration.

    For valid (nonzero, sign-consistent) elements: new_scale = |W| / |Q|.
    For mismatched elements: revert to original flat scales.
    Momentum blending prevents oscillation.
    """
    W_flat = W.reshape(-1)
    Q_flat = Q_iter.reshape(-1).float()

    nonzero_mask = Q_flat != 0
    W_nonzero = W_flat.abs() > nonzero_threshold
    sign_consistent = W_nonzero & ((W_flat * Q_flat) > 0)
    valid = nonzero_mask & sign_consistent
    mismatched = nonzero_mask & ~sign_consistent

    new_scales = current_scales.clone()
    scale_min, scale_max = flat_scales.aminmax()
    scale_floor = max(scale_min.item() * 0.1, 1e-8)
    scale_ceil = scale_max.item() * 10.0

    if valid.any():
        try:
            computed = (W_flat[valid].abs() / Q_flat[valid].abs()).clamp(scale_floor, scale_ceil)
            if momentum > 0:
                new_scales[valid] = momentum * current_scales[valid] + (1 - momentum) * computed
            else:
                new_scales[valid] = computed
        except (RuntimeError, IndexError) as e:
            # Triton kernel memory corruption can produce garbage indices
            print(f"    LDLQ: scale update failed ({e}), reverting to original scales")
            new_scales[valid] = flat_scales[valid].clamp(scale_floor, scale_ceil)

    if mismatched.any():
        try:
            new_scales[mismatched] = flat_scales[mismatched].clamp(scale_floor, scale_ceil)
        except RuntimeError:
            pass

    return new_scales
