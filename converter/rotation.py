"""Hadamard Transform for ConvRot quantization.

Applies group-wise Hadamard rotation to weight matrices and activations.
Uses dense Regular Hadamard matmul (power of 4) or Sylvester construction
(any power of 2).

Based on ConvRot (arXiv:2512.03673) — rotation suppresses outliers before
quantization, making simple RTN effective for INT4.
"""

import math
import torch
import torch.nn.functional as F

# Cache for constructed Hadamard matrices: {size: matrix}
_hadamard_cache: dict[tuple[int, str, str], torch.Tensor] = {}


def _is_power_of_four(n: int) -> bool:
    if n < 4:
        return False
    return (n & (n - 1)) == 0 and (n & 0x55555555) == n


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def make_hadamard_sylvester(n: int, dtype: torch.dtype = torch.float16, device: str = "cpu") -> torch.Tensor:
    """Construct a normalized Sylvester Hadamard matrix of size n (any power of 2)."""
    if not _is_power_of_two(n):
        raise ValueError(f"n must be a power of 2, got {n}")
    H = torch.tensor([[1.0]], dtype=dtype, device=device)
    while H.shape[0] < n:
        H = torch.kron(
            torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=dtype, device=device),
            H,
        )
    return H * (1.0 / math.sqrt(n))


def make_hadamard_regular(n: int, dtype: torch.dtype = torch.float16, device: str = "cpu") -> torch.Tensor:
    """Construct a normalized Regular Hadamard matrix of size n.

    A Regular Hadamard matrix has all row sums equal (±1 when normalized),
    which prevents row-wise outlier aggregation during rotation.
    Construction: Kronecker product of H_4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]
    normalized by 1/2 at each step. Must be a power of 4 (4, 16, 64, 256, ...).
    """
    if not _is_power_of_four(n):
        raise ValueError(f"Regular Hadamard requires power of 4, got {n}. Use rot_size=16/64/256.")

    # The 4×4 regular Hadamard kernel H_4 below is the matrix from
    # ConvRot (Ashkboos et al., arXiv:2512.03673), Theorem 3.2.
    # Each row sums to +1 (after the /2 normalisation), guaranteeing
    # the equal-row-sum property that defines a *regular* Hadamard matrix.
    H4 = torch.tensor([
        [ 1.0,  1.0,  1.0, -1.0],
        [ 1.0,  1.0, -1.0,  1.0],
        [ 1.0, -1.0,  1.0,  1.0],
        [-1.0,  1.0,  1.0,  1.0],
    ], dtype=dtype, device=device) / 2.0

    H = H4
    while H.shape[0] < n:
        H = torch.kron(H, H4)

    return H


def get_hadamard(size: int, dtype: torch.dtype = torch.float16, device: str = "cpu") -> torch.Tensor:
    """Get a cached Regular Hadamard matrix, constructing if needed."""
    key = (size, str(dtype), device)
    if key not in _hadamard_cache:
        _hadamard_cache[key] = make_hadamard_regular(size, dtype=dtype, device=device)
    return _hadamard_cache[key]


def clear_hadamard_cache() -> None:
    """Free all cached Hadamard matrices.

    Call between ``quantize_model()`` invocations to prevent unbounded
    memory growth when quantizing multiple models in one process.
    """
    _hadamard_cache.clear()


def _apply_hadamard(x: torch.Tensor, rot_size: int) -> torch.Tensor:
    """Apply Hadamard transform to the last dimension of x.

    Uses Regular Hadamard for powers of 4 (ConvRot paper) to prevent
    row-wise outlier aggregation. Falls back to Sylvester for non-power-of-4.
    """
    if _is_power_of_four(rot_size):
        H = get_hadamard(rot_size, dtype=x.dtype, device=str(x.device))
        return x @ H.T
    else:
        H = make_hadamard_sylvester(rot_size, dtype=x.dtype, device=str(x.device))
        return x @ H.T


def rotate_weights(W: torch.Tensor, rot_size: int) -> torch.Tensor:
    """Apply group-wise Hadamard Transform to a weight matrix.

    Partitions the in_features dimension into groups of rot_size and applies
    the Hadamard Transform to each group.

    Args:
        W: [out_features, in_features] weight matrix
        rot_size: group size (power of 4 for fallback, any power of 2 for fast path)

    Returns:
        W_rotated: [out_features, in_features_padded] where in_features_padded
                   is rounded up to the next multiple of rot_size
    """
    if W.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {W.dim()}D")

    if not _is_power_of_two(rot_size):
        raise ValueError(f"rot_size must be a power of 2, got {rot_size}")

    out_features, in_features = W.shape

    if in_features % rot_size != 0:
        pad_size = rot_size - (in_features % rot_size)
        W = F.pad(W, (0, pad_size))

    num_groups = W.shape[1] // rot_size
    W_grouped = W.reshape(out_features, num_groups, rot_size)
    W_rotated = _apply_hadamard(W_grouped, rot_size)

    return W_rotated.reshape(out_features, -1)


def rotate_activations(x: torch.Tensor, rot_size: int) -> torch.Tensor:
    """Apply group-wise Hadamard Transform to activation tensors.

    Args:
        x: [..., in_features] activation tensor (any leading dims)
        rot_size: group size (power of 4 for fallback, any power of 2 for fast path)

    Returns:
        x_rotated: [..., in_features_padded]
    """
    if not _is_power_of_two(rot_size):
        raise ValueError(f"rot_size must be a power of 2, got {rot_size}")

    orig_features = x.shape[-1]

    if orig_features % rot_size != 0:
        pad_size = rot_size - (orig_features % rot_size)
        x = F.pad(x, (0, pad_size))

    in_features = x.shape[-1]
    num_groups = in_features // rot_size
    leading_shape = x.shape[:-1]
    x_flat = x.reshape(-1, num_groups, rot_size)
    x_rotated = _apply_hadamard(x_flat, rot_size)

    return x_rotated.reshape(*leading_shape, in_features)


def rotate_hessian(H: torch.Tensor, rot_size: int) -> torch.Tensor:
    """Rotate a Hessian matrix to match rotated weight space: H_rot = R^T @ H @ R.

    The calibration Hessian is computed from unrotated activations (original model).
    Since the weights are rotated as W_rot = W @ R, the Hessian must be transformed
    as H_rot = R^T @ H @ R so that GPTQ's error compensation is in the correct space.

    R is a block-diagonal matrix of Hadamard blocks (orthogonal: R^T @ R = I).

    Supports both full Hessians [in, in] and block-diagonal Hessians [num_blocks, bs, bs].

    Args:
        H: Hessian matrix — 2D [in, in] or 3D [num_blocks, bs, bs]
        rot_size: Hadamard block size

    Returns:
        H_rotated: Hessian in rotated weight space (same shape as input)
    """
    if not _is_power_of_two(rot_size):
        raise ValueError(f"rot_size must be a power of 2, got {rot_size}")

    H = H.float()

    if H.dim() == 2:
        in_features = H.shape[0]
        if H.shape[0] != H.shape[1]:
            raise ValueError(f"Expected square Hessian, got {H.shape}")

        # Pad symmetrically if dimension is not divisible by rot_size
        if in_features % rot_size != 0:
            pad = rot_size - (in_features % rot_size)
            H = F.pad(H, (0, pad, 0, pad))
            in_features = H.shape[0]

        n_blocks = in_features // rot_size
        # [i, bi, j, bj] -> [i, j, bi, bj] so blocks are contiguous
        H_4d = H.reshape(n_blocks, rot_size, n_blocks, rot_size).permute(0, 2, 1, 3)
        flat = H_4d.reshape(n_blocks * n_blocks, rot_size, rot_size)
        flat = _transform_hessian_blocks(flat, rot_size)
        H_rot = flat.reshape(n_blocks, n_blocks, rot_size, rot_size).permute(0, 2, 1, 3).reshape(in_features, in_features)

        # Keep the padded size so H_rot.shape matches the padded weight
        # matrix produced by rotate_weights().  The padded zero-energy
        # columns are handled by GPTQ's damping.
        return H_rot

    elif H.dim() == 3:
        num_blocks, bs, bs2 = H.shape
        if bs != bs2:
            raise ValueError(f"Expected square blocks, got block shape ({bs}, {bs2})")
        if bs % rot_size != 0:
            raise ValueError(
                f"Block size {bs} is not divisible by rot_size {rot_size}"
            )

        if bs == rot_size:
            # Blocks already at rot_size — transform directly
            return _transform_hessian_blocks(H, rot_size)
        else:
            # Sub-block each block, transform, reassemble
            n_sub = bs // rot_size
            # [n, i, bi, j, bj] -> [n, i, j, bi, bj] so sub-blocks are contiguous
            H_5d = H.reshape(num_blocks, n_sub, rot_size, n_sub, rot_size)
            H_perm = H_5d.permute(0, 1, 3, 2, 4)
            flat = H_perm.reshape(num_blocks * n_sub * n_sub, rot_size, rot_size)
            flat = _transform_hessian_blocks(flat, rot_size)
            return flat.reshape(num_blocks, n_sub, n_sub, rot_size, rot_size).permute(0, 1, 3, 2, 4).reshape(num_blocks, bs, bs)

    else:
        raise ValueError(f"Expected 2D or 3D Hessian, got {H.dim()}D")


def _transform_hessian_blocks(H_3d: torch.Tensor, rot_size: int) -> torch.Tensor:
    """Apply R^T @ block @ R to each [rot_size, rot_size] block.

    Args:
        H_3d: [N, rot_size, rot_size] batch of Hessian blocks
        rot_size: Hadamard block size

    Returns:
        [N, rot_size, rot_size] transformed blocks
    """
    R = get_hadamard(rot_size, dtype=H_3d.dtype, device=str(H_3d.device))
    return torch.einsum("kp, nkl, le -> npe", R, H_3d, R)
