"""INT4 pack/unpack utilities.

Two INT4 values per byte, two's complement, uint8 storage.
"""

import torch

INT4_MIN = -8
INT4_MAX = 7
INT4_PACK_FACTOR = 2


class INT4PackingError(ValueError):
    """Raised when values are out of INT4 range during packing."""


def validate_int4_range(tensor: torch.Tensor) -> None:
    """Raise INT4PackingError if any values are outside [-8, 7]."""
    if tensor.numel() == 0:
        return
    if not torch.isfinite(tensor).all():
        raise INT4PackingError("Tensor contains NaN or Inf values")
    if tensor.min() < INT4_MIN or tensor.max() > INT4_MAX:
        raise INT4PackingError(
            f"Values must be in [{INT4_MIN}, {INT4_MAX}], "
            f"got [{tensor.min().item()}, {tensor.max().item()}]"
        )


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack INT4 values: two's complement, low nibble = even index, uint8.

    Args:
        values: [..., K] INT4 values in [-8, 7] (int8 dtype)

    Returns:
        packed: [..., ceil(K/2)] uint8, 2 INT4 values per byte
    """
    validate_int4_range(values)
    values = values.to(torch.int8)
    K = values.shape[-1]
    if K % 2 != 0:
        pad = torch.zeros(*values.shape[:-1], 1, dtype=values.dtype, device=values.device)
        values = torch.cat([values, pad], dim=-1)
    q_i8 = torch.where(values < 0, 2 ** 4 + values, values).to(torch.uint8)
    return q_i8[..., 0::2] | (q_i8[..., 1::2] << 4)


def unpack_int4(packed: torch.Tensor, K: int) -> torch.Tensor:
    """Unpack packed INT4 back to int8 values.

    Args:
        packed: [..., ceil(K/2)] uint8 packed INT4
        K: original last-dimension size

    Returns:
        values: [..., K] INT4 values as int8
    """
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    result = torch.zeros(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.int8, device=packed.device)
    result[..., 0::2] = low
    result[..., 1::2] = high
    return result[..., :K]
