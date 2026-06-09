"""End-to-end tests for the full INT-Crush quantize + load + forward pipeline."""

import torch
import pytest

from converter.rotation import rotate_weights
from converter.scales import calculate_scales, quantize_weights
from converter.packing import pack_int4, unpack_int4


class TestEndToEnd:
    def test_rotation_preserves_values(self):
        """Forward+inverse rotation should recover original values."""
        from converter.rotation import make_hadamard_regular

        W = torch.randn(8, 16, dtype=torch.float32)
        H = make_hadamard_regular(16, dtype=torch.float32)

        # H is orthogonal, so H @ H.T = I
        W_rot = W @ H.T
        W_recovered = W_rot @ H

        assert torch.allclose(W, W_recovered, atol=1e-5)

    def test_scales_after_rotation(self):
        """Scales should be valid after rotation."""
        W = torch.randn(16, 64, dtype=torch.float32)
        W_rot = rotate_weights(W, rot_size=16)
        scales = calculate_scales(W_rot)

        assert scales.shape == (16, 1)
        assert not torch.any(torch.isnan(scales))
        assert torch.all(scales > 0)

    def test_quantized_in_range(self):
        """Quantized values should always be in [-8, 7]."""
        W = torch.randn(32, 128, dtype=torch.float32)
        W_rot = rotate_weights(W, rot_size=64)
        scales = calculate_scales(W_rot)
        W_quant = quantize_weights(W_rot, scales)

        assert W_quant.min() >= -8
        assert W_quant.max() <= 7

    def test_quantize_pack_roundtrip(self):
        """Full pipeline: rotate -> scale -> quantize -> pack -> unpack -> verify."""
        W = torch.randn(16, 64, dtype=torch.float32)

        # 1. Rotate
        W_rot = rotate_weights(W, rot_size=16)

        # 2. Scale
        scales = calculate_scales(W_rot)

        # 3. Quantize
        W_quant = quantize_weights(W_rot, scales)

        # 4. Pack
        packed = pack_int4(W_quant)

        # 5. Unpack
        unpacked = unpack_int4(packed, K=W_quant.shape[1])

        # Verify shape and range
        assert unpacked.shape == W_quant.shape
        assert torch.equal(unpacked, W_quant)
