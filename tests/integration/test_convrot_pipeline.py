"""Full pipeline integration: ConvRot -> LDLQ -> pack -> unpack -> verify."""

import math
import torch

from converter.ldlq import ldlq_quantize_layer, _run_iterative_ldlq
from converter.rotation import rotate_weights
from converter.rounding import _invert_hessian
from converter.packing import pack_int4, unpack_int4


class TestConvRotPipeline:

    def test_convrot_ldlq_pack_unpack_roundtrip(self):
        torch.manual_seed(42)
        M, N = 32, 256
        rot_size = 64

        W = torch.randn(M, N)
        W_rot = rotate_weights(W, rot_size=rot_size)

        _qr = ldlq_quantize_layer(W_rot, block_size=32)
        packed = pack_int4(_qr.quantized_W)
        assert packed.shape == (M, N // 2)
        assert packed.dtype == torch.uint8

        unpacked = unpack_int4(packed, N)
        assert torch.equal(unpacked, _qr.quantized_W)

        W_deq = unpacked.float() * _qr.scales.float()
        assert (W_rot - W_deq).pow(2).mean().item() < 1.0

    def test_convrot_ldlq_iterative_pack(self):
        torch.manual_seed(42)
        M, N = 16, 128
        rot_size = 64

        W = torch.randn(M, N)
        W_rot = rotate_weights(W, rot_size=rot_size)

        H = W_rot.T @ W_rot / M
        diag_mean = H.diagonal().mean().clamp(min=1e-6)
        H_inv = _invert_hessian(H + 0.01 * diag_mean * torch.eye(N))
        row_scales = (W_rot.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-6)
        flat_scales = row_scales.expand(M, N).reshape(-1).clone()

        Q, _, _ = _run_iterative_ldlq(
            W_rot, H_inv, flat_scales, iterations=3, block_size=32,
            clamp_min=-8, clamp_max=7,
        )

        packed = pack_int4(Q)
        unpacked = unpack_int4(packed, N)
        assert unpacked.min() >= -8
        assert unpacked.max() <= 7

    def test_pipeline_with_different_rot_sizes(self):
        torch.manual_seed(42)
        M, N = 16, 256
        W = torch.randn(M, N)

        for rot_size in [16, 64, 256]:
            if N % rot_size != 0:
                continue
            W_rot = rotate_weights(W, rot_size=rot_size)
            _qr = ldlq_quantize_layer(W_rot, block_size=32)

            assert _qr.quantized_W.shape == (M, N)
            assert _qr.quantized_W.min() >= -8
            assert _qr.quantized_W.max() <= 7
            assert _qr.scales.shape == (M, 1)

            packed = pack_int4(_qr.quantized_W)
            unpacked = unpack_int4(packed, N)
            assert torch.equal(unpacked, _qr.quantized_W)

    def test_pipeline_asymmetric_layer_dimensions(self):
        torch.manual_seed(42)
        M, N = 32, 300
        rot_size = 64

        W = torch.randn(M, N)
        W_rot = rotate_weights(W, rot_size=rot_size)
        padded_N = W_rot.shape[1]
        assert padded_N == math.ceil(N / rot_size) * rot_size

        _qr = ldlq_quantize_layer(W_rot, block_size=32)
        assert _qr.quantized_W.shape == (M, padded_N)

        q_W_valid = _qr.quantized_W[:, :N]
        packed = pack_int4(q_W_valid)
        assert packed.shape == (M, N // 2)
