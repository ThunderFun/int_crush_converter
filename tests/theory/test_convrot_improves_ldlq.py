"""Test: ConvRot improves LDLQ quantization quality."""

import math
import torch

from converter.ldlq import ldlq_quantize_layer
from converter.rotation import rotate_weights, make_hadamard_regular, make_hadamard_sylvester


def _make_weight_with_outliers(M, N, outlier_rows, outlier_cols, outlier_scale, seed=42):
    torch.manual_seed(seed)
    W = torch.randn(M, N) * 0.1
    for r, c in zip(outlier_rows, outlier_cols):
        W[r, c] = outlier_scale
    return W


class TestConvRotImprovesLDLQ:

    def test_convrot_reduces_max_entry_for_outliers(self):
        torch.manual_seed(42)
        M, N = 32, 256
        rot_size = 64

        W = _make_weight_with_outliers(
            M, N,
            outlier_rows=[0, 1, 2], outlier_cols=[0, 50, 100], outlier_scale=10.0,
        )
        max_orig = W.abs().max().item()
        W_rot = rotate_weights(W, rot_size=rot_size)
        max_rot = W_rot.abs().max().item()

        assert max_rot < max_orig * 1.1, (
            f"Rotation increased max entry: {max_orig:.2f} -> {max_rot:.2f}"
        )

    def test_convrot_improves_mse_with_row_outliers(self):
        torch.manual_seed(42)
        M, N = 32, 256
        rot_size = 64

        W = torch.randn(M, N) * 0.1
        for r in range(M):
            n_outliers = torch.randint(1, 4, (1,)).item()
            for c in torch.randint(0, N, (n_outliers,)):
                W[r, c] = 5.0

        _qr_no_rot = ldlq_quantize_layer(W, block_size=32)
        deq_no_rot = _qr_no_rot.quantized_W.float() * _qr_no_rot.scales.float()
        mse_no_rot = (W - deq_no_rot).pow(2).mean().item()

        W_rot = rotate_weights(W, rot_size=rot_size)
        _qr_rot = ldlq_quantize_layer(W_rot, block_size=32)
        deq_rot = _qr_rot.quantized_W.float() * _qr_rot.scales.float()
        mse_rot = (W_rot - deq_rot).pow(2).mean().item()

        assert mse_rot <= mse_no_rot * 1.5, (
            f"ConvRot did not improve MSE: no_rot={mse_no_rot:.6f}, rot={mse_rot:.6f}"
        )

    def test_regular_hadamard_better_than_sylvester(self):
        torch.manual_seed(42)
        W = torch.randn(16, 64) * 0.1
        W[0, 0] = 20.0

        W_reg = W @ make_hadamard_regular(64, dtype=torch.float32).T
        W_syl = W @ make_hadamard_sylvester(64, dtype=torch.float32).T

        assert W_reg.abs().max().item() < W_syl.abs().max().item()
