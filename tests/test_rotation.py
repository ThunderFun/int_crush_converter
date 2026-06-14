"""Tests for quant.rotation — Regular Hadamard construction and group-wise RHT."""

import math
import torch
import pytest

import converter.rotation as rotation_mod
from converter.rotation import (
    _is_power_of_four,
    make_hadamard_regular,
    make_hadamard_sylvester,
    get_hadamard,
    rotate_weights,
    rotate_activations,
)


class TestIsPowerOfFour:
    def test_powers_of_four(self):
        for n in [4, 16, 64, 256, 1024]:
            assert _is_power_of_four(n)

    def test_not_powers_of_four(self):
        for n in [1, 2, 8, 32, 128, 512, 3, 5, 100]:
            assert not _is_power_of_four(n)


class TestMakeHadamardRegular:
    def test_shape(self):
        for n in [4, 16, 64]:
            H = make_hadamard_regular(n)
            assert H.shape == (n, n)

    def test_orthogonality(self):
        """H @ H.T should equal identity (scaled by 1/sqrt(n), so H @ H.T = I/n * n = I)."""
        for n in [4, 16, 64]:
            H = make_hadamard_regular(n, dtype=torch.float32)
            product = H @ H.T
            identity = torch.eye(n, dtype=torch.float32)
            assert torch.allclose(product, identity, atol=1e-5), (
                f"H @ H.T != I for n={n}, max diff: {(product - identity).abs().max()}"
            )

    def test_values_in_range(self):
        """All values should be ±1/sqrt(n)."""
        for n in [4, 16, 64]:
            H = make_hadamard_regular(n)
            val = 1.0 / math.sqrt(n)
            assert torch.allclose(H.abs(), torch.full_like(H, val), atol=1e-5)

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            make_hadamard_regular(12)  # not power of 2


class TestGetHadamard:
    def test_caching(self):
        H1 = get_hadamard(64)
        H2 = get_hadamard(64)
        assert H1 is H2  # same object from cache


class TestRotateWeights:
    def test_shape_no_padding(self):
        W = torch.randn(32, 64)
        W_rot = rotate_weights(W, rot_size=16)
        assert W_rot.shape == (32, 64)

    def test_shape_with_padding(self):
        W = torch.randn(32, 50)
        W_rot = rotate_weights(W, rot_size=16)
        assert W_rot.shape[0] == 32
        assert W_rot.shape[1] % 16 == 0  # padded to multiple of 16

    def test_preserves_norm(self):
        """Rotation is orthogonal, so Frobenius norm should be preserved."""
        W = torch.randn(16, 64, dtype=torch.float32)
        W_rot = rotate_weights(W, rot_size=64)
        assert torch.allclose(W.norm(), W_rot.norm(), rtol=1e-4)

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            rotate_weights(torch.randn(10), rot_size=16)


class TestRotateActivations:
    def test_shape_2d(self):
        x = torch.randn(8, 64)
        x_rot = rotate_activations(x, rot_size=16)
        assert x_rot.shape == (8, 64)

    def test_shape_3d(self):
        x = torch.randn(2, 8, 64)
        x_rot = rotate_activations(x, rot_size=16)
        assert x_rot.shape == (2, 8, 64)

    def test_shape_with_padding(self):
        x = torch.randn(8, 50)
        x_rot = rotate_activations(x, rot_size=16)
        assert x_rot.shape[-1] % 16 == 0

    def test_preserves_norm(self):
        x = torch.randn(8, 64, dtype=torch.float32)
        x_rot = rotate_activations(x, rot_size=64)
        assert torch.allclose(x.norm(), x_rot.norm(), rtol=1e-4)
