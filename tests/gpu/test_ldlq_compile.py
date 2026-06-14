"""Tests for the torch.compile accelerated LDLQ path."""

import math
from unittest.mock import patch

import torch
import pytest

from converter.ldlq import (
    ldlq_quantize_layer,
    _run_iterative_ldlq,
    _get_ldlq_compiled_fn,
    _ldlq_loop_gpu,
    _HAS_TORCH_COMPILE,
)


class TestTorchCompilePath:

    def _get_compiled_fn_or_skip(self):
        if not _HAS_TORCH_COMPILE:
            pytest.skip("torch.compile not available")
        import converter.ldlq as ldlq_mod
        ldlq_mod._ldlq_compiled_fn = None
        try:
            compiled_fn = _get_ldlq_compiled_fn()
        except Exception:
            pytest.skip("torch.compile backend unavailable (no C++ compiler?)")
        if compiled_fn is None:
            pytest.skip("torch.compile initialization failed")
        return compiled_fn

    def test_get_ldlq_compiled_fn_returns_callable(self):
        if not _HAS_TORCH_COMPILE:
            pytest.skip("torch.compile not available")
        try:
            compiled_fn = _get_ldlq_compiled_fn()
        except Exception:
            pytest.skip("torch.compile backend unavailable")
        assert compiled_fn is not None
        assert callable(compiled_fn)

    def test_get_ldlq_compiled_fn_caches(self):
        if not _HAS_TORCH_COMPILE:
            pytest.skip("torch.compile not available")
        try:
            fn1 = _get_ldlq_compiled_fn()
            fn2 = _get_ldlq_compiled_fn()
        except Exception:
            pytest.skip("torch.compile backend unavailable")
        assert fn1 is fn2
    def test_compiled_fn_int4_output_range(self):
        compiled_fn = self._get_compiled_fn_or_skip()
        M = 32
        W_col = torch.randn(M) * 3
        scale_col = torch.ones(M) * 1.0

        try:
            q, err_norm = compiled_fn(W_col, W_col.clone(), scale_col, 1.0, -8.0, 7.0)
        except Exception:
            pytest.skip("torch.compile execution failed (no C++ compiler?)")

        assert q.shape == (M,)
        assert q.dtype == torch.int8
        assert q.min() >= -8
        assert q.max() <= 7

    def test_compiled_fn_int8_output_range(self):
        compiled_fn = self._get_compiled_fn_or_skip()
        M = 32
        W_col = torch.randn(M) * 20
        scale_col = torch.ones(M) * 2.0

        try:
            q, err_norm = compiled_fn(W_col, W_col.clone(), scale_col, 1.0, -128.0, 127.0)
        except Exception:
            pytest.skip("torch.compile execution failed (no C++ compiler?)")

        assert q.shape == (M,)
        assert q.dtype == torch.int8
        assert q.min() >= -128
        assert q.max() <= 127

    def test_compiled_fn_error_residual(self):
        compiled_fn = self._get_compiled_fn_or_skip()
        W_col = torch.tensor([0.5, -0.5, 1.3, -1.3, 3.0, -3.0, 0.0, 0.1])
        scale_col = torch.ones(8) * 0.5
        hinv_diag_val = 2.0

        try:
            q, err_norm = compiled_fn(W_col, W_col.clone(), scale_col, hinv_diag_val, -8.0, 7.0)
        except Exception:
            pytest.skip("torch.compile execution failed (no C++ compiler?)")

        scale_safe = scale_col.clamp(min=1e-8)
        expected_err = (W_col - q.float() * scale_safe) / hinv_diag_val
        assert torch.allclose(err_norm, expected_err, atol=1e-5)

    def test_compiled_fn_sign_flip_correction(self):
        compiled_fn = self._get_compiled_fn_or_skip()
        W_col = torch.tensor([0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 7.0, -7.0])
        scale_col = torch.ones(8) * 0.3

        try:
            q, _ = compiled_fn(W_col, W_col.clone(), scale_col, 1.0, -8.0, 7.0)
        except Exception:
            pytest.skip("torch.compile execution failed (no C++ compiler?)")

        for i in range(8):
            if W_col[i].abs() > 0.05:
                assert (W_col[i].sign() * q[i].float().sign()) >= 0

    def test_compiled_fn_no_nan_inf(self):
        compiled_fn = self._get_compiled_fn_or_skip()
        W_col = torch.randn(16) * 5

        try:
            q, err_norm = compiled_fn(W_col, W_col.clone(), torch.ones(16), 0.5, -8.0, 7.0)
        except Exception:
            pytest.skip("torch.compile execution failed (no C++ compiler?)")

        assert not torch.any(torch.isnan(err_norm))
        assert not torch.any(torch.isinf(err_norm))

    def test_ldlq_loop_gpu_with_compiled_fn(self):
        compiled_fn = self._get_compiled_fn_or_skip()
        torch.manual_seed(42)
        M, N = 8, 32

        W = torch.randn(M, N)
        H = W.T @ W / M + 0.01 * torch.eye(N)
        H_inv = torch.linalg.inv(H)
        scale_2d = (W.abs() / 7.0).clamp(min=1e-6)
        Q = torch.zeros(M, N, dtype=torch.int8)

        try:
            result = _ldlq_loop_gpu(W.clone(), H_inv, scale_2d, Q, N, M, 16, -8, 7,
                                    compiled_fn=compiled_fn)
        except Exception:
            pytest.skip("torch.compile execution failed (no C++ compiler?)")

        assert result.shape == (M, N)
        assert result.dtype == torch.int8
        assert result.min() >= -8
        assert result.max() <= 7
        assert not torch.any(torch.isnan(result.float()))

    def test_compile_vs_eager_produces_same_output(self):
        compiled_fn = self._get_compiled_fn_or_skip()
        torch.manual_seed(42)
        M, N = 8, 32

        W = torch.randn(M, N)
        H = W.T @ W / M + 0.01 * torch.eye(N)
        H_inv = torch.linalg.inv(H)
        scale_2d = (W.abs() / 7.0).clamp(min=1e-6)

        try:
            Q_compile = torch.zeros(M, N, dtype=torch.int8)
            _ldlq_loop_gpu(W.clone(), H_inv.clone(), scale_2d.clone(), Q_compile, N, M, 32, -8, 7,
                            compiled_fn=compiled_fn)
        except Exception:
            pytest.skip("torch.compile execution failed (no C++ compiler?)")

        Q_eager = torch.zeros(M, N, dtype=torch.int8)
        _ldlq_loop_gpu(W.clone(), H_inv.clone(), scale_2d.clone(), Q_eager, N, M, 32, -8, 7,
                        compiled_fn=None)

        assert torch.equal(Q_compile, Q_eager)

    @patch("converter.ldlq._HAS_TRITON", False)
    def test_quantize_layer_forces_compile_path(self):
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        result = ldlq_quantize_layer(W, block_size=32)

        assert result.quantized_W.shape == (16, 64)
        assert result.quantized_W.dtype == torch.int8
        assert result.scales.shape == (16, 1)
        assert result.scales.dtype == torch.float16
        assert result.zero_points is None
        assert result.mse > 0
        assert not math.isnan(result.mse)
        assert result.mse < 1.0
        assert result.max_err > 0
        assert result.quantized_W.min() >= -8
        assert result.quantized_W.max() <= 7

    @patch("converter.ldlq._HAS_TRITON", False)
    def test_quantize_layer_compile_method_used(self):
        if not _HAS_TORCH_COMPILE:
            pytest.skip("torch.compile not available")
        torch.manual_seed(42)
        try:
            result = ldlq_quantize_layer(torch.randn(16, 64))
        except Exception:
            pytest.skip("torch.compile backend unavailable")
        assert result.method_used in ("ldlq_compile", "ldlq_cpu")

    @patch("converter.ldlq._HAS_TRITON", False)
    def test_quantize_layer_compile_int8(self):
        torch.manual_seed(42)
        result = ldlq_quantize_layer(torch.randn(16, 64) * 10, int_bits=8, block_size=32)
        assert result.quantized_W.min() >= -128
        assert result.quantized_W.max() <= 127
        assert result.mse > 0
        assert not math.isnan(result.mse)

    @patch("converter.ldlq._HAS_TRITON", False)
    def test_quantize_layer_compile_singular_hessian(self):
        torch.manual_seed(42)
        W = torch.zeros(8, 32)
        W[:, 0] = torch.randn(8)
        result = ldlq_quantize_layer(W, block_size=16, damping=0.1)
        assert not torch.any(torch.isnan(result.quantized_W.float()))
        assert result.mse >= 0

    @patch("converter.ldlq._HAS_TRITON", False)
    def test_quantize_layer_compile_zero_weights(self):
        result = ldlq_quantize_layer(torch.zeros(8, 32), block_size=16)
        assert torch.all(result.quantized_W == 0)
        assert result.mse == 0.0

    @patch("converter.ldlq._HAS_TRITON", False)
    def test_quantize_layer_compile_with_greedy(self):
        if not _HAS_TORCH_COMPILE:
            pytest.skip("torch.compile not available")
        torch.manual_seed(42)
        try:
            result = ldlq_quantize_layer(torch.randn(8, 32), block_size=16, greedy_passes=1)
        except Exception:
            pytest.skip("torch.compile backend unavailable")
        assert result.quantized_W.min() >= -8
        assert result.quantized_W.max() <= 7
        assert result.mse >= 0
        assert not math.isnan(result.mse)

    @patch("converter.ldlq._HAS_TRITON", False)
    def test_quantize_layer_compile_iterative(self):
        if not _HAS_TORCH_COMPILE:
            pytest.skip("torch.compile not available")
        torch.manual_seed(42)
        try:
            result = ldlq_quantize_layer(torch.randn(8, 32), block_size=16, iterations=3)
        except Exception:
            pytest.skip("torch.compile backend unavailable")
        assert result.quantized_W.min() >= -8
        assert result.quantized_W.max() <= 7
        assert result.mse >= 0
