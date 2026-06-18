"""Regression guards: tests that catch things going wrong if code changes.

Unlike bug-proving tests, these assert CORRECT behavior.  If a change
breaks one of these invariants, the test fails — catching regressions
before they reach production.
"""

import torch
import pytest

from converter.config import FP16_SCALE_FLOOR, MAX_FP16_SCALE
from converter.gptq import gptq_quantize_layer, gptq_quantize_layer_rtn, _prepare_hinv
from converter.ldlq import ldlq_quantize_layer
from converter.rotation import rotate_weights
from converter.scales import (
    calculate_scales, quantize_weights,
    calculate_scales_int8, quantize_weights_int8,
)
from converter.rounding import _invert_hessian


# ── GPTQ Hessian preparation ────────────────────────────────────────────────


class TestGPTQHessianCorrectness:
    """GPTQ must use the actual inverse Hessian, not a Cholesky factor."""

    def test_prepare_hinv_returns_inverse_not_cholesky(self):
        """_prepare_hinv should return H_inv, not L.T from Cholesky(H_inv)."""
        torch.manual_seed(42)
        N = 64
        X = torch.randn(128, N)
        H = X.T @ X / 128

        H_inv = _prepare_hinv(H, damping=0.01, device=H.device)

        # H_inv should satisfy H @ H_inv ≈ I
        product = H.float() @ H_inv
        identity = torch.eye(N)
        assert torch.allclose(product, identity, atol=0.1), (
            "GPTQ Hessian preparation did not return a proper inverse"
        )

    def test_gptq_hinv_diagonal_matches_inverse_diagonal(self):
        """The diagonal of the prepared Hessian should match inv(H).diagonal()."""
        torch.manual_seed(42)
        N = 32
        X = torch.randn(64, N)
        H = X.T @ X / 64

        H_inv_prepared = _prepare_hinv(H, damping=0.01, device=H.device)
        H_inv_direct = torch.linalg.inv(H + 0.01 * H.diagonal().mean() * torch.eye(N))

        # Diagonals should be close (they're the same matrix)
        diag_prepared = H_inv_prepared.diagonal()
        diag_direct = H_inv_direct.diagonal()
        max_rel_diff = ((diag_prepared - diag_direct).abs() / diag_direct.abs().clamp(min=1e-12)).max().item()
        assert max_rel_diff < 0.1, (
            f"Prepared H_inv diagonal differs from direct inverse: "
            f"max relative diff = {max_rel_diff:.4f}"
        )

    def test_gptq_matches_correct_hinv(self):
        """GPTQ output should closely match what you get with correct H_inv."""
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        X = torch.randn(128, 64)
        H = X.T @ X / 128

        result_lib = gptq_quantize_layer(W, H, int_bits=4)

        # Dequantize properly
        if result_lib.zero_points is not None:
            W_deq = (result_lib.quantized_W.float() - result_lib.zero_points.float()) * result_lib.scales.float()
        else:
            W_deq = result_lib.quantized_W.float() * result_lib.scales.float()
        mse_lib = (W - W_deq).pow(2).mean().item()

        # The library should produce reasonable MSE (not garbage from wrong H_inv)
        assert mse_lib < 1.0, f"GPTQ MSE suspiciously high: {mse_lib:.4f}"


# ── Float16 scale safety ────────────────────────────────────────────────────


class TestFloat16ScaleSafety:
    """Scales must survive float16 conversion without underflow or overflow."""

    def test_fp16_scale_floor_survives_conversion(self):
        """FP16_SCALE_FLOOR must survive float32 -> float16 -> float32."""
        scale = torch.tensor(FP16_SCALE_FLOOR, dtype=torch.float32)
        scale_fp16 = scale.to(torch.float16)
        assert scale_fp16.item() > 0.0, (
            f"FP16_SCALE_FLOOR ({FP16_SCALE_FLOOR}) underflows to 0.0 in float16"
        )

    def test_fp16_scale_floor_round_trip_accuracy(self):
        scale = torch.tensor(FP16_SCALE_FLOOR, dtype=torch.float32)
        round_trip = scale.to(torch.float16).to(torch.float32)
        rel_error = abs(round_trip.item() - FP16_SCALE_FLOOR) / FP16_SCALE_FLOOR
        assert rel_error < 0.05, (
            f"FP16_SCALE_FLOOR round-trip error too large: {rel_error*100:.1f}%"
        )

    def test_max_fp16_scale_below_fp16_max(self):
        assert MAX_FP16_SCALE < torch.finfo(torch.float16).max

    def test_max_fp16_scale_above_typical(self):
        assert MAX_FP16_SCALE > 1.0

    def test_extreme_weights_produce_finite_scales(self):
        """Even extreme weight values should produce finite float16 scales."""
        W = torch.full((4, 64), 1e8)
        for calc_fn in [calculate_scales, calculate_scales_int8]:
            scales = calc_fn(W)
            assert scales.dtype == torch.float16
            assert scales.isfinite().all()
            assert scales.min() > 0
            assert scales.max() <= MAX_FP16_SCALE

    def test_gptq_scales_always_finite(self):
        """GPTQ must always produce finite scales, even with near-zero weights."""
        torch.manual_seed(42)
        W = torch.randn(16, 64) * 1e-9
        X = torch.randn(64, 64)
        H = X.T @ X

        result = gptq_quantize_layer(W, H, int_bits=8)
        assert result.scales.isfinite().all()
        assert (result.scales == 0).sum().item() == 0
        assert result.scales.min() > 0

    def test_ldlq_scales_always_finite(self):
        torch.manual_seed(42)
        W = torch.randn(16, 64) * 1e-9
        result = ldlq_quantize_layer(W, int_bits=8)
        assert result.scales.isfinite().all()
        assert (result.scales == 0).sum().item() == 0

    def test_scale_values_above_floor_survive_fp16(self):
        """All values at or above FP16_SCALE_FLOOR must survive fp16 conversion."""
        for val in [FP16_SCALE_FLOOR, 1e-5, 1e-4, 1e-3, 1.0]:
            t32 = torch.tensor(val, dtype=torch.float32)
            t16 = t32.to(torch.float16)
            assert t16.item() > 0.0, f"Value {val} underflowed to {t16.item()} in fp16"


# ── Rotated weight quantization (no truncation) ─────────────────────────────


class TestRotatedWeightQuantization:
    """Quantizing rotated weights must quantize ALL padded columns."""

    def test_no_zero_columns_in_quantized_rotated_weight(self):
        """Quantized rotated weight must NOT have zero-padded columns."""
        torch.manual_seed(42)
        W = torch.randn(64, 128, dtype=torch.float32)
        W_rot = rotate_weights(W, rot_size=256)
        padded_in = W_rot.shape[1]
        in_features = 128

        assert padded_in > in_features, "Test requires padding"

        _qr = ldlq_quantize_layer(W_rot, int_bits=8)
        padded_cols = _qr.quantized_W[:, in_features:]
        assert not torch.all(padded_cols == 0), (
            "Padded columns are all zeros — quantizer likely truncated then zero-padded"
        )

    def test_full_quantize_has_lower_mse_than_truncated(self):
        """Quantizing the full rotated weight must give lower MSE than truncating."""
        torch.manual_seed(42)
        W = torch.randn(64, 128, dtype=torch.float32)
        W_rot = rotate_weights(W, rot_size=256)
        padded_in = W_rot.shape[1]

        # Full: quantize all columns
        _qr_full = ldlq_quantize_layer(W_rot, int_bits=8)
        dequant_full = _qr_full.quantized_W.float() * _qr_full.scales.float()
        mse_full = (W_rot - dequant_full).pow(2).mean().item()

        # Truncated: quantize only original columns, zero-pad rest
        W_orig_cols = W_rot[:, :128]
        _qr_trunc = ldlq_quantize_layer(W_orig_cols, int_bits=8)
        q_trunc_padded = torch.nn.functional.pad(
            _qr_trunc.quantized_W, (0, padded_in - 128)
        )
        dequant_trunc = q_trunc_padded.float() * _qr_trunc.scales.float()
        mse_trunc = (W_rot - dequant_trunc).pow(2).mean().item()

        assert mse_full <= mse_trunc + 1e-6, (
            f"Full quantize MSE ({mse_full:.6f}) should be <= truncated MSE ({mse_trunc:.6f})"
        )

    def test_rtn_also_quantizes_full_rotated_weight(self):
        """RTN must also quantize the full rotated weight (no truncation)."""
        torch.manual_seed(42)
        W = torch.randn(64, 128, dtype=torch.float32)
        W_rot = rotate_weights(W, rot_size=256)
        padded_in = W_rot.shape[1]

        scales = calculate_scales_int8(W_rot)
        quantized = quantize_weights_int8(W_rot, scales)

        assert quantized.shape == (64, padded_in)
        assert not torch.all(quantized[:, 128:] == 0)


# ── Quantization output invariants ───────────────────────────────────────────


class TestQuantizationInvariants:
    """Cross-cutting invariants that must hold for all quantization methods."""

    @pytest.mark.parametrize("method,args", [
        ("rtn_int4", {}),
        ("rtn_int8", {}),
        ("ldlq_int4", {}),
        ("ldlq_int8", {}),
    ])
    def test_output_dtype_is_int8(self, method, args):
        W = torch.randn(16, 64)
        if method == "rtn_int4":
            scales = calculate_scales(W)
            q = quantize_weights(W, scales)
        elif method == "rtn_int8":
            scales = calculate_scales_int8(W)
            q = quantize_weights_int8(W, scales)
        elif method == "ldlq_int4":
            q = ldlq_quantize_layer(W).quantized_W
        elif method == "ldlq_int8":
            q = ldlq_quantize_layer(W, int_bits=8).quantized_W
        assert q.dtype == torch.int8

    @pytest.mark.parametrize("method,args,lo,hi", [
        ("rtn_int4", {}, -8, 7),
        ("rtn_int8", {}, -128, 127),
        ("ldlq_int4", {}, -8, 7),
        ("ldlq_int8", {}, -128, 127),
        ("gptq_int4", {}, -8, 7),
        ("gptq_int8", {}, -128, 127),
    ])
    def test_output_range(self, method, args, lo, hi):
        W = torch.randn(16, 64)
        if method == "rtn_int4":
            scales = calculate_scales(W)
            q = quantize_weights(W, scales)
        elif method == "rtn_int8":
            scales = calculate_scales_int8(W)
            q = quantize_weights_int8(W, scales)
        elif method == "ldlq_int4":
            q = ldlq_quantize_layer(W).quantized_W
        elif method == "ldlq_int8":
            q = ldlq_quantize_layer(W, int_bits=8).quantized_W
        elif method == "gptq_int4":
            H = torch.randn(64, 64).T @ torch.randn(64, 64)
            q = gptq_quantize_layer(W, H).quantized_W
        elif method == "gptq_int8":
            H = torch.randn(64, 64).T @ torch.randn(64, 64)
            q = gptq_quantize_layer(W, H, int_bits=8).quantized_W
        assert q.min() >= lo, f"{method}: min {q.min()} < {lo}"
        assert q.max() <= hi, f"{method}: max {q.max()} > {hi}"

    @pytest.mark.parametrize("method", ["rtn", "ldlq", "gptq"])
    def test_zero_weights_quantize_to_zero(self, method):
        W = torch.zeros(8, 32)
        if method == "rtn":
            scales = calculate_scales(W)
            q = quantize_weights(W, scales)
        elif method == "ldlq":
            q = ldlq_quantize_layer(W).quantized_W
        elif method == "gptq":
            H = torch.randn(32, 32).T @ torch.randn(32, 32)
            q = gptq_quantize_layer(W, H).quantized_W
        assert torch.all(q == 0)

    @pytest.mark.parametrize("method", ["rtn", "ldlq", "gptq"])
    def test_scales_positive_finite(self, method):
        W = torch.randn(16, 64)
        if method == "rtn":
            scales = calculate_scales(W)
        elif method == "ldlq":
            result = ldlq_quantize_layer(W)
            _ = result.quantized_W
            scales = result.scales
        elif method == "gptq":
            H = torch.randn(64, 64).T @ torch.randn(64, 64)
            result = gptq_quantize_layer(W, H)
            _ = result.quantized_W
            scales = result.scales
        assert scales.dtype == torch.float16
        assert scales.isfinite().all()
        assert scales.min() > 0
