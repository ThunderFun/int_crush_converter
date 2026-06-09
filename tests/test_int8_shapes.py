"""Shape tests for the INT8 converter output and loader forward pipeline.

Tests that:
  - Converter produces correct shapes for int8 weight + fp16 scale
  - Rotated + quantized weights preserve shapes through the full pipeline
  - INT8 dequant → linear → requant round-trip preserves shapes
  - Converter does not truncate rotated weights (padded columns must be quantized)
"""

import torch
import pytest

from converter.rotation import rotate_weights
from converter.ldlq import ldlq_quantize_layer
from converter.scales import calculate_scales_int8, quantize_weights_int8


# ── Converter output shapes ──────────────────────────────────────────────────


class TestConverterOutputShapes:

    @pytest.mark.parametrize("out_features,in_features", [
        (64, 128),
        (256, 256),
        (128, 64),
        (1, 64),
    ])
    def test_rtn_int8_weight_shape(self, out_features, in_features):
        """RTN INT8 quantization preserves [out, in] shape."""
        W = torch.randn(out_features, in_features, dtype=torch.float32)
        scales = calculate_scales_int8(W)
        quantized = quantize_weights_int8(W, scales)

        assert quantized.shape == (out_features, in_features)
        assert quantized.dtype == torch.int8

    @pytest.mark.parametrize("out_features,in_features", [
        (64, 128),
        (256, 256),
        (128, 64),
    ])
    def test_rtn_int8_scale_shape(self, out_features, in_features):
        """RTN INT8 scales are [out, 1] float16."""
        W = torch.randn(out_features, in_features, dtype=torch.float32)
        scales = calculate_scales_int8(W)

        assert scales.shape == (out_features, 1)
        assert scales.dtype == torch.float16

    @pytest.mark.parametrize("out_features,in_features,rot_size", [
        (64, 128, 128),
        (64, 128, 256),
        (128, 64, 64),
        (256, 256, 256),
    ])
    def test_rotated_weight_shape(self, out_features, in_features, rot_size):
        """Rotated weight shape is [out, padded_in] where padded_in >= in."""
        W = torch.randn(out_features, in_features, dtype=torch.float32)
        W_rot = rotate_weights(W, rot_size)

        assert W_rot.shape[0] == out_features
        assert W_rot.shape[1] >= in_features
        assert W_rot.shape[1] % rot_size == 0

    @pytest.mark.parametrize("out_features,in_features", [
        (64, 128),
        (256, 256),
        (128, 64),
    ])
    def test_ldlq_int8_weight_shape(self, out_features, in_features):
        """LDLQ INT8 output is [out, in] int8 with [out, 1] fp16 scales."""
        W = torch.randn(out_features, in_features, dtype=torch.float32)
        quantized, scales = ldlq_quantize_layer(W, int_bits=8)

        assert quantized.shape == (out_features, in_features)
        assert quantized.dtype == torch.int8
        assert scales.shape == (out_features, 1)
        assert scales.dtype == torch.float16

    @pytest.mark.parametrize("out_features,in_features,rot_size", [
        (64, 128, 256),
        (128, 64, 128),
    ])
    def test_rotated_ldlq_int8_weight_shape(self, out_features, in_features, rot_size):
        """Rotated + LDLQ INT8: converter pads then quantizes full padded width."""
        W = torch.randn(out_features, in_features, dtype=torch.float32)
        W_rot = rotate_weights(W, rot_size)
        padded_in = W_rot.shape[1]

        quantized, scales = ldlq_quantize_layer(W_rot, int_bits=8)

        assert quantized.shape == (out_features, padded_in)
        assert quantized.dtype == torch.int8
        assert scales.shape == (out_features, 1)
        assert scales.dtype == torch.float16


# ── Loader forward shape pipeline ────────────────────────────────────────────


def _dequantize_int8(weight_int8, scale):
    """Dequantize INT8 weight with per-row scales."""
    scale_2d = scale.reshape(-1, 1) if scale.ndim == 1 else scale
    return weight_int8.float() * scale_2d.float()


def _requantize_int8(w_float):
    """Re-quantize float weight to INT8 with per-row scales."""
    s_w = w_float.abs().amax(dim=1).to(torch.float32).clamp(min=1e-8) / 127.0
    q = w_float / s_w.unsqueeze(1)
    w_int8 = (torch.where(q >= 0, q + 0.5, q - 0.5)
              .clamp(-128, 127).to(torch.int8))
    return w_int8, s_w


class TestLoaderForwardShapes:

    @pytest.mark.parametrize("batch,in_features,out_features", [
        (1, 128, 256),
        (4, 256, 256),
        (2, 64, 128),
        (1, 1, 64),
    ])
    def test_fallback_forward_shape(self, batch, in_features, out_features):
        """PyTorch fallback forward produces correct [batch, out] shape."""
        weight_int8 = torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
        scale = torch.rand(out_features, dtype=torch.float32).clamp(min=0.001)
        x = torch.randn(batch, in_features, dtype=torch.float32)

        w_float = _dequantize_int8(weight_int8, scale)
        out = torch.nn.functional.linear(x, w_float)

        assert out.shape == (batch, out_features)

    @pytest.mark.parametrize("batch,in_features,out_features", [
        (1, 128, 256),
        (4, 256, 256),
        (2, 64, 128),
    ])
    def test_3d_input_forward_shape(self, batch, in_features, out_features):
        """Forward with 3D input [B, S, K] produces [B, S, out]."""
        seq_len = 16
        weight_int8 = torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
        scale = torch.rand(out_features, dtype=torch.float32).clamp(min=0.001)
        x = torch.randn(batch, seq_len, in_features, dtype=torch.float32)

        w_float = _dequantize_int8(weight_int8, scale)
        x_2d = x.reshape(-1, in_features)
        out_2d = torch.nn.functional.linear(x_2d, w_float)
        out = out_2d.reshape(batch, seq_len, out_features)

        assert out.shape == (batch, seq_len, out_features)

    @pytest.mark.parametrize("batch,in_features,out_features", [
        (1, 128, 256),
        (4, 256, 256),
        (2, 64, 128),
    ])
    def test_dequant_requant_roundtrip_shape(self, batch, in_features, out_features):
        """Dequant → requant preserves weight shape."""
        weight_int8 = torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
        scale = torch.rand(out_features, dtype=torch.float32).clamp(min=0.001)

        w_float = _dequantize_int8(weight_int8, scale)
        w_requant, s_requant = _requantize_int8(w_float)

        assert w_requant.shape == (out_features, in_features)
        assert w_requant.dtype == torch.int8
        assert s_requant.shape == (out_features,)

    @pytest.mark.parametrize("out_features,in_features,rot_size", [
        (64, 128, 256),
        (128, 64, 128),
        (256, 256, 256),
    ])
    def test_rotated_forward_shape(self, out_features, in_features, rot_size):
        """Rotated forward: pad → hadamard → linear → correct output shape."""
        from converter.rotation import rotate_weights, rotate_activations

        batch = 2
        W = torch.randn(out_features, in_features, dtype=torch.float32)
        W_rot = rotate_weights(W, rot_size)
        padded_in = W_rot.shape[1]

        scales = calculate_scales_int8(W_rot)
        quantized = quantize_weights_int8(W_rot, scales)

        x = torch.randn(batch, in_features, dtype=torch.float32)
        x_rot = rotate_activations(x, rot_size)

        assert x_rot.shape == (batch, padded_in)

        w_float = _dequantize_int8(quantized, scales.reshape(-1))
        out = torch.nn.functional.linear(x_rot, w_float)

        assert out.shape == (batch, out_features)

    @pytest.mark.parametrize("out_features,in_features", [
        (64, 128),
        (256, 256),
        (128, 64),
    ])
    def test_bias_forward_shape(self, out_features, in_features):
        """Forward with bias produces correct shape."""
        weight_int8 = torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
        scale = torch.rand(out_features, dtype=torch.float32).clamp(min=0.001)
        bias = torch.randn(out_features, dtype=torch.float32)
        x = torch.randn(1, in_features, dtype=torch.float32)

        w_float = _dequantize_int8(weight_int8, scale)
        out = torch.nn.functional.linear(x, w_float, bias)

        assert out.shape == (1, out_features)

    def test_cat_shapes_after_linear(self):
        """Simulate the Flux torch.cat scenario: two linear outputs must match on non-cat dims."""
        in_features = 128
        out_features = 256

        # Both layers output the same out_features
        w1_int8 = torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
        w2_int8 = torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
        s1 = torch.rand(out_features, dtype=torch.float32).clamp(min=0.001)
        s2 = torch.rand(out_features, dtype=torch.float32).clamp(min=0.001)

        x1 = torch.randn(1, 16, in_features)   # img tokens
        x2 = torch.randn(1, 8, in_features)     # kontext tokens

        w1_float = _dequantize_int8(w1_int8, s1)
        w2_float = _dequantize_int8(w2_int8, s2)

        out1 = torch.nn.functional.linear(x1.reshape(-1, in_features), w1_float).reshape(1, 16, out_features)
        out2 = torch.nn.functional.linear(x2.reshape(-1, in_features), w2_float).reshape(1, 8, out_features)

        # This is the line that crashes in the real model
        result = torch.cat([out1, out2], dim=1)
        assert result.shape == (1, 24, out_features)

    def test_cat_shapes_mismatch_detection(self):
        """Verify that mismatched feature dims in 3D tensors DO cause cat to fail.

        Simulates the Flux torch.cat([img, kontext], dim=1) crash where
        img has [B, S1, 256] but kontext has [B, S2, 128].
        """
        in_features = 128

        w1_int8 = torch.randint(-128, 127, (256, in_features), dtype=torch.int8)
        w2_int8 = torch.randint(-128, 127, (128, in_features), dtype=torch.int8)
        s1 = torch.rand(256, dtype=torch.float32).clamp(min=0.001)
        s2 = torch.rand(128, dtype=torch.float32).clamp(min=0.001)

        x1 = torch.randn(1, 16, in_features)  # img tokens
        x2 = torch.randn(1, 8, in_features)   # kontext tokens

        w1_float = _dequantize_int8(w1_int8, s1)
        w2_float = _dequantize_int8(w2_int8, s2)

        out1 = torch.nn.functional.linear(x1.reshape(-1, in_features), w1_float).reshape(1, 16, 256)
        out2 = torch.nn.functional.linear(x2.reshape(-1, in_features), w2_float).reshape(1, 8, 128)

        with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
            torch.cat([out1, out2], dim=1)


# ── Scale reshaping tests ────────────────────────────────────────────────────


class TestScaleReshaping:

    def test_scale_2d_to_1d(self):
        """Scale [out, 1] reshaped to [out] works in dequant."""
        out_features, in_features = 64, 128
        weight_int8 = torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
        scale_2d = torch.rand(out_features, 1, dtype=torch.float16)

        scale_1d = scale_2d.float().reshape(-1)
        w_float = _dequantize_int8(weight_int8, scale_1d)

        assert w_float.shape == (out_features, in_features)

    def test_scale_broadcast_in_linear(self):
        """Per-row scale [out, 1] broadcasts correctly with weight [out, in]."""
        out_features, in_features = 64, 128
        weight_int8 = torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
        scale = torch.rand(out_features, 1, dtype=torch.float32).clamp(min=0.001)

        w_float = weight_int8.float() * scale
        assert w_float.shape == (out_features, in_features)

        x = torch.randn(1, in_features)
        out = torch.nn.functional.linear(x, w_float)
        assert out.shape == (1, out_features)

    def test_set_weight_requantize_preserves_shape(self):
        """Simulate set_weight: dequant float → requant INT8 preserves shape."""
        out_features, in_features = 64, 128
        w_float = torch.randn(out_features, in_features)

        w_int8, s_w = _requantize_int8(w_float)

        assert w_int8.shape == (out_features, in_features)
        assert w_int8.dtype == torch.int8
        assert s_w.shape == (out_features,)
        assert s_w.dtype == torch.float32


# ── Converter truncation bug ─────────────────────────────────────────────────


class TestConverterRotatedTruncation:
    """Verify that the converter quantizes ALL columns of the rotated weight.

    Bug: cli.py LDLQ/GPTQ paths do W_orig_cols = W_work[:, :orig_in_features]
    when padded_in > orig_in_features, quantize only those columns, then
    zero-pad back. This destroys the information in the padded columns,
    which are meaningful after Hadamard rotation.

    Correct behavior: quantize the full W_work matrix (all padded_in columns).
    """

    @pytest.mark.parametrize("out_features,in_features,rot_size", [
        (64, 128, 256),
        (32, 64, 128),
        (128, 100, 128),  # in_features not a multiple of rot_size
    ])
    def test_no_zero_columns_in_quantized_weight(self, out_features, in_features, rot_size):
        """Quantized rotated weight must NOT have zero-padded columns.

        The current buggy path: quantize first orig_in columns, zero-pad rest.
        After rotation, ALL columns carry information — zero-padding destroys it.
        """
        from converter.rotation import rotate_weights
        from converter.ldlq import ldlq_quantize_layer

        W = torch.randn(out_features, in_features, dtype=torch.float32)
        W_rot = rotate_weights(W, rot_size)  # [out, padded_in]
        padded_in = W_rot.shape[1]

        assert padded_in > in_features, "Test requires padding to be applied"

        # CORRECT: quantize the full rotated weight
        q_full, scales_full = ldlq_quantize_layer(W_rot, int_bits=8)

        # The padded columns must NOT be all zeros
        padded_cols = q_full[:, in_features:]
        assert not torch.all(padded_cols == 0), (
            f"Columns [{in_features}:{padded_in}] are all zeros — "
            f"the quantizer likely truncated to orig_in_features then zero-padded"
        )

    @pytest.mark.parametrize("out_features,in_features,rot_size", [
        (64, 128, 256),
        (32, 64, 128),
    ])
    def test_full_quantize_mse_better_than_truncated(self, out_features, in_features, rot_size):
        """Quantizing the full rotated weight gives lower MSE than truncating.

        Truncated path: quantize first in_features columns, zero-pad rest.
        Full path: quantize all padded_in columns.
        """
        from converter.rotation import rotate_weights
        from converter.ldlq import ldlq_quantize_layer

        W = torch.randn(out_features, in_features, dtype=torch.float32)
        W_rot = rotate_weights(W, rot_size)
        padded_in = W_rot.shape[1]

        # CORRECT: quantize full rotated weight
        q_full, scales_full = ldlq_quantize_layer(W_rot, int_bits=8)
        dequant_full = q_full.float() * scales_full.float()
        mse_full = (W_rot - dequant_full).pow(2).mean().item()

        # BUGGY: quantize only first in_features columns, zero-pad rest
        W_orig_cols = W_rot[:, :in_features]
        q_trunc, scales_trunc = ldlq_quantize_layer(W_orig_cols, int_bits=8)
        q_trunc_padded = torch.nn.functional.pad(q_trunc, (0, padded_in - in_features))
        scales_2d = scales_trunc.float()
        dequant_trunc = q_trunc_padded.float() * scales_2d
        mse_trunc = (W_rot - dequant_trunc).pow(2).mean().item()

        # Full quantization must be at least as good (lower MSE)
        assert mse_full <= mse_trunc + 1e-6, (
            f"Full quantize MSE ({mse_full:.6f}) should be <= truncated MSE ({mse_trunc:.6f})"
        )
        # Truncated path should be significantly worse due to zero columns
        assert mse_trunc > mse_full * 1.5, (
            f"Truncated MSE ({mse_trunc:.6f}) should be significantly worse than "
            f"full MSE ({mse_full:.6f}) — the zero columns contribute large error"
        )

    @pytest.mark.parametrize("out_features,in_features,rot_size", [
        (64, 128, 256),
        (32, 64, 128),
    ])
    def test_full_quantize_forward_accuracy(self, out_features, in_features, rot_size):
        """Full quantization produces more accurate forward results than truncated.

        This is the end-to-end test: quantize → load → rotate activation → matmul.
        """
        from converter.rotation import rotate_weights, rotate_activations
        from converter.ldlq import ldlq_quantize_layer

        W = torch.randn(out_features, in_features, dtype=torch.float32)
        W_rot = rotate_weights(W, rot_size)
        padded_in = W_rot.shape[1]

        x = torch.randn(4, in_features, dtype=torch.float32)
        x_rot = rotate_activations(x, rot_size)

        # Ground truth: float rotated matmul
        y_ref = torch.nn.functional.linear(x_rot, W_rot)

        # CORRECT: full quantization
        q_full, s_full = ldlq_quantize_layer(W_rot, int_bits=8)
        w_full = q_full.float() * s_full.float()
        y_full = torch.nn.functional.linear(x_rot, w_full)
        mse_full = (y_ref - y_full).pow(2).mean().item()

        # BUGGY: truncated quantization
        W_orig_cols = W_rot[:, :in_features]
        q_trunc, s_trunc = ldlq_quantize_layer(W_orig_cols, int_bits=8)
        q_trunc_padded = torch.nn.functional.pad(q_trunc, (0, padded_in - in_features))
        w_trunc = q_trunc_padded.float() * s_trunc.float()
        y_trunc = torch.nn.functional.linear(x_rot, w_trunc)
        mse_trunc = (y_ref - y_trunc).pow(2).mean().item()

        assert mse_full <= mse_trunc + 1e-6, (
            f"Full quantize output MSE ({mse_full:.6f}) should be <= "
            f"truncated output MSE ({mse_trunc:.6f})"
        )

    @pytest.mark.parametrize("out_features,in_features,rot_size", [
        (64, 128, 256),
        (32, 64, 128),
    ])
    def test_rtn_same_as_full_when_rot_size_matches(self, out_features, in_features, rot_size):
        """RTN path (no truncation) quantizes full width — should be the reference.

        RTN in cli.py does: scales = calculate_scales_int8(W_work)
                             quantized_W = quantize_weights_int8(W_work, scales)
        This operates on the full W_work (no truncation). LDLQ/GPTQ should too.
        """
        from converter.rotation import rotate_weights
        from converter.scales import calculate_scales_int8, quantize_weights_int8
        from converter.ldlq import ldlq_quantize_layer

        W = torch.randn(out_features, in_features, dtype=torch.float32)
        W_rot = rotate_weights(W, rot_size)
        padded_in = W_rot.shape[1]

        # RTN: full width (this is what cli.py does for quant_method == "rtn")
        scales_rtn = calculate_scales_int8(W_rot)
        q_rtn = quantize_weights_int8(W_rot, scales_rtn)

        assert q_rtn.shape == (out_features, padded_in)
        padded_cols_rtn = q_rtn[:, in_features:]
        assert not torch.all(padded_cols_rtn == 0), (
            "RTN path produces zero-padded columns — same bug as LDLQ/GPTQ"
        )

    @pytest.mark.parametrize("out_features,in_features,rot_size", [
        (64, 128, 256),
        (32, 64, 128),
    ])
    def test_cli_ldlq_truncates_padded_columns(self, out_features, in_features, rot_size):
        """Simulate the actual cli.py LDLQ path — it truncates then zero-pads.

        This is the BUG. The converter does:
            W_orig_cols = W_work[:, :orig_in_features]   # truncate!
            q_orig, scales = ldlq_quantize_layer(W_orig_cols, ...)
            quantized_W = pad(q_orig, (0, pad_size))     # zero-pad!

        The padded columns should contain quantized values, not zeros.
        """
        from converter.rotation import rotate_weights
        from converter.ldlq import ldlq_quantize_layer

        W = torch.randn(out_features, in_features, dtype=torch.float32)
        W_rot = rotate_weights(W, rot_size)
        padded_in = W_rot.shape[1]

        # This is what cli.py does for LDLQ when padded_in > orig_in_features:
        W_orig_cols = W_rot[:, :in_features]
        q_orig, scales = ldlq_quantize_layer(W_orig_cols, int_bits=8)
        pad_size = padded_in - in_features
        quantized_W_buggy = torch.nn.functional.pad(q_orig, (0, pad_size))

        # The padded columns are zeros — this is the bug
        padded_cols = quantized_W_buggy[:, in_features:]
        assert torch.all(padded_cols == 0), (
            "cli.py LDLQ path zero-pads columns — confirmed bug"
        )

        # Compare to correct path (quantize full rotated weight)
        q_full, scales_full = ldlq_quantize_layer(W_rot, int_bits=8)

        # Correct path has non-zero padded columns
        assert not torch.all(q_full[:, in_features:] == 0), (
            "Full quantization should NOT have zero padded columns"
        )

        # The buggy path has much higher reconstruction error
        dequant_buggy = quantized_W_buggy.float() * scales.float()
        dequant_full = q_full.float() * scales_full.float()
        mse_buggy = (W_rot - dequant_buggy).pow(2).mean().item()
        mse_full = (W_rot - dequant_full).pow(2).mean().item()

        assert mse_buggy > mse_full, (
            f"Buggy truncated MSE ({mse_buggy:.6f}) > full MSE ({mse_full:.6f})"
        )

    @pytest.mark.parametrize("out_features,in_features,rot_size", [
        (64, 128, 256),
        (32, 64, 128),
    ])
    def test_cli_rtn_does_NOT_truncate(self, out_features, in_features, rot_size):
        """RTN path in cli.py quantizes full W_work — no truncation bug.

        This confirms RTN is correct and only LDLQ/GPTQ have the bug.
        """
        from converter.rotation import rotate_weights
        from converter.scales import calculate_scales_int8, quantize_weights_int8

        W = torch.randn(out_features, in_features, dtype=torch.float32)
        W_rot = rotate_weights(W, rot_size)
        padded_in = W_rot.shape[1]

        # This is what cli.py does for RTN:
        scales = calculate_scales_int8(W_rot)
        quantized_W = quantize_weights_int8(W_rot, scales)

        # Full width, no zeros
        assert quantized_W.shape == (out_features, padded_in)
        assert not torch.all(quantized_W[:, in_features:] == 0)

    @pytest.mark.parametrize("out_features,in_features,rot_size", [
        (64, 128, 256),
        (32, 64, 128),
    ])
    def test_fix_ldlq_no_truncation(self, out_features, in_features, rot_size):
        """After the fix, LDLQ quantizes full W_work — no zero-padded columns.

        This is the FIXED behavior: ldlq_quantize_layer(W_work, ...) on the
        full rotated weight, not on the truncated W_orig_cols.
        """
        from converter.rotation import rotate_weights
        from converter.ldlq import ldlq_quantize_layer

        W = torch.randn(out_features, in_features, dtype=torch.float32)
        W_rot = rotate_weights(W, rot_size)
        padded_in = W_rot.shape[1]

        # Fixed path: quantize full rotated weight
        q_full, scales = ldlq_quantize_layer(W_rot, int_bits=8)

        assert q_full.shape == (out_features, padded_in)
        padded_cols = q_full[:, in_features:]
        assert not torch.all(padded_cols == 0), (
            "After fix, LDLQ should quantize all columns — no zero-padding"
        )
