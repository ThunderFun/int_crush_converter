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
        _qr = ldlq_quantize_layer(W, int_bits=8)
        quantized = _qr.quantized_W
        scales = _qr.scales
        zero_points = _qr.zero_points

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

        _qr = ldlq_quantize_layer(W_rot, int_bits=8)
        quantized = _qr.quantized_W
        scales = _qr.scales
        zero_points = _qr.zero_points

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

