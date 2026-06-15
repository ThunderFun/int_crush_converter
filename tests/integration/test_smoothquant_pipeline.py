"""Integration tests for SmoothQuant pipeline integration."""

import os

import pytest
import torch
from safetensors.torch import load_file, save_file

from converter.pipeline import quantize_model
from converter.types import QuantizeConfig


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_tiny_safetensors(path: str, layer_names=None, shapes=None) -> str:
    if layer_names is None:
        layer_names = ["layers.0.q_proj.weight", "layers.0.k_proj.weight"]
    if shapes is None:
        shapes = [(16, 64), (16, 64)]
    state = {}
    for name, shape in zip(layer_names, shapes):
        state[name] = torch.randn(*shape, dtype=torch.float32)
    save_file(state, path)
    return path


def _make_calibration_with_amax(
    path: str,
    layer_names: list[str],
    in_features_list: list[int],
    include_amax_per_channel: bool = True,
) -> str:
    """Create mock calibration data with optional per-channel amax."""
    hessians = {}
    shapes = {}
    layer_types = {}
    amax_per_channel = {}
    for name, in_feat in zip(layer_names, in_features_list):
        X = torch.randn(32, in_feat)
        H = X.T @ X / 32.0
        hessians[name] = H
        shapes[name] = [in_feat, in_feat]
        layer_types[name] = "linear"
        if include_amax_per_channel:
            amax_per_channel[name] = X.abs().amax(dim=0)
    data = {
        "hessians": hessians,
        "shapes": shapes,
        "layer_types": layer_types,
        "metadata": {"num_samples": 32},
    }
    if include_amax_per_channel:
        data["amax_per_channel"] = amax_per_channel
    torch.save(data, path)
    return path


# ── SmoothQuant RTN path ─────────────────────────────────────────────────────


class TestSmoothQuantRTN:

    def test_smoothquant_rtn_int8(self, tmp_path):
        """SmoothQuant + RTN INT8 produces valid output with _smooth tensors."""
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        _make_tiny_safetensors(input_path)

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            int_bits=8, quant_method="rtn", smoothquant=True, smooth_alpha=0.5,
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight" in result
        assert "layers.0.q_proj.weight_scale" in result
        assert "layers.0.q_proj.weight_smooth" in result
        assert "layers.0.k_proj.weight_smooth" in result
        # Smoothing factors should be float16 and have the right shape
        assert result["layers.0.q_proj.weight_smooth"].dtype == torch.float16
        assert result["layers.0.q_proj.weight_smooth"].shape == (64,)

    def test_smoothquant_rtn_int4(self, tmp_path):
        """SmoothQuant + RTN INT4 produces output (with warning)."""
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        _make_tiny_safetensors(input_path)

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            int_bits=4, quant_method="rtn", smoothquant=True, smooth_alpha=0.5,
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight_smooth" in result

    def test_smoothquant_rtn_no_calibration_weight_only(self, tmp_path):
        """SmoothQuant without calibration uses weight-only smoothing."""
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        _make_tiny_safetensors(input_path)

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            int_bits=8, quant_method="rtn", smoothquant=True, smooth_alpha=0.5,
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight_smooth" in result
        assert result["layers.0.q_proj.weight_smooth"].shape == (64,)


# ── SmoothQuant GPTQ path ────────────────────────────────────────────────────


class TestSmoothQuantGPTQ:

    def test_smoothquant_gptq_with_amax(self, tmp_path):
        """SmoothQuant + GPTQ with per-channel amax from calibration."""
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        cal_path = str(tmp_path / "cal.pt")

        _make_tiny_safetensors(input_path)
        _make_calibration_with_amax(
            cal_path,
            ["layers.0.q_proj", "layers.0.k_proj"],
            [64, 64],
            include_amax_per_channel=True,
        )

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            int_bits=8, quant_method="gptq", calibration_path=cal_path,
            smoothquant=True, smooth_alpha=0.5,
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight_smooth" in result
        # GPTQ INT8 symmetric: no zp (only present with --asymmetric)

    def test_smoothquant_gptq_without_amax_uses_hessian(self, tmp_path):
        """SmoothQuant + GPTQ falls back to Hessian diagonal when no amax."""
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        cal_path = str(tmp_path / "cal.pt")

        _make_tiny_safetensors(input_path)
        _make_calibration_with_amax(
            cal_path,
            ["layers.0.q_proj", "layers.0.k_proj"],
            [64, 64],
            include_amax_per_channel=False,  # No amax_per_channel key
        )

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            int_bits=8, quant_method="gptq", calibration_path=cal_path,
            smoothquant=True, smooth_alpha=0.5,
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight_smooth" in result

    def test_smoothquant_gptq_with_rotation(self, tmp_path):
        """SmoothQuant + GPTQ + ConvRot rotation."""
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        cal_path = str(tmp_path / "cal.pt")

        _make_tiny_safetensors(input_path, shapes=[(16, 64)])
        _make_calibration_with_amax(
            cal_path,
            ["layers.0.q_proj"],
            [64],
            include_amax_per_channel=True,
        )

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=16,
            int_bits=8, quant_method="gptq", calibration_path=cal_path,
            smoothquant=True, smooth_alpha=0.5,
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight_smooth" in result
        # Smoothing factors should have the original (unpadded) in_features
        assert result["layers.0.q_proj.weight_smooth"].shape == (64,)


# ── SmoothQuant + PermuQuant ─────────────────────────────────────────────────


class TestSmoothQuantPermuQuant:

    def test_smoothquant_with_permuquant(self, tmp_path):
        """SmoothQuant + PermuQuant produces valid output."""
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        _make_tiny_safetensors(input_path)

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            int_bits=8, quant_method="rtn", smoothquant=True,
            smooth_alpha=0.5, use_permuquant=True,
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight_smooth" in result

    def test_smoothquant_with_permuquant_and_rotation(self, tmp_path):
        """SmoothQuant + PermuQuant + ConvRot rotation."""
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        _make_tiny_safetensors(input_path, shapes=[(16, 64)])

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=16,
            int_bits=8, quant_method="rtn", smoothquant=True,
            smooth_alpha=0.5, use_permuquant=True,
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight_smooth" in result


# ── SmoothQuant metadata ─────────────────────────────────────────────────────


class TestSmoothQuantMetadata:

    def test_smoothquant_metadata_present(self, tmp_path):
        """Output safetensors contains smoothquant metadata."""
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        _make_tiny_safetensors(input_path)

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            int_bits=8, quant_method="rtn", smoothquant=True, smooth_alpha=0.75,
        ))

        from safetensors import safe_open
        with safe_open(os.path.join(output_dir, "model.safetensors"), framework="numpy") as f:
            meta = f.metadata()
        assert meta["int_crush.smoothquant"] == "true"
        assert meta["int_crush.smooth_alpha"] == "0.75"

    def test_smoothquant_in_method_string(self, tmp_path):
        """Method string includes 'smoothq' when SmoothQuant is enabled."""
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        _make_tiny_safetensors(input_path)

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            int_bits=8, quant_method="rtn", smoothquant=True,
        ))

        from safetensors import safe_open
        with safe_open(os.path.join(output_dir, "model.safetensors"), framework="numpy") as f:
            meta = f.metadata()
        assert "smoothq" in meta["int_crush.method"]

    def test_smoothquant_gptq_method_string(self, tmp_path):
        """Method string with SmoothQuant + GPTQ."""
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        cal_path = str(tmp_path / "cal.pt")

        _make_tiny_safetensors(input_path)
        _make_calibration_with_amax(
            cal_path, ["layers.0.q_proj", "layers.0.k_proj"], [64, 64],
        )

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            int_bits=8, quant_method="gptq", calibration_path=cal_path,
            smoothquant=True,
        ))

        from safetensors import safe_open
        with safe_open(os.path.join(output_dir, "model.safetensors"), framework="numpy") as f:
            meta = f.metadata()
        assert "smoothq" in meta["int_crush.method"]
        assert "gptq" in meta["int_crush.method"]

    def test_smoothquant_with_permuquant_method_string(self, tmp_path):
        """Method string with SmoothQuant + PermuQuant."""
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        _make_tiny_safetensors(input_path)

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            int_bits=8, quant_method="rtn", smoothquant=True, use_permuquant=True,
        ))

        from safetensors import safe_open
        with safe_open(os.path.join(output_dir, "model.safetensors"), framework="numpy") as f:
            meta = f.metadata()
        assert "smoothq" in meta["int_crush.method"]
        assert "permuq" in meta["int_crush.method"]

    def test_no_smoothquant_metadata_when_disabled(self, tmp_path):
        """No smoothquant metadata when feature is disabled."""
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        _make_tiny_safetensors(input_path)

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            int_bits=8, quant_method="rtn", smoothquant=False,
        ))

        from safetensors import safe_open
        with safe_open(os.path.join(output_dir, "model.safetensors"), framework="numpy") as f:
            meta = f.metadata()
        assert "int_crush.smoothquant" not in meta


# ── SmoothQuant alpha variants ───────────────────────────────────────────────


class TestSmoothQuantAlpha:

    def test_different_alpha_values(self, tmp_path):
        """Different alpha values produce different smoothing factors (with calibration)."""
        input_path = str(tmp_path / "input.safetensors")
        output_dir_1 = str(tmp_path / "output_1")
        output_dir_2 = str(tmp_path / "output_2")
        cal_path = str(tmp_path / "cal.pt")

        _make_tiny_safetensors(input_path)
        _make_calibration_with_amax(
            cal_path,
            ["layers.0.q_proj", "layers.0.k_proj"],
            [64, 64],
            include_amax_per_channel=True,
        )

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir_1, rot_size=0,
            int_bits=8, quant_method="rtn", calibration_path=cal_path,
            smoothquant=True, smooth_alpha=0.3,
        ))
        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir_2, rot_size=0,
            int_bits=8, quant_method="rtn", calibration_path=cal_path,
            smoothquant=True, smooth_alpha=0.7,
        ))

        result_1 = load_file(os.path.join(output_dir_1, "model.safetensors"))
        result_2 = load_file(os.path.join(output_dir_2, "model.safetensors"))

        # The smoothing factors should differ with calibration data
        s1 = result_1["layers.0.q_proj.weight_smooth"].float()
        s2 = result_2["layers.0.q_proj.weight_smooth"].float()
        assert not torch.allclose(s1, s2, atol=1e-6)


# ── Progress callback ────────────────────────────────────────────────────────


class TestSmoothQuantProgress:

    def test_smoothquant_in_progress_summary(self, tmp_path):
        """ProgressSummary includes smoothquant_layers count."""
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        _make_tiny_safetensors(input_path)

        from converter.types import ProgressSummary
        summaries = []

        def callback(info):
            if isinstance(info, ProgressSummary):
                summaries.append(info)

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            int_bits=8, quant_method="rtn", smoothquant=True,
        ), progress_callback=callback)

        assert len(summaries) == 1
        assert summaries[0].smoothquant_layers == 2  # Two quantizable layers

    def test_smoothquant_zero_when_disabled(self, tmp_path):
        """smoothquant_layers is 0 when SmoothQuant is disabled."""
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        _make_tiny_safetensors(input_path)

        from converter.types import ProgressSummary
        summaries = []

        def callback(info):
            if isinstance(info, ProgressSummary):
                summaries.append(info)

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            int_bits=8, quant_method="rtn", smoothquant=False,
        ), progress_callback=callback)

        assert len(summaries) == 1
        assert summaries[0].smoothquant_layers == 0


# ── Mathematical equivalence ─────────────────────────────────────────────────


class TestSmoothQuantEquivalence:

    def test_quantized_output_approximates_original(self, tmp_path):
        """Quantized + smoothed output should be close to the original."""
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")

        # Use small weights so quantization error is low
        state = {
            "layers.0.q_proj.weight": torch.randn(16, 64, dtype=torch.float32) * 0.1,
        }
        save_file(state, input_path)

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            int_bits=8, quant_method="rtn", smoothquant=True, smooth_alpha=0.5,
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        q = result["layers.0.q_proj.weight"].float()
        s = result["layers.0.q_proj.weight_scale"].float()
        s_smooth = result["layers.0.q_proj.weight_smooth"].float()

        # Dequantize: W_approx = Q * scale / smooth
        # The stored weight is already smoothed then quantized, so:
        # W_smooth = Q * scale → W_orig ≈ W_smooth / smooth
        W_approx = (q.float() * s) / s_smooth.unsqueeze(0)
        W_orig = state["layers.0.q_proj.weight"]

        # Should be close (INT8 quantization is high precision)
        mse = ((W_approx - W_orig) ** 2).mean().item()
        assert mse < 0.01, f"MSE too high: {mse}"
