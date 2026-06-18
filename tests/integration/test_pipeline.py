"""Tests for the quantize_model() pipeline — RTN, GPTQ, and LDLQ paths."""

import os

import pytest
import torch
from safetensors.torch import load_file

from converter.pipeline import quantize_model, DEFAULT_SKIP_PATTERNS
from converter.types import QuantizeConfig


# ── RTN path ─────────────────────────────────────────────────────────────────


class TestQuantizeModelRTN:

    def test_rtn_basic(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0, quant_method="rtn",
        ))

        out_path = os.path.join(output_dir, "model.safetensors")
        assert os.path.exists(out_path)
        result = load_file(out_path)
        assert "layers.0.q_proj.weight" in result
        assert "layers.0.q_proj.weight_scale" in result

    def test_rtn_with_rotation(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors(shapes=[(16, 64), (16, 64)])
        output_dir = str(tmp_path / "output")

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=16, quant_method="rtn",
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight_scale" in result

    def test_rtn_int8(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0, int_bits=8, quant_method="rtn",
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight_scale" in result

    def test_rtn_skips_norms(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors(
            layer_names=["model.norm.weight", "layers.0.q_proj.weight"],
            shapes=[(64,), (16, 64)],
        )
        output_dir = str(tmp_path / "output")

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0, quant_method="rtn",
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "model.norm.weight" in result
        assert "model.norm.weight_scale" not in result
        assert "layers.0.q_proj.weight_scale" in result

    def test_rtn_metadata(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0, quant_method="rtn",
        ))

        from safetensors import safe_open
        with safe_open(os.path.join(output_dir, "model.safetensors"), framework="numpy") as f:
            meta = f.metadata()

        assert "int_crush.format_version" in meta
        assert "int_crush.method" in meta

    def test_rtn_asymmetric(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            asymmetric=True, quant_method="rtn",
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight_zp" in result

    def test_rtn_bias_preserved(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors(state={
            "layers.0.q_proj.weight": torch.randn(16, 64),
            "layers.0.q_proj.bias": torch.randn(16),
        })
        output_dir = str(tmp_path / "output")

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0, quant_method="rtn",
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.bias" in result


# ── GPTQ path ────────────────────────────────────────────────────────────────


class TestQuantizeModelGPTQ:

    def test_gptq_basic(self, tmp_path, tmp_safetensors, tmp_calibration):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")
        cal_path = tmp_calibration(["layers.0.q_proj", "layers.0.k_proj"], [64, 64])

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            quant_method="gptq", calibration_path=cal_path,
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight" in result
        assert "layers.0.q_proj.weight_scale" in result
        assert "layers.0.q_proj.weight_zp" in result

    def test_gptq_with_rotation(self, tmp_path, tmp_safetensors, tmp_calibration):
        input_path = tmp_safetensors(shapes=[(16, 64)])
        output_dir = str(tmp_path / "output")
        cal_path = tmp_calibration(["layers.0.q_proj"], [64])

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=16,
            quant_method="gptq", calibration_path=cal_path,
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight" in result

    def test_gptq_metadata(self, tmp_path, tmp_safetensors, tmp_calibration):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")
        cal_path = tmp_calibration(["layers.0.q_proj", "layers.0.k_proj"], [64, 64])

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            quant_method="gptq", calibration_path=cal_path,
        ))

        from safetensors import safe_open
        with safe_open(os.path.join(output_dir, "model.safetensors"), framework="numpy") as f:
            meta = f.metadata()
        assert "int_crush.gptq_block_size" in meta
        assert "int_crush.damping_ratio" in meta

    def test_gptq_missing_calibration_falls_back_to_rtn(self, tmp_path, tmp_safetensors, tmp_calibration):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")
        cal_path = tmp_calibration(["layers.0.v_proj"], [64])

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            quant_method="gptq", calibration_path=cal_path,
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight" in result

    def test_gptq_no_calibration_raises(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")

        with pytest.raises(ValueError, match="GPTQ requires"):
            quantize_model(QuantizeConfig(
                input_path=input_path, output_dir=output_dir, rot_size=0,
                quant_method="gptq", calibration_path=None,
            ))


# ── LDLQ path ────────────────────────────────────────────────────────────────


class TestQuantizeModelLDLQ:

    def test_ldlq_basic(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0, quant_method="ldlq",
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight" in result
        assert "layers.0.q_proj.weight_scale" in result

    def test_ldlq_with_rotation(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors(shapes=[(16, 64)])
        output_dir = str(tmp_path / "output")

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=16, quant_method="ldlq",
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight" in result

    def test_ldlq_int8(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            int_bits=8, quant_method="ldlq",
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight" in result

    def test_ldlq_metadata(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0, quant_method="ldlq",
        ))

        from safetensors import safe_open
        with safe_open(os.path.join(output_dir, "model.safetensors"), framework="numpy") as f:
            meta = f.metadata()
        assert "int_crush.ldlq_block_size" in meta
        assert "int_crush.ldlq_iterations" in meta
