"""Tests for quantize_model() error handling and config validation."""

import pytest
import torch

from converter.pipeline import quantize_model
from converter.types import QuantizeConfig


class TestQuantizeModelErrors:

    def test_invalid_quant_method(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")

        with pytest.raises(ValueError, match="quant_method"):
            quantize_model(QuantizeConfig(
                input_path=input_path, output_dir=output_dir, quant_method="invalid",
            ))

    def test_invalid_rot_size(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")

        with pytest.raises(ValueError, match="rot_size"):
            quantize_model(QuantizeConfig(
                input_path=input_path, output_dir=output_dir, rot_size=3,
            ))

    def test_invalid_group_size(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")

        with pytest.raises(ValueError, match="quant_group_size"):
            quantize_model(QuantizeConfig(
                input_path=input_path, output_dir=output_dir, quant_group_size=100,
            ))

    def test_invalid_int_bits(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")

        with pytest.raises(ValueError, match="int_bits"):
            quantize_model(QuantizeConfig(
                input_path=input_path, output_dir=output_dir, int_bits=6,
            ))

    def test_exclude_patterns(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors(
            layer_names=["layers.0.q_proj.weight", "special.layer.weight"],
            shapes=[(16, 64), (16, 64)],
        )
        output_dir = str(tmp_path / "output")

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            exclude_patterns=["special"],
        ))

        from safetensors.torch import load_file
        result = load_file(str(tmp_path / "output" / "model.safetensors"))
        assert "special.layer.weight" in result
        assert "special.layer.weight_scale" not in result
        assert "layers.0.q_proj.weight_scale" in result
