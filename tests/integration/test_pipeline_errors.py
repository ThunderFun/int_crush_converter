"""Tests for quantize_model() error handling and config validation."""

import pytest
import torch
from safetensors.torch import save_file

from converter.pipeline import quantize_model
from converter.types import QuantizeConfig


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


class TestQuantizeModelErrors:

    def test_invalid_quant_method(self, tmp_path):
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        _make_tiny_safetensors(input_path)

        with pytest.raises(ValueError, match="quant_method"):
            quantize_model(QuantizeConfig(
                input_path=input_path, output_dir=output_dir, quant_method="invalid",
            ))

    def test_invalid_rot_size(self, tmp_path):
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        _make_tiny_safetensors(input_path)

        with pytest.raises(ValueError, match="rot_size"):
            quantize_model(QuantizeConfig(
                input_path=input_path, output_dir=output_dir, rot_size=3,
            ))

    def test_invalid_group_size(self, tmp_path):
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        _make_tiny_safetensors(input_path)

        with pytest.raises(ValueError, match="quant_group_size"):
            quantize_model(QuantizeConfig(
                input_path=input_path, output_dir=output_dir, quant_group_size=100,
            ))

    def test_invalid_int_bits(self, tmp_path):
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        _make_tiny_safetensors(input_path)

        with pytest.raises(ValueError, match="int_bits"):
            quantize_model(QuantizeConfig(
                input_path=input_path, output_dir=output_dir, int_bits=6,
            ))

    def test_exclude_patterns(self, tmp_path):
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        _make_tiny_safetensors(
            input_path,
            layer_names=["layers.0.q_proj.weight", "special.layer.weight"],
            shapes=[(16, 64), (16, 64)],
        )

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            exclude_patterns=["special"],
        ))

        from safetensors.torch import load_file
        result = load_file(str(tmp_path / "output" / "model.safetensors"))
        assert "special.layer.weight" in result
        assert "special.layer.weight_scale" not in result
        assert "layers.0.q_proj.weight_scale" in result
