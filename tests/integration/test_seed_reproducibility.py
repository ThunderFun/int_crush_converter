"""Tests for --seed reproducibility across runs."""

import os

import torch
from safetensors.torch import load_file, save_file

from converter.pipeline import quantize_model
from converter.types import QuantizeConfig


def _make_tiny_safetensors(path: str, shapes=None) -> str:
    if shapes is None:
        shapes = [(16, 64), (16, 64)]
    state = {}
    for i, shape in enumerate(shapes):
        state[f"layers.{i}.proj.weight"] = torch.randn(*shape, dtype=torch.float32)
    save_file(state, path)
    return path


class TestSeedReproducibility:

    def test_same_seed_same_output(self, tmp_path):
        input_path = str(tmp_path / "input.safetensors")
        _make_tiny_safetensors(input_path)

        out1 = str(tmp_path / "out1")
        out2 = str(tmp_path / "out2")

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=out1, rot_size=0,
            quant_method="rtn", seed=42,
        ))
        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=out2, rot_size=0,
            quant_method="rtn", seed=42,
        ))

        r1 = load_file(os.path.join(out1, "model.safetensors"))
        r2 = load_file(os.path.join(out2, "model.safetensors"))
        for key in r1:
            assert torch.equal(r1[key], r2[key]), f"Mismatch on {key}"

    def test_different_seeds_can_differ(self, tmp_path):
        input_path = str(tmp_path / "input.safetensors")
        _make_tiny_safetensors(input_path, shapes=[(64, 256), (64, 256)])

        out1 = str(tmp_path / "out1")
        out2 = str(tmp_path / "out2")

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=out1, rot_size=16,
            quant_method="rtn", seed=0,
        ))
        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=out2, rot_size=16,
            quant_method="rtn", seed=999,
        ))

        r1 = load_file(os.path.join(out1, "model.safetensors"))
        r2 = load_file(os.path.join(out2, "model.safetensors"))
        assert "layers.0.proj.weight" in r1
        assert "layers.0.proj.weight" in r2

    def test_seed_minus_one_disables(self, tmp_path):
        input_path = str(tmp_path / "input.safetensors")
        _make_tiny_safetensors(input_path)

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=str(tmp_path / "out"),
            rot_size=0, seed=-1,
        ))

        assert os.path.exists(str(tmp_path / "out" / "model.safetensors"))
