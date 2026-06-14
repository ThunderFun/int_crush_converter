"""Tests for PermuQuant + block-diagonal Hessian integration (warns, not crash)."""

import logging
import os

import pytest
import torch
from safetensors.torch import load_file, save_file

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


def _make_block_diagonal_calibration(path, layer_names, in_features_list, block_size=16):
    hessians = {}
    shapes = {}
    layer_types = {}
    for name, in_feat in zip(layer_names, in_features_list):
        num_blocks = (in_feat + block_size - 1) // block_size
        blocks = []
        for _ in range(num_blocks):
            X = torch.randn(32, block_size)
            blocks.append(X.T @ X)
        hessians[name] = torch.stack(blocks)
        shapes[name] = [in_feat, in_feat]
        layer_types[name] = "linear"
    data = {
        "hessians": hessians, "shapes": shapes,
        "layer_types": layer_types, "metadata": {},
    }
    torch.save(data, path)
    return path


class TestPermuQuantBlockDiagonal:

    def test_permuquant_block_diagonal_no_crash(self, tmp_path):
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        cal_path = str(tmp_path / "cal.pt")

        _make_tiny_safetensors(input_path)
        _make_block_diagonal_calibration(
            cal_path,
            layer_names=["layers.0.q_proj", "layers.0.k_proj"],
            in_features_list=[64, 64],
            block_size=16,
        )

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            quant_method="gptq", calibration_path=cal_path,
            use_permuquant=True, tau=0.0,
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        for name in ["layers.0.q_proj.weight", "layers.0.k_proj.weight"]:
            assert name in result
            assert name.replace(".weight", ".weight_scale") in result

    def test_permuquant_block_diagonal_warns(self, tmp_path, caplog):
        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        cal_path = str(tmp_path / "cal.pt")

        _make_tiny_safetensors(
            input_path,
            layer_names=["layers.0.q_proj.weight"],
            shapes=[(16, 64)],
        )
        _make_block_diagonal_calibration(
            cal_path,
            layer_names=["layers.0.q_proj"],
            in_features_list=[64],
            block_size=16,
        )

        with caplog.at_level(logging.DEBUG, logger="converter"):
            quantize_model(QuantizeConfig(
                input_path=input_path, output_dir=output_dir, rot_size=0,
                quant_method="gptq", calibration_path=cal_path,
                use_permuquant=True, tau=0.0,
            ))

        assert any("block-diagonal Hessian" in record.message for record in caplog.records), (
            f"Expected block-diagonal message in logs, got: {caplog.text}"
        )
