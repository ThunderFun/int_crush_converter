"""Tests for PermuQuant + block-diagonal Hessian integration (warns, not crash)."""

import logging
import os

import pytest
import torch
from safetensors.torch import load_file

from converter.pipeline import quantize_model
from converter.types import QuantizeConfig


class TestPermuQuantBlockDiagonal:

    def test_permuquant_block_diagonal_no_crash(self, tmp_path, tmp_safetensors, tmp_calibration):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")
        cal_path = tmp_calibration(
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

    def test_permuquant_block_diagonal_warns(self, tmp_path, caplog, tmp_safetensors, tmp_calibration):
        input_path = tmp_safetensors(
            layer_names=["layers.0.q_proj.weight"],
            shapes=[(16, 64)],
        )
        output_dir = str(tmp_path / "output")
        cal_path = tmp_calibration(
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
