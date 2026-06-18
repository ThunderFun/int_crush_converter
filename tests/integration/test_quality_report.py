"""Tests for --quality-report JSON output."""

import json
import os

import torch

from converter.pipeline import quantize_model
from converter.types import QuantizeConfig


class TestQualityReport:

    def test_quality_report_basic(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")
        report_path = str(tmp_path / "report.json")

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            quant_method="rtn", quality_report_path=report_path,
        ))

        assert os.path.exists(report_path)
        with open(report_path) as f:
            report = json.load(f)

        assert "layers" in report
        assert "summary" in report
        assert report["summary"]["quantized"] == 2
        assert report["summary"]["skipped"] == 0
        assert len(report["layers"]) == 2

        layer = report["layers"][0]
        for key in ("name", "method", "mse", "max_err", "scale_range", "fallbacks", "shape"):
            assert key in layer

    def test_quality_report_metrics_sane(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")
        report_path = str(tmp_path / "report.json")

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            quant_method="rtn", quality_report_path=report_path,
        ))

        with open(report_path) as f:
            report = json.load(f)

        for layer in report["layers"]:
            assert isinstance(layer["mse"], float)
            assert isinstance(layer["max_err"], float)
            assert layer["mse"] >= 0
            assert layer["max_err"] >= 0
            assert not (layer["mse"] == 0 and layer["max_err"] == 0), (
                f"Both MSE and max_err are zero for {layer['name']} — "
                "dequantization verification likely skipped"
            )

    def test_quality_report_with_skips(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors(
            layer_names=["model.norm.weight", "layers.0.q_proj.weight"],
            shapes=[(64,), (16, 64)],
        )
        output_dir = str(tmp_path / "output")
        report_path = str(tmp_path / "report.json")

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            quality_report_path=report_path,
        ))

        with open(report_path) as f:
            report = json.load(f)
        assert report["summary"]["quantized"] == 1
        assert report["summary"]["skipped"] == 1
        assert len(report["layers"]) == 1

    def test_quality_report_ldlq_method(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors(shapes=[(16, 64)])
        output_dir = str(tmp_path / "output")
        report_path = str(tmp_path / "report.json")

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            quant_method="ldlq", quality_report_path=report_path,
        ))

        with open(report_path) as f:
            report = json.load(f)
        assert report["layers"][0]["method"].startswith("ldlq_")

    def test_no_report_when_not_requested(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
        ))

        assert not os.path.exists(os.path.join(output_dir, "report.json"))
