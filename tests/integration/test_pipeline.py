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


# ── GPTQ with DLR Hessian path ───────────────────────────────────────────────


class TestQuantizeModelGPTQDLR:
    """Pipeline tests for GPTQ with DLR (Diagonal + Low-Rank) calibration."""

    @staticmethod
    def _make_dlr_calibration(layer_names: list[str], in_features_list: list[int],
                               rank: int = 8) -> str:
        """Create a calibration .pt file with DLR Hessians."""
        import tempfile
        from converter.dlr import make_dlr_dict

        hessians = {}
        shapes = {}
        layer_types = {}
        torch.manual_seed(42)
        for name, in_feat in zip(layer_names, in_features_list):
            X = torch.randn(32, in_feat)
            H = X.T @ X
            D = H.diagonal().clone()
            eigvals, eigvecs = torch.linalg.eigh(H)
            r = min(rank, in_feat)
            U = eigvecs[:, -r:] * eigvals[-r:].clamp(min=0).sqrt().unsqueeze(0)
            hessians[name] = make_dlr_dict(D, U)
            shapes[name] = [in_feat, in_feat]
            layer_types[name] = "linear"
        data = {
            "hessians": hessians,
            "shapes": shapes,
            "layer_types": layer_types,
            "metadata": {"hessian_format": "dlr", "dlr_rank": rank},
        }
        path = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
        torch.save(data, path)
        return path

    def test_gptq_dlr_basic(self, tmp_path, tmp_safetensors):
        """Full pipeline: DLR calibration → GPTQ → safetensors output."""
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")
        cal_path = self._make_dlr_calibration(
            ["layers.0.q_proj", "layers.0.k_proj"], [64, 64], rank=8
        )

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            quant_method="gptq", calibration_path=cal_path,
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight" in result
        assert "layers.0.q_proj.weight_scale" in result
        assert "layers.0.q_proj.weight_zp" in result  # INT4 asymmetric

    def test_gptq_dlr_int8(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")
        cal_path = self._make_dlr_calibration(
            ["layers.0.q_proj", "layers.0.k_proj"], [64, 64], rank=8
        )

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=0,
            int_bits=8, quant_method="gptq", calibration_path=cal_path,
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight" in result
        assert "layers.0.q_proj.weight_scale" in result
        assert "layers.0.q_proj.weight_zp" not in result  # INT8 symmetric

    def test_gptq_dlr_with_rotation(self, tmp_path, tmp_safetensors):
        """DLR + pre-rotated calibration + rotation at quantization time."""
        input_path = tmp_safetensors(shapes=[(16, 64)])
        output_dir = str(tmp_path / "output")
        cal_path = self._make_dlr_calibration(["layers.0.q_proj"], [64], rank=8)

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=16,
            quant_method="gptq", calibration_path=cal_path,
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight" in result
        assert "layers.0.q_proj.weight_scale" in result

    def test_gptq_dlr_pre_rotated_calibration(self, tmp_path, tmp_safetensors):
        """DLR calibration already in rotated space (hessian_rotated=True)."""
        from converter.rotation import rotate_activations
        from converter.dlr import make_dlr_dict

        input_path = tmp_safetensors(shapes=[(16, 64)])
        output_dir = str(tmp_path / "output")
        rot_size = 64

        # Create DLR calibration with pre-rotated Hessian
        torch.manual_seed(42)
        X = torch.randn(32, 64)
        X_rot = rotate_activations(X, rot_size)
        H_rot = X_rot.T @ X_rot
        D = H_rot.diagonal().clone()
        eigvals, eigvecs = torch.linalg.eigh(H_rot)
        U = eigvecs[:, -8:] * eigvals[-8:].clamp(min=0).sqrt().unsqueeze(0)

        import tempfile
        cal_data = {
            "hessians": {"layers.0.q_proj": make_dlr_dict(D, U)},
            "shapes": {"layers.0.q_proj": [64, 64]},
            "layer_types": {"layers.0.q_proj": "linear"},
            "metadata": {"hessian_rotated": True, "rot_size": rot_size,
                         "hessian_format": "dlr", "dlr_rank": 8},
        }
        cal_path = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
        torch.save(cal_data, cal_path)

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=output_dir, rot_size=rot_size,
            quant_method="gptq", calibration_path=cal_path,
        ))

        result = load_file(os.path.join(output_dir, "model.safetensors"))
        assert "layers.0.q_proj.weight" in result


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
