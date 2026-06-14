"""Tests for structured progress callback for quantize_model()."""

import math
import os
from dataclasses import FrozenInstanceError

import pytest
import torch
from safetensors.torch import load_file

from converter.pipeline import quantize_model
from converter.types import ProgressInfo, ProgressSummary, QuantizeConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_calls(calls):
    """Split raw callback calls into (progress_infos, summaries)."""
    infos = [c for c in calls if isinstance(c, ProgressInfo)]
    summaries = [c for c in calls if isinstance(c, ProgressSummary)]
    return infos, summaries


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

class TestProgressCallbackContract:
    """Core contract: callback fires with correct types and counts."""

    def test_callback_receives_progress_info_per_layer(self, tmp_path, tmp_safetensors):
        calls = []
        input_path = tmp_safetensors()
        quantize_model(
            QuantizeConfig(input_path=input_path, output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        infos, summaries = _collect_calls(calls)
        assert len(infos) == 2
        assert len(summaries) == 1

    def test_callback_receives_summary_at_end(self, tmp_path, tmp_safetensors):
        calls = []
        input_path = tmp_safetensors()
        quantize_model(
            QuantizeConfig(input_path=input_path, output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        assert isinstance(calls[-1], ProgressSummary)
        infos, summaries = _collect_calls(calls)
        assert len(summaries) == 1
        assert len(calls) == len(infos) + 1

    def test_callback_not_called_when_none(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")
        quantize_model(QuantizeConfig(input_path=input_path, output_dir=output_dir, rot_size=0))
        assert os.path.exists(os.path.join(output_dir, "model.safetensors"))

    def test_callback_explicit_none(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors()
        output_dir = str(tmp_path / "output")
        quantize_model(
            QuantizeConfig(input_path=input_path, output_dir=output_dir, rot_size=0),
            progress_callback=None,
        )
        assert os.path.exists(os.path.join(output_dir, "model.safetensors"))


# ---------------------------------------------------------------------------
# ProgressInfo field validation
# ---------------------------------------------------------------------------

class TestProgressInfoFields:
    """Validate ProgressInfo field values and types."""

    def test_current_layer_1based(self, tmp_path, tmp_safetensors):
        calls = []
        quantize_model(
            QuantizeConfig(input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        assert infos[0].current_layer == 1

    def test_current_layer_monotonic(self, tmp_path, tmp_safetensors):
        calls = []
        quantize_model(
            QuantizeConfig(input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        for i, info in enumerate(infos, start=1):
            assert info.current_layer == i

    def test_total_layers_constant(self, tmp_path, tmp_safetensors):
        calls = []
        quantize_model(
            QuantizeConfig(input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        totals = {info.total_layers for info in infos}
        assert len(totals) == 1

    def test_total_layers_matches_quantized_count(self, tmp_path, tmp_safetensors):
        calls = []
        quantize_model(
            QuantizeConfig(input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        assert infos[0].total_layers == len(infos)

    def test_layer_names_match_quantized(self, tmp_path, tmp_safetensors):
        calls = []
        layer_names = ["layers.0.q_proj.weight", "layers.0.k_proj.weight"]
        input_path = tmp_safetensors(layer_names=layer_names)
        quantize_model(
            QuantizeConfig(input_path=input_path, output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        assert {info.layer_name for info in infos} == set(layer_names)

    def test_layer_shape_is_tuple(self, tmp_path, tmp_safetensors):
        calls = []
        input_path = tmp_safetensors(shapes=[(16, 64)])
        quantize_model(
            QuantizeConfig(input_path=input_path, output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        assert isinstance(infos[0].layer_shape, tuple)
        assert infos[0].layer_shape == (16, 64)

    def test_elapsed_seconds_non_negative(self, tmp_path, tmp_safetensors):
        calls = []
        quantize_model(
            QuantizeConfig(input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        for info in infos:
            assert info.elapsed_seconds >= 0.0

    def test_elapsed_seconds_non_decreasing(self, tmp_path, tmp_safetensors):
        calls = []
        quantize_model(
            QuantizeConfig(input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        for i in range(1, len(infos)):
            assert infos[i].elapsed_seconds >= infos[i - 1].elapsed_seconds

    def test_eta_non_negative(self, tmp_path, tmp_safetensors):
        calls = []
        quantize_model(
            QuantizeConfig(input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        for info in infos:
            assert info.estimated_remaining_seconds >= 0.0

    def test_eta_zero_on_first_layer(self, tmp_path, tmp_safetensors):
        calls = []
        quantize_model(
            QuantizeConfig(input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        assert infos[0].estimated_remaining_seconds == 0.0

    def test_eta_trends_toward_zero(self, tmp_path, tmp_safetensors):
        calls = []
        names = [f"layers.{i}.proj.weight" for i in range(3)]
        input_path = tmp_safetensors(layer_names=names, shapes=[(16, 64)] * 3)
        quantize_model(
            QuantizeConfig(input_path=input_path, output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        assert infos[-1].estimated_remaining_seconds <= infos[1].estimated_remaining_seconds

    def test_method_valid(self, tmp_path, tmp_safetensors):
        calls = []
        quantize_model(
            QuantizeConfig(input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        valid_methods = {"rtn", "gptq_triton", "gptq_pytorch", "ldlq_triton", "ldlq_compile", "ldlq_cpu"}
        infos, _ = _collect_calls(calls)
        for info in infos:
            assert info.method in valid_methods, f"Unknown method: {info.method}"

    def test_mse_nonnegative(self, tmp_path, tmp_safetensors):
        calls = []
        quantize_model(
            QuantizeConfig(input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        for info in infos:
            assert info.mse >= 0.0

    def test_mse_finite(self, tmp_path, tmp_safetensors):
        calls = []
        quantize_model(
            QuantizeConfig(input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        for info in infos:
            assert math.isfinite(info.mse)

    def test_progress_info_frozen(self, tmp_path, tmp_safetensors):
        calls = []
        quantize_model(
            QuantizeConfig(input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        with pytest.raises(FrozenInstanceError):
            infos[0].current_layer = 99


# ---------------------------------------------------------------------------
# Skipped layers
# ---------------------------------------------------------------------------

class TestProgressCallbackWithSkips:

    def test_skipped_layers_excluded_from_progress(self, tmp_path, tmp_safetensors):
        calls = []
        input_path = tmp_safetensors(
            layer_names=["model.norm.weight", "layers.0.q_proj.weight"],
            shapes=[(64,), (16, 64)],
        )
        quantize_model(
            QuantizeConfig(input_path=input_path, output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        assert len(infos) == 1
        assert infos[0].layer_name == "layers.0.q_proj.weight"

    def test_1d_tensors_excluded(self, tmp_path):
        from safetensors.torch import save_file
        calls = []
        input_path = str(tmp_path / "input.safetensors")
        save_file({
            "layers.0.bias": torch.randn(16),
            "layers.0.weight": torch.randn(16, 64),
        }, input_path)
        quantize_model(
            QuantizeConfig(input_path=input_path, output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        assert len(infos) == 1
        assert infos[0].layer_name == "layers.0.weight"

    def test_summary_reflects_skips(self, tmp_path, tmp_safetensors):
        calls = []
        input_path = tmp_safetensors(
            layer_names=["model.norm.weight", "layers.0.q_proj.weight"],
            shapes=[(64,), (16, 64)],
        )
        quantize_model(
            QuantizeConfig(input_path=input_path, output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        _, summaries = _collect_calls(calls)
        assert summaries[0].skipped_layers == 1

    def test_total_layers_equals_quantized_not_total(self, tmp_path, tmp_safetensors):
        calls = []
        input_path = tmp_safetensors(
            layer_names=["model.norm.weight", "layers.0.q_proj.weight"],
            shapes=[(64,), (16, 64)],
        )
        quantize_model(
            QuantizeConfig(input_path=input_path, output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        assert infos[0].total_layers == 1


# ---------------------------------------------------------------------------
# All layers skipped
# ---------------------------------------------------------------------------

class TestProgressCallbackAllSkipped:

    def test_no_progress_info_when_all_skipped(self, tmp_path, tmp_safetensors):
        calls = []
        input_path = tmp_safetensors(
            layer_names=["model.norm.weight"],
            shapes=[(64,)],
        )
        quantize_model(
            QuantizeConfig(input_path=input_path, output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        assert len(infos) == 0

    def test_summary_still_fired_when_all_skipped(self, tmp_path, tmp_safetensors):
        calls = []
        input_path = tmp_safetensors(
            layer_names=["model.norm.weight"],
            shapes=[(64,)],
        )
        quantize_model(
            QuantizeConfig(input_path=input_path, output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        _, summaries = _collect_calls(calls)
        assert len(summaries) == 1
        assert summaries[0].total_layers == 0


# ---------------------------------------------------------------------------
# Method strings per quantization path
# ---------------------------------------------------------------------------

class TestProgressCallbackMethods:

    def test_rtn_method_string(self, tmp_path, tmp_safetensors):
        calls = []
        quantize_model(
            QuantizeConfig(
                input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"),
                rot_size=0, quant_method="rtn",
            ),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        for info in infos:
            assert info.method == "rtn"

    def test_ldlq_method_string(self, tmp_path, tmp_safetensors):
        calls = []
        quantize_model(
            QuantizeConfig(
                input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"),
                rot_size=0, quant_method="ldlq",
            ),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        for info in infos:
            assert info.method.startswith("ldlq")

    def test_gptq_method_string(self, tmp_path, tmp_safetensors, tmp_calibration):
        calls = []
        input_path = tmp_safetensors(shapes=[(16, 64)])
        cal_path = tmp_calibration(["layers.0.q_proj"], [64])
        quantize_model(
            QuantizeConfig(
                input_path=input_path, output_dir=str(tmp_path / "output"),
                rot_size=0, quant_method="gptq", calibration_path=cal_path,
            ),
            progress_callback=calls.append,
        )
        infos, _ = _collect_calls(calls)
        for info in infos:
            assert info.method.startswith("gptq") or info.method == "rtn"


# ---------------------------------------------------------------------------
# ProgressSummary field validation
# ---------------------------------------------------------------------------

class TestProgressCallbackSummary:

    def test_summary_fields_present(self, tmp_path, tmp_safetensors):
        calls = []
        quantize_model(
            QuantizeConfig(input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        _, summaries = _collect_calls(calls)
        s = summaries[0]
        for attr in ("total_layers", "skipped_layers", "permuted_layers", "gptq_layers",
                      "rtn_fallback_layers", "elapsed_seconds", "input_size_gb",
                      "output_size_gb", "compression_ratio"):
            assert isinstance(getattr(s, attr), (int, float))

    def test_summary_total_layers_matches_quantized(self, tmp_path, tmp_safetensors):
        calls = []
        quantize_model(
            QuantizeConfig(input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        infos, summaries = _collect_calls(calls)
        assert summaries[0].total_layers == len(infos)

    def test_summary_elapsed_positive(self, tmp_path, tmp_safetensors):
        calls = []
        quantize_model(
            QuantizeConfig(input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        _, summaries = _collect_calls(calls)
        assert summaries[0].elapsed_seconds > 0.0

    def test_summary_sizes_non_negative(self, tmp_path, tmp_safetensors):
        calls = []
        quantize_model(
            QuantizeConfig(input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        _, summaries = _collect_calls(calls)
        assert summaries[0].input_size_gb >= 0.0
        assert summaries[0].output_size_gb >= 0.0

    def test_summary_compression_ratio_positive(self, tmp_path, tmp_safetensors):
        calls = []
        quantize_model(
            QuantizeConfig(input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        _, summaries = _collect_calls(calls)
        assert summaries[0].compression_ratio > 0.0

    def test_summary_frozen(self, tmp_path, tmp_safetensors):
        calls = []
        quantize_model(
            QuantizeConfig(input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"), rot_size=0),
            progress_callback=calls.append,
        )
        _, summaries = _collect_calls(calls)
        with pytest.raises(FrozenInstanceError):
            summaries[0].total_layers = 99


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestProgressCallbackErrors:

    def test_callback_exception_propagates(self, tmp_path, tmp_safetensors):
        def boom(info):
            raise RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            quantize_model(
                QuantizeConfig(
                    input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"), rot_size=0,
                ),
                progress_callback=boom,
            )

    def test_no_partial_output_on_callback_error(self, tmp_path, tmp_safetensors):
        def boom(info):
            raise RuntimeError("boom")
        with pytest.raises(RuntimeError):
            quantize_model(
                QuantizeConfig(
                    input_path=tmp_safetensors(), output_dir=str(tmp_path / "output"), rot_size=0,
                ),
                progress_callback=boom,
            )
        assert not os.path.exists(os.path.join(str(tmp_path / "output"), "model.safetensors"))


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestProgressCallbackBackwardCompat:

    def test_no_callback_same_output(self, tmp_path, tmp_safetensors):
        input_path = tmp_safetensors()
        out1 = str(tmp_path / "out1")
        out2 = str(tmp_path / "out2")

        quantize_model(QuantizeConfig(
            input_path=input_path, output_dir=out1, rot_size=0, seed=42,
        ))
        quantize_model(
            QuantizeConfig(input_path=input_path, output_dir=out2, rot_size=0, seed=42),
            progress_callback=None,
        )

        r1 = load_file(os.path.join(out1, "model.safetensors"))
        r2 = load_file(os.path.join(out2, "model.safetensors"))
        for key in r1:
            assert torch.equal(r1[key], r2[key]), f"Mismatch on {key}"
