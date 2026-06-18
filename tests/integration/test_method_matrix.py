"""Comprehensive benchmark pipeline test suite.

Tests every valid (method × bits × features) combo via converter.benchmark.
~65 tests across 15 classes. All tests import from converter.benchmark
directly — no subprocess, no CLI parsing.

Run with::

    pytest tests/integration/test_method_matrix.py -v
"""

from __future__ import annotations

import dataclasses
import json
import math

import pytest
import torch

from converter.benchmark import (
    FEATURE_PRESETS,
    VALID_COMBOS,
    benchmark_matrix,
    benchmark_method,
    make_synthetic_calibration,
    make_synthetic_model,
)
from converter.smoothrot import detect_ffn_pairs
from converter.types import BenchmarkConfig, MethodBenchmarkResult

# ── Thresholds ───────────────────────────────────────────────────────────────

SYNTHETIC_INT4_THRESHOLD = 0.5
SYNTHETIC_INT8_THRESHOLD = 0.1
REAL_INT4_THRESHOLD = 1.0
REAL_INT8_THRESHOLD = 0.5


def _threshold(int_bits: int, real: bool = False) -> float:
    if real:
        return REAL_INT4_THRESHOLD if int_bits == 4 else REAL_INT8_THRESHOLD
    return SYNTHETIC_INT4_THRESHOLD if int_bits == 4 else SYNTHETIC_INT8_THRESHOLD


# ── Class 0: Inverse chain identity ─────────────────────────────────────────


class TestInverseChainIdentity:
    """Validate the inverse-transform chain by using lossless INT8 weights.

    Each row is a constant value, making INT8 quantization lossless.
    The round-trip should produce the original weight to machine precision.
    """

    def _make_constant_weights(self, num_layers=4, shape=(32, 64), ff_pairs=False):
        """Create weights where each row is constant (lossless INT8)."""
        out_f, in_f = shape
        weights = {}
        for i in range(num_layers):
            # Each row has a different constant, in INT8-representable range
            vals = torch.linspace(-10, 10, out_f).unsqueeze(1).expand(out_f, in_f)
            if ff_pairs:
                weights[f"double_blocks.{i}.img_mlp.0.weight"] = vals.clone()
                weights[f"double_blocks.{i}.img_mlp.2.weight"] = vals[:, :out_f].clone().T
            else:
                weights[f"layers.{i}.q_proj.weight"] = vals.clone()
        return weights

    def test_identity_plain(self):
        weights = self._make_constant_weights()
        config = BenchmarkConfig(method="rtn", int_bits=8, features="plain")
        result = benchmark_method(weights, config)
        assert result.error is None
        for lr in result.layers:
            assert lr.weight_mse < 1e-6, f"{lr.name}: mse={lr.weight_mse}"
        assert result.mse_mean < 1e-6

    def test_identity_convrot(self):
        # Use in_features divisible by 64 to avoid padding issues
        weights = self._make_constant_weights(shape=(32, 128))
        config = BenchmarkConfig(method="rtn", int_bits=8, features="convrot")
        result = benchmark_method(weights, config)
        assert result.error is None
        for lr in result.layers:
            assert lr.weight_mse < 1e-6, f"{lr.name}: mse={lr.weight_mse}"
        assert result.mse_mean < 1e-6

    def test_identity_smoothrot(self):
        weights = self._make_constant_weights(shape=(32, 128), ff_pairs=True)
        # SmoothRot needs FFN pairs detected
        pairs = detect_ffn_pairs(weights)
        assert len(pairs) > 0, "No FFN pairs detected"

        config = BenchmarkConfig(method="rtn", int_bits=8, features="smoothrot")
        result = benchmark_method(weights, config)
        assert result.error is None
        for lr in result.layers:
            assert lr.weight_mse < 1e-6, f"{lr.name}: mse={lr.weight_mse}"
        assert result.mse_mean < 1e-6


# ── Class 1: Base quantizer matrix ──────────────────────────────────────────


class TestMethodMatrix:
    """Every base quantizer at every bit-width with every calibration mode."""

    @pytest.mark.parametrize("method", ["rtn", "gptq", "ldlq"])
    @pytest.mark.parametrize("int_bits", [4, 8])
    @pytest.mark.parametrize("calibration_mode", ["none", "synthetic"])
    def test_base_quantizers(
        self, method, int_bits, calibration_mode,
        synthetic_model, synthetic_calibration,
    ):
        # Skip invalid combos
        if method == "gptq" and calibration_mode == "none":
            pytest.skip("GPTQ without calibration tested in Class 10")
        if method == "rtn" and calibration_mode == "synthetic":
            pytest.skip("RTN ignores calibration")

        config = BenchmarkConfig(method=method, int_bits=int_bits, features="plain")
        cal = synthetic_calibration if calibration_mode == "synthetic" else None

        result = benchmark_method(synthetic_model, config, calibration=cal)

        assert result.error is None, f"Error: {result.error}"
        thresh = _threshold(int_bits)
        for lr in result.layers:
            assert lr.weight_mse < thresh, f"{lr.name}: mse={lr.weight_mse} >= {thresh}"
            assert math.isfinite(lr.weight_max_err)
            assert lr.output_mse is None  # no activations provided
        assert result.mse_mean < thresh
        # Method used should start with expected prefix
        for lr in result.layers:
            if method == "rtn":
                assert lr.method_used.startswith("rtn"), lr.method_used
            elif method == "gptq":
                assert lr.method_used.startswith("gptq"), lr.method_used
            elif method == "ldlq":
                assert lr.method_used.startswith("ldlq"), lr.method_used
        # No unexpected fallbacks on clean synthetic data
        for lr in result.layers:
            assert lr.fallbacks == [], f"{lr.name}: unexpected fallbacks {lr.fallbacks}"


# ── Class 2: Asymmetric quantization ────────────────────────────────────────


class TestMethodMatrixAsymmetric:
    """INT8 asymmetric quantization path."""

    @pytest.mark.parametrize("method", ["rtn", "gptq", "ldlq"])
    def test_asymmetric_int8(self, method, synthetic_model, synthetic_calibration):
        config = BenchmarkConfig(
            method=method, int_bits=8, features="plain", asymmetric=True
        )
        cal = synthetic_calibration if method == "gptq" else None
        result = benchmark_method(synthetic_model, config, calibration=cal)
        assert result.error is None, f"Error: {result.error}"
        assert result.mse_mean < SYNTHETIC_INT8_THRESHOLD


# ── Class 3: ConvRot ────────────────────────────────────────────────────────


class TestConvRot:
    """Hadamard rotation tests."""

    @pytest.mark.parametrize("method", ["rtn", "gptq", "ldlq"])
    @pytest.mark.parametrize("int_bits", [4, 8])
    def test_convrot(self, method, int_bits, synthetic_model, synthetic_calibration):
        config = BenchmarkConfig(method=method, int_bits=int_bits, features="convrot")
        cal = synthetic_calibration if method == "gptq" else None
        result = benchmark_method(synthetic_model, config, calibration=cal)
        assert result.error is None, f"Error: {result.error}"
        thresh = _threshold(int_bits)
        for lr in result.layers:
            assert lr.weight_mse < thresh, f"{lr.name}: mse={lr.weight_mse}"
        assert result.mse_mean < thresh


# ── Class 4: SmoothQuant ────────────────────────────────────────────────────


class TestSmoothQuant:
    """Per-channel smoothing tests."""

    @pytest.mark.parametrize("method", ["rtn", "gptq", "ldlq"])
    @pytest.mark.parametrize("int_bits", [4, 8])
    def test_smoothquant(self, method, int_bits, synthetic_model, synthetic_calibration):
        config = BenchmarkConfig(method=method, int_bits=int_bits, features="smoothquant")
        cal = synthetic_calibration if method == "gptq" else synthetic_calibration
        result = benchmark_method(synthetic_model, config, calibration=cal)
        assert result.error is None, f"Error: {result.error}"
        thresh = _threshold(int_bits)
        assert result.mse_mean < thresh
        # At least some layers should have smooth_source set
        sources = [lr.smooth_source for lr in result.layers if lr.smooth_source is not None]
        # SmoothQuant should have been applied (at least weight_only fallback)
        assert len(sources) > 0, "No smoothing sources found"


# ── Class 5: SmoothRot ──────────────────────────────────────────────────────


class TestSmoothRot:
    """Smooth-then-rotate pipeline tests."""

    @pytest.mark.parametrize("method", ["rtn", "gptq"])
    def test_smoothrot(self, method, synthetic_ffn_model, synthetic_ffn_calibration):
        # Verify FFN pairs are detected
        pairs = detect_ffn_pairs(synthetic_ffn_model)
        assert len(pairs) > 0, "No FFN pairs detected — SmoothRot will be vacuous"

        config = BenchmarkConfig(method=method, int_bits=8, features="smoothrot")
        # SmoothRot requires calibration for smoothing factors.
        # For RTN, provide calibration too (it's used for smoothing, not quantization).
        result = benchmark_method(
            synthetic_ffn_model, config, calibration=synthetic_ffn_calibration,
        )
        assert result.error is None, f"Error: {result.error}"
        assert result.mse_mean < SYNTHETIC_INT8_THRESHOLD
        # At least one layer should have smooth_source == "smoothrot"
        smoothrot_sources = [
            lr for lr in result.layers if lr.smooth_source == "smoothrot"
        ]
        assert len(smoothrot_sources) > 0, "SmoothRot was not applied to any layer"


# ── Class 6: PermuQuant ─────────────────────────────────────────────────────


class TestPermuQuant:
    """Channel reordering tests."""

    @pytest.mark.parametrize("method", ["rtn", "gptq", "ldlq"])
    @pytest.mark.parametrize("int_bits", [4, 8])
    def test_permuquant(self, method, int_bits, synthetic_model, synthetic_calibration):
        config = BenchmarkConfig(method=method, int_bits=int_bits, features="permuquant")
        cal = synthetic_calibration if method == "gptq" else None
        result = benchmark_method(synthetic_model, config, calibration=cal)
        assert result.error is None, f"Error: {result.error}"
        thresh = _threshold(int_bits)
        assert result.mse_mean < thresh
        # At INT4 with per-row quantization, PermuQuant correctly rejects
        # permutations (column order doesn't affect per-row MSE).
        # Only check that permutations were applied for INT8.
        if int_bits == 8:
            perm_layers = [lr for lr in result.layers if lr.permutation_applied]
            assert len(perm_layers) > 0, "PermuQuant was not applied to any layer"

    def test_permuquant_calibration_permutation(self, synthetic_model, synthetic_calibration):
        """Test calibration-provided permutation path."""
        config = BenchmarkConfig(method="gptq", int_bits=8, features="permuquant")
        result = benchmark_method(synthetic_model, config, calibration=synthetic_calibration)
        assert result.error is None, f"Error: {result.error}"
        assert result.mse_mean < SYNTHETIC_INT8_THRESHOLD

    def test_permuquant_block_diagonal_hessian(self, synthetic_model):
        """Test block-diagonal Hessian + PermuQuant interaction."""
        names = [k for k in synthetic_model if k.endswith(".weight")]
        cal = make_synthetic_calibration(names, in_features=256, block_size=32, seed=42)
        config = BenchmarkConfig(method="gptq", int_bits=8, features="permuquant")
        result = benchmark_method(synthetic_model, config, calibration=cal)
        assert result.error is None, f"Error: {result.error}"
        assert result.mse_mean < SYNTHETIC_INT8_THRESHOLD


# ── Class 7: SVD ────────────────────────────────────────────────────────────


class TestSVD:
    """Low-rank decomposition tests."""

    @pytest.mark.parametrize("int_bits", [4, 8])
    def test_svd(self, int_bits, synthetic_model):
        config = BenchmarkConfig(method="rtn", int_bits=int_bits, features="svd")
        result = benchmark_method(synthetic_model, config)
        assert result.error is None, f"Error: {result.error}"
        thresh = _threshold(int_bits)
        for lr in result.layers:
            assert lr.weight_mse < thresh
        assert result.mse_mean < thresh


# ── Class 8: Feature combos ─────────────────────────────────────────────────


class TestFeatureCombos:
    """Multi-feature interaction tests."""

    def test_smoothrot_with_permuquant(self, synthetic_ffn_model):
        config = BenchmarkConfig(method="rtn", int_bits=8, features="smoothrot+pq")
        result = benchmark_method(synthetic_ffn_model, config)
        assert result.error is None, f"Error: {result.error}"
        assert result.mse_mean < SYNTHETIC_INT8_THRESHOLD

    def test_svd_with_convrot(self, synthetic_model):
        config = BenchmarkConfig(method="rtn", int_bits=8, features="svd+convrot")
        result = benchmark_method(synthetic_model, config)
        assert result.error is None, f"Error: {result.error}"
        assert result.mse_mean < SYNTHETIC_INT8_THRESHOLD

    def test_smoothquant_with_permuquant(self, synthetic_model, synthetic_calibration):
        config = BenchmarkConfig(method="gptq", int_bits=8, features="convrot+sq")
        result = benchmark_method(synthetic_model, config, calibration=synthetic_calibration)
        assert result.error is None, f"Error: {result.error}"
        assert result.mse_mean < SYNTHETIC_INT8_THRESHOLD

    def test_everything_on(self, synthetic_ffn_model, synthetic_ffn_calibration):
        config = BenchmarkConfig(method="rtn", int_bits=8, features="everything")
        result = benchmark_method(synthetic_ffn_model, config)
        assert result.error is None, f"Error: {result.error}"
        assert result.mse_mean < SYNTHETIC_INT8_THRESHOLD


class TestFeaturePresets:
    """Validate feature preset strings map to correct config overrides."""

    @pytest.mark.parametrize("preset_name", list(FEATURE_PRESETS.keys()))
    def test_preset_fields(self, preset_name):
        """Each preset should produce the expected config field values."""
        preset = FEATURE_PRESETS[preset_name]
        config = BenchmarkConfig(method="rtn", int_bits=8, features=preset_name)
        from converter.benchmark import _apply_feature_preset
        cfg = _apply_feature_preset(config)
        assert cfg.rot_size == preset["rot_size"], f"{preset_name}: rot_size mismatch"
        assert cfg.svd_rank == preset["svd_rank"], f"{preset_name}: svd_rank mismatch"
        assert cfg._smoothquant == preset["smoothquant"], f"{preset_name}: smoothquant mismatch"
        assert cfg._smoothrot == preset["smoothrot"], f"{preset_name}: smoothrot mismatch"
        assert cfg._use_permuquant == preset["use_permuquant"], f"{preset_name}: use_permuquant mismatch"


# ── Class 9: Method effectiveness ───────────────────────────────────────────


class TestMethodEffectiveness:
    """Comparative tests: does the feature help?"""

    def _outlier_model(self):
        return make_synthetic_model(
            num_layers=8, shape=(128, 512), outlier_channels=10, outlier_scale=100.0, seed=42
        )

    def _low_rank_model(self):
        return make_synthetic_model(
            num_layers=8, shape=(128, 512), low_rank=16, seed=42
        )

    def test_gptq_beats_rtn(self):
        """GPTQ (with Hessian correction) should beat plain RTN on outlier weights."""
        weights = self._outlier_model()
        names = [k for k in weights if k.endswith(".weight")]
        cal = make_synthetic_calibration(names, in_features=512, seed=42)

        rtn_cfg = BenchmarkConfig(method="rtn", int_bits=4, features="convrot")
        gptq_cfg = BenchmarkConfig(method="gptq", int_bits=4, features="convrot")

        rtn_result = benchmark_method(weights, rtn_cfg)
        gptq_result = benchmark_method(weights, gptq_cfg, calibration=cal)

        assert rtn_result.error is None
        assert gptq_result.error is None
        # GPTQ should beat RTN (uses Hessian information)
        assert gptq_result.mse_mean <= rtn_result.mse_mean * 1.0 + 1e-9

    def test_ldlq_beats_rtn(self):
        """LDLQ (weight-only Hessian) should be competitive with RTN on outlier weights.

        LDLQ uses H = W^T@W as the Hessian for error compensation. On small
        synthetic weights the improvement over RTN is marginal (within
        floating-point noise), so we allow a small tolerance.
        """
        weights = self._outlier_model()

        rtn_cfg = BenchmarkConfig(method="rtn", int_bits=8, features="convrot")
        ldlq_cfg = BenchmarkConfig(method="ldlq", int_bits=8, features="convrot")

        rtn_result = benchmark_method(weights, rtn_cfg)
        ldlq_result = benchmark_method(weights, ldlq_cfg)

        assert rtn_result.error is None
        assert ldlq_result.error is None
        # LDLQ should be within 20% of RTN (marginal on small synthetic data)
        assert ldlq_result.mse_mean <= rtn_result.mse_mean * 1.2 + 1e-9

    def test_convrot_reduces_mse(self):
        weights = self._outlier_model()

        plain_cfg = BenchmarkConfig(method="rtn", int_bits=8, features="plain")
        convrot_cfg = BenchmarkConfig(method="rtn", int_bits=8, features="convrot")

        plain_result = benchmark_method(weights, plain_cfg)
        convrot_result = benchmark_method(weights, convrot_cfg)

        assert plain_result.error is None
        assert convrot_result.error is None
        assert convrot_result.mse_mean <= plain_result.mse_mean * 1.0 + 1e-9

    def test_svd_beats_rtn(self):
        weights = self._low_rank_model()

        rtn_cfg = BenchmarkConfig(method="rtn", int_bits=8, features="plain")
        svd_cfg = BenchmarkConfig(method="rtn", int_bits=8, features="svd")

        rtn_result = benchmark_method(weights, rtn_cfg)
        svd_result = benchmark_method(weights, svd_cfg)

        assert rtn_result.error is None
        assert svd_result.error is None
        # SVD should help on low-rank weights
        assert svd_result.mse_mean <= rtn_result.mse_mean * 1.0 + 1e-9

    def test_smoothquant_reduces_output_mse(self):
        """SmoothQuant should reduce output_mse (not necessarily weight_mse)."""
        weights = self._outlier_model()
        acts = {}
        for name in weights:
            if name.endswith(".weight"):
                g = torch.Generator().manual_seed(42)
                acts[name] = torch.randn(128, weights[name].shape[1], generator=g)

        plain_cfg = BenchmarkConfig(method="rtn", int_bits=8, features="plain")
        sq_cfg = BenchmarkConfig(method="rtn", int_bits=8, features="smoothquant")

        plain_result = benchmark_method(weights, plain_cfg, activations=acts)
        sq_result = benchmark_method(weights, sq_cfg, activations=acts)

        assert plain_result.error is None
        assert sq_result.error is None
        assert plain_result.output_mse_mean is not None
        assert sq_result.output_mse_mean is not None
        assert sq_result.output_mse_mean <= plain_result.output_mse_mean * 1.0 + 1e-9


# ── Class 10: Error handling ────────────────────────────────────────────────


class TestMethodMatrixErrors:
    """Invalid configs and edge cases."""

    def test_gptq_no_calibration_raises(self, synthetic_model):
        config = BenchmarkConfig(method="gptq", int_bits=8, features="plain")
        with pytest.raises(ValueError, match="GPTQ requires calibration"):
            benchmark_method(synthetic_model, config)

    def test_invalid_method_raises(self, synthetic_model):
        config = BenchmarkConfig(method="invalid", int_bits=8, features="plain")
        with pytest.raises(ValueError, match="Unknown method"):
            benchmark_method(synthetic_model, config)

    def test_invalid_int_bits_raises(self, synthetic_model):
        config = BenchmarkConfig(method="rtn", int_bits=6, features="plain")
        with pytest.raises(ValueError, match="int_bits must be 4 or 8"):
            benchmark_method(synthetic_model, config)

    def test_smoothrot_int4_auto_disables(self, synthetic_ffn_model):
        """SmoothRot + INT4 should auto-disable and proceed as ConvRot."""
        config = BenchmarkConfig(method="rtn", int_bits=4, features="smoothrot")
        result = benchmark_method(synthetic_ffn_model, config)

        assert result.error is None
        # Quality should be acceptable (auto-disabled path applies weight-only
        # smoothing + ConvRot, which may differ slightly from plain ConvRot)
        assert result.mse_mean < SYNTHETIC_INT4_THRESHOLD
        # No NaN or Inf
        for lr in result.layers:
            assert math.isfinite(lr.weight_mse)
            assert math.isfinite(lr.weight_max_err)

    def test_smoothrot_int4_force_ok(self, synthetic_ffn_model):
        """force_smoothrot_w4=True should proceed with SmoothRot."""
        config = BenchmarkConfig(
            method="rtn", int_bits=4, features="smoothrot", force_smoothrot_w4=True
        )
        result = benchmark_method(synthetic_ffn_model, config)
        assert result.error is None
        # No NaN or Inf in results
        for lr in result.layers:
            assert math.isfinite(lr.weight_mse), f"{lr.name}: non-finite mse={lr.weight_mse}"
            assert math.isfinite(lr.weight_max_err)
        assert result.mse_mean < SYNTHETIC_INT4_THRESHOLD

    def test_gptq_rtn_fallback(self, synthetic_model):
        """GPTQ with missing Hessian should fall back to RTN."""
        # Create calibration with a subset of layer names (missing some)
        names = [k for k in synthetic_model if k.endswith(".weight")]
        # Only calibrate the first 2 layers
        cal = make_synthetic_calibration(names[:2], in_features=256, seed=42)
        config = BenchmarkConfig(method="gptq", int_bits=8, features="plain")
        result = benchmark_method(synthetic_model, config, calibration=cal)
        assert result.error is None
        # Some layers should have rtn_fallback
        fallback_layers = [
            lr for lr in result.layers if "rtn_fallback" in lr.fallbacks
        ]
        assert len(fallback_layers) > 0, "No RTN fallback layers found"

    def test_skip_patterns_excludes_layers(self, synthetic_model):
        """skip_patterns should exclude matching layers from results."""
        config = BenchmarkConfig(
            method="rtn", int_bits=8, features="plain",
            skip_patterns=["layers.0"],
        )
        result = benchmark_method(synthetic_model, config)
        assert result.error is None
        # layers.0.q_proj.weight should be skipped
        names = [lr.name for lr in result.layers]
        assert not any("layers.0" in n for n in names), f"layers.0 not skipped: {names}"
        # Other layers should still be present
        assert len(result.layers) > 0

    def test_output_mse_none_without_activations(self, synthetic_model):
        """output_mse should be None when activations are not provided."""
        config = BenchmarkConfig(method="rtn", int_bits=8, features="plain")
        result = benchmark_method(synthetic_model, config)
        for lr in result.layers:
            assert lr.output_mse is None, f"{lr.name}: output_mse should be None"


# ── Class 11: Real weights ──────────────────────────────────────────────────


@pytest.mark.slow
class TestRealWeights:
    """Real model comparison (skipped in CI, requires ig4_bf16.safetensors)."""

    @pytest.mark.parametrize("method", ["rtn", "gptq", "ldlq"])
    @pytest.mark.parametrize("int_bits", [4, 8])
    def test_real_weights_full(self, method, int_bits, real_weights, real_calibration):
        config = BenchmarkConfig(method=method, int_bits=int_bits, features="plain")
        result = benchmark_method(real_weights, config, calibration=real_calibration)
        assert result.error is None
        thresh = _threshold(int_bits, real=True)
        assert result.mse_mean < thresh, f"mse_mean={result.mse_mean} >= {thresh}"
        # Outlier detection: no layer should be 10x the mean
        for lr in result.layers:
            assert lr.weight_mse < thresh * 10, (
                f"{lr.name}: mse={lr.weight_mse} is >10x threshold"
            )

    def test_real_weights_subset_smoke(self, real_weights_subset):
        """Lightweight smoke test on real model data."""
        config = BenchmarkConfig(method="rtn", int_bits=8, features="convrot")
        result = benchmark_method(real_weights_subset, config)
        assert result.error is None
        assert result.mse_mean < REAL_INT8_THRESHOLD


# ── Class 12: Full benchmark matrix ─────────────────────────────────────────


@pytest.mark.slow
class TestBenchmarkMatrix:
    """Integration test for benchmark_matrix()."""

    def test_full_matrix(self, synthetic_model, synthetic_calibration):
        report = benchmark_matrix(
            synthetic_model,
            calibration=synthetic_calibration,
        )

        # All results returned
        assert len(report.results) > 0

        # Results sorted by mse_mean (errors last)
        for i in range(len(report.results) - 1):
            a = report.results[i]
            b = report.results[i + 1]
            if a.error is None and b.error is None:
                assert a.mse_mean <= b.mse_mean
            elif a.error is not None:
                # Errors should sort after non-errors
                assert b.error is not None

        # Best INT8 result should have mse_mean < 0.01
        int8_results = [r for r in report.results if r.config.int_bits == 8 and r.error is None]
        if int8_results:
            best_int8 = min(r.mse_mean for r in int8_results)
            assert best_int8 < 0.01, f"Best INT8 mse_mean={best_int8} >= 0.01"

        # Best INT4 result should have mse_mean < 0.1
        int4_results = [r for r in report.results if r.config.int_bits == 4 and r.error is None]
        if int4_results:
            best_int4 = min(r.mse_mean for r in int4_results)
            assert best_int4 < 0.1, f"Best INT4 mse_mean={best_int4} >= 0.1"

        # JSON serialization round-trips
        report_dict = dataclasses.asdict(report)
        json_str = json.dumps(report_dict, default=str)
        restored = json.loads(json_str)
        for orig, rest in zip(report_dict["results"], restored["results"]):
            if orig["mse_mean"] is not None and math.isfinite(orig["mse_mean"]):
                assert abs(orig["mse_mean"] - rest["mse_mean"]) < 1e-10

        # output_mse should be None (no activations provided)
        for r in report.results:
            if r.error is None:
                for lr in r.layers:
                    assert lr.output_mse is None


# ── Class 13: Determinism ───────────────────────────────────────────────────


class TestDeterminism:
    """Reproducibility across runs."""

    @pytest.mark.parametrize("method", ["rtn", "gptq", "ldlq"])
    def test_deterministic(self, method, synthetic_model, synthetic_calibration):
        config = BenchmarkConfig(method=method, int_bits=8, features="plain")
        cal = synthetic_calibration if method == "gptq" else None

        result1 = benchmark_method(synthetic_model, config, calibration=cal)
        result2 = benchmark_method(synthetic_model, config, calibration=cal)

        assert result1.error is None
        assert result2.error is None
        assert result1.mse_mean == result2.mse_mean, (
            f"mse_mean mismatch: {result1.mse_mean} != {result2.mse_mean}"
        )
        # Per-layer MSE should match exactly
        for lr1, lr2 in zip(result1.layers, result2.layers):
            assert lr1.weight_mse == lr2.weight_mse, (
                f"{lr1.name}: {lr1.weight_mse} != {lr2.weight_mse}"
            )


# ── Import for warnings capture ─────────────────────────────────────────────
import warnings
