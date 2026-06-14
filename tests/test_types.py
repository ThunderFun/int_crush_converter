"""Tests for converter.types — QuantizationResult field access."""

import torch

from converter.types import QuantizationResult


class TestQuantizationResult:
    """Verify QuantizationResult fields are accessible and correct."""

    def _make_result(self) -> QuantizationResult:
        return QuantizationResult(
            quantized_W=torch.randint(-8, 7, (4, 8), dtype=torch.int8),
            scales=torch.ones(4, 1, dtype=torch.float16),
            zero_points=None,
            mse=0.01,
            max_err=0.5,
            method_used="ldlq_triton",
            fallbacks=[],
        )

    def test_named_attribute_access(self):
        """All fields must be accessible as named attributes."""
        result = self._make_result()
        assert torch.equal(result.quantized_W, result.quantized_W)
        assert torch.equal(result.scales, result.scales)
        assert result.zero_points is None
        assert result.mse == 0.01
        assert result.max_err == 0.5
        assert result.method_used == "ldlq_triton"
        assert result.fallbacks == []

    def test_all_fields_accessible(self):
        """All 7 fields must be accessible as named attributes."""
        result = QuantizationResult(
            quantized_W=torch.zeros(2, 4, dtype=torch.int8),
            scales=torch.ones(2, 1, dtype=torch.float16),
            zero_points=torch.zeros(2, 1, dtype=torch.float16),
            mse=0.05,
            max_err=1.2,
            method_used="gptq_triton",
            fallbacks=["oom_cpu"],
        )
        assert result.quantized_W.shape == (2, 4)
        assert result.scales.shape == (2, 1)
        assert result.zero_points.shape == (2, 1)
        assert result.mse == 0.05
        assert result.max_err == 1.2
        assert result.method_used == "gptq_triton"
        assert result.fallbacks == ["oom_cpu"]

    def test_not_iterable(self):
        """QuantizationResult must not be iterable (no __iter__)."""
        result = self._make_result()
        assert not hasattr(result, '__iter__') or not callable(getattr(result, '__iter__', None))
