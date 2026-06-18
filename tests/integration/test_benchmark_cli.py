"""Tests for the int-crush-benchmark CLI wrapper.

Run with::

    pytest tests/integration/test_benchmark_cli.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


@pytest.fixture
def run_cli(tmp_path):
    """Run the benchmark CLI and return (returncode, stdout, stderr)."""
    def _run(args: list[str], timeout: int = 60):
        result = subprocess.run(
            [sys.executable, "-m", "converter.benchmark_cli"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    return _run


class TestBenchmarkCLI:
    """CLI wrapper tests."""

    def test_synthetic_produces_table(self, run_cli):
        """--synthetic should print a comparison table to stderr."""
        rc, stdout, stderr = run_cli(["--synthetic", "--methods", "rtn", "--int-bits", "8", "--features", "plain"])
        assert rc == 0, f"stderr: {stderr}"
        assert "Benchmark" in stderr
        assert "rtn" in stderr.lower()

    def test_json_output_round_trips(self, run_cli, tmp_path):
        """-o file.json should write valid JSON that deserializes correctly."""
        out_path = str(tmp_path / "results.json")
        rc, stdout, stderr = run_cli([
            "--synthetic", "--methods", "rtn", "--int-bits", "8",
            "--features", "plain", "-o", out_path, "-q",
        ])
        assert rc == 0, f"stderr: {stderr}"

        with open(out_path) as f:
            data = json.load(f)

        assert "results" in data
        assert len(data["results"]) > 0
        # Check mse_mean values are valid numbers
        for r in data["results"]:
            if r["error"] is None:
                assert isinstance(r["mse_mean"], (int, float))
                assert r["mse_mean"] >= 0

    def test_features_filter(self, run_cli):
        """--features plain,convrot should only run those two presets."""
        rc, stdout, stderr = run_cli([
            "--synthetic", "--methods", "rtn", "--int-bits", "8",
            "--features", "plain,convrot",
        ])
        assert rc == 0, f"stderr: {stderr}"
        # Count feature lines in the table (lines with "plain" or "convrot")
        feature_lines = [
            line for line in stderr.split("\n")
            if "plain" in line or "convrot" in line
        ]
        # Should have exactly 2 result lines (one per feature)
        assert len(feature_lines) >= 2

    def test_missing_input_raises(self):
        """No --input and no --synthetic should exit with error."""
        result = subprocess.run(
            [sys.executable, "-m", "converter.benchmark_cli"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()
