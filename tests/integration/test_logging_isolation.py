"""Tests for logging isolation — importing as library must not clobber root logger."""

import logging

import pytest
import torch
from safetensors.torch import save_file

from converter.log import logger


def _make_tiny_safetensors(path: str) -> str:
    state = {
        "layers.0.q_proj.weight": torch.randn(16, 64, dtype=torch.float32),
        "layers.0.k_proj.weight": torch.randn(16, 64, dtype=torch.float32),
    }
    save_file(state, path)
    return path


class TestLoggingIsolation:

    def test_import_does_not_clobber_root_logger(self):
        root = logging.getLogger()
        initial_handlers = list(root.handlers)

        import converter  # noqa: F401

        assert list(root.handlers) == initial_handlers, (
            "Importing 'converter' modified root logger handlers"
        )

    def test_package_logger_has_null_handler(self):
        assert any(isinstance(h, logging.NullHandler) for h in logger.handlers), (
            "Package logger should have a NullHandler"
        )

    def test_cli_main_configures_package_logger(self, tmp_path, monkeypatch):
        from converter.cli import main

        logger.handlers.clear()

        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        _make_tiny_safetensors(input_path)

        monkeypatch.setattr("sys.argv", [
            "int-crush-convert", "-i", input_path, "-o", output_dir, "--quiet",
        ])

        main()

        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
        ]
        assert len(stream_handlers) >= 1, (
            "CLI main() should have added a StreamHandler to the package logger"
        )

    def test_cli_logging_produces_output(self, tmp_path, monkeypatch, capsys):
        from converter.cli import main

        logger.handlers.clear()

        input_path = str(tmp_path / "input.safetensors")
        output_dir = str(tmp_path / "output")
        _make_tiny_safetensors(input_path)

        monkeypatch.setattr("sys.argv", [
            "int-crush-convert", "-i", input_path, "-o", output_dir,
        ])

        main()

        captured = capsys.readouterr()
        assert len(captured.out) > 0 or len(captured.err) > 0, (
            "CLI produced no output — logging may be misconfigured"
        )
