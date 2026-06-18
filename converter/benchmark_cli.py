"""CLI wrapper for converter.benchmark.

Entry point: ``int-crush-benchmark``

Usage::

    int-crush-benchmark -i model.safetensors -c cal.pt [options]
    int-crush-benchmark --synthetic [options]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from .benchmark import (
    FEATURE_PRESETS,
    benchmark_matrix,
    benchmark_method,
    make_synthetic_calibration,
    make_synthetic_model,
    _override_preset_rot_size,
)
from .calibration_io import load_calibration
from .types import BenchmarkConfig, BenchmarkReport

# ANSI color codes — auto-disabled when stderr is not a TTY.
_USE_COLOR = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

_RESET  = "\033[0m"
_DIM    = "\033[2m"
_BOLD   = "\033[1m"
_GREEN  = "\033[38;2;120;230;120m"
_YELLOW = "\033[38;2;240;220;90m"
_RED    = "\033[38;2;200;80;80m"
_CYAN   = "\033[38;2;80;190;200m"


def _c(text: str, color: str) -> str:
    """Wrap text in an ANSI color, or return plain if colors are off."""
    if not _USE_COLOR:
        return text
    return f"{color}{text}{_RESET}"


def _mse_color(mse: float) -> str:
    """Pick a color for a relative MSE value: green=good, yellow=ok, red=bad."""
    if mse < 1e-4:
        return _GREEN
    elif mse < 1e-2:
        return _YELLOW
    else:
        return _RED


def _quality_score(mse_mean: float, max_err: float) -> float:
    """Composite score for ranking: lower is better.

    Penalizes combos where a single layer has catastrophic error (high max_err).
    ``mse_mean * (1 + max_err)`` — a combo with max_err=1.0 gets 2× the
    mse_mean penalty, catching cases where the mean is fine but one layer
    is completely broken.
    """
    return mse_mean * (1.0 + max_err)


def _format_table(report: BenchmarkReport) -> str:
    """Format a BenchmarkReport as a human-readable colored table."""
    lines = []
    model_desc = report.model_path or "synthetic"
    cal_desc = report.calibration_path or "none"

    shape_desc = "varies"
    if report.layer_shapes:
        shapes = set(report.layer_shapes.values())
        if len(shapes) == 1:
            shape_desc = str(list(shapes)[0])
        else:
            shape_desc = f"{len(shapes)} shapes"

    lines.append(_c(f"Benchmark — {model_desc} ({report.num_layers} layers, shape {shape_desc})", _BOLD))
    if report.calibration_path:
        lines.append(f"Calibration: {cal_desc}")
    header = (
        f"{'Method':<10s} {'Bits':>4s} {'Features':<18s} "
        f"{'MSE (mean)':>12s} {'MSE (p95)':>12s} {'Max Err':>10s} "
        f"{'vs Plain':>8s} {'SNR(dB)':>8s} {'Time':>6s}"
    )
    sep_len = len(header)
    lines.append("=" * sep_len)
    lines.append(header)
    lines.append("-" * sep_len)

    # Find best per bit-width for highlighting, and baseline plain MSE
    best_int8 = None
    best_int4 = None
    plain_mse: dict[int, float] = {}  # int_bits -> mse_mean of plain RTN
    for r in report.results:
        if r.error is not None:
            continue
        if r.config.int_bits == 8:
            if best_int8 is None or _quality_score(r.mse_mean, r.max_err) < _quality_score(best_int8.mse_mean, best_int8.max_err):
                best_int8 = r
        else:
            if best_int4 is None or _quality_score(r.mse_mean, r.max_err) < _quality_score(best_int4.mse_mean, best_int4.max_err):
                best_int4 = r
        if r.config.features == "plain" and r.config.method == "rtn":
            plain_mse[r.config.int_bits] = r.mse_mean

    for r in report.results:
        cfg = r.config
        if r.error is not None:
            lines.append(
                f"{cfg.method:<10s} {cfg.int_bits:>4d} {cfg.features:<18s} "
                f"{_c('ERROR', _RED):>21s} {_c(r.error, _DIM)}"
            )
            continue

        snr_vals = [lr.snr_db for lr in r.layers if lr.snr_db != float("inf")]
        snr_mean = sum(snr_vals) / len(snr_vals) if snr_vals else float("nan")

        # Highlight the best row
        is_best = (r is best_int8) or (r is best_int4)
        prefix = _c(">>>", _GREEN) + " " if is_best else "    "

        # vs Plain: % improvement over plain RTN for this bit-width
        baseline = plain_mse.get(cfg.int_bits)
        if baseline is not None and baseline > 0:
            improvement = (baseline - r.mse_mean) / baseline * 100
            if improvement > 0.05:
                vs_plain = _c(f"{improvement:>+7.1f}%", _GREEN)
            elif improvement < -0.05:
                vs_plain = _c(f"{improvement:>+7.1f}%", _RED)
            else:
                vs_plain = _c(f"{improvement:>+7.1f}%", _DIM)
        else:
            vs_plain = _c("   base", _DIM)

        mse_str = _c(f"{r.mse_mean:>12.2e}", _mse_color(r.mse_mean))
        p95_str = _c(f"{r.mse_p95:>12.2e}", _mse_color(r.mse_p95))
        max_str = _c(f"{r.max_err:>10.2e}", _mse_color(r.max_err))

        line = (
            f"{prefix}{cfg.method:<10s} {cfg.int_bits:>4d} {cfg.features:<18s} "
            f"{mse_str} {p95_str} {max_str} "
            f"{vs_plain} {snr_mean:>8.1f} {r.elapsed_seconds:>5.1f}s"
        )
        if is_best:
            line = _c(line, _BOLD)
        lines.append(line)

    lines.append("-" * sep_len)

    # Best INT8 and INT4 (using composite score)
    if best_int8:
        lines.append(
            _c(f"Best INT8: {best_int8.config.method} + {best_int8.config.features} "
               f"(MSE={best_int8.mse_mean:.2e}, max_err={best_int8.max_err:.2e})", _GREEN)
        )
    if best_int4:
        lines.append(
            _c(f"Best INT4: {best_int4.config.method} + {best_int4.config.features} "
               f"(MSE={best_int4.mse_mean:.2e}, max_err={best_int4.max_err:.2e})", _GREEN)
        )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``int-crush-benchmark``."""
    parser = argparse.ArgumentParser(
        prog="int-crush-benchmark",
        description="Benchmark quantization methods side-by-side.",
    )
    parser.add_argument(
        "-i", "--input", type=str, default=None,
        help="Input safetensors file (required unless --synthetic).",
    )
    parser.add_argument(
        "-c", "--calibration", type=str, default=None,
        help="Calibration .pt file.",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Write JSON report to this path.",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic weights (no --input needed).",
    )
    parser.add_argument(
        "--layers", type=int, default=8,
        help="Number of synthetic layers (default: 8).",
    )
    parser.add_argument(
        "--shape", type=str, default="64,256",
        help="Synthetic layer shape as 'out,in' (default: 64,256).",
    )
    parser.add_argument(
        "--methods", type=str, default="rtn,gptq,ldlq",
        help="Comma-separated methods to test (default: rtn,gptq,ldlq).",
    )
    parser.add_argument(
        "--int-bits", type=str, default="4,8",
        help="Comma-separated bit-widths (default: 4,8).",
    )
    parser.add_argument(
        "--features", type=str, default=None,
        help="Comma-separated feature presets (default: all).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--rot-size", type=int, default=None,
        help="Override Hadamard rot_size for rotation presets (e.g. 64, 256). "
             "When set, all rotation presets use this value instead of their default. "
             "Also overrides the calibration's rot_size for adaptation.",
    )
    parser.add_argument(
        "--max-layers", type=int, default=None,
        help="Limit benchmark to N random quantizable layers (for speed on large models).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show per-layer details.",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress table, only write JSON.",
    )

    args = parser.parse_args(argv)

    # Validate inputs
    if not args.synthetic and args.input is None:
        parser.error("--input is required unless --synthetic is used.")

    # Parse comma-separated args
    methods = [m.strip() for m in args.methods.split(",")]
    int_bits = [int(b.strip()) for b in args.int_bits.split(",")]
    features = None
    if args.features:
        features = [f.strip() for f in args.features.split(",")]

    # Parse shape
    shape = tuple(int(x) for x in args.shape.split(","))
    if len(shape) != 2:
        parser.error("--shape must be 'out,in' (e.g. 64,256)")

    # Load data
    if args.synthetic:
        weights = make_synthetic_model(
            num_layers=args.layers, shape=shape, seed=args.seed
        )
        calibration = None
        if "gptq" in methods:
            names = [k for k in weights if k.endswith(".weight")]
            calibration = make_synthetic_calibration(
                names, in_features=shape[1], seed=args.seed
            )
    else:
        from safetensors.torch import load_file
        weights = load_file(args.input)
        calibration = None
        if args.calibration:
            calibration = load_calibration(args.calibration)

    # Show calibration metadata
    if calibration is not None and not args.quiet:
        cal_meta = calibration.get("metadata", {})
        cal_rot_size = int(cal_meta.get("rot_size", 0))
        cal_rotated = bool(cal_meta.get("hessian_rotated", False))
        cal_samples = int(cal_meta.get("num_samples", 0))
        cal_steps = int(cal_meta.get("num_steps", 0))
        cal_block = int(cal_meta.get("hessian_block_size", 0))
        rot_desc = f"rot_size={cal_rot_size}" if cal_rotated else "no rotation"
        print(
            f"Calibration: {cal_samples} samples, {cal_steps} steps, "
            f"{rot_desc}, block_size={cal_block}",
            file=sys.stderr,
        )

    # ── Build feature presets (with optional --rot-size override) ──
    effective_presets = FEATURE_PRESETS
    if args.rot_size is not None:
        if not ((args.rot_size & (args.rot_size - 1)) == 0 and args.rot_size > 0):
            parser.error(f"--rot-size must be a power of 2, got {args.rot_size}")
        effective_presets = _override_preset_rot_size(FEATURE_PRESETS, args.rot_size)
        if not args.quiet:
            print(
                f"Overriding rotation presets to rot_size={args.rot_size}",
                file=sys.stderr,
            )

    # Show which presets will be skipped for GPTQ/LDLQ
    if calibration is not None and not args.quiet:
        cal_meta = calibration.get("metadata", {})
        cal_rotated = bool(cal_meta.get("hessian_rotated", False))
        cal_rot_size = int(cal_meta.get("rot_size", 0))
        if cal_rotated:
            no_rot_feats = [
                f for f in (features or effective_presets.keys())
                if effective_presets.get(f, {}).get("rot_size", 0) == 0
            ]
            rot_feats = [
                f for f in (features or effective_presets.keys())
                if effective_presets.get(f, {}).get("rot_size", 0) > 0
            ]
            if ("gptq" in methods or "ldlq" in methods):
                if rot_feats:
                    # Determine effective rot_size for rotation presets
                    sample_rot = next(
                        effective_presets[f]["rot_size"]
                        for f in rot_feats
                        if effective_presets.get(f, {}).get("rot_size", 0) > 0
                    )
                    if sample_rot == cal_rot_size:
                        print(
                            f"  GPTQ/LDLQ: rotation presets match calibration "
                            f"(rot_size={cal_rot_size})",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            _c(
                                f"  GPTQ/LDLQ: rotation presets rot_size={sample_rot} "
                                f"!= calibration rot_size={cal_rot_size} — will skip",
                                _YELLOW,
                            ),
                            file=sys.stderr,
                        )
                if no_rot_feats:
                    print(
                        _c(
                            f"  GPTQ/LDLQ: skipping {len(no_rot_feats)} no-rotation "
                            f"presets (calibration is rotated)",
                            _YELLOW,
                        ),
                        file=sys.stderr,
                    )

    # --max-layers: subsample N random quantizable layers for speed
    if args.max_layers is not None:
        from .benchmark import _should_skip, DEFAULT_SKIP_PATTERNS
        import random as _random
        _random.seed(args.seed)
        quantizable = [
            k for k in weights
            if not _should_skip(k, DEFAULT_SKIP_PATTERNS) and weights[k].dim() == 2
        ]
        if len(quantizable) > args.max_layers:
            selected = set(_random.sample(quantizable, args.max_layers))
            # Keep all non-quantizable layers (norms, embeds, etc.) plus the
            # selected subset.  This way skip_patterns still works correctly
            # and calibration name-mapping still finds the selected layers.
            weights = {
                k: v for k, v in weights.items()
                if k in selected or weights[k].dim() != 2 or _should_skip(k, DEFAULT_SKIP_PATTERNS)
            }
            if not args.quiet:
                print(
                    f"Subsampled {args.max_layers} layers from {len(quantizable)} "
                    f"quantizable layers (seed={args.seed})",
                    file=sys.stderr,
                )

    # Progress callbacks
    def _on_combo_done(result, combo_idx: int, total_combos: int) -> None:
        if args.quiet:
            return
        cfg = result.config
        if result.error is not None:
            print(
                f"[{combo_idx}/{total_combos}] {cfg.method} {cfg.int_bits}-bit "
                f"{cfg.features}: ERROR — {result.error}",
                file=sys.stderr,
            )
        else:
            print(
                f"[{combo_idx}/{total_combos}] {cfg.method} {cfg.int_bits}-bit "
                f"{cfg.features}: mse={result.mse_mean:.2e}  "
                f"({result.elapsed_seconds:.1f}s)",
                file=sys.stderr,
            )

    # Run benchmark
    report = benchmark_matrix(
        weights,
        calibration=calibration,
        methods=methods,
        int_bits=int_bits,
        features=features,
        feature_presets=effective_presets,
        progress_callback=_on_combo_done,
    )
    # Fill in model/calibration paths
    report.model_path = args.input
    report.calibration_path = args.calibration

    # Output — verbose per-layer details first (if requested), then summary table last
    if args.verbose and not args.quiet:
        print("\nPer-layer details:", file=sys.stderr)
        for r in report.results:
            if r.error is not None:
                continue
            print(f"\n  {r.config.method} {r.config.int_bits}-bit {r.config.features}:", file=sys.stderr)
            for lr in r.layers:
                print(
                    f"    {lr.name:<40s} mse={lr.weight_mse:.2e} "
                    f"snr={lr.snr_db:.1f}dB method={lr.method_used}",
                    file=sys.stderr,
                )
        print("", file=sys.stderr)

    if not args.quiet:
        table = _format_table(report)
        print(table, file=sys.stderr)

    if args.output:
        report_dict = dataclasses.asdict(report)
        with open(args.output, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
        if not args.quiet:
            print(f"\nJSON report written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
