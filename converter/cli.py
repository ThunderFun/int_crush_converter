"""CLI entry point for INT-Crush quantization with PermuQuant and GPTQ.

Usage:
    python -m converter -i in.safetensors -o out/ --rot-size 256 --permuquant
    python -m converter -i in.safetensors -o out/ --rot-size 256 -c cal.pt --quant-method gptq
"""

import argparse
import logging
import os
import sys

from .log import logger
from .pipeline import quantize_model, DEFAULT_SKIP_PATTERNS
from .types import ProgressInfo, ProgressSummary, QuantizeConfig

# ANSI escape codes.
# Progress output goes to stderr so stdout stays clean for piped output.
# Auto-disabled when stderr is not a TTY (piped output, CI logs, etc.).
_USE_COLOR = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

_RESET  = "\033[0m"
_DIM    = "\033[2m"
_BOLD   = "\033[1m"
# 24-bit colors for text.
_GREEN  = "\033[38;2;120;230;120m"
_YELLOW = "\033[38;2;240;220;90m"
_RED    = "\033[38;2;200;80;80m"
_BLUE   = "\033[38;2;100;160;230m"
_CYAN   = "\033[38;2;80;190;200m"

# Anchor points for the progress gradient (percentage → RGB).
# Smoothly interpolated via 24-bit true color.
_GRADIENT_ANCHORS = [
    (0,   (200, 80,  80)),    # red
    (50,  (220, 200, 70)),    # yellow
    (100, (100, 200, 100)),   # green
]


def _c(text: str, color: str) -> str:
    """Wrap *text* in an ANSI color, or return it plain if colors are off."""
    if not _USE_COLOR:
        return text
    return f"{color}{text}{_RESET}"


def _pct_color(pct: float) -> str:
    """Pick a smooth 24-bit color from red (0%) to green (100%)."""
    pct = max(0.0, min(100.0, pct))
    # Find the two anchors we're between
    for i in range(len(_GRADIENT_ANCHORS) - 1):
        p0, c0 = _GRADIENT_ANCHORS[i]
        p1, c1 = _GRADIENT_ANCHORS[i + 1]
        if pct <= p1:
            t = (pct - p0) / (p1 - p0)
            r = int(c0[0] + (c1[0] - c0[0]) * t)
            g = int(c0[1] + (c1[1] - c0[1]) * t)
            b = int(c0[2] + (c1[2] - c0[2]) * t)
            return f"\033[38;2;{r};{g};{b}m"
    return _GREEN


def _configure_cli_logging(level: int) -> None:
    """Attach a StreamHandler to the package logger for CLI use.

    This deliberately avoids ``logging.basicConfig`` so that importing
    ``converter`` as a library never mutates the root logger or clobbers
    the host application's logging configuration.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)


class _PerLayerFilter(logging.Filter):
    """Suppress the per-layer 'Quantizing ...' log line.

    When progress callback is active, the callback prints its own
    per-layer line with percentage and ETA.  The bare 'Quantizing ...'
    log message would be redundant, so this filter drops it while
    keeping all other INFO messages (loading, saving, summary).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.levelno == logging.INFO
                    and record.getMessage().startswith("Quantizing "))


def _cli_progress_callback(info: ProgressInfo | ProgressSummary) -> None:
    """Print per-layer progress and summary to stderr."""
    if isinstance(info, ProgressInfo):
        pct = info.current_layer / info.total_layers * 100
        eta = info.estimated_remaining_seconds
        if eta >= 60:
            eta_str = f"{int(eta // 60)}m{int(eta % 60):02d}s"
        else:
            eta_str = f"{eta:.0f}s"

        # MSE severity: green < 0.0001, yellow < 0.001, red >= 0.001
        mse = info.mse
        if mse < 0.0001:
            mse_color = _GREEN
        elif mse < 0.001:
            mse_color = _YELLOW
        else:
            mse_color = _RED

        print(
            f"  [{_c(f'{pct:5.1f}%', _pct_color(pct))}] {_c(info.layer_name, _DIM)}\n"
            f"           {info.method:15s} "
            f"MSE={_c(f'{mse:<12.6f}', mse_color)} "
            f"ETA {_c(eta_str, _YELLOW)}",
            file=sys.stderr,
        )
    elif isinstance(info, ProgressSummary):
        elapsed = info.elapsed_seconds
        if elapsed >= 60:
            elapsed_str = f"{int(elapsed // 60)}m{int(elapsed % 60):02d}s"
        else:
            elapsed_str = f"{elapsed:.1f}s"
        summary = (
            f"Quantized {info.total_layers} layers in {elapsed_str}, "
            f"{info.compression_ratio:.1f}x compression"
        )
        print(f"\n{_c(summary, _BOLD)}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize a model using INT-Crush + PermuQuant + GPTQ/LDLQ"
    )
    parser.add_argument("-i", "--input", required=True, help="Input safetensors file")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--rot-size", type=int, default=0,
                        help="Regular Hadamard group size (0=no rotation, power of 4 for "
                             "Regular Hadamard, any power of 2 for Sylvester fallback. "
                             "Default: 0)")
    parser.add_argument("--int-bits", type=int, default=4, choices=[4, 8],
                        help="Quantization bit-width: 4 (INT4) or 8 (INT8, default: 4)")
    parser.add_argument("--perm-group-size", type=int, default=128,
                        help="PermuQuant acceptance evaluation group size (default: 128). "
                             "Controls the granularity used when evaluating whether a channel "
                             "permutation improves quantization error.")
    parser.add_argument("--quant-group-size", type=int, default=128,
                        help="RTN quantization group size (default: 128). "
                             "Forced to per-row (in_features) for INT4 regardless of this value.")
    parser.add_argument("--permuquant", action="store_true",
                        help="Enable PermuQuant channel reordering")
    parser.add_argument("--tau", type=float, default=0.0,
                        help="PermuQuant acceptance threshold (default: 0.0)")
    parser.add_argument("--skip-patterns", type=str, default=None,
                        help="Comma-separated skip patterns (default: embed,norm,modulation,lm_head,output,proj_out)")
    parser.add_argument("--exclude-patterns", type=str, default=None,
                        help="Comma-separated additional patterns to exclude from quantization (added to defaults)")
    parser.add_argument("-c", "--calibration", type=str, default=None,
                        help="Path to .pt calibration file (for GPTQ)")
    parser.add_argument("--quant-method", type=str, default="rtn", choices=["rtn", "gptq", "ldlq"],
                        help="Quantization method: 'rtn' (round-to-nearest), 'gptq' (calibration-based), or 'ldlq' (weight-only, no calibration) (default: rtn)")
    parser.add_argument("--gptq-block-size", type=int, default=128,
                        help="GPTQ block size (default: 128)")
    parser.add_argument("--damping", type=float, default=0.01,
                        help="GPTQ/LDLQ damping ratio (default: 0.01)")
    parser.add_argument("--hessian-method", type=str, default="hinv",
                        choices=["hinv", "cholesky"],
                        help="Hessian preparation for GPTQ: 'hinv' (full inverse H⁻¹, default, "
                             "generally better) or 'cholesky' (Cholesky factor of H⁻¹, "
                             "may help on some layers)")
    parser.add_argument("--ldlq-iterations", type=int, default=1,
                        help="LDLQ iterations with scale refinement (default: 1)")
    parser.add_argument("--greedy-passes", type=int, default=0,
                        help="Greedy local search passes after LDLQ (default: 0, recommended: 5-10)")
    parser.add_argument("--rank-threshold", type=float, default=0.01,
                        help="Eigenvalue threshold for low-rank greedy (default: 0.01). "
                             "Lower values keep more eigenvalues (better quality, slower). "
                             "Only used with --greedy-passes > 0 and Triton available.")
    parser.add_argument("--comfy-compat", action="store_true",
                        help="Write comfy_quant metadata for ComfyUI-INT8-Fast compatibility (INT8 + ConvRot only)")
    parser.add_argument("--piso", action="store_true",
                        help="Use PiSO data-aware scale optimization (requires -c calibration). "
                             "Computes per-row scales that minimize output error using the "
                             "Hessian diagonal. Replaces absmax scales for better INT8 quality.")
    parser.add_argument("--asymmetric", action="store_true",
                        help="Use asymmetric quantization (scale + zero-point). Improves quality for skewed distributions.")
    parser.add_argument("--clipping-ratios", type=str, default=None,
                        help="Comma-separated clipping ratios to search for lowest-MSE scale per group. "
                             "e.g. '0.8,0.85,0.9,0.95,1.0'. Smaller ratios clip outliers for finer resolution.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output (algorithm internals, per-layer metrics)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress all output except warnings and errors")
    parser.add_argument("--quality-report", type=str, default=None, metavar="REPORT.json",
                        help="Write per-layer quantization metrics as JSON to this file "
                             "(saved in --output directory unless an absolute path is given)")
    parser.add_argument("--smoothquant", action="store_true",
                        help="Apply SmoothQuant per-channel smoothing before quantization. "
                             "Reduces per-row weight dynamic range by migrating activation "
                             "outlier magnitude into weight columns. Best with -c calibration.")
    parser.add_argument("--smooth-alpha", type=float, default=0.5,
                        help="SmoothQuant migration strength (default: 0.5). "
                             "0.0 = all difficulty to weights, 1.0 = all to activations. "
                             "0.5 works for most models; use 0.75 for severe outliers.")
    parser.add_argument("--smoothrot", action="store_true",
                        help="Explicit confirmation of smooth-then-rotate order (now the default "
                             "when --smoothquant + --rot-size > 0). Also enables FFN pair detection "
                             "and smoothrot_factors storage.")
    parser.add_argument("--smoothrot-alpha", type=float, default=None,
                        help="SmoothRot migration strength (default: inherits --smooth-alpha). "
                             "The SmoothRot paper recommends 0.45-0.6 for LLaMA-family models.")
    parser.add_argument("--force-smoothrot-w4", action="store_true",
                        help="Force smooth-then-rotate for W4 despite quality risk (not recommended)")
    parser.add_argument("--no-progress", action="store_true", default=False,
                        help="Disable per-layer progress with ETA (default: progress is on)")
    parser.add_argument("--seed", type=int, default=42, metavar="N",
                        help="Random seed for reproducible output (default: 42, -1 to disable)")
    parser.add_argument("--svd-rank", type=int, default=0, metavar="R",
                        help="SVD-absorbed low-rank rank (0=disabled, default: 0). "
                             "16 for INT8, 32 for INT4. Decomposes each weight as "
                             "W ≈ L1@L2 + residual; L1/L2 stored in FP16, "
                             "residual quantized normally.")

    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        log_level = logging.WARNING
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    _configure_cli_logging(log_level)

    if args.rot_size != 0 and (args.rot_size & (args.rot_size - 1)) != 0:
        parser.error(f"--rot-size must be 0 or a power of 2, got {args.rot_size}")

    # SmoothRot validation: auto-enable smooth-then-rotate when both
    # --smoothquant and --rot-size > 0 are active (correct pipeline order).
    if args.smoothquant and args.rot_size > 0:
        args.smoothrot = True  # auto-enable for correct pipeline order
    if args.smoothrot:
        if args.rot_size == 0:
            parser.error("--smoothrot requires --rot-size > 0")
        if not args.smoothquant:
            args.smoothquant = True  # auto-enable smoothquant
        if args.int_bits == 4 and not args.force_smoothrot_w4:
            logger.warning(
                "Smooth-then-rotate at W4 (INT4) can hurt quality due to limited "
                "dynamic range. The old rotate-then-smooth was even worse. "
                "Consider --int-bits 8. Use --force-smoothrot-w4 to suppress."
            )

    skip_patterns = None
    if args.skip_patterns:
        skip_patterns = [p.strip() for p in args.skip_patterns.split(",")]

    exclude_patterns = None
    if args.exclude_patterns:
        exclude_patterns = [p.strip() for p in args.exclude_patterns.split(",")]

    clipping_ratios = None
    if args.clipping_ratios:
        clipping_ratios = [float(x.strip()) for x in args.clipping_ratios.split(",")]

    quality_report_path = args.quality_report
    if quality_report_path and os.sep not in quality_report_path and "/" not in quality_report_path:
        quality_report_path = os.path.join(args.output, quality_report_path)

    config = QuantizeConfig(
        input_path=args.input,
        output_dir=args.output,
        rot_size=args.rot_size,
        perm_group_size=args.perm_group_size,
        quant_group_size=args.quant_group_size,
        use_permuquant=args.permuquant,
        tau=args.tau,
        skip_patterns=skip_patterns,
        exclude_patterns=exclude_patterns,
        calibration_path=args.calibration,
        quant_method=args.quant_method,
        gptq_block_size=args.gptq_block_size,
        damping=args.damping,
        hessian_method=args.hessian_method,
        int_bits=args.int_bits,
        ldlq_iterations=args.ldlq_iterations,
        greedy_passes=args.greedy_passes,
        rank_threshold=args.rank_threshold,
        comfy_compat=args.comfy_compat,
        piso_scales=args.piso,
        asymmetric=args.asymmetric,
        clipping_ratios=clipping_ratios,
        smoothquant=args.smoothquant,
        smooth_alpha=args.smooth_alpha,
        quality_report_path=quality_report_path,
        seed=args.seed,
        smoothrot=args.smoothrot,
        smoothrot_alpha=args.smoothrot_alpha,
        force_smoothrot_w4=args.force_smoothrot_w4,
        svd_rank=args.svd_rank,
    )

    progress_callback = None
    if not args.no_progress and not args.quiet:
        progress_callback = _cli_progress_callback
        # Suppress the per-layer "Quantizing ..." log line — the progress
        # callback prints its own line with percentage, method, MSE, and ETA.
        logger.addFilter(_PerLayerFilter())

    quantize_model(config, progress_callback=progress_callback)


if __name__ == "__main__":
    main()
