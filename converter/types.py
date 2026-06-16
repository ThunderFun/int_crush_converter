"""Shared data types for the converter package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch


@dataclass
class QuantizeConfig:
    """Configuration for :func:`quantize_model`."""

    input_path: str
    """Path to input safetensors file."""

    output_dir: str
    """Output directory for quantized model."""

    rot_size: int = 256
    """Regular Hadamard group size (power of 4: 16/64/256)."""

    perm_group_size: int = 128
    """PermuQuant acceptance evaluation group size."""

    quant_group_size: int = 128
    """RTN quantization group size. Forced to per-row (in_features) for INT4."""

    use_permuquant: bool = False
    """Whether to apply PermuQuant channel reordering."""

    tau: float = 0.0
    """PermuQuant acceptance threshold."""

    skip_patterns: list[str] | None = None
    """Custom skip patterns (default: common patterns)."""

    exclude_patterns: list[str] | None = None
    """Additional patterns to exclude from quantization."""

    calibration_path: str | None = None
    """Path to .pt calibration file (required for GPTQ)."""

    quant_method: str = "rtn"
    """Quantization method: "rtn", "gptq", or "ldlq"."""

    gptq_block_size: int = 128
    """GPTQ/LDLQ block size."""

    damping: float = 0.01
    """GPTQ/LDLQ damping ratio."""

    hessian_method: str = "hinv"
    """Hessian preparation for GPTQ: "hinv" (full H⁻¹, default) or
    "cholesky" (Cholesky factor of H⁻¹, matches GPTQ paper Algorithm 1)."""

    int_bits: int = 4
    """Quantization bit-width (4 or 8)."""

    ldlq_iterations: int = 1
    """Number of LDLQ iterations with scale refinement."""

    greedy_passes: int = 0
    """Number of greedy local search passes after LDLQ."""

    rank_threshold: float = 0.01
    """Eigenvalue threshold for low-rank greedy."""

    comfy_compat: bool = False
    """Write comfy_quant metadata for ComfyUI-INT8-Fast compatibility."""

    asymmetric: bool = False
    """Use asymmetric quantization (scale + zero-point)."""

    clipping_ratios: list[float] | None = None
    """Clipping ratios to search for lowest MSE."""

    piso_scales: bool = False
    """Use PiSO data-aware scale optimization (requires calibration data).

    Computes per-row scales that minimize output reconstruction error
    using the Hessian diagonal from calibration.  Replaces absmax scales
    with data-aware optimal scales.  Only effective with GPTQ or when
    calibration data is available.  See arXiv:2606.10890."""

    quality_report_path: str | None = None
    """If set, write per-layer quantization metrics as JSON to this path."""

    smoothquant: bool = False
    """Apply SmoothQuant per-channel smoothing before quantization.

    Computes per-input-channel smoothing factors that migrate activation
    outlier magnitude into the weight columns, reducing per-row dynamic
    range.  Requires calibration data for best results (per-channel amax
    or Hessian diagonal).  See arXiv:2211.10438."""

    smooth_alpha: float = 0.5
    """SmoothQuant migration strength alpha.  Controls how much quantization
    difficulty is migrated from activations to weights.

    0.0 = all to weights, 1.0 = all to activations, 0.5 = even split.
    Default 0.5 is the sweet spot for most models (OPT, BLOOM, LLaMA).
    Use 0.75 for models with severe activation outliers (e.g., GLM-130B)."""

    seed: int = 42
    """Random seed for reproducibility. Set to -1 to disable seeding."""


@dataclass
class QuantizationResult:
    """Result of quantizing a single weight layer.

    Returned by :func:`ldlq_quantize_layer`, :func:`gptq_quantize_layer`,
    and :func:`gptq_quantize_layer_rtn`.

    Access fields by name::

        result = ldlq_quantize_layer(W)
        result.quantized_W
        result.scales
        result.mse
        result.method_used
    """

    quantized_W: torch.Tensor
    """Quantized weight values as int8 (INT4 range [-8,7] or INT8 range [-128,127])."""

    scales: torch.Tensor
    """Per-row (or per-group) dequantization scales. See config.SCALE_DTYPE."""

    zero_points: torch.Tensor | None = None
    """Per-row (or per-group) zero-points for asymmetric quantization. None for symmetric."""

    mse: float = 0.0
    """Mean squared error between original and dequantized weights."""

    max_err: float = 0.0
    """Maximum absolute error between original and dequantized weights."""

    method_used: str = ""
    """Backend that produced the result, e.g. "ldlq_triton", "ldlq_compile",
    "ldlq_cpu", "gptq_triton", "gptq_pytorch", "rtn", "rtn_fallback"."""

    fallbacks: list[str] = field(default_factory=list)
    """Fallbacks triggered during quantization, e.g. ["oom_cpu", "hessian_nan_rtn"]."""


@dataclass(frozen=True)
class ProgressInfo:
    """Snapshot of quantization progress, delivered once per quantized layer.

    Passed to ``progress_callback`` after each layer is fully processed
    (rotated, permuted, quantized, verified, packed, and stored).
    ``current_layer`` is 1-based: the first call delivers ``(1, N)``.
    """

    current_layer: int
    """1-based index of the layer just completed (1, 2, …, total_layers)."""

    total_layers: int
    """Total number of quantizable layers (2D, non-skipped) in the model."""

    layer_name: str
    """State-dict key of the layer just quantized
    (e.g. ``'layers.0.q_proj.weight'``)."""

    layer_shape: tuple[int, ...]
    """Shape of the weight tensor (e.g. ``(4096, 4096)``)."""

    elapsed_seconds: float
    """Wall-clock seconds since the quantization loop started."""

    estimated_remaining_seconds: float
    """Estimated seconds to finish remaining layers.
    ``0.0`` on the first layer.  Based on linear extrapolation of
    average time per completed layer."""

    method: str
    """Quantization backend that produced this layer's result.
    One of: ``'rtn'``, ``'gptq_triton'``, ``'gptq_pytorch'``,
    ``'ldlq_triton'``, ``'ldlq_compile'``, ``'ldlq_cpu'``."""

    mse: float
    """Mean squared error from the verification step."""


@dataclass(frozen=True)
class ProgressSummary:
    """Final summary delivered via ``progress_callback`` after all layers.

    The callback receives this once, after ``save_file`` completes,
    as the last event.  Consumers dispatch on
    ``isinstance(info, ProgressInfo)`` vs ``isinstance(info, ProgressSummary)``.
    """

    total_layers: int
    """Number of quantized layers (same as the count of ProgressInfo calls)."""

    skipped_layers: int
    """Layers stored as-is (norms, embeddings, 1D tensors, exclude-pattern matches)."""

    permuted_layers: int
    """Layers where PermuQuant channel reordering was applied."""

    gptq_layers: int
    """Layers quantized via GPTQ or LDLQ (includes RTN fallbacks)."""

    rtn_fallback_layers: int
    """Layers that fell back from GPTQ to RTN due to missing Hessian."""

    smoothquant_layers: int
    """Layers where SmoothQuant per-channel smoothing was applied."""

    elapsed_seconds: float
    """Total wall-clock time for the quantization loop (excludes setup and save)."""

    input_size_gb: float
    """Input model size in GiB."""

    output_size_gb: float
    """Output model size in GiB."""

    compression_ratio: float
    """input_size_gb / output_size_gb."""


class ProgressCallback(Protocol):
    """Protocol for progress callbacks from :func:`quantize_model`.

    Called once per quantized layer with a :class:`ProgressInfo` and
    once at the end with a :class:`ProgressSummary`.  Any callable
    matching this signature satisfies the protocol — no inheritance
    required.  Example::

        def my_callback(info):
            if isinstance(info, ProgressInfo):
                pct = info.current_layer / info.total_layers * 100
                print(f"  [{pct:.0f}%] {info.layer_name}")
            else:
                print(f"Done! {info.total_layers} layers in {info.elapsed_seconds:.1f}s")

        quantize_model(config, progress_callback=my_callback)

    The callback is synchronous and called from the quantization thread.
    If it raises an exception, quantization aborts and the exception
    propagates to the caller.
    """

    def __call__(self, info: ProgressInfo | ProgressSummary) -> None: ...
