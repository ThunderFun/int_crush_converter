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
    """Write .comfy_quant metadata in ComfyUI's native INT8 format.

    Uses ``{"format": "int8_tensorwise", "convrot": true, ...}``.
    Requires ``int_bits=8``."""

    comfy_int8_fast: bool = False
    """Write comfy_quant metadata for ComfyUI-INT8-Fast compatibility.

    Uses the legacy ``{"per_row": true, ...}`` format.  Requires ``int_bits=8``."""

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
    Default 0.5 works well for most architectures. Use 0.75 for models
    with severe activation outliers."""

    seed: int = 42
    """Random seed for reproducibility. Set to -1 to disable seeding."""

    smoothrot: bool = False
    """Explicit confirmation of smooth-then-rotate pipeline order.

    Smooth-then-rotate is now the **default** pipeline order when
    ``smoothquant=True`` and ``rot_size > 0``.  This flag enables
    additional FFN-pair-specific optimizations (pair detection,
    ``smoothrot_factors`` storage) but is not required for the correct
    order.

    When enabled, the pipeline detects FFN up/down pairs, computes
    smoothing factors for the down-projection, and stores them as
    ``{name}_smoothrot_factors`` tensors for inference-side ``1/s``
    application (before the online Hadamard).

    Automatically disabled for ``int_bits=4`` (W4) unless
    ``force_smoothrot_w4=True``."""

    smoothrot_alpha: float | None = None
    """SmoothRot migration strength.  If ``None``, inherits ``smooth_alpha``.

    The SmoothRot paper recommends per-model tuning in the 0.45–0.6 range.
    Test with the ablation suite for your target architecture."""

    force_smoothrot_w4: bool = False
    """Force-enable SmoothRot for W4 (INT4) quantization despite the safeguard.

    NOT recommended — SmoothRot can hurt quality at W4 due to limited
    INT4 dynamic range (only 16 levels).  Use only for experimentation."""

    svd_rank: int = 0
    """SVD-absorbed low-rank branch rank. 0 = disabled (default).

    When enabled, each 2D weight is decomposed as::

        W ≈ L1 @ L2 + residual

    where L1 [out, r] and L2 [r, in] are stored in FP16 and the
    residual is quantized normally.  The low-rank branch absorbs the
    dominant singular values, leaving a cleaner residual with lower
    dynamic range for quantization.

    Typical values: 16 (INT8), 32 (INT4).  Based on SVDQuant
    (arXiv:2411.05007).  Adds two FP16 tensors per layer at
    negligible inference overhead."""


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

    svd_layers: int
    """Layers where SVD-absorbed low-rank decomposition was applied."""

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


# ── Benchmark types ──────────────────────────────────────────────────────────


@dataclass
class BenchmarkConfig:
    """Configuration for :func:`benchmark_method` / :func:`benchmark_matrix`."""

    method: str
    """Quantization method: ``"rtn"``, ``"gptq"``, or ``"ldlq"``."""

    int_bits: int
    """Quantization bit-width (4 or 8)."""

    features: str = "plain"
    """Feature preset string (see :data:`FEATURE_PRESETS` for valid values)."""

    rot_size: int = 0
    """Hadamard rotation group size. 0 = no rotation, 64/256 for ConvRot."""

    smooth_alpha: float = 0.5
    """SmoothQuant / SmoothRot migration strength."""

    svd_rank: int = 0
    """SVD-absorbed low-rank branch rank. 0 = disabled."""

    perm_group_size: int = 128
    """PermuQuant acceptance evaluation group size."""

    tau: float = 0.0
    """PermuQuant acceptance threshold."""

    asymmetric: bool = False
    """Use asymmetric quantization (scale + zero-point)."""

    gptq_block_size: int = 128
    """GPTQ/LDLQ block size."""

    damping: float = 0.01
    """GPTQ/LDLQ damping ratio."""

    hessian_method: str = "hinv"
    """Hessian preparation for GPTQ: ``"hinv"`` or ``"cholesky"``."""

    ldlq_iterations: int = 1
    """Number of LDLQ iterations with scale refinement."""

    greedy_passes: int = 0
    """LDLQ greedy coordinate descent passes (0 = disabled)."""

    rank_threshold: float = 0.01
    """Eigenvalue threshold for low-rank greedy."""

    clipping_ratios: list[float] | None = None
    """Scale search ratios for lowest-MSE clipping."""

    force_smoothrot_w4: bool = False
    """Allow SmoothRot at INT4 (not recommended)."""

    skip_patterns: list[str] | None = None
    """Custom skip patterns. None → use :data:`DEFAULT_SKIP_PATTERNS`."""

    seed: int = 42
    """Random seed for reproducibility."""


@dataclass
class LayerBenchmarkResult:
    """Result for a single layer under one method config."""

    name: str
    """State-dict key (e.g. ``"layers.0.q_proj.weight"``)."""

    shape: tuple[int, int]
    """Weight tensor shape ``(out_features, in_features)``."""

    weight_mse: float
    """Relative MSE: ``||W - W_deq||² / ||W||²``."""

    weight_max_err: float
    """Relative max error: ``max|W - W_deq| / max|W|``."""

    output_mse: float | None
    """Relative output MSE (if activations provided). None otherwise."""

    snr_db: float
    """Signal-to-noise ratio in dB."""

    method_used: str
    """Backend that produced this result (e.g. ``"rtn"``, ``"gptq_triton"``)."""

    fallbacks: list[str]
    """Fallbacks triggered (e.g. ``["oom_cpu"]``)."""

    smooth_source: str | None
    """Smoothing source: ``"per_channel_amax"``, ``"hessian_diag"``,
    ``"weight_only"``, ``"smoothrot"``, or ``None``."""

    permutation_applied: bool
    """True if PermuQuant reordered columns."""

    svd_rank: int
    """0 if SVD not applied, else the rank used."""


@dataclass
class MethodBenchmarkResult:
    """Result for one (method, bits, features) combo across all layers."""

    config: BenchmarkConfig
    """The benchmark configuration used."""

    layers: list[LayerBenchmarkResult]
    """Per-layer results."""

    mse_mean: float
    """Mean ``weight_mse`` across layers (relative)."""

    mse_p95: float
    """95th percentile ``weight_mse``."""

    max_err: float
    """Maximum ``weight_max_err`` across layers."""

    output_mse_mean: float | None
    """Mean ``output_mse`` across layers (if activations provided)."""

    elapsed_seconds: float
    """Wall-clock time for this combo."""

    error: str | None
    """Error message if the whole combo failed. None on success."""


@dataclass
class BenchmarkReport:
    """Full side-by-side comparison of all method combos."""

    model_path: str | None
    """Path to input safetensors (None for synthetic)."""

    calibration_path: str | None
    """Path to calibration .pt file (None if not provided)."""

    num_layers: int
    """Number of quantized layers."""

    layer_shapes: dict[str, tuple]
    """Mapping from layer name to shape tuple."""

    results: list[MethodBenchmarkResult]
    """Per-combo results, sorted by ``mse_mean``."""
