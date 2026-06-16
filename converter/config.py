"""Named constants for the INT-Crush quantization algorithms.

Centralizes magic numbers so they're documented and easy to tune.
Triton kernel constants (``tl.constexpr``) cannot be imported at runtime
and are kept inline with comments referencing these names.
"""

import torch

# --- Scale and numerical floors ---

INT4_SCALE_DIVISOR = 7.0
"""Divisor for symmetric INT4 scale: scale = max(|row|) / 7.0.
Max representable positive INT4 value is 7 (range [-8, 7])."""

INT8_SCALE_DIVISOR = 127.0
"""Divisor for symmetric INT8 scale: scale = max(|row|) / 127.0.
Max representable positive INT8 value is 127 (range [-128, 127])."""

# --- Scale storage dtype ---

SCALE_DTYPE = torch.float16
"""Dtype for stored dequantization scales."""

SMOOTH_FACTOR_DTYPE = torch.float16
"""Dtype for stored SmoothQuant smoothing factors. These are per-input-channel
values s_i such that the smoothed weight is W_smooth = W @ diag(s) and the
effective activation is X_smooth = X @ diag(1/s). Stored so the inference
engine can undo the smoothing."""

SCALE_MIN = 1e-5
"""Minimum scale value for storage. Prevents division-by-zero in dequant.
Matches FP16 safe minimum (6e-5 normalized, 1e-5 for subnormal margin)."""

SCALE_MAX = 65500.0
"""Maximum scale value for storage."""

# Legacy aliases — prefer SCALE_MIN / SCALE_MAX / SCALE_DTYPE.
FP16_SCALE_FLOOR = SCALE_MIN
MAX_FP16_SCALE = SCALE_MAX

SCALE_FLOOR = 1e-8
"""Minimum scale value to prevent division by zero. Used in LDLQ internal
rounding (per-element scales, not stored externally). For stored scales,
use SCALE_MIN."""

DIAG_MEAN_FLOOR = 1e-6
"""Minimum mean diagonal value for Hessian damping. Prevents zero damping
on near-zero weight matrices."""

DENOMINATOR_FLOOR = 1e-8
"""Floor for least-squares denominator in LDLQ scale refinement.
Prevents NaN when all quantized values in a row are zero."""

HINV_DIAG_FLOOR = 1e-8
"""Floor for inverse Hessian diagonal in GPTQ/LDLQ column loop.
Prevents division by zero on singular Hessian blocks."""

# --- Error propagation ---

ERR_CLAMP_RANGE = 100.0
"""Clamp range for normalized quantization error in GPTQ/LDLQ.
Prevents numerical explosion from ill-conditioned Hessians.
Applied symmetrically: err.clamp(-ERR_CLAMP_RANGE, ERR_CLAMP_RANGE)."""

SANITIZE_FLOOR = 1e-8
"""Replacement value for NaN/Inf in scale sanitization."""

SANITIZE_CEIL = 1e30
"""Replacement value for +Inf in scale sanitization."""

# --- Weight outlier sanitization ---

WEIGHT_OUTLIER_CLAMP = 1000.0
"""Maximum absolute weight value before rotation/quantization.

Weights exceeding this are clamped.  Normal transformer weights are in
[-1, 1] with rare exceptions up to ~10.  Values like 2.25e33 are
corrupted model data that would cause the per-row scale
to blow up, making the MSE computation overflow float32
(since ``(scale/2)² > 3.4e38`` when scale > ~3.7e19).

The clamp must happen *before* the Hadamard rotation so extreme values
don't poison an entire 256-column block.
"""

# --- Sign-flip correction ---

SIGN_FLIP_THRESHOLD = 1e-8
"""Minimum |weight| for sign-flip correction in LDLQ rounding.
Below this, the weight is treated as zero and sign is ambiguous."""

# --- Convergence and divergence ---

CONVERGENCE_EPS = 1e-10
"""Small epsilon added to prev_mse to avoid division by zero in
convergence improvement calculation."""

CONVERGENCE_IMPROVEMENT_THRESHOLD = 1e-4
"""Minimum relative MSE improvement to continue iterative LDLQ.
Below this, the iteration is considered converged."""

# --- Scale update ---

SCALE_FLOOR_MULTIPLIER = 0.1
"""Scale floor = SCALE_MIN * this multiplier. Prevents scales from
shrinking to zero during iterative refinement."""

SCALE_CEIL_MULTIPLIER = 10.0
"""Scale ceil = SCALE_MAX * this multiplier. Prevents scales from
growing unbounded during iterative refinement."""

ABS_SCALE_FLOOR = 1e-8
"""Absolute minimum scale during iterative refinement, regardless
of the weight distribution."""

# --- Greedy search ---

EIGENVALUE_FLOOR = 1e-12
"""Minimum eigenvalue for low-rank decomposition. Below this,
the Hessian is effectively zero and low-rank is not applicable."""

STALL_THRESHOLD = 0.99
"""Proxy loss ratio threshold for stall detection in low-rank greedy.
If proxy >= prev_proxy * STALL_THRESHOLD, the low-rank pass is
considered stalled and falls back to full-rank v2."""

LOWRANK_MAX_RANK_FRAC = 0.3
"""Maximum fraction of N for low-rank to be useful. If effective
rank > LOWRANK_MAX_RANK_FRAC * N, fall back to full-rank greedy."""

LOWRANK_MAX_K = 512
"""Absolute cap on effective rank K for the low-rank greedy kernel.
Above this, the two O(K) scalar loops in the Triton kernel dominate;
fall back to the faster full-rank path."""

INITIAL_BEST_COST = 1e30
"""Initial best cost value in greedy candidate evaluation.
Must be large enough that any real cost is lower."""

# --- SmoothQuant ---

SMOOTHQUANT_AMAX_FLOOR = 1e-8
"""Minimum per-channel activation amax to prevent division by zero in
SmoothQuant smoothing factor computation."""

SMOOTHQUANT_HESSIAN_FLOOR = SMOOTHQUANT_AMAX_FLOOR ** 2
"""Floor for the product s_i * s_j in the SmoothQuant Hessian
transformation.  Each smoothing factor is individually clamped to
SMOOTHQUANT_AMAX_FLOOR, so the minimum pairwise product is its square."""

# --- Triton availability ---

try:
    import triton
    import triton.language as tl  # noqa: F401
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

# --- Triton kernel sizes (passed as tl.constexpr at call sites) ---

TRITON_BLOCK_ROWS_GPTQ = 64
"""Rows per Triton program in the GPTQ block kernel (gptq_triton.py)."""

TRITON_BLOCK_ROWS_LDLQ = 64
"""Rows per Triton program in the LDLQ block kernel (ldlq_triton.py)."""

TRITON_BLOCK_ROWS_GREEDY = 256
"""Rows per Triton program in the greedy column kernel (greedy.py)."""

GREEDY_COL_BLOCK = 64
"""Column block size for full-rank greedy Triton (greedy.py)."""

GREEDY_RECOMPUTE_EVERY = 8
"""Recompute cross terms every N columns to limit drift (greedy.py)."""

TRITON_BLOCK_ROWS_PISO = 64
"""Rows per Triton program in the PiSO grid-search kernel (piso_triton.py)."""

PISO_D_BLOCK = 512
"""Column block size for the D-dimension loop in the PiSO Triton kernel.
Each program loads BLOCK_ROWS × D_BLOCK floats per iteration, keeping
the working set in L2 cache across candidate evaluations."""
