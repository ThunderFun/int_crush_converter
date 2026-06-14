"""Named constants for the INT-Crush quantization algorithms.

Centralizes magic numbers so they're documented and easy to tune.
Triton kernel constants (``tl.constexpr``) cannot be imported at runtime
and are kept inline with comments referencing these names.
"""

# --- Scale and numerical floors ---

INT4_SCALE_DIVISOR = 7.0
"""Divisor for symmetric INT4 scale: scale = max(|row|) / 7.0.
Max representable positive INT4 value is 7 (range [-8, 7])."""

INT8_SCALE_DIVISOR = 127.0
"""Divisor for symmetric INT8 scale: scale = max(|row|) / 127.0.
Max representable positive INT8 value is 127 (range [-128, 127])."""

MAX_FP16_SCALE = 65000.0
"""Maximum safe float16 scale value. Keeps scales well below the fp16 max
(65504) to leave headroom for dequant multiplication without overflow."""

FP16_SCALE_FLOOR = 1e-6
"""Minimum scale for values cast to float16. 1e-6 is the smallest value
that survives fp16 cast (fp16 smallest subnormal ≈ 6e-8; 1e-8 → 0.0).
Used in scales.py for per-row/per-group scale computation."""

SCALE_FLOOR = 1e-8
"""Minimum scale value to prevent division by zero. Used in LDLQ internal
rounding (per-element scales, not stored as float16). NOT safe for float16
storage — use FP16_SCALE_FLOOR for any scale that will be cast to float16."""

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
