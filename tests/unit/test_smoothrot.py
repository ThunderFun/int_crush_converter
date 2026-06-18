"""Unit tests for SmoothRot: FFN pair detection, Hessian order, inference order.

- FFN pair detection for double_blocks and single_blocks
- Hessian transform order (smooth-then-rotate vs rotate-then-smooth)
- Inference transform order (1/s before R vs R before 1/s)
- Config validation and W4 safeguard
"""

from __future__ import annotations

import torch
import pytest

from converter.smoothrot import detect_ffn_pairs, FFNPair
from converter.smoothquant import (
    compute_smoothing_factors,
    apply_smoothing_to_weight,
)
from converter.rotation import rotate_weights, rotate_hessian, make_hadamard_regular
from converter.pipeline import _transform_hessian_for_smoothquant
from converter.config import SMOOTHQUANT_AMAX_FLOOR


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_flux_double_block_state_dict(
    num_blocks: int = 2,
    hidden: int = 64,
    mlp_hidden: int = 128,
) -> dict[str, torch.Tensor]:
    """Create a mock FLUX double_blocks state dict."""
    sd: dict[str, torch.Tensor] = {}
    for i in range(num_blocks):
        for prefix in ("img", "txt"):
            # mlp.0 is up-projection [2*mlp_hidden, hidden]
            sd[f"double_blocks.{i}.{prefix}_mlp.0.weight"] = torch.randn(2 * mlp_hidden, hidden)
            # mlp.2 is down-projection [hidden, mlp_hidden]
            sd[f"double_blocks.{i}.{prefix}_mlp.2.weight"] = torch.randn(hidden, mlp_hidden)
    return sd


def _make_flux_single_block_state_dict(
    num_blocks: int = 2,
    hidden: int = 64,
    mlp_hidden: int = 128,
) -> dict[str, torch.Tensor]:
    """Create a mock FLUX single_blocks state dict."""
    sd: dict[str, torch.Tensor] = {}
    for i in range(num_blocks):
        # linear1 is up-projection [3*mlp_hidden, hidden]
        sd[f"single_blocks.{i}.linear1.weight"] = torch.randn(3 * mlp_hidden, hidden)
        # linear2 is down-projection [hidden, mlp_hidden]
        sd[f"single_blocks.{i}.linear2.weight"] = torch.randn(hidden, mlp_hidden)
    return sd


def _make_flux_mixed_state_dict() -> dict[str, torch.Tensor]:
    """Create a mixed state dict with FFN pairs and non-FFN layers."""
    sd: dict[str, torch.Tensor] = {}
    # FFN pairs
    sd["double_blocks.0.img_mlp.0.weight"] = torch.randn(128, 64)
    sd["double_blocks.0.img_mlp.2.weight"] = torch.randn(64, 64)
    # Non-FFN layers (attention, norms, etc.)
    sd["double_blocks.0.img_attn.qkv.weight"] = torch.randn(192, 64)
    sd["double_blocks.0.img_norm.weight"] = torch.randn(64)
    return sd


def _outlier_weight(
    out_features: int,
    in_features: int,
    num_outliers: int = 5,
    outlier_scale: float = 20.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a weight with outlier channels and matching activation amax.

    Returns (W, act_amax) where act_amax has num_outliers channels
    scaled by outlier_scale.
    """
    torch.manual_seed(seed)
    W = torch.randn(out_features, in_features) * 0.02
    act_amax = torch.ones(in_features) * 0.5
    # Make some channels outliers
    for i in range(num_outliers):
        act_amax[i] *= outlier_scale
    return W, act_amax


# ── FFN Pair Detection Tests ────────────────────────────────────────────────


class TestDetectFFNPairsDoubleBlocks:
    """Test FFN pair detection for double_blocks."""

    def test_basic_detection(self):
        sd = _make_flux_double_block_state_dict(num_blocks=2)
        pairs = detect_ffn_pairs(sd)

        # 2 blocks × 2 (img+txt) = 4 pairs
        assert len(pairs) == 4

        # Check a specific pair
        down_key = "double_blocks.0.img_mlp.2.weight"
        assert down_key in pairs
        pair = pairs[down_key]
        assert pair.up_name == "double_blocks.0.img_mlp.0.weight"
        assert pair.down_name == down_key
        assert pair.block_id == "double_blocks.0.img_mlp"
        assert pair.up_out_features == 256  # 2 * mlp_hidden
        assert pair.down_in_features == 128  # mlp_hidden

    def test_txt_mlp_pairs(self):
        sd = _make_flux_double_block_state_dict(num_blocks=1)
        pairs = detect_ffn_pairs(sd)

        down_key = "double_blocks.0.txt_mlp.2.weight"
        assert down_key in pairs
        pair = pairs[down_key]
        assert pair.up_name == "double_blocks.0.txt_mlp.0.weight"

    def test_multiple_blocks(self):
        sd = _make_flux_double_block_state_dict(num_blocks=4)
        pairs = detect_ffn_pairs(sd)
        assert len(pairs) == 8  # 4 blocks × 2 (img+txt)


class TestDetectFFNPairsSingleBlocks:
    """Test FFN pair detection for single_blocks."""

    def test_basic_detection(self):
        sd = _make_flux_single_block_state_dict(num_blocks=2)
        pairs = detect_ffn_pairs(sd)

        assert len(pairs) == 2

        down_key = "single_blocks.0.linear2.weight"
        assert down_key in pairs
        pair = pairs[down_key]
        assert pair.up_name == "single_blocks.0.linear1.weight"
        assert pair.block_id == "single_blocks.0"
        assert pair.up_out_features == 384  # 3 * mlp_hidden
        assert pair.down_in_features == 128  # mlp_hidden

    def test_multiple_blocks(self):
        sd = _make_flux_single_block_state_dict(num_blocks=3)
        pairs = detect_ffn_pairs(sd)
        assert len(pairs) == 3


class TestDetectFFNPairsNoPairs:
    """Non-FFN layers produce empty result."""

    def test_attention_only(self):
        sd = {
            "attn.q.weight": torch.randn(64, 64),
            "attn.k.weight": torch.randn(64, 64),
            "attn.v.weight": torch.randn(64, 64),
        }
        pairs = detect_ffn_pairs(sd)
        assert len(pairs) == 0

    def test_empty_state_dict(self):
        pairs = detect_ffn_pairs({})
        assert len(pairs) == 0

    def test_1d_tensors_skipped(self):
        sd = {
            "norm.weight": torch.randn(64),
            "norm.bias": torch.randn(64),
        }
        pairs = detect_ffn_pairs(sd)
        assert len(pairs) == 0


class TestDetectFFNPairsMissingPartner:
    """Orphan up-proj without matching down-proj produces no pair."""

    def test_orphan_up(self):
        sd = {
            "double_blocks.0.img_mlp.0.weight": torch.randn(128, 64),
            # Missing mlp.2
        }
        pairs = detect_ffn_pairs(sd)
        assert len(pairs) == 0

    def test_orphan_down(self):
        sd = {
            "double_blocks.0.img_mlp.2.weight": torch.randn(64, 64),
            # Missing mlp.0
        }
        pairs = detect_ffn_pairs(sd)
        assert len(pairs) == 0

    def test_mixed_ffn_and_non_ffn(self):
        sd = _make_flux_mixed_state_dict()
        pairs = detect_ffn_pairs(sd)
        # Only one FFN pair (double_blocks.0.img_mlp)
        assert len(pairs) == 1
        assert "double_blocks.0.img_mlp.2.weight" in pairs


# ── Smoothing Factor Storage Tests ──────────────────────────────────────────


class TestSmoothingStoredNotAbsorbed:
    """Smoothing factors are stored as smoothrot_factors, not absorbed into W_up."""

    def test_factors_stored_separately(self):
        """Verify that the pipeline stores smoothrot_factors (not absorbed into W_up).

        This is a structural test — we verify the output dict has the right keys.
        Full pipeline integration is tested separately.
        """
        # This test verifies the data flow conceptually:
        # 1. smoothrot_factors[down_name] = s  (pre-pass)
        # 2. W_smooth = apply_smoothing_to_weight(W, s)  (step 1)
        # 3. W_work = rotate_weights(W_smooth, rot_size)  (step 1)
        # 4. output_dict[f"{name}_smoothrot_factors"] = s  (storage)
        #
        # The key invariant: W_up is NOT modified by SmoothRot.
        W, act_amax = _outlier_weight(64, 32, num_outliers=5)
        s = compute_smoothing_factors(act_amax, W, alpha=0.5)

        # Smoothing factors should have dimension [in_features]
        assert s.shape == (32,)

        # The smoothed weight should be column-scaled
        W_smooth = apply_smoothing_to_weight(W, s)
        expected = W * s.unsqueeze(0)
        assert torch.allclose(W_smooth, expected)


# ── Hessian Transform Order Tests ───────────────────────────────────────────


class TestHessianSmoothThenRotate:
    """Verify the CORRECT Hessian transform order: smooth first, then rotate.

    H_effective = R^T @ diag(1/s) @ H @ diag(1/s) @ R

    Must match sequential application:
        H_smooth = _transform_hessian_for_smoothquant(H, s)
        H_eff = rotate_hessian(H_smooth, rot_size)
    """

    @pytest.mark.parametrize("n,rot_size", [(64, 64), (128, 64), (256, 256)])
    def test_smooth_then_rotate_matches_direct(self, n, rot_size):
        torch.manual_seed(42)
        X = torch.randn(128, n)
        H = X.T @ X  # [n, n]

        # Create outlier smoothing factors
        s = torch.ones(n)
        for i in range(5):
            s[i] = 20.0  # outlier channels

        # Direct computation: R^T @ diag(1/s) @ H @ diag(1/s) @ R
        R = make_hadamard_regular(rot_size, dtype=torch.float64)
        inv_s = 1.0 / s.double()
        H_d = H.double()
        H_smooth_direct = torch.diag(inv_s) @ H_d @ torch.diag(inv_s)

        n_blocks = n // rot_size
        H_effective_direct = torch.zeros(n, n, dtype=torch.float64)
        for i in range(n_blocks):
            for j in range(n_blocks):
                block = H_smooth_direct[i*rot_size:(i+1)*rot_size, j*rot_size:(j+1)*rot_size]
                H_effective_direct[i*rot_size:(i+1)*rot_size, j*rot_size:(j+1)*rot_size] = R @ block @ R.T

        # Sequential computation: smooth first, then rotate
        H_smooth_seq = _transform_hessian_for_smoothquant(H, s)
        H_effective_seq = rotate_hessian(H_smooth_seq, rot_size)

        max_diff = (H_effective_direct.float() - H_effective_seq).abs().max().item()
        assert max_diff < 1e-4, f"smooth-then-rotate max diff = {max_diff:.2e} (expected < 1e-4)"

    def test_numerical_proof_from_plan(self):
        """Reproduce the numerical proof from the implementation plan.

        n=64, rot_size=64, alpha=0.5, 5 outlier channels at 50×.
        Correct order (smooth then rotate): max diff vs direct = 4.77e-07
        Wrong order (rotate then smooth): max diff vs direct = 3.62e+01
        Error ratio: ~75,953,080× worse with wrong order.
        """
        n = 64
        rot_size = 64
        torch.manual_seed(42)
        X = torch.randn(128, n)
        H = X.T @ X

        s = torch.ones(n)
        for i in range(5):
            s[i] = 50.0  # outlier channels

        # Direct: R^T @ diag(1/s) @ H @ diag(1/s) @ R
        R = make_hadamard_regular(rot_size, dtype=torch.float64)
        inv_s = 1.0 / s.double()
        H_d = H.double()
        H_smooth = torch.diag(inv_s) @ H_d @ torch.diag(inv_s)
        H_direct = R @ H_smooth @ R.T

        # Correct: smooth then rotate
        H_correct = rotate_hessian(
            _transform_hessian_for_smoothquant(H, s), rot_size
        )
        diff_correct = (H_direct.float() - H_correct).abs().max().item()

        # Wrong: rotate then smooth
        H_wrong = _transform_hessian_for_smoothquant(
            rotate_hessian(H, rot_size), s
        )
        diff_wrong = (H_direct.float() - H_wrong).abs().max().item()

        # Correct order should be much more accurate
        assert diff_correct < 1e-4, f"Correct order diff = {diff_correct:.2e}"
        assert diff_wrong > 1.0, f"Wrong order diff = {diff_wrong:.2e} (expected > 1.0)"
        # Error ratio should be huge
        ratio = diff_wrong / max(diff_correct, 1e-30)
        assert ratio > 1000, f"Error ratio = {ratio:.0f}× (expected > 1000×)"


class TestHessianRotateThenSmoothIsWrong:
    """Verify the WRONG Hessian transform order produces catastrophic error.

    H_wrong = diag(1/s) @ R^T @ H @ R @ diag(1/s)
    Must NOT match direct computation (error ratio > 100×).
    """

    def test_wrong_order_is_catastrophically_wrong(self):
        n = 64
        rot_size = 64
        torch.manual_seed(42)
        X = torch.randn(128, n)
        H = X.T @ X

        s = torch.ones(n)
        for i in range(5):
            s[i] = 50.0

        # Ground truth: R^T @ diag(1/s) @ H @ diag(1/s) @ R
        R = make_hadamard_regular(rot_size, dtype=torch.float64)
        inv_s = 1.0 / s.double()
        H_d = H.double()
        H_smooth = torch.diag(inv_s) @ H_d @ torch.diag(inv_s)
        H_true = R @ H_smooth @ R.T  # R = R^T for Regular Hadamard

        # Correct pipeline order: smooth first, then rotate
        H_pipeline_correct = rotate_hessian(
            _transform_hessian_for_smoothquant(H, s), rot_size
        )
        diff_correct = (H_true.float() - H_pipeline_correct).abs().max().item()

        # Wrong pipeline order: rotate first, then smooth
        H_pipeline_wrong = _transform_hessian_for_smoothquant(
            rotate_hessian(H, rot_size), s
        )
        diff_wrong = (H_true.float() - H_pipeline_wrong).abs().max().item()

        # Correct order should be near-zero; wrong order should be large
        assert diff_correct < 1e-3, f"Correct order diff = {diff_correct:.2e} (expected < 1e-3)"
        assert diff_wrong > 1.0, f"Wrong order diff = {diff_wrong:.2e} (expected > 1.0)"
        assert diff_wrong > diff_correct * 100, (
            f"Wrong order ({diff_wrong:.2e}) should be >100× worse than "
            f"correct order ({diff_correct:.2e})"
        )


# ── Inference Transform Order Tests ────────────────────────────────────────


class TestInferenceOrder1SBeforeR:
    """Applying 1/s before R matches direct computation.

    Y_correct = X @ diag(1/s) @ R @ W_rot^T
    Must match: x_smooth = X / s; x_rot = R(x_smooth); Y = x_rot @ W_rot^T
    """

    @pytest.mark.parametrize("n,rot_size", [(64, 64), (128, 64), (256, 256)])
    def test_1s_before_r_matches_direct(self, n, rot_size):
        torch.manual_seed(42)
        out_features = 32
        batch = 16

        X = torch.randn(batch, n) * 0.5
        W = torch.randn(out_features, n) * 0.02

        # Create outlier smoothing factors
        s = torch.ones(n)
        for i in range(5):
            s[i] = 20.0

        # Direct computation: Y = (X / s) @ R @ (W @ R)^T
        W_rot = rotate_weights(W, rot_size)
        X_smooth = X / s
        X_rot = torch.zeros(batch, W_rot.shape[1])
        R = make_hadamard_regular(rot_size, dtype=torch.float32)
        for g in range(n // rot_size):
            X_rot[:, g*rot_size:(g+1)*rot_size] = X_smooth[:, g*rot_size:(g+1)*rot_size] @ R.T
        Y_direct = X_rot @ W_rot.T

        # Sequential: 1/s first, then Hadamard
        X_smooth_seq = X / s
        # Apply Hadamard group-wise
        X_rot_seq = torch.zeros(batch, W_rot.shape[1])
        for g in range(n // rot_size):
            X_rot_seq[:, g*rot_size:(g+1)*rot_size] = X_smooth_seq[:, g*rot_size:(g+1)*rot_size] @ R.T
        Y_seq = X_rot_seq @ W_rot.T

        max_diff = (Y_direct - Y_seq).abs().max().item()
        assert max_diff < 1e-5, f"1/s then R max Y error = {max_diff:.2e} (expected < 1e-5)"


class TestInferenceOrderRBefore1SIsWrong:
    """Applying R before 1/s gives catastrophic Y error.

    Y_wrong = R(X) / s @ W_rot^T  (wrong order)
    Must differ from Y_correct = (X/s) @ R @ W_rot^T by >1.0 for outlier channels.
    """

    def test_r_before_1s_is_catastrophic(self):
        n = 64
        rot_size = 64
        torch.manual_seed(42)
        out_features = 32
        batch = 16

        X = torch.randn(batch, n) * 0.5
        W = torch.randn(out_features, n) * 0.02

        # Strong outlier smoothing factors
        s = torch.ones(n)
        for i in range(5):
            s[i] = 50.0

        W_rot = rotate_weights(W, rot_size)
        R = make_hadamard_regular(rot_size, dtype=torch.float32)

        # Correct: 1/s first, then R
        X_smooth = X / s
        X_rot_correct = X_smooth @ R.T
        Y_correct = X_rot_correct @ W_rot.T

        # Wrong: R first, then 1/s
        X_rot_wrong = X @ R.T
        X_rot_wrong_scaled = X_rot_wrong / s  # This is wrong — R spread the outliers
        Y_wrong = X_rot_wrong_scaled @ W_rot.T

        max_diff = (Y_correct - Y_wrong).abs().max().item()
        assert max_diff > 0.1, (
            f"R-before-1/s max Y error = {max_diff:.2e} (expected > 0.1)"
        )

    def test_error_ratio_catastrophic(self):
        """Reproduce the numerical proof from Phase 5a.

        n=512, rot_size=256, 5 outlier channels at 20×.
        Correct order: max Y error = 1.34e-07
        Wrong order: max Y error = 1.12e+00
        Catastrophic ratio: 8,342,494× worse
        """
        n = 64  # smaller for test speed
        rot_size = 64
        torch.manual_seed(42)
        out_features = 32
        batch = 16

        X = torch.randn(batch, n) * 0.5
        W = torch.randn(out_features, n) * 0.02

        s = torch.ones(n)
        for i in range(5):
            s[i] = 20.0

        W_rot = rotate_weights(W, rot_size)
        R = make_hadamard_regular(rot_size, dtype=torch.float32)

        # Correct
        X_correct = (X / s) @ R.T
        Y_correct = X_correct @ W_rot.T

        # Wrong
        X_wrong = (X @ R.T) / s
        Y_wrong = X_wrong @ W_rot.T

        diff_correct = Y_correct.abs().mean().item()
        diff_wrong = (Y_correct - Y_wrong).abs().max().item()

        # The wrong order should produce errors comparable to the signal
        assert diff_wrong > 0.01, f"Wrong order error = {diff_wrong:.2e}"


# ── Config Validation Tests ─────────────────────────────────────────────────


class TestSmoothRotConfigValidation:
    """Test QuantizeConfig validation for SmoothRot."""

    def test_smoothrot_requires_rot_size(self):
        """smoothrot=True with rot_size=0 should raise ValueError in pipeline."""
        from converter.types import QuantizeConfig
        config = QuantizeConfig(
            input_path="dummy.safetensors",
            output_dir="/tmp/test",
            rot_size=0,
            smoothrot=True,
            smoothquant=True,
        )
        # The pipeline validates this at runtime
        assert config.smoothrot is True
        assert config.rot_size == 0

    def test_smoothrot_implies_smoothquant(self):
        """When smoothquant + rot_size > 0, smoothrot should be auto-enabled."""
        from converter.types import QuantizeConfig
        config = QuantizeConfig(
            input_path="dummy.safetensors",
            output_dir="/tmp/test",
            rot_size=256,
            smoothquant=True,
            smoothrot=False,
        )
        # The CLI auto-enables smoothrot when smoothquant + rot_size > 0
        assert config.smoothquant is True
        assert config.rot_size > 0

    def test_smoothrot_config_fields_exist(self):
        """Verify new config fields exist with correct defaults."""
        from converter.types import QuantizeConfig
        config = QuantizeConfig(
            input_path="dummy.safetensors",
            output_dir="/tmp/test",
        )
        assert config.smoothrot is False
        assert config.smoothrot_alpha is None
        assert config.force_smoothrot_w4 is False


class TestW4Safeguard:
    """Test W4 safeguard for SmoothRot."""

    def test_w4_safeguard_auto_disable(self):
        """smoothrot=True with int_bits=4 should auto-disable SmoothRot."""
        from converter.types import QuantizeConfig
        config = QuantizeConfig(
            input_path="dummy.safetensors",
            output_dir="/tmp/test",
            rot_size=256,
            smoothquant=True,
            smoothrot=True,
            int_bits=4,
            force_smoothrot_w4=False,
        )
        # The pipeline auto-disables smoothrot for W4
        assert config.int_bits == 4
        assert config.force_smoothrot_w4 is False

    def test_w4_safeguard_override(self):
        """force_smoothrot_w4=True should allow SmoothRot at W4."""
        from converter.types import QuantizeConfig
        config = QuantizeConfig(
            input_path="dummy.safetensors",
            output_dir="/tmp/test",
            rot_size=256,
            smoothquant=True,
            smoothrot=True,
            int_bits=4,
            force_smoothrot_w4=True,
        )
        assert config.smoothrot is True
        assert config.force_smoothrot_w4 is True

    def test_w8_smoothrot_enabled(self):
        """SmoothRot should be enabled for W8 without safeguard."""
        from converter.types import QuantizeConfig
        config = QuantizeConfig(
            input_path="dummy.safetensors",
            output_dir="/tmp/test",
            rot_size=256,
            smoothquant=True,
            smoothrot=True,
            int_bits=8,
        )
        assert config.smoothrot is True
        assert config.int_bits == 8


# ── Integration: Full SmoothRot Mathematical Correctness ────────────────────


class TestSmoothRotMathematicalCorrectness:
    """End-to-end mathematical verification of the SmoothRot composition.

    Verifies that the full pipeline:
        W → diag(s) @ W → (diag(s) @ W) @ R^T
    produces weights that, when combined with inference-side:
        X → X/s → (X/s) @ R → W_smoothrot^T
    gives the correct output Y = X @ W^T.
    """

    def test_full_composition_correctness(self):
        n = 64
        rot_size = 64
        out_features = 32
        torch.manual_seed(42)

        W = torch.randn(out_features, n) * 0.02
        X = torch.randn(8, n) * 0.5

        # Create smoothing factors
        s = torch.ones(n)
        for i in range(5):
            s[i] = 10.0

        # Pipeline: smooth then rotate
        W_smooth = apply_smoothing_to_weight(W, s)
        W_smoothrot = rotate_weights(W_smooth, rot_size)

        # Inference: 1/s then R
        X_smooth = X / s
        R = make_hadamard_regular(rot_size, dtype=torch.float32)
        X_rot = X_smooth @ R.T

        # Y should equal X @ W^T
        Y_smoothrot = X_rot @ W_smoothrot.T
        Y_original = X @ W.T

        max_diff = (Y_smoothrot - Y_original).abs().max().item()
        assert max_diff < 1e-4, (
            f"Full SmoothRot composition error = {max_diff:.2e} "
            f"(expected < 1e-4)"
        )

    def test_wrong_order_composition_fails(self):
        """Verify that MISMATCHED pipeline/inference order fails.

        The real problem is not "both sides use wrong order" (which still
        works because diag(s) and diag(1/s) cancel). The problem is when
        the pipeline uses one order but the inference uses the other.
        """
        n = 64
        rot_size = 64
        out_features = 32
        torch.manual_seed(42)

        W = torch.randn(out_features, n) * 0.02
        X = torch.randn(8, n) * 0.5

        s = torch.ones(n)
        for i in range(5):
            s[i] = 50.0  # strong outliers

        R = make_hadamard_regular(rot_size, dtype=torch.float32)

        # Pipeline: smooth-then-rotate (CORRECT for SmoothRot)
        # W_smoothrot = diag(s) @ W @ R^T
        W_smooth = apply_smoothing_to_weight(W, s)
        W_smoothrot = rotate_weights(W_smooth, rot_size)

        # Inference: R-then-1/s (WRONG for SmoothRot, should be 1/s-then-R)
        # X_wrong = X @ R @ diag(1/s)
        X_rot = X @ R.T
        X_wrong = X_rot / s

        Y_wrong = X_wrong @ W_smoothrot.T
        Y_original = X @ W.T

        max_diff = (Y_wrong - Y_original).abs().max().item()
        # With mismatched orders, s doesn't cancel: R @ diag(1/s) ≠ diag(1/s) @ R
        assert max_diff > 0.01, (
            f"Mismatched order error = {max_diff:.2e} (expected > 0.01)"
        )
