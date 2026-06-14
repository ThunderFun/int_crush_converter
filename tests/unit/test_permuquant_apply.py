"""Tests for PermuQuant permutation application functions."""

import torch

from converter.permuquant import (
    apply_permutation_to_weight,
    apply_permutation_to_norm,
    apply_permutation_to_norm_bias,
    apply_permutation_to_linear_output,
    permute_state_dict,
    channel_second_moments,
    find_permutation_weight,
    compute_group_quant_error,
)


class TestApplyPermutationToWeight:

    def test_basic_permutation(self):
        W = torch.randn(4, 8)
        perm = torch.tensor([3, 1, 4, 0, 7, 2, 6, 5])
        assert torch.equal(apply_permutation_to_weight(W, perm), W[:, perm])

    def test_identity_permutation(self):
        W = torch.randn(8, 16)
        perm = torch.arange(16)
        assert torch.equal(apply_permutation_to_weight(W, perm), W)

    def test_shape_preserved(self):
        W = torch.randn(16, 64)
        perm = torch.randperm(64)
        assert apply_permutation_to_weight(W, perm).shape == W.shape


class TestApplyPermutationToNorm:

    def test_basic_permutation(self):
        norm_weight = torch.randn(16)
        perm = torch.randperm(16)
        assert torch.equal(apply_permutation_to_norm(norm_weight, perm), norm_weight[perm])

    def test_identity_permutation(self):
        norm_weight = torch.randn(32)
        perm = torch.arange(32)
        assert torch.equal(apply_permutation_to_norm(norm_weight, perm), norm_weight)

    def test_single_element(self):
        norm_weight = torch.tensor([5.0])
        assert torch.equal(apply_permutation_to_norm(norm_weight, torch.tensor([0])), norm_weight)


class TestApplyPermutationToNormBias:

    def test_basic_permutation(self):
        norm_bias = torch.randn(16)
        perm = torch.randperm(16)
        assert torch.equal(apply_permutation_to_norm_bias(norm_bias, perm), norm_bias[perm])

    def test_identity_permutation(self):
        norm_bias = torch.randn(32)
        perm = torch.arange(32)
        assert torch.equal(apply_permutation_to_norm_bias(norm_bias, perm), norm_bias)


class TestApplyPermutationToLinearOutput:

    def test_basic_permutation(self):
        W = torch.randn(8, 16)
        perm = torch.randperm(8)
        assert torch.equal(apply_permutation_to_linear_output(W, perm), W[perm, :])

    def test_identity_permutation(self):
        W = torch.randn(16, 32)
        perm = torch.arange(16)
        assert torch.equal(apply_permutation_to_linear_output(W, perm), W)

    def test_permutes_rows_not_columns(self):
        W = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        perm = torch.tensor([2, 1, 0])
        result = apply_permutation_to_linear_output(W, perm)
        assert torch.equal(result[0], W[2])
        assert torch.equal(result[1], W[1])
        assert torch.equal(result[2], W[0])

    def test_column_order_preserved(self):
        torch.manual_seed(42)
        W = torch.randn(8, 16)
        perm = torch.randperm(8)
        result = apply_permutation_to_linear_output(W, perm)
        for i in range(8):
            assert torch.equal(result[i], W[perm[i]])


class TestPermuteStateDict:

    def test_basic_weight_permutation(self):
        original_weight = torch.randn(8, 16)
        state_dict = {"layer.weight": original_weight.clone()}
        perm = torch.randperm(16)
        result = permute_state_dict(state_dict, {"layer": perm})
        assert torch.equal(result["layer.weight"], original_weight[:, perm])

    def test_with_norm_map(self):
        orig_linear = torch.randn(8, 16)
        orig_norm_w = torch.randn(16)
        orig_norm_b = torch.randn(16)
        state_dict = {
            "linear.weight": orig_linear.clone(),
            "norm.weight": orig_norm_w.clone(),
            "norm.bias": orig_norm_b.clone(),
        }
        perm = torch.randperm(16)
        result = permute_state_dict(state_dict, {"linear": perm}, norm_map={"linear": "norm"})
        assert torch.equal(result["linear.weight"], orig_linear[:, perm])
        assert torch.equal(result["norm.weight"], orig_norm_w[perm])
        assert torch.equal(result["norm.bias"], orig_norm_b[perm])

    def test_with_linear_map(self):
        orig_l2 = torch.randn(8, 16)
        orig_l1 = torch.randn(16, 32)
        state_dict = {
            "linear2.weight": orig_l2.clone(),
            "linear1.weight": orig_l1.clone(),
        }
        perm = torch.randperm(16)
        result = permute_state_dict(
            state_dict, {"linear2": perm}, linear_map={"linear2": "linear1"},
        )
        assert torch.equal(result["linear2.weight"], orig_l2[:, perm])
        assert torch.equal(result["linear1.weight"], orig_l1[perm, :])

    def test_missing_weight_key_skipped(self):
        state_dict = {"other.weight": torch.randn(4, 8)}
        result = permute_state_dict(state_dict, {"nonexistent": torch.randperm(8)})
        assert torch.equal(result["other.weight"], state_dict["other.weight"])

    def test_norm_bias_without_weight(self):
        orig_linear = torch.randn(8, 16)
        orig_bias = torch.randn(16)
        state_dict = {"linear.weight": orig_linear.clone(), "norm.bias": orig_bias.clone()}
        perm = torch.randperm(16)
        result = permute_state_dict(state_dict, {"linear": perm}, norm_map={"linear": "norm"})
        assert torch.equal(result["norm.bias"], orig_bias[perm])
        assert "norm.weight" not in result

    def test_multiple_layers(self):
        orig_a = torch.randn(4, 8)
        orig_b = torch.randn(4, 16)
        state_dict = {"a.weight": orig_a.clone(), "b.weight": orig_b.clone()}
        perm_a = torch.randperm(8)
        perm_b = torch.randperm(16)
        result = permute_state_dict(state_dict, {"a": perm_a, "b": perm_b})
        assert torch.equal(result["a.weight"], orig_a[:, perm_a])
        assert torch.equal(result["b.weight"], orig_b[:, perm_b])

    def test_empty_layer_perms(self):
        original = torch.randn(4, 8)
        state_dict = {"layer.weight": original.clone()}
        result = permute_state_dict(state_dict, {})
        assert torch.equal(result["layer.weight"], original)

    def test_defaults_for_none_maps(self):
        original = torch.randn(4, 8)
        state_dict = {"layer.weight": original.clone()}
        perm = torch.randperm(8)
        result = permute_state_dict(state_dict, {"layer": perm}, norm_map=None, linear_map=None)
        assert torch.equal(result["layer.weight"], original[:, perm])

    def test_norm_takes_precedence_over_linear(self):
        orig_linear = torch.randn(8, 16)
        orig_norm_w = torch.randn(16)
        orig_prev = torch.randn(16, 32)
        state_dict = {
            "linear.weight": orig_linear.clone(),
            "norm.weight": orig_norm_w.clone(),
            "prev.weight": orig_prev.clone(),
        }
        perm = torch.randperm(16)
        result = permute_state_dict(
            state_dict, {"linear": perm},
            norm_map={"linear": "norm"}, linear_map={"linear": "prev"},
        )
        assert torch.equal(result["norm.weight"], orig_norm_w[perm])
        assert torch.equal(result["prev.weight"], orig_prev)


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:

    def test_single_column_matrix(self):
        W = torch.randn(8, 1)
        mu2 = channel_second_moments(W)
        assert mu2.shape == (1,)
        perm = find_permutation_weight(W)
        assert perm.shape == (1,)
        assert perm[0].item() == 0
        assert torch.equal(apply_permutation_to_weight(W, perm), W)

    def test_identity_permutation_roundtrip(self):
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        assert torch.equal(W, apply_permutation_to_weight(W, torch.arange(64)))

    def test_zero_weight_matrix_moments(self):
        W = torch.zeros(8, 32)
        mu2 = channel_second_moments(W)
        assert torch.allclose(mu2, torch.zeros(32))
        perm = find_permutation_weight(W)
        assert sorted(perm.tolist()) == list(range(32))

    def test_zero_weight_quant_error(self):
        W = torch.zeros(8, 128)
        for int_bits in [4, 8]:
            assert compute_group_quant_error(W, group_size=128, int_bits=int_bits) < 1e-6

    def test_single_column_quant_error(self):
        W = torch.randn(8, 1)
        assert compute_group_quant_error(W, group_size=1, int_bits=4) >= 0.0

    def test_permutation_preserves_data(self):
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        perm = find_permutation_weight(W)
        W_perm = apply_permutation_to_weight(W, perm)
        mu2_orig = channel_second_moments(W)
        mu2_perm = channel_second_moments(W_perm)
        assert torch.allclose(mu2_orig.sort().values, mu2_perm.sort().values)
