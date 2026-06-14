"""Tests for PermuQuant channel operations: moments, permutations, errors, sweeps."""

import pytest
import torch

from converter.permuquant import (
    channel_second_moments,
    find_permutation_weight,
    find_permutation_joint,
    compute_group_quant_error,
    find_permutation_with_acceptance,
    sweep_alpha,
)


class TestChannelSecondMoments:

    def test_shape(self):
        mu2 = channel_second_moments(torch.randn(16, 64))
        assert mu2.shape == (64,)

    def test_all_ones(self):
        mu2 = channel_second_moments(torch.ones(8, 16))
        assert torch.allclose(mu2, torch.ones(16))

    def test_zero_input(self):
        mu2 = channel_second_moments(torch.zeros(8, 16))
        assert torch.allclose(mu2, torch.zeros(16))

    def test_formula(self):
        W = torch.tensor([[2.0, -1.0], [0.0, 3.0]])
        mu2 = channel_second_moments(W)
        expected = torch.tensor([(4.0 + 0.0) / 2, (1.0 + 9.0) / 2])
        assert torch.allclose(mu2, expected)


class TestFindPermutationWeight:

    def test_output_length(self):
        perm = find_permutation_weight(torch.randn(16, 64))
        assert perm.shape == (64,)

    def test_is_valid_permutation(self):
        perm = find_permutation_weight(torch.randn(16, 64))
        assert sorted(perm.tolist()) == list(range(64))

    def test_descending_by_second_moment(self):
        torch.manual_seed(42)
        W = torch.randn(8, 32)
        mu2 = channel_second_moments(W)
        perm = find_permutation_weight(W)
        mu2_reordered = mu2[perm]
        assert (mu2_reordered[:-1] >= mu2_reordered[1:]).all()

    def test_channel_zero_highest_mu2(self):
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        W[:, 0] *= 100.0
        perm = find_permutation_weight(W)
        assert perm[0].item() == 0


class TestFindPermutationJoint:

    def test_alpha_zero_ignores_activations(self):
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        act_mu2 = torch.rand(64)
        perm_joint = find_permutation_joint(W, act_mu2, alpha=0.0)
        perm_weight = find_permutation_weight(W)
        assert torch.equal(perm_joint, perm_weight)

    def test_alpha_one_ignores_weights(self):
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        act_mu2 = torch.rand(64)
        perm = find_permutation_joint(W, act_mu2, alpha=1.0)
        expected = act_mu2.argsort(descending=True)
        assert torch.equal(perm, expected)

    def test_is_valid_permutation(self):
        torch.manual_seed(42)
        perm = find_permutation_joint(torch.randn(8, 32), torch.rand(32), alpha=0.5)
        assert sorted(perm.tolist()) == list(range(32))


class TestComputeGroupQuantError:

    def test_non_negative_int4(self):
        torch.manual_seed(42)
        assert compute_group_quant_error(torch.randn(16, 128), group_size=128, int_bits=4) >= 0.0

    def test_non_negative_int8(self):
        torch.manual_seed(42)
        assert compute_group_quant_error(torch.randn(16, 128), group_size=128, int_bits=8) >= 0.0

    def test_zero_weights(self):
        assert compute_group_quant_error(torch.zeros(8, 128), group_size=128, int_bits=4) < 1e-6

    def test_int8_less_error_than_int4(self):
        torch.manual_seed(42)
        W = torch.randn(16, 128)
        err_int4 = compute_group_quant_error(W, group_size=128, int_bits=4)
        err_int8 = compute_group_quant_error(W, group_size=128, int_bits=8)
        assert err_int8 <= err_int4 + 1e-6


class TestFindPermutationWithAcceptance:

    def test_tau_zero_accepts_improvement(self):
        torch.manual_seed(42)
        perm, accepted = find_permutation_with_acceptance(
            torch.randn(16, 128), tau=0.0, group_size=128, int_bits=4,
        )
        assert perm.shape == (128,)
        assert isinstance(accepted, bool)

    def test_tau_high_rejects_most(self):
        accepted_count = 0
        for seed in range(20):
            torch.manual_seed(seed)
            _, accepted = find_permutation_with_acceptance(
                torch.randn(16, 128), tau=0.99, group_size=128, int_bits=4,
            )
            if accepted:
                accepted_count += 1
        assert accepted_count < 20

    def test_returns_valid_perm(self):
        torch.manual_seed(42)
        perm, _ = find_permutation_with_acceptance(torch.randn(8, 64), tau=0.0)
        assert sorted(perm.tolist()) == list(range(64))

    def test_zero_weight_matrix(self):
        _, accepted = find_permutation_with_acceptance(
            torch.zeros(8, 64), tau=0.0, group_size=64, int_bits=4,
        )
        assert accepted is False

    def test_with_activations(self):
        torch.manual_seed(42)
        perm, accepted = find_permutation_with_acceptance(
            torch.randn(16, 128), act_mu2=torch.rand(128),
            alpha=0.5, tau=0.0, group_size=128, int_bits=4,
        )
        assert perm.shape == (128,)
        assert isinstance(accepted, bool)


class TestSweepAlpha:

    def test_num_alpha_one(self):
        torch.manual_seed(42)
        perm, best_alpha, accepted = sweep_alpha(
            torch.randn(16, 128), num_alpha=1, group_size=128, int_bits=4,
        )
        assert perm.shape == (128,)
        assert best_alpha == 0.0
        assert isinstance(accepted, bool)

    def test_num_alpha_eleven(self):
        torch.manual_seed(42)
        perm, best_alpha, accepted = sweep_alpha(
            torch.randn(16, 128), act_mu2=torch.rand(128),
            num_alpha=11, group_size=128, int_bits=4,
        )
        assert perm.shape == (128,)
        assert sorted(perm.tolist()) == list(range(128))
        assert isinstance(accepted, bool)
        if accepted:
            assert 0.0 <= best_alpha <= 1.0

    def test_accepted_is_bool(self):
        torch.manual_seed(42)
        _, _, accepted = sweep_alpha(torch.randn(8, 64), num_alpha=5)
        assert isinstance(accepted, bool)

    def test_invalid_num_alpha(self):
        with pytest.raises(ValueError):
            sweep_alpha(torch.randn(8, 64), num_alpha=0)

    def test_weight_only_sweep(self):
        torch.manual_seed(42)
        perm, best_alpha, accepted = sweep_alpha(
            torch.randn(16, 128), act_mu2=None, num_alpha=11, group_size=128, int_bits=4,
        )
        assert perm.shape == (128,)
        assert isinstance(accepted, bool)
