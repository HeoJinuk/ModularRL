"""Exact trajectory ratios, ordinary/weighted estimates and resumable IS stats."""

import io
import math

import numpy as np
import pytest
import torch

from modular_rl.component.estimator import StateValueEstimator, TabularQEstimator, TabularVEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import MonteCarloReturnGenerator
from modular_rl.data import Episode, Transition
from modular_rl.learner import MCPredictionLearner, OffPolicyMCPredictionLearner
from modular_rl.model import Model


def make_learner(method="weighted", mode="first", dtype=torch.float64, gamma=1, policy=None):
    model = Model(IdentityRepresentation(), TabularVEstimator(4)).to(dtype=dtype)
    return OffPolicyMCPredictionLearner(model, MonteCarloReturnGenerator(gamma),
                                      np.full((4, 2), .5) if policy is None else policy, method, mode)


def episode(states=(1,), rewards=(4.,), probabilities=(.25,), actions=None, terminated=True, truncated=False):
    actions = [0] * len(states) if actions is None else actions
    next_states = list(states[1:]) + [0]
    return Episode(tuple(Transition(torch.tensor(state, dtype=torch.int32), torch.tensor(action, dtype=torch.int32),
                                    np.float64(reward), torch.tensor(next_state, dtype=torch.int32),
                                    terminated if time == len(states) - 1 else False,
                                    truncated if time == len(states) - 1 else False,
                                    torch.tensor(math.log(probability), dtype=torch.float64))
                         for time, (state, action, reward, next_state, probability) in
                         enumerate(zip(states, actions, rewards, next_states, probabilities))))


@pytest.mark.parametrize("method,expected", [("ordinary", 8.), ("weighted", 4.)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_one_step_reward_four_ratio_two_is_exact(method, expected, dtype):
    learner = make_learner(method, dtype=dtype)
    table = learner.model.estimator.v_table
    pointer = table.data_ptr()
    sample = episode()
    original_log_prob = sample.transitions[0].log_prob.clone()
    returns = learner.learn(sample)
    torch.testing.assert_close(returns, torch.tensor([4.], dtype=dtype))
    assert table[1] == expected and table.data_ptr() == pointer
    assert learner.visit_counts[1] == 1 and learner.visit_counts.dtype == torch.long
    assert learner.weight_sums[1] == 2 and learner.weighted_return_sums[1] == 8
    assert learner.weight_sums.dtype == learner.weighted_return_sums.dtype == torch.float64
    assert not returns.requires_grad and returns.grad_fn is None
    torch.testing.assert_close(sample.transitions[0].log_prob, original_log_prob)
    assert sample.transitions[0].observation.dtype == sample.transitions[0].action.dtype == torch.int32
    assert learner.visit_counts[0] == table[0] == 0


@pytest.mark.parametrize("method,expected", [("ordinary", 5.), ("weighted", 10 / 3)])
def test_incremental_statistics_accumulate_different_behavior_weights(method, expected):
    learner = make_learner(method)
    learner.learn(episode())
    learner.learn(episode(rewards=(2.,), probabilities=(.5,)))
    assert learner.model.estimator.v_table[1].item() == pytest.approx(expected)
    assert learner.visit_counts[1] == 2 and learner.weight_sums[1] == 3
    assert learner.weighted_return_sums[1] == 10


@pytest.mark.parametrize("method", ["ordinary", "weighted"])
@pytest.mark.parametrize("gamma,returns", [(0, [1., 4.]), (.5, [3., 4.]), (1, [5., 4.])])
def test_suffix_weight_includes_current_and_all_later_actions(method, gamma, returns):
    learner = make_learner(method, gamma=gamma)
    result = learner.learn(episode((1, 2), (1., 4.), (.25, .25)))
    torch.testing.assert_close(result, torch.tensor(returns, dtype=torch.float64))
    torch.testing.assert_close(learner.weight_sums[1:3], torch.tensor([4., 2.], dtype=torch.float64))
    expected = np.array(returns) * ([4, 2] if method == "ordinary" else [1, 1])
    np.testing.assert_allclose(learner.model.estimator.v_table.numpy()[1:3], expected)


@pytest.mark.parametrize("method,mode,value,count,weight_sum", [
    ("ordinary", "first", 48., 1, 8.), ("weighted", "first", 6., 1, 8.),
    ("ordinary", "every", 27., 2, 10.), ("weighted", "every", 5.4, 2, 10.),
])
def test_repeated_state_first_visit_is_earliest_despite_reverse_ratios(method, mode, value, count, weight_sum):
    learner = make_learner(method, mode)
    result = learner.learn(episode((1, 2, 1), (1., 2., 3.), (.25, .25, .25), actions=(0, 1, 1)))
    torch.testing.assert_close(result, torch.tensor([6., 5., 3.], dtype=torch.float64))
    assert learner.model.estimator.v_table[1].item() == pytest.approx(value)
    assert learner.visit_counts[1] == count and learner.weight_sums[1] == pytest.approx(weight_sum)
    assert learner.model.estimator.v_table[2] == (20 if method == "ordinary" else 5)


@pytest.mark.parametrize("method", ["ordinary", "weighted"])
def test_zero_target_action_kills_that_and_all_earlier_suffix_weights(method):
    policy = np.full((4, 2), .5)
    policy[2] = [1, 0]
    learner = make_learner(method, policy=policy)
    learner.model.estimator.v_table.fill_(99)
    learner.learn(episode((1, 2, 3), (1., 2., 3.), (.25, .25, .25), actions=(0, 1, 0)))
    torch.testing.assert_close(learner.weight_sums, torch.tensor([0., 0., 0., 2.], dtype=torch.float64))
    assert learner.visit_counts[1] == learner.visit_counts[2] == learner.visit_counts[3] == 1
    assert learner.model.estimator.v_table[1] == learner.model.estimator.v_table[2] == (0 if method == "ordinary" else 99)
    assert learner.model.estimator.v_table[3] == (6 if method == "ordinary" else 3)


@pytest.mark.parametrize("method,first_value", [("ordinary", 0), ("weighted", 99)])
def test_zero_weight_counts_as_sample_and_later_positive_weight_is_not_a_pseudo_sample(method, first_value):
    policy = np.full((4, 2), .5)
    policy[1] = [0, 1]
    learner = make_learner(method, policy=policy)
    learner.model.estimator.v_table.fill_(99)
    learner.learn(episode(rewards=(3.,), probabilities=(.5,)))
    assert learner.visit_counts[1] == 1 and learner.weight_sums[1] == 0
    assert learner.model.estimator.v_table[1] == first_value
    learner.learn(episode(actions=(1,), probabilities=(.5,)))
    assert learner.visit_counts[1] == 2 and learner.weight_sums[1] == 2
    assert learner.weighted_return_sums[1] == 8 and learner.model.estimator.v_table[1] == 4


@pytest.mark.parametrize("mode", ["first", "every"])
@pytest.mark.parametrize("method", ["ordinary", "weighted"])
def test_behavior_equals_target_reduces_to_on_policy_mc_prediction(method, mode):
    learner = make_learner(method, mode)
    on_policy = MCPredictionLearner(Model(IdentityRepresentation(), TabularVEstimator(4)).double(),
                                   MonteCarloReturnGenerator(1), mode)
    for _ in range(3):
        sample = episode((1, 2, 1), (1., 2., 3.), (.5, .5, .5))
        learner.learn(sample)
        on_policy.learn(sample)
    torch.testing.assert_close(learner.model.estimator.v_table, on_policy.model.estimator.v_table)
    torch.testing.assert_close(learner.visit_counts, on_policy.visit_counts)
    torch.testing.assert_close(learner.weight_sums, learner.visit_counts.double())


def test_owned_target_snapshot_and_no_grad_or_model_queries(monkeypatch):
    policy = torch.full((4, 2), .5, dtype=torch.float64, requires_grad=True)
    learner = make_learner(policy=policy)
    with torch.no_grad():
        policy[1] = torch.tensor([1., 0.])
    monkeypatch.setattr(learner.model, "forward", lambda *args: pytest.fail("no bootstrap or policy query"))
    learner.model.estimator.v_table.requires_grad_(True)
    result = learner.learn(episode())
    assert learner.target_policy[1, 0] == .5 and not learner.target_policy.requires_grad
    assert not result.requires_grad and learner.model.estimator.v_table.grad is None
    assert policy.grad is None


@pytest.mark.parametrize("terminated,truncated", [(False, False), (False, True)])
def test_open_and_truncated_only_episodes_fail_without_changes(terminated, truncated):
    learner = make_learner()
    with pytest.raises(ValueError, match="incomplete"):
        learner.learn(episode(terminated=terminated, truncated=truncated))
    assert learner.visit_counts.sum() == learner.weight_sums.sum() == learner.model.estimator.v_table.sum() == 0


def test_natural_termination_on_time_limit_is_complete():
    learner = make_learner()
    learner.learn(episode(truncated=True))
    assert learner.model.estimator.v_table[1] == 4


@pytest.mark.parametrize("log_b,error", [
    (None, ValueError), (torch.tensor(float('-inf')), ValueError), (torch.tensor(float('inf')), ValueError),
    (torch.tensor(float('nan')), ValueError), (torch.tensor(.1), ValueError), (torch.tensor(0), TypeError),
    (torch.tensor(True), TypeError), (torch.tensor([-.5]), TypeError),
])
def test_missing_zero_or_invalid_behavior_probability_is_rejected(log_b, error):
    learner = make_learner()
    sample = Episode((Transition(1, 0, 4., 0, True, False, log_b),))
    with pytest.raises(error, match="log_prob"):
        learner.learn(sample)
    assert learner.visit_counts.sum() == learner.weight_sums.sum() == learner.model.estimator.v_table.sum() == 0


@pytest.mark.parametrize("policy,error", [
    (np.zeros((4, 0)), ValueError), (np.ones((3, 2)) / 2, ValueError), (np.ones(4), ValueError),
    (np.ones((4, 2)), ValueError), (np.full((4, 2), -.5), ValueError), (np.full((4, 2), np.nan), ValueError),
    (np.full((4, 2), np.inf), ValueError), (np.full((4, 2), .5 + 1j), TypeError), (np.ones((4, 2), bool), TypeError),
])
def test_invalid_target_policy_is_rejected_at_construction(policy, error):
    with pytest.raises(error, match="target_policy"):
        make_learner(policy=policy)


@pytest.mark.parametrize("field,value,error", [
    ("state", 1., TypeError), ("state", True, TypeError), ("state", -1, IndexError), ("state", 4, IndexError),
    ("action", 0., TypeError), ("action", True, TypeError), ("action", -1, IndexError), ("action", 2, IndexError),
])
def test_bad_state_action_indices_are_rejected_before_updates(field, value, error):
    learner = make_learner()
    state, action = (value, 0) if field == "state" else (1, value)
    sample = Episode((Transition(state, action, 4., type(state)(0), True, False, torch.tensor(-math.log(4.))),))
    with pytest.raises(error, match=field):
        learner.learn(sample)
    assert learner.visit_counts.sum() == 0


def test_method_mode_and_model_contracts_fail_clearly():
    with pytest.raises(ValueError, match="method"):
        make_learner(method="per-decision")
    with pytest.raises(ValueError, match="visit_mode"):
        make_learner(mode="last")
    with pytest.raises(TypeError, match="Episode"):
        make_learner().learn(episode().transitions)
    for estimator in (TabularQEstimator(4, 2), StateValueEstimator(1)):
        with pytest.raises(TypeError, match="TabularVEstimator"):
            OffPolicyMCPredictionLearner(Model(IdentityRepresentation(), estimator), MonteCarloReturnGenerator(1), np.full((4, 2), .5))


def test_unrepresentable_positive_weight_raises_before_any_update():
    learner = make_learner()
    with pytest.raises(FloatingPointError, match="weights overflowed"):
        learner.learn(Episode((Transition(1, 0, 4., 0, True, False, torch.tensor(-1000., dtype=torch.float64)),)))
    assert learner.visit_counts.sum() == learner.weight_sums.sum() == learner.model.estimator.v_table.sum() == 0


def test_underflow_is_not_confused_with_exact_zero_target_probability():
    policy = np.tile([1e-300, 1.], (4, 1))
    learner = make_learner(policy=policy)
    with pytest.raises(FloatingPointError, match="underflowed"):
        learner.learn(episode((1, 2), (1., 1.), (.5, .5)))
    assert learner.visit_counts.sum() == 0


def test_weighted_return_overflow_does_not_commit_partial_state_or_stats_updates():
    learner = make_learner()
    sample = Episode((Transition(1, 0, np.float64(1e100), 0, True, False, torch.tensor(-700., dtype=torch.float64)),))
    with pytest.raises(FloatingPointError, match="statistics or values"):
        learner.learn(sample)
    assert learner.visit_counts.sum() == learner.weight_sums.sum() == learner.weighted_return_sums.sum() == 0
    assert learner.model.estimator.v_table.sum() == 0


def test_accumulated_weight_overflow_keeps_the_previous_valid_checkpoint_state():
    learner = make_learner()
    sample = Episode((Transition(1, 0, 1., 0, True, False, torch.tensor(math.log(.5) - math.log(1e308))),))
    learner.learn(sample)
    saved = (learner.model.estimator.v_table.clone(), learner.visit_counts.clone(),
             learner.weight_sums.clone(), learner.weighted_return_sums.clone())
    with pytest.raises(FloatingPointError, match="statistics or values"):
        learner.learn(sample)
    for current, original in zip((learner.model.estimator.v_table, learner.visit_counts,
                                  learner.weight_sums, learner.weighted_return_sums), saved):
        torch.testing.assert_close(current, original, rtol=0, atol=0)


def test_float64_stats_keep_weighted_float32_value_finite_but_unrepresentable_ordinary_value_fails():
    sample = Episode((Transition(1, 0, 4., 0, True, False, torch.tensor(-90., dtype=torch.float64)),))
    weighted = make_learner(dtype=torch.float32)
    weighted.learn(sample)
    assert weighted.model.estimator.v_table[1] == 4 and weighted.weight_sums.dtype == torch.float64
    ordinary = make_learner("ordinary", dtype=torch.float32)
    with pytest.raises(FloatingPointError, match="values"):
        ordinary.learn(sample)
    assert ordinary.visit_counts.sum() == 0


@pytest.mark.parametrize("method", ["ordinary", "weighted"])
@pytest.mark.parametrize("mode", ["first", "every"])
def test_model_and_all_stats_checkpoint_resume_identical_learning(method, mode):
    learner = make_learner(method, mode, gamma=.9)
    sample = episode((1, 2, 1), (1., 2., 3.), (.25, .25, .25))
    learner.learn(sample)
    checkpoint = io.BytesIO()
    torch.save({"model": learner.model.state_dict(), "target_policy": learner.target_policy,
                "method": method, "visit_mode": mode, "gamma": .9,
                **{name: getattr(learner, name) for name in ("visit_counts", "weight_sums", "weighted_return_sums")}}, checkpoint)
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    restored = make_learner(saved["method"], saved["visit_mode"], gamma=saved["gamma"], policy=saved["target_policy"])
    restored.model.load_state_dict(saved["model"], strict=True)
    for name in ("visit_counts", "weight_sums", "weighted_return_sums"):
        getattr(restored, name).copy_(saved[name])
        assert name not in restored.model.state_dict()
    for candidate in (learner, restored):
        candidate.learn(sample)
        candidate.learn(episode())
    torch.testing.assert_close(restored.model.estimator.v_table, learner.model.estimator.v_table, rtol=0, atol=0)
    for name in ("visit_counts", "weight_sums", "weighted_return_sums"):
        torch.testing.assert_close(getattr(restored, name), getattr(learner, name), rtol=0, atol=0)


def test_live_table_dtype_change_preserves_float64_stats_and_integer_counts():
    learner = make_learner(dtype=torch.float32)
    learner.learn(episode())
    learner.model.double()
    result = learner.learn(episode())
    assert result.dtype == torch.float64 and learner.visit_counts.dtype == torch.long
    assert learner.weight_sums.dtype == learner.weighted_return_sums.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("method", ["ordinary", "weighted"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_episode_and_target_policy_can_update_cuda_table_and_stats(method, dtype):
    learner = make_learner(method, dtype=dtype)
    learner.model.cuda()
    result = learner.learn(episode())
    assert result.device.type == learner.visit_counts.device.type == learner.weight_sums.device.type == "cuda"
    assert learner.weighted_return_sums.device.type == "cuda" and result.dtype == dtype
    assert learner.model.estimator.v_table[1].item() == (8 if method == "ordinary" else 4)
