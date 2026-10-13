"""Actual next-action TD targets, terminal masks and frozen/online updates."""

import io
from dataclasses import replace

import numpy as np
import pytest
import torch

from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import DiscreteQEstimator, TabularQEstimator, TabularVEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import Transition, TransitionBatch
from modular_rl.learner import TabularQLearner, TabularSARSALearner
from modular_rl.model import Model


def make_learner(dtype=torch.float64, gamma=.9, alpha=.5, selector=None):
    model = Model(IdentityRepresentation(), TabularQEstimator(4, 2), selector).to(dtype=dtype)
    model.estimator.q_table[1, 0] = 4
    model.estimator.q_table[2] = torch.tensor([10., 2.])
    return TabularSARSALearner(model, TD0TargetGenerator(gamma), TabularQUpdateRule(alpha))


def batch(terminated=False, truncated=False):
    return Transition(np.int32(1), np.int32(0), 1., np.int32(2), terminated, truncated).to_batch()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("next_action,target,error,value", [(1, 2.8, -1.2, 3.4), (0, 10., 6., 7.)])
def test_sarsa_uses_actual_next_action_instead_of_max_q(dtype, next_action, target, error, value, monkeypatch):
    learner = make_learner(dtype)
    table = learner.model.estimator.q_table
    pointer = table.data_ptr()
    original_generate = learner.target_generator.generate

    def generate(reward, next_value, terminated):
        assert not torch.is_grad_enabled()
        result = original_generate(reward, next_value, terminated)
        torch.testing.assert_close(result, torch.tensor([target], dtype=dtype))
        return result

    monkeypatch.setattr(learner.target_generator, "generate", generate)
    result = learner.learn(batch(), next_action=next_action)
    torch.testing.assert_close(result, torch.tensor([error], dtype=dtype))
    assert table[1, 0].item() == pytest.approx(value)
    assert table.data_ptr() == pointer and result.shape == (1,)
    assert result.dtype == dtype and result.grad_fn is None and not result.requires_grad


def test_sarsa_and_q_learning_differ_when_exploration_selects_the_lower_q_action():
    sarsa, other = make_learner(), make_learner()
    q_learning = TabularQLearner(other.model, other.target_generator, other.update_rule)
    assert sarsa.learn(batch(), next_action=1).item() == pytest.approx(-1.2)
    assert q_learning.learn(batch()).item() == 6
    assert sarsa.model.estimator.q_table[1, 0].item() == pytest.approx(3.4)
    assert other.model.estimator.q_table[1, 0] == 7


@pytest.mark.parametrize("truncated", [False, True])
def test_all_terminal_rows_allow_none_without_next_state_model_call(truncated, monkeypatch):
    learner = make_learner()
    monkeypatch.setattr(learner.model, "forward", lambda *args: pytest.fail("terminal needs no bootstrap"))
    result = learner.learn(batch(True, truncated), next_action=None)
    torch.testing.assert_close(result, torch.tensor([-3.], dtype=torch.float64))
    assert learner.model.estimator.q_table[1, 0] == 2.5


def test_truncation_bootstraps_from_the_final_observation_and_requires_next_action():
    learner = make_learner()
    with pytest.raises(ValueError, match="including truncation"):
        learner.learn(batch(truncated=True))
    result = learner.learn(batch(truncated=True), next_action=torch.tensor([1], dtype=torch.int32))
    assert result.item() == pytest.approx(-1.2)
    assert learner.model.estimator.q_table[1, 0].item() == pytest.approx(3.4)


def test_mixed_terminal_rows_use_valid_ignored_placeholders():
    learner = make_learner()
    sample = TransitionBatch(torch.tensor([1, 1]), torch.tensor([0, 1]), torch.tensor([1., 1.]),
                             torch.tensor([2, 2]), torch.tensor([True, False]), torch.tensor([False, True]))
    with pytest.raises(ValueError, match="required"):
        learner.learn(sample)
    result = learner.learn(sample, next_action=torch.tensor([0, 1]))
    torch.testing.assert_close(result, torch.tensor([-3., 2.8], dtype=torch.float64))
    torch.testing.assert_close(learner.model.estimator.q_table[1], torch.tensor([2.5, 1.4], dtype=torch.float64))


def test_all_terminal_batch_without_actions_returns_each_reward_error():
    learner = make_learner()
    sample = TransitionBatch(torch.tensor([1, 2]), torch.tensor([0, 1]), torch.tensor([1., 3.]),
                             torch.tensor([3, 3]), torch.tensor([True, True]), torch.tensor([False, True]))
    torch.testing.assert_close(learner.learn(sample), torch.tensor([-3., 1.], dtype=torch.float64))


def test_batch_bootstrap_is_frozen_while_online_calls_observe_previous_updates():
    frozen, online = make_learner(gamma=1, alpha=1), make_learner(gamma=1, alpha=1)
    for learner in (frozen, online):
        learner.model.estimator.q_table.zero_()
        learner.model.estimator.q_table[1, 0] = 10
    sample = TransitionBatch(torch.tensor([1, 0]), torch.tensor([0, 0]), torch.zeros(2),
                             torch.tensor([3, 1]), torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool))
    result = frozen.learn(sample, next_action=torch.tensor([0, 0]))
    torch.testing.assert_close(result, torch.tensor([-10., 10.], dtype=torch.float64))
    online.learn(Transition(1, 0, 0., 3, True, False).to_batch(), next_action=None)
    online.learn(Transition(0, 0, 0., 1, False, False).to_batch(), next_action=0)
    assert frozen.model.estimator.q_table[0, 0] == 10 and online.model.estimator.q_table[0, 0] == 0


def test_duplicate_pairs_keep_frozen_targets_and_sequential_update_rule_errors():
    frozen, online = make_learner(gamma=1), make_learner(gamma=1)
    sample = TransitionBatch(torch.tensor([1, 1]), torch.tensor([0, 0]), torch.tensor([1., 3.]),
                             torch.tensor([1, 1]), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
    result = frozen.learn(sample, next_action=torch.tensor([0, 0]))
    torch.testing.assert_close(result, torch.tensor([1., 2.5], dtype=torch.float64))
    assert frozen.model.estimator.q_table[1, 0] == 5.75
    for reward in (1., 3.):
        online.learn(Transition(1, 0, reward, 1, False, False).to_batch(), next_action=0)
    assert online.model.estimator.q_table[1, 0] == 6


@pytest.mark.parametrize("gamma,alpha,expected", [(0, 1, 1.), (1, 1, 3.), (.9, np.float64(.5), 3.4)])
def test_existing_gamma_and_alpha_contracts(gamma, alpha, expected):
    learner = make_learner(gamma=gamma, alpha=alpha)
    learner.learn(batch(), next_action=1)
    assert learner.model.estimator.q_table[1, 0].item() == pytest.approx(expected)


@pytest.mark.parametrize("next_action,error", [
    (1., TypeError), (True, TypeError), (torch.tensor([1.]), TypeError), (torch.tensor([1j]), TypeError),
    (torch.tensor([True]), TypeError), (-1, IndexError), (2, IndexError),
    (torch.tensor([[1]]), ValueError), (torch.tensor([0, 1]), ValueError), (torch.tensor([]), ValueError),
])
def test_invalid_next_action_fails_before_table_update(next_action, error):
    learner = make_learner()
    original = learner.model.estimator.q_table.clone()
    with pytest.raises(error, match="next_action"):
        learner.learn(batch(), next_action=next_action)
    torch.testing.assert_close(learner.model.estimator.q_table, original, rtol=0, atol=0)


def test_scalar_next_action_cannot_broadcast_over_batch_and_terminal_ids_are_validated():
    learner = make_learner()
    sample = TransitionBatch(torch.tensor([1, 1]), torch.tensor([0, 1]), torch.ones(2),
                             torch.tensor([2, 2]), torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="shape"):
        learner.learn(sample, next_action=1)
    with pytest.raises(IndexError, match="terminal placeholders"):
        learner.learn(sample, next_action=torch.tensor([-1, 1]))


@pytest.mark.parametrize("name,value,error", [
    ("observation", torch.tensor([1.]), TypeError), ("action", torch.tensor([0.]), TypeError),
    ("next_observation", torch.tensor([2.]), TypeError), ("reward", torch.tensor([1]), TypeError),
    ("terminated", torch.tensor([0]), TypeError), ("truncated", torch.tensor([False, False]), ValueError),
    ("observation", torch.tensor(1), ValueError), ("observation", torch.tensor([], dtype=torch.long), ValueError),
])
def test_batch_contract_matches_existing_tabular_learners(name, value, error):
    with pytest.raises(error):
        make_learner().learn(replace(batch(), **{name: value}), next_action=1)


@pytest.mark.parametrize("estimator", [TabularVEstimator(4), DiscreteQEstimator(1, 2)])
def test_wrong_model_connection_fails(estimator):
    with pytest.raises(TypeError, match="TabularQEstimator"):
        TabularSARSALearner(Model(IdentityRepresentation(), estimator), TD0TargetGenerator(.9), TabularQUpdateRule(.5))


def test_non_batch_input_fails():
    with pytest.raises(TypeError, match="TransitionBatch"):
        make_learner().learn(Transition(1, 0, 1., 2, False, False), next_action=1)


def test_learning_changes_the_existing_behavior_selector_for_future_selections():
    learner = make_learner(alpha=1, selector=EpsilonGreedyActionSelector(0))
    assert learner.model.act(1).action.item() == 0
    learner.learn(Transition(1, 1, 20., 3, True, False).to_batch(), next_action=None)
    assert learner.model.act(1).action.item() == 1


def test_components_are_called_in_order_without_sampling_and_target_graphs(monkeypatch):
    learner = make_learner(selector=EpsilonGreedyActionSelector(.1))
    calls, rng = [], torch.get_rng_state().clone()
    original_forward, generate, update = learner.model.forward, learner.target_generator.generate, learner.update_rule.update
    monkeypatch.setattr(learner.model.action_selector, "select", lambda *args: pytest.fail("no learner sampling"))

    def forward(observation):
        calls.append("forward")
        assert not torch.is_grad_enabled()
        return original_forward(observation)

    def target(*args):
        calls.append("target")
        return generate(*args)

    def write(*args):
        calls.append("update")
        assert not args[-1].requires_grad
        return update(*args)

    monkeypatch.setattr(learner.model, "forward", forward)
    monkeypatch.setattr(learner.target_generator, "generate", target)
    monkeypatch.setattr(learner.update_rule, "update", write)
    reward = torch.tensor([1.], requires_grad=True)
    sample = replace(batch(), reward=reward)
    result = learner.learn(sample, next_action=torch.tensor([1], dtype=torch.int32))
    assert calls == ["forward", "target", "update"] and not result.requires_grad
    assert sample.reward is reward and reward.requires_grad and reward.grad is None
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_model_checkpoint_restore_and_live_dtype(dtype):
    learner = make_learner(dtype)
    learner.learn(batch(), next_action=1)
    file = io.BytesIO()
    torch.save(learner.model.state_dict(), file)
    file.seek(0)
    restored = make_learner(dtype)
    restored.model.load_state_dict(torch.load(file, weights_only=True), strict=True)
    for item in (learner, restored):
        item.learn(batch(), next_action=0)
    torch.testing.assert_close(learner.model.estimator.q_table, restored.model.estimator.q_table, rtol=0, atol=0)
    restored.model.double()
    assert restored.learn(batch(), next_action=1).dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_inputs_can_update_cuda_model(dtype):
    learner = make_learner(dtype)
    learner.model.cuda()
    result = learner.learn(batch(), next_action=torch.tensor([1], dtype=torch.int32))
    assert result.device.type == "cuda" and result.dtype == dtype
    assert learner.model.estimator.q_table[1, 0].item() == pytest.approx(3.4)
