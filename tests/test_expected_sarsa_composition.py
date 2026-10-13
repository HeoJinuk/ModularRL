"""Existing Q/probabilities/TD0/update contracts compose into Expected SARSA."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from examples.gridworld_expected_sarsa import learn_expected_sarsa
from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import DiscreteQEstimator, TabularQEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import Transition, TransitionBatch
from modular_rl.learner import TabularQLearner
from modular_rl.model import Model


def components(epsilon=.2, dtype=torch.float64, gamma=.9, alpha=.5, n_actions=2):
    selector = EpsilonGreedyActionSelector(epsilon)
    model = Model(IdentityRepresentation(), TabularQEstimator(4, n_actions), selector).to(dtype=dtype)
    model.estimator.q_table[1, 0] = 4
    model.estimator.q_table[2] = torch.tensor([10., 2.] if n_actions == 2 else [0., 3., 3., 1.])
    return model, TD0TargetGenerator(gamma), TabularQUpdateRule(alpha)


def learn(model, batch, target, rule):
    return learn_expected_sarsa(model, batch, target, rule, policy_selector=model.action_selector)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("epsilon,next_value", [(0, 10.), (1, 6.), (.2, 9.2)])
def test_expected_value_and_td_target_are_exact_component_compositions(epsilon, next_value, dtype, monkeypatch):
    model, target, rule = components(epsilon, dtype)
    table, rng = model.estimator.q_table, torch.get_rng_state().clone()
    pointer, generate = table.data_ptr(), target.generate
    calls = []

    def check(reward, bootstrap, terminated):
        calls.append(bootstrap.clone())
        assert not torch.is_grad_enabled() and table[1, 0] == 4
        torch.testing.assert_close(bootstrap, torch.tensor([next_value], dtype=dtype))
        return generate(reward, bootstrap, terminated)

    monkeypatch.setattr(target, "generate", check)
    result = learn(model, Transition(np.int32(1), np.int32(0), 1., np.int32(2), False, False).to_batch(), target, rule)
    assert len(calls) == 1
    assert result.item() == pytest.approx(1 + .9 * next_value - 4)
    assert table[1, 0].item() == pytest.approx(4 + .5 * (1 + .9 * next_value - 4))
    assert table.data_ptr() == pointer and result.dtype == dtype and result.shape == (1,)
    assert result.grad_fn is None and not result.requires_grad
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("epsilon,expected", [(0, 3.), (1, 1.75), (.2, 2.75)])
def test_epsilon_greedy_ties_include_exploration_in_the_expectation(epsilon, expected):
    model, target, rule = components(epsilon, gamma=1, alpha=1, n_actions=4)
    result = learn(model, Transition(0, 0, 0., 2, False, False).to_batch(), target, rule)
    assert result.item() == pytest.approx(expected)
    assert model.estimator.q_table[0, 0].item() == pytest.approx(expected)
    probabilities = model.action_selector.probabilities(model.forward(2))
    assert probabilities[1] == probabilities[2]  # Shared greedy tie mass.


def test_zero_epsilon_matches_existing_q_learning_on_the_same_batch():
    expected_model, target, rule = components(0)
    q_model, q_target, q_rule = components(0)
    sample = TransitionBatch(torch.tensor([0, 1, 1]), torch.tensor([0, 0, 1]), torch.tensor([2., 1., 3.]),
                             torch.tensor([1, 2, 3]), torch.tensor([False, False, True]), torch.tensor([False, True, False]))
    expected_error = learn(expected_model, sample, target, rule)
    q_error = TabularQLearner(q_model, q_target, q_rule).learn(sample)
    torch.testing.assert_close(expected_error, q_error, rtol=0, atol=0)
    torch.testing.assert_close(expected_model.estimator.q_table, q_model.estimator.q_table, rtol=0, atol=0)


@pytest.mark.parametrize("terminated,truncated,expected", [(True, False, -3.), (True, True, -3.), (False, True, 5.28)])
def test_termination_and_time_limit_have_independent_bootstrap_contracts(terminated, truncated, expected):
    model, target, rule = components()
    result = learn(model, Transition(1, 0, 1., 2, terminated, truncated).to_batch(), target, rule)
    assert result.item() == pytest.approx(expected)


def test_batch_freezes_both_policy_and_q_before_any_table_writes():
    model, target, rule = components(gamma=1, alpha=1)
    model.estimator.q_table.zero_()
    model.estimator.q_table[1] = torch.tensor([10., 2.])
    sample = TransitionBatch(torch.tensor([1, 0]), torch.tensor([0, 0]), torch.zeros(2),
                             torch.tensor([3, 1]), torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool))
    result = learn(model, sample, target, rule)
    torch.testing.assert_close(result, torch.tensor([-10., 9.2], dtype=torch.float64))
    assert model.estimator.q_table[1, 0] == 0 and model.estimator.q_table[0, 0].item() == pytest.approx(9.2)
    # After the first terminal update, recomputed policy would favor action 1.
    online, online_target, online_rule = components(gamma=1, alpha=1)
    online.estimator.q_table.copy_(torch.tensor([[0., 0.], [10., 2.], [0., 0.], [0., 0.]]))
    learn(online, Transition(1, 0, 0., 3, True, False).to_batch(), online_target, online_rule)
    learn(online, Transition(0, 0, 0., 1, False, False).to_batch(), online_target, online_rule)
    assert online.estimator.q_table[0, 0].item() == pytest.approx(1.8)


def test_duplicate_pairs_keep_the_existing_sequential_update_rule_contract():
    model, target, rule = components(0, gamma=1)
    sample = TransitionBatch(torch.tensor([1, 1]), torch.tensor([0, 0]), torch.tensor([1., 3.]),
                             torch.tensor([1, 1]), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
    error = learn(model, sample, target, rule)
    torch.testing.assert_close(error, torch.tensor([1., 2.5], dtype=torch.float64))
    assert model.estimator.q_table[1, 0] == 5.75


def test_one_estimate_is_shared_by_probability_query_and_target_calculation(monkeypatch):
    model, target, rule = components()
    calls, estimates = [], []
    forward, probabilities, generate, update = model.forward, model.action_selector.probabilities, target.generate, rule.update

    def estimate(observation):
        calls.append("forward")
        value = forward(observation)
        estimates.append(value)
        return value

    def policy(value):
        calls.append("probabilities")
        assert value is estimates[0] and not torch.is_grad_enabled()
        return probabilities(value)

    def td(*args):
        calls.append("target")
        return generate(*args)

    def write(*args):
        calls.append("update")
        return update(*args)

    monkeypatch.setattr(model, "forward", estimate)
    monkeypatch.setattr(model.action_selector, "probabilities", policy)
    monkeypatch.setattr(target, "generate", td)
    monkeypatch.setattr(rule, "update", write)
    monkeypatch.setattr(model, "act", lambda *args: pytest.fail("no sampling during learning"))
    learn(model, Transition(1, 0, 1., 2, False, False).to_batch(), target, rule)
    assert calls == ["forward", "probabilities", "target", "update"]


def test_on_policy_selector_identity_is_enforced_even_with_matching_epsilon():
    model, target, rule = components()
    sample = Transition(1, 0, 1., 2, False, False).to_batch()
    before = model.estimator.q_table.clone()
    with pytest.raises(ValueError, match="same object"):
        learn_expected_sarsa(model, sample, target, rule, policy_selector=EpsilonGreedyActionSelector(.2))
    torch.testing.assert_close(model.estimator.q_table, before, rtol=0, atol=0)


def test_same_selector_epsilon_change_is_used_and_inputs_keep_their_graph_and_dtype():
    model, target, rule = components(0)
    model.action_selector.epsilon = 1
    reward = torch.tensor([1.], requires_grad=True)
    sample = replace(Transition(1, 0, 1., 2, False, False).to_batch(), reward=reward)
    result = learn(model, sample, target, rule)
    assert result.item() == pytest.approx(2.4) and not result.requires_grad
    assert sample.reward is reward and reward.requires_grad and reward.grad is None
    assert sample.reward.dtype == torch.float32


@pytest.mark.parametrize("name,value,error", [
    ("observation", torch.tensor([1.]), TypeError), ("action", torch.tensor([0.]), TypeError),
    ("next_observation", torch.tensor([2.]), TypeError), ("reward", torch.tensor([1]), TypeError),
    ("terminated", torch.tensor([0]), TypeError), ("truncated", torch.tensor([False, False]), ValueError),
    ("observation", torch.tensor(1), ValueError), ("observation", torch.tensor([], dtype=torch.long), ValueError),
])
def test_batch_contract_is_checked_before_any_update(name, value, error):
    model, target, rule = components()
    with pytest.raises(error):
        learn(model, replace(Transition(1, 0, 1., 2, False, False).to_batch(), **{name: value}), target, rule)


@pytest.mark.parametrize("output,error", [(torch.ones(1, 1), ValueError), (torch.ones(1, 2), ValueError), ([.5, .5], TypeError)])
def test_probability_shape_dtype_and_type_do_not_silently_broadcast(output, error, monkeypatch):
    model, target, rule = components()
    original = model.estimator.q_table.clone()
    monkeypatch.setattr(model.action_selector, "probabilities", lambda estimate: output)
    with pytest.raises(error, match="probabilities"):
        learn(model, Transition(1, 0, 1., 2, False, False).to_batch(), target, rule)
    torch.testing.assert_close(model.estimator.q_table, original, rtol=0, atol=0)


def test_missing_probability_interface_wrong_model_and_non_batch_fail():
    model, target, rule = components()
    sample = Transition(1, 0, 1., 2, False, False).to_batch()
    model.action_selector = None
    with pytest.raises(TypeError, match="discrete probabilities"):
        learn(model, sample, target, rule)
    wrong = Model(IdentityRepresentation(), DiscreteQEstimator(1, 2), EpsilonGreedyActionSelector(.1))
    with pytest.raises(TypeError, match="TabularQEstimator"):
        learn(wrong, sample, target, rule)
    model, target, rule = components()
    with pytest.raises(TypeError, match="TransitionBatch"):
        learn(model, sample.observation, target, rule)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_batch_can_update_cuda_model_without_rng_consumption(dtype):
    model, target, rule = components(dtype=dtype)
    model.cuda()
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    result = learn(model, Transition(1, 0, 1., 2, False, False).to_batch(), target, rule)
    assert result.device.type == "cuda" and result.dtype == dtype and result.item() == pytest.approx(5.28)
    torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
