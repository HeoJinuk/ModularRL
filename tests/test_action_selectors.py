"""Behavior-policy contracts for tabular, DQN, and categorical actor-critic use."""

import math

import numpy as np
import pytest
import torch
from torch.distributions import Categorical, Normal

from modular_rl.component.action_selector import (
    CategoricalActionSelector,
    EpsilonGreedyActionSelector,
    GreedyActionSelector,
)
from modular_rl.component.estimator import (
    CategoricalPolicyEstimator,
    StateValueEstimator,
    TabularQEstimator,
)
from modular_rl.component.estimator.actor_critic import ActorCriticEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.data import ActionOutput, EstimatorOutput, ValueOutput
from modular_rl.model import Model


@pytest.fixture(autouse=True)
def seeded_torch_rng():
    # Sampling tests neither depend on nor change other tests' RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        yield


def q_output(q):
    return EstimatorOutput(values=ValueOutput(q=(q,)))


@pytest.mark.parametrize("selector,expected", [
    (GreedyActionSelector(), [0., 0.5, 0.5, 0.]),
    (EpsilonGreedyActionSelector(0.), [0., 0.5, 0.5, 0.]),
    (EpsilonGreedyActionSelector(1.), [0.25, 0.25, 0.25, 0.25]),
    (EpsilonGreedyActionSelector(0.2), [0.05, 0.45, 0.45, 0.05]),
])
@pytest.mark.parametrize("batch_shape", [(), (7,), (2, 3)])
def test_q_probabilities_actions_and_log_prob_preserve_batch_axes(selector, expected, batch_shape):
    q = torch.tensor([0., 3., 3., 1.], dtype=torch.float64).expand(*batch_shape, 4)
    estimate = q_output(q)
    expected_probs = torch.tensor(expected, dtype=torch.float64).expand_as(q)
    probs = selector.probabilities(estimate)
    torch.testing.assert_close(probs, expected_probs)
    torch.testing.assert_close(probs.sum(-1), torch.ones(batch_shape, dtype=torch.float64))

    result = selector.select(estimate)
    assert isinstance(result, ActionOutput)
    assert result.action.dtype == torch.long
    assert result.action.shape == batch_shape
    assert result.action.device == q.device
    assert result.log_prob.shape == batch_shape
    assert result.log_prob.dtype == torch.float64
    chosen = expected_probs.gather(-1, result.action.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(result.log_prob, chosen.log())

    rng = torch.random.get_rng_state()
    deterministic = selector.select(estimate, deterministic=True)
    assert deterministic.action.shape == batch_shape
    assert torch.all(deterministic.action == 1)  # first tied maximum, even at epsilon=1
    assert deterministic.log_prob is None
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_greedy_unique_maximum_and_integer_q_values():
    selector = GreedyActionSelector()
    estimate = q_output(torch.tensor([-5, -3, 8, -1]))
    torch.testing.assert_close(selector.probabilities(estimate), torch.tensor([0., 0., 1., 0.]))
    result = selector.select(estimate)
    assert result.action.item() == 2
    assert result.log_prob.item() == 0.


def test_q_comparison_keeps_double_precision():
    estimate = q_output(torch.tensor([1., 1. + 1e-10], dtype=torch.float64))
    assert GreedyActionSelector().select(estimate).action.item() == 1


@pytest.mark.parametrize("selector,expected", [
    (GreedyActionSelector(), [0., 0.5, 0.5, 0.]),
    (EpsilonGreedyActionSelector(0.), [0., 0.5, 0.5, 0.]),
    (EpsilonGreedyActionSelector(1.), [0.25, 0.25, 0.25, 0.25]),
    (EpsilonGreedyActionSelector(0.2), [0.05, 0.45, 0.45, 0.05]),
    (CategoricalActionSelector(), [0.1, 0.2, 0.3, 0.4]),
])
def test_empirical_sampling_matches_theoretical_probabilities(selector, expected):
    n = 30_000
    if isinstance(selector, CategoricalActionSelector):
        policy = Categorical(probs=torch.tensor(expected).expand(n, 4))
        estimate = EstimatorOutput(policy=policy)
    else:
        estimate = q_output(torch.tensor([0., 3., 3., 1.]).expand(n, 4))
    result = selector.select(estimate)
    counts = torch.bincount(result.action, minlength=4)
    theoretical = torch.tensor(expected)
    # A six-standard-deviation binomial bound avoids flaky sampling assertions.
    tolerance = 6 * torch.sqrt(theoretical * (1 - theoretical) / n) + 1 / n
    assert torch.all((counts / n - theoretical).abs() <= tolerance)
    assert torch.all(counts[theoretical == 0] == 0)


def test_tiny_epsilon_log_prob_is_not_clamped(monkeypatch):
    selector = EpsilonGreedyActionSelector(2e-10)
    estimate = q_output(torch.tensor([0., 3., 3., 1.], dtype=torch.float64))
    # Force a rare but possible exploration action to test its likelihood.
    monkeypatch.setattr(Categorical, "sample", lambda self: torch.tensor(0))
    result = selector.select(estimate)
    assert result.log_prob.item() == pytest.approx(math.log(selector.epsilon / 4), abs=1e-12)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_epsilon_range_checked_at_construction_and_assignment(value):
    with pytest.raises(ValueError, match="epsilon"):
        EpsilonGreedyActionSelector(value)
    selector = EpsilonGreedyActionSelector(0.5)
    with pytest.raises(ValueError, match="epsilon"):
        selector.epsilon = value
    assert selector.epsilon == 0.5


@pytest.mark.parametrize("value", [None, "0.1", True, 1j, torch.tensor(0.1)])
def test_epsilon_rejects_non_numeric_configuration(value):
    with pytest.raises(TypeError, match="epsilon"):
        EpsilonGreedyActionSelector(value)
    selector = EpsilonGreedyActionSelector()
    with pytest.raises(TypeError, match="epsilon"):
        selector.epsilon = value


def test_epsilon_assignment_changes_probabilities_and_survives_checkpoint():
    selector = EpsilonGreedyActionSelector(0.)
    selector.epsilon = 1.
    estimate = q_output(torch.tensor([0., 9., 2.]))
    torch.testing.assert_close(selector.probabilities(estimate), torch.full((3,), 1 / 3))
    restored = EpsilonGreedyActionSelector()
    restored.load_state_dict(selector.state_dict())
    assert restored.epsilon == 1.
    assert restored.select(estimate, deterministic=True).action.item() == 1


@pytest.mark.parametrize("selector", [GreedyActionSelector(), EpsilonGreedyActionSelector()])
@pytest.mark.parametrize("q,error", [
    (np.array([1., 2.]), TypeError),
    ([1., 2.], TypeError),
    ({"atoms": torch.tensor([1., 2.]), "logits": torch.zeros(3, 2)}, TypeError),
    (torch.tensor(1.), ValueError),
    (torch.empty(2, 0), ValueError),
])
def test_q_payload_contract_for_probabilities_and_both_selection_modes(selector, q, error):
    estimate = q_output(q)
    with pytest.raises(error):
        selector.probabilities(estimate)
    for deterministic in (False, True):
        with pytest.raises(error):
            selector.select(estimate, deterministic=deterministic)


@pytest.mark.parametrize("selector", [GreedyActionSelector(), EpsilonGreedyActionSelector()])
@pytest.mark.parametrize("critics", [(), (torch.ones(3), torch.ones(3))])
def test_missing_or_multiple_q_estimates_rejected(selector, critics):
    estimate = EstimatorOutput(values=ValueOutput(v=torch.tensor(100.), q=critics))
    with pytest.raises(ValueError):
        selector.probabilities(estimate)
    for deterministic in (False, True):
        with pytest.raises(ValueError):
            selector.select(estimate, deterministic=deterministic)


@pytest.mark.parametrize("batch_shape", [(), (7,), (2, 3)])
def test_categorical_probabilities_sampling_log_prob_and_deterministic_mode(batch_shape):
    probs = torch.tensor([0.1, 0.4, 0.4, 0.1], dtype=torch.float64).expand(*batch_shape, 4)
    policy = Categorical(probs=probs)
    estimate = EstimatorOutput(policy=policy)
    selector = CategoricalActionSelector()
    torch.testing.assert_close(selector.probabilities(estimate), probs)
    result = selector.select(estimate)
    assert result.action.dtype == torch.long
    assert result.action.shape == batch_shape
    assert result.log_prob.shape == batch_shape
    torch.testing.assert_close(result.log_prob, policy.log_prob(result.action))
    chosen = probs.gather(-1, result.action.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(result.log_prob, chosen.log())

    rng = torch.random.get_rng_state()
    result = selector.select(estimate, deterministic=True)
    assert torch.all(result.action == 1)
    assert result.log_prob is None
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("payload", [
    None, torch.tensor([0.2, 0.8]), Normal(torch.tensor(0.), torch.tensor(1.)),
])
def test_categorical_rejects_unsupported_policy(payload):
    selector = CategoricalActionSelector()
    estimate = EstimatorOutput(policy=payload, values=ValueOutput(q=(torch.ones(2),)))
    with pytest.raises(TypeError, match="Categorical"):
        selector.probabilities(estimate)
    for deterministic in (False, True):
        with pytest.raises(TypeError, match="Categorical"):
            selector.select(estimate, deterministic=deterministic)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_categorical_accepts_low_precision_rounding(dtype):
    torch.manual_seed(17)
    policy = Categorical(logits=torch.randn(40, 7, dtype=dtype))
    estimate = EstimatorOutput(policy=policy)
    selector = CategoricalActionSelector()
    torch.testing.assert_close(selector.probabilities(estimate), policy.probs)
    result = selector.select(estimate)
    torch.testing.assert_close(result.log_prob, policy.log_prob(result.action))
    assert result.action.shape == (40,)


@pytest.mark.parametrize("selector", [
    GreedyActionSelector(), EpsilonGreedyActionSelector(), CategoricalActionSelector(),
])
def test_torch_rng_controls_sampling(selector):
    estimate = EstimatorOutput(
        policy=Categorical(logits=torch.zeros(40, 4)),
        values=ValueOutput(q=(torch.ones(40, 4),)),
    )
    torch.manual_seed(42)
    first = selector.select(estimate)
    torch.manual_seed(42)
    second = selector.select(estimate)
    assert torch.equal(first.action, second.action)
    torch.testing.assert_close(first.log_prob, second.log_prob)


def test_policy_estimator_returns_distribution_without_sampling_and_selector_preserves_gradient():
    estimator = CategoricalPolicyEstimator(2, 3)
    features = torch.ones(5, 2, requires_grad=True)
    rng = torch.random.get_rng_state()
    estimate = estimator(features)
    assert isinstance(estimate.policy, Categorical)
    assert torch.equal(torch.random.get_rng_state(), rng)
    result = CategoricalActionSelector().select(estimate)
    (-result.log_prob.mean()).backward()
    assert features.grad is not None
    assert estimator.network.weight.grad is not None


def test_actor_critic_selection_uses_policy_and_ignores_values():
    estimator = ActorCriticEstimator(CategoricalPolicyEstimator(2, 3), StateValueEstimator(2))
    with torch.no_grad():
        estimator.policy.network.weight.zero_()
        estimator.policy.network.bias.copy_(torch.tensor([0., 2., 1.]))
    model = Model(IdentityRepresentation(), estimator, CategoricalActionSelector())
    features = torch.ones(2, 4, 2)
    estimate = model(features)
    assert estimate.values.v.shape == (2, 4)
    altered = EstimatorOutput(
        policy=estimate.policy,
        values=ValueOutput(v=torch.full((2, 4), float("nan")), q=(object(), object())),
    )
    selector = model.action_selector
    torch.testing.assert_close(selector.probabilities(altered), estimate.policy.probs)
    torch.manual_seed(99)
    first = selector.select(estimate)
    torch.manual_seed(99)
    second = selector.select(altered)
    assert torch.equal(first.action, second.action)
    torch.testing.assert_close(first.log_prob, second.log_prob)
    chosen = model.act(features, deterministic=True)
    assert chosen.action.shape == (2, 4)
    assert torch.all(chosen.action == 1)
    assert chosen.log_prob is None
    assert not model.act(features).log_prob.requires_grad


@pytest.mark.parametrize("selector", [GreedyActionSelector(), EpsilonGreedyActionSelector(1.)])
def test_tabular_model_act_uses_q_for_deterministic_selection(selector):
    estimator = TabularQEstimator(3, 4)
    estimator.q_table.copy_(torch.tensor([[0., 3., 3., 0.], [0., 0., 1., 4.], [9., 0., 0., 0.]]))
    model = Model(IdentityRepresentation(), estimator, selector)
    assert model.act(np.int64(0), deterministic=True).action.item() == 1
    result = model.act(np.array([[0, 1], [2, 0]]), deterministic=True)
    assert torch.equal(result.action, torch.tensor([[1, 3], [0, 1]]))
    assert result.log_prob is None


@pytest.mark.parametrize("estimator,selector,observation,error,message", [
    (StateValueEstimator(2), GreedyActionSelector(), torch.ones(2), ValueError, "values.q"),
    (StateValueEstimator(2), EpsilonGreedyActionSelector(), torch.ones(2), ValueError, "values.q"),
    (CategoricalPolicyEstimator(2, 3), GreedyActionSelector(), torch.ones(2), ValueError, "values.q"),
    (CategoricalPolicyEstimator(2, 3), EpsilonGreedyActionSelector(), torch.ones(2), ValueError, "values.q"),
    (TabularQEstimator(3, 2), CategoricalActionSelector(), 0, TypeError, "Categorical"),
])
def test_incompatible_estimator_selector_connection_fails_clearly(
    estimator, selector, observation, error, message
):
    model = Model(IdentityRepresentation(), estimator, selector)
    estimate = model(observation)
    with pytest.raises(error, match=message):
        selector.probabilities(estimate)
    for deterministic in (False, True):
        with pytest.raises(error, match=message):
            model.act(observation, deterministic=deterministic)


@pytest.mark.parametrize("selector", [
    GreedyActionSelector(), EpsilonGreedyActionSelector(), CategoricalActionSelector(),
])
@pytest.mark.parametrize("method,kwargs", [
    ("probabilities", {}),
    ("select", {}),
    ("select", {"deterministic": True}),
])
def test_input_is_validated_once_per_public_call(selector, method, kwargs, monkeypatch):
    estimate = EstimatorOutput(
        policy=Categorical(logits=torch.zeros(2, 3, 4)),
        values=ValueOutput(q=(torch.ones(2, 3, 4),)),
    )
    validate = selector._validate
    calls = []

    def track_validation(value):
        calls.append(value)
        validate(value)

    monkeypatch.setattr(selector, "_validate", track_validation)
    getattr(selector, method)(estimate, **kwargs)
    assert len(calls) == 1
    assert calls[0] is estimate
