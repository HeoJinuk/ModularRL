"""Final configuration, Gaussian numerics, and conditional-critic contracts."""

import math

import numpy as np
import pytest
import torch
from torch import nn

from modular_rl.component.action_selector import CategoricalActionSelector
from modular_rl.component.estimator import (
    ActorCriticEstimator,
    CategoricalPolicyEstimator,
    ContinuousQEstimator,
    DeterministicPolicyEstimator,
    DiscreteAdvantageEstimator,
    DiscreteQEstimator,
    GaussianPolicyEstimator,
    StateValueEstimator,
)
from modular_rl.component.representation import IdentityRepresentation, MLPRepresentation
from modular_rl.model import Model


DIMENSION_CASES = [
    pytest.param(CategoricalPolicyEstimator, "n_actions", id="categorical"),
    pytest.param(GaussianPolicyEstimator, "action_dim", id="gaussian"),
    pytest.param(DeterministicPolicyEstimator, "action_dim", id="deterministic"),
    pytest.param(StateValueEstimator, None, id="V"),
    pytest.param(DiscreteQEstimator, "n_actions", id="discrete-Q"),
    pytest.param(DiscreteAdvantageEstimator, "n_actions", id="advantage"),
    pytest.param(ContinuousQEstimator, "action_dim", id="continuous-Q"),
]


@pytest.fixture(autouse=True)
def fixed_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        yield


@pytest.mark.parametrize("estimator_type,output_dimension", DIMENSION_CASES)
@pytest.mark.parametrize("custom", [False, True], ids=["default", "custom"])
@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_neural_dimensions_use_the_same_positive_integer_rule(
    estimator_type, output_dimension, custom, invalid
):
    kwargs = {"feature_dim": 4}
    if output_dimension is not None:
        kwargs[output_dimension] = 2
    if custom:
        kwargs["network"] = nn.Identity()
    dimension_names = ["feature_dim"] + ([output_dimension] if output_dimension else [])
    for name in dimension_names:
        with pytest.raises(ValueError, match="positive integer"):
            estimator_type(**{**kwargs, name: invalid})


@pytest.mark.parametrize("estimator_type,output_dimension", DIMENSION_CASES)
@pytest.mark.parametrize("custom", [False, True], ids=["default", "custom"])
def test_numpy_integer_dimensions_remain_supported(estimator_type, output_dimension, custom):
    kwargs = {"feature_dim": np.int64(4)}
    output_size = 1 if output_dimension is None or estimator_type is ContinuousQEstimator else 2
    if output_dimension is not None:
        kwargs[output_dimension] = np.int64(2)
    if custom:
        input_size = 6 if estimator_type is ContinuousQEstimator else 4
        kwargs["network"] = nn.Linear(input_size, output_size)
    estimator = estimator_type(**kwargs)
    features, actions = torch.ones(4), torch.ones(2)
    estimate = estimator.evaluate(features, actions) if estimator_type is ContinuousQEstimator else estimator(features)
    assert estimate is not None


@pytest.mark.parametrize("invalid", [-1, True, 1.5])
def test_hidden_dim_is_validated_only_for_the_default_continuous_q_network(invalid):
    with pytest.raises(ValueError, match="hidden_dim must be a positive integer"):
        ContinuousQEstimator(4, 2, hidden_dim=invalid)
    estimator = ContinuousQEstimator(4, 2, hidden_dim=invalid, network=nn.Linear(6, 1))
    assert estimator.evaluate(torch.ones(4), torch.ones(2)).values.q[0].shape == ()


@pytest.mark.parametrize("initial", [float("nan"), float("inf"), float("-inf")])
def test_gaussian_rejects_nonfinite_log_std_initialization(initial):
    with pytest.raises(ValueError, match="log_std_init must be finite"):
        GaussianPolicyEstimator(2, 2, log_std_init=initial)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("value", [-1000.0, 1000.0, float("nan"), float("inf"), float("-inf")])
def test_gaussian_rejects_unsafe_scales_after_parameter_or_checkpoint_updates(dtype, value):
    estimator = GaussianPolicyEstimator(2, 2, network=nn.Identity()).to(dtype=dtype)
    with torch.no_grad():
        estimator.log_std[1] = value
    checkpoint_before = estimator.log_std.detach().clone()
    with pytest.raises(ValueError, match="log_std produces an unsafe variance"):
        estimator(torch.zeros(3, 2, dtype=dtype))
    torch.testing.assert_close(estimator.log_std, checkpoint_before, equal_nan=True)


@pytest.mark.parametrize("value", [-100.0, -45.0, 44.3, 100.0])
def test_float32_gaussian_fails_before_underflow_or_overflow_corrupts_log_prob(value):
    estimator = GaussianPolicyEstimator(2, 2, value, network=nn.Identity())
    with pytest.raises(ValueError, match="unsafe variance for torch.float32"):
        estimator(torch.zeros(2))


@pytest.mark.parametrize("dtype,value", [
    (torch.float32, -20.0), (torch.float32, 20.0),
    (torch.float64, -100.0), (torch.float64, 100.0),
])
@pytest.mark.parametrize("batch_shape", [(), (1,), (2, 1, 3)])
def test_gaussian_safe_scales_keep_shape_log_prob_and_gradients(dtype, value, batch_shape):
    estimator = GaussianPolicyEstimator(2, 2, value, network=nn.Linear(2, 2)).to(dtype=dtype)
    with torch.no_grad():
        estimator.network.weight.zero_()
        estimator.network.bias.zero_()
    features = torch.ones(batch_shape + (2,), dtype=dtype, requires_grad=True)
    policy = estimator(features).policy
    assert policy.batch_shape == batch_shape
    assert policy.event_shape == (2,)
    torch.testing.assert_close(policy.base_dist.scale, estimator.log_std.exp().expand_as(policy.mean))
    actions = policy.mean.detach() + 0.3 * policy.base_dist.scale.detach()
    log_prob = policy.log_prob(actions)
    expected = (
        -0.5 * ((actions - policy.mean) / policy.base_dist.scale).square()
        - estimator.log_std - 0.5 * math.log(2 * math.pi)
    ).sum(-1)
    torch.testing.assert_close(log_prob, expected)
    assert log_prob.shape == batch_shape
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(policy.entropy()).all()
    log_prob.sum().backward()
    for parameter in estimator.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    torch.testing.assert_close(estimator.log_std.detach(), torch.full((2,), value, dtype=dtype))


def test_continuous_q_does_not_fit_the_feature_only_actor_critic_forward_contract():
    combined = ActorCriticEstimator(GaussianPolicyEstimator(4, 2), ContinuousQEstimator(4, 2))
    model = Model(IdentityRepresentation(), combined)
    with pytest.raises(NotImplementedError, match="requires evaluate"):
        model.forward(torch.ones(4))
    with pytest.raises(TypeError, match="no action-conditioned evaluate"):
        model.evaluate_q(torch.ones(4), torch.ones(2))


@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
def test_separate_actor_and_continuous_critic_models_share_features_and_gradients(batch_shape):
    representation = MLPRepresentation(3, hidden_dims=(4,), activation=nn.Tanh)
    actor = Model(representation, GaussianPolicyEstimator(4, 2, network=nn.Linear(4, 2)))
    critic_network = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 1))
    critic = Model(representation, ContinuousQEstimator(4, 2, network=critic_network))
    observations = torch.ones(batch_shape + (3,), requires_grad=True)
    policy = actor.forward(observations).policy
    actions = policy.rsample()
    estimate = critic.evaluate_q(observations, actions)
    assert estimate.values.q[0].shape == batch_shape
    estimate.values.q[0].sum().backward()
    assert observations.grad is not None and torch.isfinite(observations.grad).all()
    for parameter in list(actor.parameters()) + list(critic.parameters()):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert actor.representation is critic.representation


@pytest.mark.parametrize("batch_shape", [(), (3,)])
def test_model_act_keeps_custom_actor_critic_output_and_deterministic_behavior(batch_shape):
    estimator = ActorCriticEstimator(
        CategoricalPolicyEstimator(4, 2, network=nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))),
        StateValueEstimator(4, network=nn.Linear(4, 1)),
    )
    model = Model(MLPRepresentation(3, hidden_dims=(4,)), estimator, CategoricalActionSelector())
    observations = np.ones(batch_shape + (3,), dtype=np.float32)
    estimate = model.forward(observations)
    selected = model.act(observations)
    assert selected.action.shape == batch_shape
    torch.testing.assert_close(selected.log_prob, estimate.policy.log_prob(selected.action))
    assert not selected.log_prob.requires_grad
    selected = model.act(observations, deterministic=True)
    torch.testing.assert_close(selected.action, estimate.policy.probs.argmax(-1))
    assert selected.log_prob is None


@pytest.mark.parametrize("batch_shape", [(), (3,)])
def test_gaussian_actor_critic_with_state_value_uses_the_existing_forward_contract(batch_shape):
    estimator = ActorCriticEstimator(
        GaussianPolicyEstimator(4, 2, network=nn.Linear(4, 2)),
        StateValueEstimator(4, network=nn.Linear(4, 1)),
    )
    estimate = Model(IdentityRepresentation(), estimator).forward(torch.ones(batch_shape + (4,)))
    assert estimate.policy.batch_shape == batch_shape
    assert estimate.policy.event_shape == (2,)
    assert estimate.values.v.shape == batch_shape
