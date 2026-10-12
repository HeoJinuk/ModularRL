"""Injected networks keep estimator semantics, gradients and module ownership."""

import io

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


# Class, legacy positional arguments, network input/output dimensions, payload.
CASES = [
    pytest.param(CategoricalPolicyEstimator, (4, 3), 4, 3, "probabilities", id="categorical"),
    pytest.param(GaussianPolicyEstimator, (4, 2, -0.3), 4, 2, "mean", id="gaussian"),
    pytest.param(DeterministicPolicyEstimator, (4, 2), 4, 2, "action", id="deterministic"),
    pytest.param(StateValueEstimator, (4,), 4, 1, "v", id="V"),
    pytest.param(DiscreteQEstimator, (4, 3), 4, 3, "q", id="discrete-Q"),
    pytest.param(DiscreteAdvantageEstimator, (4, 3), 4, 3, "advantage", id="advantage"),
    pytest.param(ContinuousQEstimator, (4, 2, 8), 6, 1, "q", id="continuous-Q"),
]


@pytest.fixture(autouse=True)
def fixed_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        yield


def evaluate(estimator, features, actions):
    if isinstance(estimator, ContinuousQEstimator):
        return estimator.evaluate(features, actions)
    return estimator(features)


def evaluate_model(model, observations, actions):
    if isinstance(model.estimator, ContinuousQEstimator):
        return model.evaluate_q(observations, actions)
    return model.forward(observations)


def payload(estimate, kind):
    if kind == "probabilities":
        return estimate.policy.probs
    if kind == "mean":
        return estimate.policy.mean
    if kind == "action":
        return estimate.policy
    if kind == "q":
        assert len(estimate.values.q) == 1
        return estimate.values.q[0]
    return getattr(estimate.values, kind)


class BufferedMLP(nn.Module):
    """A user's module with both trainable parameters and a persistent buffer."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, 7), nn.Tanh(), nn.Linear(7, output_dim))
        self.register_buffer("offset", torch.arange(output_dim, dtype=torch.float32) / 10)

    def forward(self, features):
        return self.layers(features) + self.offset


@pytest.mark.parametrize("estimator_type,args,input_dim,output_dim,kind", CASES)
def test_legacy_constructors_and_explicit_none_keep_default_networks(
    estimator_type, args, input_dim, output_dim, kind
):
    torch.manual_seed(11)
    legacy = estimator_type(*args)
    torch.manual_seed(11)
    explicit_none = estimator_type(*args, network=None)
    expected_keys = {"network.weight", "network.bias"}
    if estimator_type is ContinuousQEstimator:
        assert [type(layer) for layer in legacy.network] == [nn.Linear, nn.ReLU, nn.Linear]
        assert legacy.network[0].in_features == input_dim
        assert legacy.network[-1].out_features == output_dim
        expected_keys = {"network.0.weight", "network.0.bias", "network.2.weight", "network.2.bias"}
    else:
        assert isinstance(legacy.network, nn.Linear)
        assert legacy.network.in_features == input_dim
        assert legacy.network.out_features == output_dim
    if estimator_type is GaussianPolicyEstimator:
        expected_keys.add("log_std")
        torch.testing.assert_close(legacy.log_std, torch.full((2,), -0.3))
    assert set(legacy.state_dict()) == expected_keys
    for key, value in legacy.state_dict().items():
        torch.testing.assert_close(value, explicit_none.state_dict()[key])
    explicit_none.load_state_dict(legacy.state_dict(), strict=True)
    features, actions = torch.ones(1, 4), torch.ones(1, 2)
    torch.testing.assert_close(
        payload(evaluate(legacy, features, actions), kind),
        payload(evaluate(explicit_none, features, actions), kind),
    )


@pytest.mark.parametrize("estimator_type,args,input_dim,output_dim,kind", CASES)
@pytest.mark.parametrize("batch_shape", [(), (1,), (2, 1, 3)])
def test_custom_mlp_preserves_numeric_meaning_and_all_batch_dimensions(
    estimator_type, args, input_dim, output_dim, kind, batch_shape
):
    network = nn.Sequential(nn.Linear(input_dim, 7), nn.Tanh(), nn.Linear(7, output_dim))
    estimator = estimator_type(*args, network=network)
    assert estimator.network is network
    features, actions = torch.randn(*batch_shape, 4), torch.randn(*batch_shape, 2)
    network_input = torch.cat((features, actions), -1) if estimator_type is ContinuousQEstimator else features
    raw = network(network_input)
    expected = raw
    if kind == "probabilities":
        expected = raw.softmax(-1)
    elif output_dim == 1:
        expected = raw.squeeze(-1)
    actual = payload(evaluate(estimator, features, actions), kind)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == batch_shape + (() if output_dim == 1 else (output_dim,))


@pytest.mark.parametrize("estimator_type,args,input_dim,output_dim,kind", CASES)
def test_injection_does_not_execute_the_network_or_initialize_unused_defaults(
    estimator_type, args, input_dim, output_dim, kind
):
    network = BufferedMLP(input_dim, output_dim)
    calls = []
    network.register_forward_pre_hook(lambda module, inputs: calls.append(inputs[0]))
    rng_before = torch.random.get_rng_state().clone()
    estimator = estimator_type(*args, network=network)
    assert calls == []
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    evaluate(estimator, torch.ones(2, 4), torch.ones(2, 2))
    assert len(calls) == 1


@pytest.mark.parametrize("estimator_type,args,input_dim,output_dim,kind", CASES)
def test_custom_network_gradients_and_parameters_are_owned_by_model(
    estimator_type, args, input_dim, output_dim, kind
):
    network = BufferedMLP(input_dim, output_dim)
    model = Model(MLPRepresentation(3, hidden_dims=(4,), activation=nn.Tanh),
                  estimator_type(*args, network=network)).to(device="cpu", dtype=torch.float64)
    model_parameter_ids = {id(parameter) for parameter in model.parameters()}
    assert all(id(parameter) in model_parameter_ids for parameter in network.parameters())
    assert network.offset.dtype == torch.float64
    assert network.offset.device.type == "cpu"
    observations = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    actions = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    estimate = evaluate_model(model, observations, actions)
    loss = payload(estimate, kind).square().mean()
    if estimator_type is GaussianPolicyEstimator:
        loss = loss - estimate.policy.log_prob(torch.ones(2, 2, dtype=torch.float64)).mean()
    loss.backward()
    assert observations.grad is not None and observations.grad.abs().sum() > 0
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    if estimator_type is ContinuousQEstimator:
        assert actions.grad is not None and actions.grad.abs().sum() > 0


@pytest.mark.parametrize("estimator_type,args,input_dim,output_dim,kind", CASES)
def test_model_saves_and_restores_custom_parameters_and_buffers(
    estimator_type, args, input_dim, output_dim, kind
):
    original = Model(IdentityRepresentation(), estimator_type(*args, network=BufferedMLP(input_dim, output_dim))).double()
    with torch.no_grad():
        original.estimator.network.offset.add_(0.8)
        if estimator_type is GaussianPolicyEstimator:
            original.estimator.log_std.add_(0.4)
    assert "estimator.network.offset" in original.state_dict()
    observations, actions = np.ones((2, 4), dtype=np.float32), np.ones((2, 2), dtype=np.float32)
    expected = evaluate_model(original, observations, actions)
    checkpoint = io.BytesIO()
    torch.save(original.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = Model(IdentityRepresentation(), estimator_type(*args, network=BufferedMLP(input_dim, output_dim))).double()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    actual = evaluate_model(restored, observations, actions)
    torch.testing.assert_close(payload(actual, kind), payload(expected, kind))
    torch.testing.assert_close(restored.estimator.network.offset, original.estimator.network.offset)
    if estimator_type is GaussianPolicyEstimator:
        torch.testing.assert_close(actual.policy.stddev, expected.policy.stddev)


def test_actor_critic_combines_injected_networks_and_connects_to_selector():
    policy_network = BufferedMLP(4, 3)
    value_network = BufferedMLP(4, 1)
    estimator = ActorCriticEstimator(
        CategoricalPolicyEstimator(4, 3, network=policy_network),
        StateValueEstimator(4, network=value_network),
    )
    model = Model(MLPRepresentation(3, hidden_dims=(4,)), estimator, CategoricalActionSelector())
    features_seen = []
    for network in (policy_network, value_network):
        network.register_forward_pre_hook(lambda module, inputs: features_seen.append(inputs[0]))
    result = model.forward(np.ones((2, 3), dtype=np.float32))
    assert features_seen[0] is features_seen[1]
    torch.testing.assert_close(result.policy.probs, policy_network(features_seen[0]).softmax(-1))
    torch.testing.assert_close(result.values.v, value_network(features_seen[0]).squeeze(-1))
    result.values.v.sum().backward()
    assert all(parameter.grad is not None for parameter in value_network.parameters())
    action = model.action_selector.select(result)
    torch.testing.assert_close(action.log_prob, result.policy.log_prob(action.action))
    assert {id(p) for p in policy_network.parameters()}.issubset({id(p) for p in model.parameters()})


def test_continuous_q_passes_state_then_action_to_custom_network():
    network = nn.Linear(6, 1, bias=False)
    with torch.no_grad():
        network.weight.copy_(torch.tensor([[1., 2., 3., 4., 5., 6.]]))
    estimator = ContinuousQEstimator(4, 2, hidden_dim=0, network=network)
    output = estimator.evaluate(torch.tensor([1., 2., 3., 4.]), torch.tensor([5., 6.]))
    torch.testing.assert_close(output.values.q[0], torch.tensor(91.0))


def test_empty_sequential_is_kept_as_the_custom_network():
    network = nn.Sequential()
    estimator = DiscreteQEstimator(feature_dim=3, n_actions=3, network=network)
    features = torch.tensor([2., -1., 4.])
    assert estimator.network is network
    assert estimator(features).values.q[0] is features


@pytest.mark.parametrize("estimator_type,args,input_dim,output_dim,kind", CASES)
def test_wrong_network_output_dimension_has_a_clear_error(
    estimator_type, args, input_dim, output_dim, kind
):
    estimator = estimator_type(*args, network=nn.Linear(input_dim, output_dim + 1))
    with pytest.raises(ValueError, match="network output must have shape.*got"):
        evaluate(estimator, torch.ones(2, 4), torch.ones(2, 2))


def test_network_cannot_silently_flatten_batch_dimensions():
    network = nn.Sequential(nn.Flatten(0, 1), nn.Linear(4, 3))
    estimator = DiscreteQEstimator(4, 3, network=network)
    with pytest.raises(ValueError, match=r"shape \(2, 1, 3\), got \(2, 3\)"):
        estimator(torch.ones(2, 1, 4))


@pytest.mark.parametrize("estimator_type,args,input_dim,output_dim,kind", CASES)
def test_network_must_be_a_registered_module(estimator_type, args, input_dim, output_dim, kind):
    with pytest.raises(TypeError, match="network must be an nn.Module or None"):
        estimator_type(*args, network=lambda features: features)


class TupleNetwork(nn.Module):
    def forward(self, features):
        return (features, features)


@pytest.mark.parametrize("estimator_type,args,input_dim,output_dim,kind", CASES)
def test_network_must_return_a_tensor(estimator_type, args, input_dim, output_dim, kind):
    estimator = estimator_type(*args, network=TupleNetwork())
    with pytest.raises(TypeError, match="network must return"):
        evaluate(estimator, torch.ones(2, 4), torch.ones(2, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_model_to_cuda_moves_all_injected_parameters_and_buffers():
    for case in CASES:
        estimator_type, args, input_dim, output_dim, kind = case.values
        network = BufferedMLP(input_dim, output_dim)
        model = Model(IdentityRepresentation(), estimator_type(*args, network=network)).to("cuda")
        estimate = evaluate_model(model, np.ones((2, 4), dtype=np.float32), np.ones((2, 2), dtype=np.float32))
        assert payload(estimate, kind).device.type == "cuda"
        assert network.offset.device.type == "cuda"
        assert all(parameter.device.type == "cuda" for parameter in network.parameters())
