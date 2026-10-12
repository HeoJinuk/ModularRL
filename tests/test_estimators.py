"""Estimator contracts, composition, gradients and checkpoint compatibility."""

import io
import math

import numpy as np
import pytest
import torch
from torch.distributions import Categorical, Independent

from modular_rl.component.action_selector import CategoricalActionSelector, GreedyActionSelector
from modular_rl.component.estimator import (
    ActorCriticEstimator,
    AdvantageEstimator,
    BaseEstimator,
    BasePolicyEstimator,
    BaseValueEstimator,
    CategoricalPolicyEstimator,
    ContinuousQEstimator,
    DeterministicPolicyEstimator,
    DiscreteAdvantageEstimator,
    DiscreteQEstimator,
    GaussianPolicyEstimator,
    PolicyEstimator,
    StateValueEstimator,
    TabularQEstimator,
    TabularVEstimator,
    ValueEstimator,
)
from modular_rl.component.representation import IdentityRepresentation, MLPRepresentation
from modular_rl.data import EstimatorOutput, ValueOutput
from modular_rl.model import Model


NEURAL_FACTORIES = [
    pytest.param(lambda: StateValueEstimator(4), id="V"),
    pytest.param(lambda: DiscreteQEstimator(4, 3), id="discrete-Q"),
    pytest.param(lambda: ContinuousQEstimator(4, 2, hidden_dim=8), id="continuous-Q"),
    pytest.param(lambda: DiscreteAdvantageEstimator(4, 3), id="advantage"),
    pytest.param(lambda: CategoricalPolicyEstimator(4, 3), id="categorical"),
    pytest.param(lambda: GaussianPolicyEstimator(4, 2), id="gaussian"),
    pytest.param(lambda: DeterministicPolicyEstimator(4, 2), id="deterministic"),
    pytest.param(
        lambda: ActorCriticEstimator(CategoricalPolicyEstimator(4, 3), StateValueEstimator(4)),
        id="actor-critic",
    ),
]
ALL_FACTORIES = NEURAL_FACTORIES + [
    pytest.param(lambda: TabularVEstimator(6), id="tabular-V"),
    pytest.param(lambda: TabularQEstimator(6, 3), id="tabular-Q"),
]


@pytest.fixture(autouse=True)
def fixed_torch_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        yield


def estimate_for(estimator, features, actions):
    if isinstance(estimator, ContinuousQEstimator):
        return estimator.evaluate(features, actions)
    return estimator(features)


def output_tensors(estimate):
    """Expose numeric payloads for dtype/device and save/restore comparisons."""
    tensors = []
    if isinstance(estimate.policy, Categorical):
        tensors.append(estimate.policy.probs)
    elif isinstance(estimate.policy, Independent):
        tensors.extend((estimate.policy.mean, estimate.policy.stddev))
    elif isinstance(estimate.policy, torch.Tensor):
        tensors.append(estimate.policy)
    if estimate.values.v is not None:
        tensors.append(estimate.values.v)
    tensors.extend(estimate.values.q)
    if estimate.values.advantage is not None:
        tensors.append(estimate.values.advantage)
    return tensors


def test_base_names_preserve_existing_roles_and_subclasses():
    assert BasePolicyEstimator is PolicyEstimator
    assert BaseValueEstimator is ValueEstimator
    assert isinstance(CategoricalPolicyEstimator(4, 3), BasePolicyEstimator)
    assert isinstance(StateValueEstimator(4), BaseValueEstimator)
    assert isinstance(DiscreteAdvantageEstimator(4, 3), AdvantageEstimator)
    with pytest.raises(TypeError, match="abstract"):
        BaseEstimator()
    with pytest.raises(TypeError, match="abstract"):
        AdvantageEstimator()


@pytest.mark.parametrize("kind", ["V", "Q"])
@pytest.mark.parametrize("indices", [torch.tensor(2), torch.tensor([2, 0, 5]), torch.tensor([[2, 0], [5, 1]])])
def test_tabular_lookup_preserves_all_state_dimensions(kind, indices):
    estimator = TabularVEstimator(6) if kind == "V" else TabularQEstimator(6, 3)
    table = estimator.v_table if kind == "V" else estimator.q_table
    table.copy_(torch.arange(table.numel()).reshape_as(table))
    result = estimator(indices)
    actual = result.values.v if kind == "V" else result.values.q[0]
    torch.testing.assert_close(actual, table[indices])
    assert actual.shape == indices.shape + (() if kind == "V" else (3,))
    assert not actual.requires_grad
    assert not list(estimator.parameters())
    assert result.policy is None


@pytest.mark.parametrize("factory", [TabularVEstimator, lambda n: TabularQEstimator(n, 3)])
@pytest.mark.parametrize("indices", [torch.tensor(2.0), torch.tensor([1, 2], dtype=torch.int32)])
def test_tabular_requires_long_indices(factory, indices):
    with pytest.raises(TypeError, match="torch.long"):
        factory(6)(indices)


@pytest.mark.parametrize("factory", [TabularVEstimator, lambda n: TabularQEstimator(n, 3)])
@pytest.mark.parametrize("indices", [torch.tensor(-1), torch.tensor(6), torch.tensor([0, 6])])
def test_tabular_rejects_out_of_range_indices(factory, indices):
    with pytest.raises(IndexError, match="State index out of bounds"):
        factory(6)(indices)


@pytest.mark.parametrize("factory", NEURAL_FACTORIES)
@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 5)])
def test_neural_shapes_and_semantic_fields(factory, batch_shape):
    estimator = factory()
    features = torch.randn(*batch_shape, 4)
    actions = torch.randn(*batch_shape, 2)
    result = estimate_for(estimator, features, actions)
    assert isinstance(result, EstimatorOutput)
    assert isinstance(result.values.q, tuple)
    if isinstance(estimator, (StateValueEstimator, ActorCriticEstimator)):
        assert result.values.v.shape == batch_shape
    if isinstance(estimator, DiscreteQEstimator):
        assert len(result.values.q) == 1
        assert result.values.q[0].shape == batch_shape + (3,)
    if isinstance(estimator, ContinuousQEstimator):
        assert len(result.values.q) == 1
        assert result.values.q[0].shape == batch_shape
    if isinstance(estimator, DiscreteAdvantageEstimator):
        assert result.values.advantage.shape == batch_shape + (3,)
        assert result.values.v is None and result.values.q == ()
    if isinstance(result.policy, Categorical):
        assert result.policy.batch_shape == batch_shape
        assert result.policy.event_shape == ()
        assert result.policy.probs.shape == batch_shape + (3,)
        torch.testing.assert_close(result.policy.probs.sum(-1), torch.ones(batch_shape))
    elif isinstance(result.policy, Independent):
        assert result.policy.batch_shape == batch_shape
        assert result.policy.event_shape == (2,)
        assert result.policy.mean.shape == batch_shape + (2,)
        assert result.policy.log_prob(actions).shape == batch_shape
    elif result.policy is not None:
        assert isinstance(result.policy, torch.Tensor)
        assert result.policy.shape == batch_shape + (2,)


@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 5)])
def test_gaussian_log_probability_sums_over_action_coordinates(batch_shape):
    estimator = GaussianPolicyEstimator(4, 2, log_std_init=-0.7)
    with torch.no_grad():
        estimator.network.weight.zero_()
        estimator.network.bias.copy_(torch.tensor([0.25, -0.5]))
    policy = estimator(torch.ones(*batch_shape, 4)).policy
    actions = torch.full(batch_shape + (2,), 0.8)
    expected = (
        -0.5 * ((actions - torch.tensor([0.25, -0.5])) / math.exp(-0.7)).square()
        + 0.7 - 0.5 * math.log(2 * math.pi)
    ).sum(-1)
    torch.testing.assert_close(policy.log_prob(actions), expected)
    torch.testing.assert_close(policy.stddev, torch.full_like(actions, math.exp(-0.7)))
    assert policy.entropy().shape == batch_shape


@pytest.mark.parametrize("factory", [CategoricalPolicyEstimator, GaussianPolicyEstimator, DeterministicPolicyEstimator])
def test_policy_forward_does_not_sample_or_change_rng(factory):
    estimator = factory(4, 2)
    features = torch.ones(3, 4)
    rng_before = torch.random.get_rng_state().clone()
    estimator(features)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_deterministic_policy_preserves_raw_unbounded_action_tensor():
    estimator = DeterministicPolicyEstimator(4, 2)
    with torch.no_grad():
        estimator.network.weight.zero_()
        estimator.network.bias.copy_(torch.tensor([-5.0, 7.0]))
    torch.testing.assert_close(estimator(torch.ones(4)).policy, torch.tensor([-5.0, 7.0]))


@pytest.mark.parametrize("features,actions,message", [
    (torch.tensor(1.0), torch.ones(2), "features must have shape"),
    (torch.ones(3), torch.ones(2), "features must have shape"),
    (torch.ones(4), torch.tensor(1.0), "actions must have shape"),
    (torch.ones(4), torch.ones(3), "actions must have shape"),
    (torch.ones(3, 4), torch.ones(2), "matching batch dimensions"),
    (torch.ones(2, 3, 4), torch.ones(2, 1, 2), "matching batch dimensions"),
])
def test_continuous_q_rejects_wrong_shapes_without_broadcasting(features, actions, message):
    with pytest.raises(ValueError, match=message):
        ContinuousQEstimator(4, 2).evaluate(features, actions)


@pytest.mark.parametrize("features,actions", [
    (torch.ones(4, dtype=torch.long), torch.ones(2)),
    (torch.ones(4), torch.ones(2, dtype=torch.long)),
])
def test_continuous_q_requires_float_state_and_action(features, actions):
    with pytest.raises(TypeError, match="floating-point"):
        ContinuousQEstimator(4, 2).evaluate(features, actions)


def test_continuous_q_cannot_evaluate_without_actions():
    estimator = ContinuousQEstimator(4, 2)
    with pytest.raises(NotImplementedError, match="evaluate"):
        estimator(torch.ones(4))
    with pytest.raises(NotImplementedError, match="evaluate"):
        Model(IdentityRepresentation(), estimator).forward(torch.ones(4))


@pytest.mark.parametrize("factory", [
    lambda: TabularVEstimator(0),
    lambda: ContinuousQEstimator(0, 2),
    lambda: ContinuousQEstimator(4, 0),
    lambda: ContinuousQEstimator(4, 2, hidden_dim=0),
    lambda: DiscreteAdvantageEstimator(4, 0),
    lambda: GaussianPolicyEstimator(4, 0),
    lambda: DeterministicPolicyEstimator(4, 0),
])
def test_new_estimators_reject_empty_dimensions(factory):
    with pytest.raises(ValueError, match="positive"):
        factory()


@pytest.mark.parametrize("factory", NEURAL_FACTORIES)
def test_neural_gradients_reach_features_actions_and_parameters(factory):
    estimator = factory()
    features = torch.randn(5, 4, requires_grad=True)
    actions = torch.randn(5, 2, requires_grad=True)
    result = estimate_for(estimator, features, actions)
    loss = sum(tensor.square().mean() for tensor in output_tensors(result))
    if isinstance(result.policy, Categorical):
        loss = loss - result.policy.log_prob(torch.zeros(5, dtype=torch.long)).mean()
    elif isinstance(result.policy, Independent):
        loss = loss - result.policy.log_prob(torch.ones(5, 2)).mean()
    loss.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0
    for parameter in estimator.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    if isinstance(estimator, ContinuousQEstimator):
        assert actions.grad is not None and torch.isfinite(actions.grad).all()
        assert actions.grad.abs().sum() > 0


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")),
])
def test_estimators_keep_device_and_float_dtype(device):
    for case in ALL_FACTORIES:
        estimator = case.values[0]().to(device=device, dtype=torch.float64)
        features = torch.ones(5, 4, device=device, dtype=torch.float64)
        if isinstance(estimator, (TabularVEstimator, TabularQEstimator)):
            features = torch.tensor([0, 2, 5], device=device)
        actions = torch.ones(5, 2, device=device, dtype=torch.float64)
        for tensor in output_tensors(estimate_for(estimator, features, actions)):
            assert tensor.device.type == device
            assert tensor.dtype == torch.float64


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_state_dict_serialization_restores_estimates(factory):
    original = factory()
    with torch.no_grad():
        for tensor in list(original.parameters()) + list(original.buffers()):
            tensor.add_(0.3)
    features = torch.ones(5, 4)
    if isinstance(original, (TabularVEstimator, TabularQEstimator)):
        features = torch.tensor([0, 2, 5])
    actions = torch.ones(5, 2)
    expected = output_tensors(estimate_for(original, features, actions))
    checkpoint = io.BytesIO()
    torch.save(original.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = factory()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    actual = output_tensors(estimate_for(restored, features, actions))
    assert len(actual) == len(expected)
    for first, second in zip(actual, expected):
        torch.testing.assert_close(first, second)


class FixedPolicy(PolicyEstimator):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, features):
        return self.output


class FixedValue(ValueEstimator):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, features):
        return self.output


@pytest.mark.parametrize("policy", [Categorical(logits=torch.zeros(3)), torch.tensor([-2.0, 3.0])])
def test_actor_critic_preserves_raw_policy_and_all_value_payloads(policy):
    distributional_q = {"atoms": torch.tensor([-1.0, 1.0]), "logits": torch.zeros(3, 2)}
    values = ValueOutput(v=object(), q=(distributional_q, torch.ones(3)), advantage=object())
    estimator = ActorCriticEstimator(FixedPolicy(EstimatorOutput(policy=policy)), FixedValue(EstimatorOutput(values=values)))
    result = estimator(torch.ones(4))
    assert result.policy is policy
    assert result.values is values
    assert result.values.q[0] is distributional_q


@pytest.mark.parametrize("policy_output,value_output,message", [
    (EstimatorOutput(), EstimatorOutput(values=ValueOutput(v=torch.tensor(0.0))), "PolicyEstimator did not provide policy"),
    (EstimatorOutput(policy=torch.ones(2)), EstimatorOutput(), "ValueEstimator did not provide values"),
])
def test_actor_critic_rejects_missing_role_outputs(policy_output, value_output, message):
    estimator = ActorCriticEstimator(FixedPolicy(policy_output), FixedValue(value_output))
    with pytest.raises(ValueError, match=message):
        estimator(torch.ones(4))


def test_actor_critic_reuses_one_representation_and_the_same_features():
    representation = MLPRepresentation(3, hidden_dims=(4,))
    estimator = ActorCriticEstimator(CategoricalPolicyEstimator(4, 3), StateValueEstimator(4))
    representation_calls, child_features = [], []
    representation.register_forward_hook(lambda module, inputs, output: representation_calls.append(output))
    estimator.policy.register_forward_pre_hook(lambda module, inputs: child_features.append(inputs[0]))
    estimator.value.register_forward_pre_hook(lambda module, inputs: child_features.append(inputs[0]))
    model = Model(representation, estimator, CategoricalActionSelector())
    result = model.forward(np.ones((5, 3), dtype=np.float32))
    assert len(representation_calls) == 1
    assert child_features[0] is child_features[1] is representation_calls[0]
    action = model.action_selector.select(result)
    torch.testing.assert_close(action.log_prob, result.policy.log_prob(action.action))
    assert action.action.shape == (5,)


@pytest.mark.parametrize("observation", [2, np.array([2, 0, 5], dtype=np.int32), torch.tensor([[2, 0], [5, 1]])])
def test_tabular_v_model_adapts_python_numpy_and_tensor_states(observation):
    model = Model(IdentityRepresentation(), TabularVEstimator(6))
    model.estimator.v_table.copy_(torch.arange(6.0))
    torch.testing.assert_close(model.forward(observation).values.v, torch.as_tensor(observation).float())


@pytest.mark.parametrize("factory", [
    lambda: GaussianPolicyEstimator(4, 2),
    lambda: DeterministicPolicyEstimator(4, 2),
    lambda: DiscreteAdvantageEstimator(4, 3),
])
@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 5)])
def test_new_heads_connect_to_model_forward(factory, batch_shape):
    model = Model(MLPRepresentation(3, hidden_dims=(4,)), factory()).double()
    estimate = model.forward(np.ones(batch_shape + (3,), dtype=np.float32))
    for tensor in output_tensors(estimate):
        assert tensor.shape[:-1] == batch_shape
        assert tensor.dtype == torch.float64


@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 5)])
def test_continuous_q_model_evaluate_q_adapts_inputs_and_keeps_gradients(batch_shape):
    model = Model(MLPRepresentation(3, hidden_dims=(4,)), ContinuousQEstimator(4, 2, hidden_dim=8)).double()
    observation = np.ones(batch_shape + (3,), dtype=np.float32)
    actions = np.ones(batch_shape + (2,), dtype=np.float32)
    q = model.evaluate_q(observation, actions).values.q[0]
    assert q.shape == batch_shape
    assert q.dtype == torch.float64
    observation_tensor = torch.ones(batch_shape + (3,), dtype=torch.float64, requires_grad=True)
    action_tensor = torch.ones(batch_shape + (2,), dtype=torch.float64, requires_grad=True)
    model.evaluate_q(observation_tensor, action_tensor).values.q[0].sum().backward()
    assert observation_tensor.grad is not None
    assert action_tensor.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("factory", [GaussianPolicyEstimator, DeterministicPolicyEstimator])
def test_continuous_policies_require_a_compatible_action_selector(factory):
    model = Model(IdentityRepresentation(), factory(4, 2), CategoricalActionSelector())
    with pytest.raises(TypeError, match="Categorical policy"):
        model.act(torch.ones(4))


def test_advantage_is_not_implicitly_used_as_discrete_q():
    estimate = DiscreteAdvantageEstimator(4, 3)(torch.ones(4))
    with pytest.raises(ValueError, match="values.q"):
        GreedyActionSelector().select(estimate)
