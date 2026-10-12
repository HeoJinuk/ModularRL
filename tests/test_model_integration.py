"""Model composes inference components without taking over learning behavior."""

import io
import math

import numpy as np
import pytest
import torch
from torch import nn

from modular_rl.component.action_selector import (
    CategoricalActionSelector,
    EpsilonGreedyActionSelector,
    GreedyActionSelector,
)
from modular_rl.component.estimator import (
    ActorCriticEstimator,
    CategoricalPolicyEstimator,
    ContinuousQEstimator,
    DiscreteQEstimator,
    GaussianPolicyEstimator,
    StateValueEstimator,
    TabularQEstimator,
)
from modular_rl.component.representation import IdentityRepresentation, MLPRepresentation
from modular_rl.data import ActionOutput, EstimatorOutput
from modular_rl.model import InputAdapter, Model


KINDS = ("tabular", "discrete-Q", "actor-critic", "continuous-Q")


@pytest.fixture(autouse=True)
def fixed_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        yield


def make_model(kind, input_adapter=None):
    if kind == "tabular":
        estimator = TabularQEstimator(6, 4)
        estimator.q_table.copy_(torch.arange(6.0)[:, None] + torch.tensor([0., 2., 2., -1.]))
        return Model(IdentityRepresentation(), estimator, EpsilonGreedyActionSelector(0.2), input_adapter)

    representation = MLPRepresentation(3, hidden_dims=(5,), activation=nn.Tanh)
    if kind == "discrete-Q":
        return Model(representation, DiscreteQEstimator(5, 4), GreedyActionSelector(), input_adapter)
    if kind == "actor-critic":
        estimator = ActorCriticEstimator(CategoricalPolicyEstimator(5, 4), StateValueEstimator(5))
        return Model(representation, estimator, CategoricalActionSelector(), input_adapter)
    return Model(representation, ContinuousQEstimator(5, 2, hidden_dim=8), input_adapter=input_adapter)


def inputs(kind, batch_shape, backend="numpy"):
    count = math.prod(batch_shape)
    if kind == "tabular":
        observations = (np.arange(count, dtype=np.int32) % 6).reshape(batch_shape)
    else:
        observations = np.linspace(-0.5, 0.5, count * 3).reshape(batch_shape + (3,))
    actions = np.linspace(-0.3, 0.3, count * 2).reshape(batch_shape + (2,))
    if backend == "tensor":
        observations, actions = torch.from_numpy(observations), torch.from_numpy(actions)
    return observations, actions


def estimate(model, kind, observations, actions):
    if kind == "continuous-Q":
        return model.evaluate_q(observations, actions)
    return model.forward(observations)


def numeric_outputs(result):
    outputs = list(result.values.q)
    if result.values.v is not None:
        outputs.append(result.values.v)
    if result.policy is not None:
        outputs.append(result.policy.probs)
    return outputs


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("batch_shape", [(), (1,), (2, 3)])
@pytest.mark.parametrize("backend", ["numpy", "tensor"])
def test_requested_compositions_preserve_shapes_and_selection(kind, batch_shape, backend):
    model = make_model(kind)
    observations, actions = inputs(kind, batch_shape, backend)
    result = estimate(model, kind, observations, actions)
    assert isinstance(result, EstimatorOutput)
    assert isinstance(result.values.q, tuple)
    if kind == "actor-critic":
        assert result.values.v.shape == batch_shape
        assert result.policy.probs.shape == batch_shape + (4,)
        assert result.values.q == ()
    else:
        assert result.policy is None
        assert result.values.v is None
        assert len(result.values.q) == 1
        assert result.values.q[0].shape == batch_shape + (() if kind == "continuous-Q" else (4,))
    if kind == "tabular":
        states = torch.as_tensor(observations, dtype=torch.long)
        torch.testing.assert_close(result.values.q[0], states[..., None].float() + torch.tensor([0., 2., 2., -1.]))
    for tensor in numeric_outputs(result):
        assert tensor.dtype == torch.float32
    if kind == "continuous-Q":
        return

    selected = model.act(observations)
    assert isinstance(selected, ActionOutput)
    assert selected.action.shape == selected.log_prob.shape == batch_shape
    assert selected.action.dtype == torch.long
    assert not selected.log_prob.requires_grad
    probabilities = model.action_selector.probabilities(result)
    expected_log_prob = probabilities.gather(-1, selected.action.unsqueeze(-1)).squeeze(-1).log()
    torch.testing.assert_close(selected.log_prob, expected_log_prob)
    deterministic = model.act(observations, deterministic=True)
    maximum = result.policy.probs if kind == "actor-critic" else result.values.q[0]
    torch.testing.assert_close(deterministic.action, maximum.argmax(-1))
    assert deterministic.log_prob is None


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_model_to_controls_adapter_and_all_registered_components(kind, dtype):
    model = make_model(kind).to(device="cpu", dtype=dtype)
    observations, actions = inputs(kind, (2, 3))
    encoded = model.encode(observations)
    assert encoded.device.type == "cpu"
    assert encoded.dtype == (torch.long if kind == "tabular" else dtype)
    for tensor in list(model.parameters()) + list(model.buffers()):
        assert tensor.device.type == "cpu"
        if tensor.is_floating_point():
            assert tensor.dtype == dtype
    for tensor in numeric_outputs(estimate(model, kind, observations, actions)):
        assert tensor.dtype == dtype
    if model.action_selector is not None:
        selected = model.act(observations)
        assert selected.action.device.type == "cpu"
        assert selected.log_prob.dtype == dtype


@pytest.mark.parametrize("kind", ["discrete-Q", "actor-critic", "continuous-Q"])
def test_dtype_conversion_preserves_observation_and_action_gradients(kind):
    model = make_model(kind).double()
    if kind == "continuous-Q":
        model.estimator.network = nn.Sequential(nn.Linear(7, 8), nn.Tanh(), nn.Linear(8, 1)).double()
    observations = torch.randn(4, 3, dtype=torch.float32, requires_grad=True)
    actions = torch.randn(4, 2, dtype=torch.float32, requires_grad=True)
    result = estimate(model, kind, observations, actions)
    loss = sum(tensor.square().mean() for tensor in result.values.q)
    if kind == "actor-critic":
        loss = result.values.v.square().mean() - result.policy.log_prob(torch.zeros(4, dtype=torch.long)).mean()
    loss.backward()
    assert observations.grad.dtype == torch.float32
    assert torch.isfinite(observations.grad).all() and observations.grad.abs().sum() > 0
    for parameter in model.parameters():
        assert parameter.grad is not None and parameter.grad.dtype == torch.float64
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    if kind == "continuous-Q":
        assert actions.grad.dtype == torch.float32
        assert torch.isfinite(actions.grad).all() and actions.grad.abs().sum() > 0


class RecordingAdapter(InputAdapter):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.calls = []

    def __call__(self, observation, *, device, float_dtype):
        tensor = super().__call__(observation, device=device, float_dtype=float_dtype)
        self.events.append("adapter")
        self.calls.append((observation, tensor))
        return tensor


@pytest.mark.parametrize("method", ["forward", "act", "evaluate_q"])
def test_public_methods_follow_one_pipeline_without_repeated_stage_calls(method, monkeypatch):
    events = []
    adapter = RecordingAdapter(events)
    kind = "continuous-Q" if method == "evaluate_q" else "discrete-Q"
    model = make_model(kind, adapter)
    features_seen, estimates_seen = [], []

    def representation_hook(module, arguments, output):
        events.append("representation")
        features_seen.append(output)

    def estimator_hook(module, arguments, output):
        events.append("estimator")
        assert arguments[0] is features_seen[0]
        estimates_seen.append(output)

    model.representation.register_forward_hook(representation_hook)
    model.estimator.register_forward_hook(estimator_hook)
    observations, actions = inputs(kind, (2,))
    if method == "evaluate_q":
        original = model.estimator.evaluate

        def recorded_evaluate(features, action_tensor):
            events.append("evaluate")
            assert features is features_seen[0]
            assert action_tensor is adapter.calls[1][1]
            return original(features, action_tensor)

        monkeypatch.setattr(model.estimator, "evaluate", recorded_evaluate)
        model.evaluate_q(observations, actions)
        assert events == ["adapter", "representation", "adapter", "evaluate"]
        assert adapter.calls[1][0] is actions
    elif method == "act":
        original = model.action_selector.select

        def recorded_select(result, *, deterministic=False):
            events.append("selector")
            assert result is estimates_seen[0]
            assert not torch.is_grad_enabled()
            return original(result, deterministic=deterministic)

        monkeypatch.setattr(model.action_selector, "select", recorded_select)
        model.act(observations)
        assert events == ["adapter", "representation", "estimator", "selector"]
    else:
        result = model.forward(observations)
        assert result is estimates_seen[0]
        assert events == ["adapter", "representation", "estimator"]
    assert adapter.calls[0][0] is observations
    assert len(features_seen) == 1


@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
def test_act_disables_autograd_without_changing_component_modes(training):
    estimator = DiscreteQEstimator(5, 4, network=nn.Sequential(nn.Dropout(0.3), nn.Linear(5, 4)))
    model = Model(MLPRepresentation(3, hidden_dims=(5,)), estimator, GreedyActionSelector())
    model.train(training)
    modes_before = {name: module.training for name, module in model.named_modules()}
    grad_modes = []
    model.estimator.register_forward_pre_hook(
        lambda module, arguments: grad_modes.append(torch.is_grad_enabled())
    )
    model.act(torch.ones(3, requires_grad=True), deterministic=True)
    assert grad_modes == [False]
    assert torch.is_grad_enabled()
    assert {name: module.training for name, module in model.named_modules()} == modes_before


@pytest.mark.parametrize("kind", KINDS)
def test_full_model_checkpoint_restores_estimates_and_behavior_policy(kind):
    original = make_model(kind).double()
    with torch.no_grad():
        for parameter in original.parameters():
            parameter.add_(0.15)
        if kind == "tabular":
            original.estimator.q_table.add_(1.0)
            original.action_selector.epsilon = 0.65
    observations, actions = inputs(kind, (2, 3))
    expected = estimate(original, kind, observations, actions)
    saved = original.state_dict()
    assert "_input_reference" not in saved
    checkpoint = io.BytesIO()
    torch.save(saved, checkpoint)
    checkpoint.seek(0)
    restored = make_model(kind).double()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    actual = estimate(restored, kind, observations, actions)
    for first, second in zip(numeric_outputs(actual), numeric_outputs(expected)):
        torch.testing.assert_close(first, second)
    if kind == "tabular":
        assert restored.action_selector.epsilon == pytest.approx(0.65)
    if original.action_selector is not None:
        torch.manual_seed(42)
        expected_action = original.act(observations)
        torch.manual_seed(42)
        restored_action = restored.act(observations)
        torch.testing.assert_close(restored_action.action, expected_action.action)
        torch.testing.assert_close(restored_action.log_prob, expected_action.log_prob)


def test_explicit_adapter_dtype_remains_authoritative_for_integer_box_observations():
    model = make_model("discrete-Q", InputAdapter(dtype=torch.float64)).double()
    observations = np.ones((2, 3), dtype=np.int32)
    assert model.forward(observations).values.q[0].dtype == torch.float64
    override = make_model("discrete-Q", InputAdapter(dtype=torch.float32)).double()
    with pytest.raises(RuntimeError, match="same dtype"):
        override.forward(observations)


@pytest.mark.parametrize("method,component,operation,kind", [
    ("encode", "representation", "forward", "discrete-Q"),
    ("forward", "estimator", "forward", "discrete-Q"),
    ("act", "action_selector", "select", "discrete-Q"),
    ("evaluate_q", "estimator", "evaluate", "continuous-Q"),
])
def test_model_keeps_clear_component_output_contract_errors(method, component, operation, kind, monkeypatch):
    model = make_model(kind)
    monkeypatch.setattr(getattr(model, component), operation, lambda *args, **kwargs: None)
    observations, actions = inputs(kind, ())
    arguments = (observations, actions) if method == "evaluate_q" else (observations,)
    with pytest.raises(TypeError, match="must return"):
        getattr(model, method)(*arguments)


@pytest.mark.parametrize("kind", ["tabular", "discrete-Q", "actor-critic"])
def test_evaluate_q_is_optional_and_does_not_change_other_forward_contracts(kind):
    model = make_model(kind)
    observations, actions = inputs(kind, ())
    with pytest.raises(TypeError, match="no action-conditioned evaluate"):
        model.evaluate_q(observations, actions)
    assert isinstance(model.forward(observations), EstimatorOutput)


def test_separate_continuous_policy_and_critic_require_no_temporary_action_selector():
    policy = Model(MLPRepresentation(3, hidden_dims=(5,)), GaussianPolicyEstimator(5, 2))
    critic = make_model("continuous-Q")
    observations = np.ones((2, 3), dtype=np.float32)
    provided_actions = np.zeros((2, 2), dtype=np.float32)
    assert policy.forward(observations).policy.event_shape == (2,)
    assert critic.evaluate_q(observations, provided_actions).values.q[0].shape == (2,)
    for model in (policy, critic):
        assert model.action_selector is None
        with pytest.raises(RuntimeError, match="requires an ActionSelector"):
            model.act(observations)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_model_to_cuda_accepts_numpy_inputs_across_all_compositions():
    for dtype in (torch.float32, torch.float64):
        for kind in KINDS:
            model = make_model(kind).to(device="cuda", dtype=dtype)
            observations, actions = inputs(kind, (2, 3))
            for tensor in list(model.parameters()) + list(model.buffers()):
                assert tensor.device.type == "cuda"
            for tensor in numeric_outputs(estimate(model, kind, observations, actions)):
                assert tensor.device.type == "cuda" and tensor.dtype == dtype
            if model.action_selector is not None:
                selected = model.act(observations)
                assert selected.action.device.type == selected.log_prob.device.type == "cuda"
