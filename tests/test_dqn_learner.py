"""Deterministic Bellman, optimization and independent target-model contracts."""

import io
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.buffer import ReplayBuffer
from modular_rl.component.estimator import DiscreteQEstimator, TabularQEstimator
from modular_rl.component.representation import IdentityRepresentation, MLPRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TDLossUpdateRule
from modular_rl.data import EstimatorOutput, TDLossOutput, Transition, TransitionBatch, ValueOutput
from modular_rl.learner import DQNLearner
from modular_rl.model import Model


def make_models(dtype=torch.float32, selector=None, network=None):
    online = Model(IdentityRepresentation(), DiscreteQEstimator(1, 2, network=network), selector).to(dtype=dtype)
    target = deepcopy(online)
    if network is None:
        with torch.no_grad():
            online.estimator.network.weight.copy_(torch.tensor([[2.0], [4.0]], dtype=dtype))
            target.estimator.network.weight.copy_(torch.tensor([[10.0], [20.0]], dtype=dtype))
            online.estimator.network.bias.zero_()
            target.estimator.network.bias.zero_()
    return online, target


def make_learner(dtype=torch.float32, gamma=0.5, selector=None, network=None):
    online, target = make_models(dtype, selector, network)
    return DQNLearner(online, target, torch.optim.SGD(online.parameters(), lr=0.01),
                      TD0TargetGenerator(gamma), TDLossUpdateRule())


def make_batch(states=(1.0,), actions=(0,), rewards=(1.0,), next_states=(3.0,),
               terminated=(False,), truncated=(False,), dtype=torch.float32):
    return TransitionBatch(torch.tensor(states, dtype=dtype).unsqueeze(-1),
                           torch.tensor(actions, dtype=torch.int32), torch.tensor(rewards, dtype=dtype),
                           torch.tensor(next_states, dtype=dtype).unsqueeze(-1),
                           torch.tensor(terminated), torch.tensor(truncated))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_single_replay_transition_uses_bellman_target_and_updates_only_online(dtype):
    learner = make_learner(dtype)
    replay = ReplayBuffer(2, seed=7)
    replay.add(Transition(np.array([1.0], dtype=np.float32), np.int16(0), 1.0,
                          np.array([3.0], dtype=np.float32), False, False))
    batch = replay.sample(1)
    target_before = {key: value.clone() for key, value in learner.target_model.state_dict().items()}
    result = learner.learn(batch)
    # Q_online(s,a)=2; max Q_target(s',.)=60; target=1+.5*60=31.
    torch.testing.assert_close(result.td_error, torch.tensor([29.0], dtype=dtype))
    torch.testing.assert_close(result.loss, torch.tensor(841.0, dtype=dtype))
    assert isinstance(result, TDLossOutput) and result.loss.shape == () and result.td_error.shape == (1,)
    assert result.td_error.dtype == result.loss.dtype == dtype
    assert not result.loss.requires_grad and not result.td_error.requires_grad
    assert result.loss.grad_fn is result.td_error.grad_fn is None
    torch.testing.assert_close(learner.online_model.estimator.network.weight,
                               torch.tensor([[2.58], [4.0]], dtype=dtype))
    assert learner.online_model.estimator.network.weight.grad is not None
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in learner.target_model.parameters())
    for key, value in learner.target_model.state_dict().items():
        torch.testing.assert_close(value, target_before[key])
    assert batch.action.dtype == torch.int16 and batch.reward.dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_gather_and_signed_td_error_keep_row_order(dtype, monkeypatch):
    learner = make_learner(dtype)
    batch = make_batch(states=(1.0, 2.0), actions=(1, 0), rewards=(1.0, 2.0), next_states=(3.0, 4.0),
                       terminated=(False, True), truncated=(True, False))
    compute = learner.update_rule.compute

    def record(prediction, target):
        torch.testing.assert_close(prediction, torch.tensor([4.0, 4.0], dtype=dtype))
        torch.testing.assert_close(target, torch.tensor([31.0, 2.0], dtype=dtype))
        return compute(prediction, target)

    monkeypatch.setattr(learner.update_rule, "compute", record)
    result = learner.learn(batch)
    torch.testing.assert_close(result.td_error, torch.tensor([27.0, -2.0], dtype=dtype))
    torch.testing.assert_close(result.loss, torch.tensor(366.5, dtype=dtype))


@pytest.mark.parametrize("terminated,truncated,expected", [
    (False, False, 29.0), (False, True, 29.0), (True, False, -1.0), (True, True, -1.0),
])
def test_only_termination_stops_bootstrap(terminated, truncated, expected):
    result = make_learner().learn(make_batch(terminated=(terminated,), truncated=(truncated,)))
    torch.testing.assert_close(result.td_error, torch.tensor([expected]))


def test_component_calls_and_optimizer_step_follow_the_dqn_order(monkeypatch):
    learner = make_learner()
    calls = []
    online_forward, target_forward = learner.online_model.forward, learner.target_model.forward
    generate, compute = learner.target_generator.generate, learner.update_rule.compute
    zero_grad, step = learner.optimizer.zero_grad, learner.optimizer.step

    def online(observation):
        assert torch.is_grad_enabled()
        calls.append("online")
        return online_forward(observation)

    def target(observation):
        assert not torch.is_grad_enabled() and not learner.target_model.training
        calls.append("target")
        return target_forward(observation)

    def target_value(*args):
        assert not torch.is_grad_enabled()
        calls.append("generate")
        return generate(*args)

    def loss(*args):
        result = compute(*args)
        assert result.loss.requires_grad and result.td_error.requires_grad
        result.loss.register_hook(lambda gradient: calls.append("backward"))
        calls.append("compute")
        return result

    def zero():
        calls.append("zero_grad")
        return zero_grad()

    def optimize():
        calls.append("step")
        return step()

    monkeypatch.setattr(learner.online_model, "forward", online)
    monkeypatch.setattr(learner.target_model, "forward", target)
    monkeypatch.setattr(learner.target_generator, "generate", target_value)
    monkeypatch.setattr(learner.update_rule, "compute", loss)
    monkeypatch.setattr(learner.optimizer, "zero_grad", zero)
    monkeypatch.setattr(learner.optimizer, "step", optimize)
    learner.learn(make_batch())
    assert calls == ["online", "target", "generate", "compute", "zero_grad", "backward", "step"]


def test_learning_does_not_select_actions_or_sample_replay(monkeypatch):
    learner = make_learner(selector=EpsilonGreedyActionSelector(0.5))

    def forbidden(*args, **kwargs):
        raise AssertionError("learn must not sample or select actions")

    monkeypatch.setattr(ReplayBuffer, "sample", forbidden)
    monkeypatch.setattr(learner.online_model.action_selector, "select", forbidden)
    monkeypatch.setattr(learner.target_model.action_selector, "select", forbidden)
    learner.learn(make_batch())


def test_target_graph_is_disabled_without_changing_the_input_batch():
    learner = make_learner(torch.float64)
    batch = make_batch()
    reward = batch.reward.requires_grad_()
    next_observation = batch.next_observation.requires_grad_()
    result = learner.learn(batch)
    assert reward.grad is None and next_observation.grad is None
    assert reward.requires_grad and next_observation.requires_grad
    assert batch.reward.dtype == torch.float32 and result.loss.dtype == torch.float64


def test_constructor_clears_preexisting_target_gradients_before_freezing():
    online, target = make_models()
    target(torch.ones(1, 1)).values.q[0].sum().backward()
    assert all(parameter.grad is not None for parameter in target.parameters())
    learner = DQNLearner(online, target, torch.optim.SGD(online.parameters(), lr=.01),
                         TD0TargetGenerator(.5), TDLossUpdateRule())
    learner.learn(make_batch())
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in target.parameters())


def test_successive_steps_clear_old_online_gradients():
    learner = make_learner()
    learner.learn(make_batch())
    learner.online_model.estimator.network.weight.grad.fill_(10000)
    prediction = learner.online_model(torch.tensor([[1.0]])).values.q[0][0, 0].item()
    learner.learn(make_batch(terminated=(True,)))
    torch.testing.assert_close(learner.online_model.estimator.network.weight.grad,
                               torch.tensor([[2 * (prediction - 1.0)], [0.0]]))


def test_constructor_does_not_implicitly_synchronize_and_sync_copies_without_aliasing():
    learner = make_learner(selector=EpsilonGreedyActionSelector(0.3))
    pointers = [parameter.data_ptr() for parameter in learner.target_model.parameters()]
    assert learner.target_model.estimator.network.weight[0, 0].item() == 10.0
    learner.online_model.action_selector.epsilon = 0.1
    learner.target_model.train()
    learner.sync_target()
    assert not learner.target_model.training
    assert [parameter.data_ptr() for parameter in learner.target_model.parameters()] == pointers
    for key, value in learner.online_model.state_dict().items():
        torch.testing.assert_close(value, learner.target_model.state_dict()[key])
    with torch.no_grad():
        learner.online_model.estimator.network.weight.add_(1)
    assert learner.target_model.estimator.network.weight[0, 0].item() == 2.0


def test_sync_copies_persistent_and_nonpersistent_buffers_and_learn_preserves_target_buffers():
    network = nn.Sequential(nn.Linear(1, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Linear(4, 2))
    network.register_buffer("temporary", torch.tensor([3.0]), persistent=False)
    learner = make_learner(network=network)
    learner.online_model.estimator.network.temporary.fill_(8)
    learner.online_model.estimator.network[1].running_mean.fill_(2)
    learner.sync_target()
    torch.testing.assert_close(learner.target_model.estimator.network.temporary, torch.tensor([8.0]))
    torch.testing.assert_close(learner.target_model.estimator.network[1].running_mean, torch.full((4,), 2.0))
    before = {key: value.clone() for key, value in learner.target_model.named_buffers()}
    learner.target_model.train()  # learn reasserts target eval, even after external mode changes.
    learner.learn(make_batch(states=(1.0, 2.0), actions=(0, 1), rewards=(1.0, 2.0), next_states=(3.0, 4.0),
                              terminated=(False, False), truncated=(False, False)))
    assert learner.online_model.training and not learner.target_model.training
    for key, value in learner.target_model.named_buffers():
        torch.testing.assert_close(value, before[key])


@pytest.mark.parametrize("sharing", ["same_model", "parameter", "storage", "buffer"])
def test_models_must_not_share_parameters_or_buffer_storage(sharing):
    online, target = make_models()
    if sharing == "same_model":
        target = online
    elif sharing == "parameter":
        target.estimator.network.weight = online.estimator.network.weight
    elif sharing == "storage":
        target.estimator.network.weight = nn.Parameter(online.estimator.network.weight.detach())
    else:
        online.register_buffer("shared", torch.ones(1))
        target.register_buffer("shared", online.shared)
    with pytest.raises(ValueError, match="independent"):
        DQNLearner(online, target, torch.optim.SGD(online.parameters(), lr=.01),
                    TD0TargetGenerator(.9), TDLossUpdateRule())
    assert all(parameter.requires_grad for parameter in online.parameters())


def test_optimizer_must_exclude_target_and_unrelated_parameters():
    online, target = make_models()
    for extra in (list(target.parameters()), [nn.Parameter(torch.ones(1))]):
        with pytest.raises(ValueError, match="only Online"):
            DQNLearner(online, target, torch.optim.SGD(list(online.parameters()) + extra, lr=.01),
                        TD0TargetGenerator(.9), TDLossUpdateRule())


@pytest.mark.parametrize("target", [
    Model(IdentityRepresentation(), DiscreteQEstimator(1, 3)),
    Model(MLPRepresentation(1, (2,)), DiscreteQEstimator(2, 2)),
])
def test_incompatible_target_structure_fails_at_construction(target):
    online, _ = make_models()
    with pytest.raises(ValueError, match="matching (shape|structure)"):
        DQNLearner(online, target, torch.optim.SGD(online.parameters(), lr=.01),
                    TD0TargetGenerator(.9), TDLossUpdateRule())


def test_sync_checks_structure_before_copying_any_weights():
    learner = make_learner()
    initial = learner.target_model.estimator.network.weight.detach().clone()
    learner.target_model.estimator.network.bias = nn.Parameter(torch.ones(3))
    with pytest.raises(ValueError, match="matching shape"):
        learner.sync_target()
    torch.testing.assert_close(learner.target_model.estimator.network.weight, initial)


def test_model_and_optimizer_connection_errors_are_clear():
    online, target = make_models()
    with pytest.raises(TypeError, match="Online and Target Models"):
        DQNLearner(object(), target, torch.optim.SGD(online.parameters(), lr=.01),
                    TD0TargetGenerator(.9), TDLossUpdateRule())
    with pytest.raises(TypeError, match="Optimizer"):
        DQNLearner(online, target, object(), TD0TargetGenerator(.9), TDLossUpdateRule())
    tabular = Model(IdentityRepresentation(), TabularQEstimator(3, 2))
    with pytest.raises(ValueError, match="trainable floating"):
        DQNLearner(tabular, deepcopy(tabular), torch.optim.SGD(online.parameters(), lr=.01),
                    TD0TargetGenerator(.9), TDLossUpdateRule())
    with pytest.raises(TypeError, match="floating dtype"):
        DQNLearner(online, target.double(), torch.optim.SGD(online.parameters(), lr=.01),
                    TD0TargetGenerator(.9), TDLossUpdateRule())


@pytest.mark.parametrize("which", ["online_model", "target_model"])
@pytest.mark.parametrize("output,error,message", [
    (EstimatorOutput(), ValueError, "requires Q-values"),
    (EstimatorOutput(values=ValueOutput(v=torch.ones(1))), ValueError, "requires Q-values"),
    (EstimatorOutput(values=ValueOutput(q=(torch.ones(1, 2), torch.ones(1, 2)))), ValueError, "exactly one"),
    (EstimatorOutput(values=ValueOutput(q=(torch.distributions.Normal(0., 1.),))), TypeError, "Q Tensor"),
    (EstimatorOutput(values=ValueOutput(q=(torch.ones(1, 2, dtype=torch.long),))), TypeError, "floating"),
    (EstimatorOutput(values=ValueOutput(q=(torch.ones(2),))), ValueError, "shape"),
    (EstimatorOutput(values=ValueOutput(q=(torch.ones(1, 2, 3),))), ValueError, "shape"),
    (EstimatorOutput(values=ValueOutput(q=(torch.ones(2, 2),))), ValueError, "shape"),
    (EstimatorOutput(values=ValueOutput(q=(torch.ones(1, 0),))), ValueError, "shape"),
    (EstimatorOutput(values=ValueOutput(q=(torch.ones(1, 2, dtype=torch.float64),))), ValueError, "dtype and device"),
])
def test_invalid_scalar_discrete_q_fails_before_optimizer_step(which, output, error, message, monkeypatch):
    learner = make_learner()
    initial = learner.online_model.estimator.network.weight.detach().clone()
    monkeypatch.setattr(getattr(learner, which), "forward", lambda observation: output)
    with pytest.raises(error, match=message):
        learner.learn(make_batch())
    torch.testing.assert_close(learner.online_model.estimator.network.weight, initial)
    assert all(parameter.grad is None for parameter in learner.online_model.parameters())


def test_target_action_axis_must_match_online(monkeypatch):
    learner = make_learner()
    monkeypatch.setattr(learner.target_model, "forward",
                        lambda observation: EstimatorOutput(values=ValueOutput(q=(torch.ones(1, 3),))))
    with pytest.raises(ValueError, match="same.*shape"):
        learner.learn(make_batch())


@pytest.mark.parametrize("field,value,error,message", [
    ("observation", torch.tensor(1.0), ValueError, "batch dimension"),
    ("observation", torch.empty(0, 1), ValueError, "nonempty"),
    ("observation", np.ones((1, 1)), TypeError, "observation must be a Tensor"),
    ("next_observation", torch.ones(1, 2), ValueError, "same shape"),
    ("action", torch.tensor([0.0]), TypeError, "integer index"),
    ("action", torch.tensor([True]), TypeError, "integer index"),
    ("action", torch.tensor([[0]]), ValueError, "shape"),
    ("action", torch.tensor([-1]), ValueError, "action indices"),
    ("action", torch.tensor([2]), ValueError, "action indices"),
    ("reward", torch.tensor([1]), TypeError, "floating dtype"),
    ("reward", torch.tensor(1.0), ValueError, "shape"),
    ("terminated", torch.tensor([0]), TypeError, "bool dtype"),
    ("truncated", torch.tensor([False, False]), ValueError, "shape"),
])
def test_invalid_batches_fail_before_optimization(field, value, error, message):
    learner = make_learner()
    with pytest.raises(error, match=message):
        learner.learn(replace(make_batch(), **{field: value}))
    assert all(parameter.grad is None for parameter in learner.online_model.parameters())


def test_learn_requires_a_transition_batch():
    with pytest.raises(TypeError, match="TransitionBatch"):
        make_learner().learn({"observation": 1})


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_both_models_and_optimizer_state_can_be_saved_and_restored(dtype):
    learner = make_learner(dtype)
    learner.optimizer = torch.optim.Adam(learner.online_model.parameters(), lr=.001)
    learner.learn(make_batch())
    learner.sync_target()
    checkpoint = io.BytesIO()
    torch.save({"online": learner.online_model.state_dict(), "target": learner.target_model.state_dict(),
                "optimizer": learner.optimizer.state_dict()}, checkpoint)
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    restored = make_learner(dtype)
    restored.optimizer = torch.optim.Adam(restored.online_model.parameters(), lr=.001)
    restored.online_model.load_state_dict(saved["online"], strict=True)
    restored.target_model.load_state_dict(saved["target"], strict=True)
    restored.optimizer.load_state_dict(saved["optimizer"])
    result, repeated = learner.learn(make_batch()), restored.learn(make_batch())
    torch.testing.assert_close(result.td_error, repeated.td_error)
    torch.testing.assert_close(result.loss, repeated.loss)
    for first, second in zip(learner.online_model.parameters(), restored.online_model.parameters()):
        torch.testing.assert_close(first, second)


def test_model_dtype_moves_before_optimizer_creation_are_supported():
    online, target = make_models()
    online.double()
    target.double()
    learner = DQNLearner(online, target, torch.optim.Adam(online.parameters()),
                         TD0TargetGenerator(.5), TDLossUpdateRule())
    assert learner.learn(make_batch()).loss.dtype == torch.float64


def test_replay_to_mlp_model_target_generator_and_loss_updates_representation_and_head():
    torch.manual_seed(11)
    representation = MLPRepresentation(2, hidden_dims=(8,), activation=nn.Tanh)
    online = Model(representation, DiscreteQEstimator(8, 2))
    target = deepcopy(online)
    learner = DQNLearner(online, target, torch.optim.Adam(online.parameters(), lr=.001),
                         TD0TargetGenerator(.9), TDLossUpdateRule())
    learner.sync_target()
    replay = ReplayBuffer(2, seed=7)
    replay.add(Transition(np.array([.1, .2], dtype=np.float64), 0, 1.0,
                          np.array([.3, .4], dtype=np.float64), False, True))
    replay.add(Transition(np.array([.5, .6], dtype=np.float64), 1, 2.0,
                          np.array([.7, .8], dtype=np.float64), True, False))
    initial = {name: parameter.detach().clone() for name, parameter in online.named_parameters()}
    result = learner.learn(replay.sample(2))
    assert result.td_error.shape == (2,) and result.loss.shape == ()
    for name, parameter in online.named_parameters():
        assert parameter.grad is not None and not torch.equal(parameter, initial[name])
    for name, parameter in target.named_parameters():
        torch.testing.assert_close(parameter, initial[name])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_batch_is_prepared_for_cuda_and_sync_remains_independent(dtype):
    online, target = make_models(dtype)
    online.cuda()
    target.cuda()
    learner = DQNLearner(online, target, torch.optim.SGD(online.parameters(), lr=.01),
                         TD0TargetGenerator(.5), TDLossUpdateRule())
    result = learner.learn(make_batch())
    assert result.loss.device.type == result.td_error.device.type == "cuda"
    torch.testing.assert_close(result.td_error.cpu(), torch.tensor([29.0], dtype=dtype))
    assert all(parameter.grad is None for parameter in target.parameters())
    learner.sync_target()
    for first, second in zip(online.parameters(), target.parameters()):
        torch.testing.assert_close(first, second)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_mismatched_model_device_fails_at_construction():
    online, target = make_models()
    target.cuda()
    with pytest.raises(ValueError, match="same device"):
        DQNLearner(online, target, torch.optim.SGD(online.parameters(), lr=.01),
                    TD0TargetGenerator(.5), TDLossUpdateRule())
