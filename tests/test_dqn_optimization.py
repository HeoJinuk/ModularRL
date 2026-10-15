"""Shared DQN optimization and exact equivalence to the pre-refactor updates."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from examples.gridworld_dyna_dqn import learn_real_n_step, make_n_step_experience
from modular_rl.component.estimator import DiscreteQEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TDLossUpdateRule
from modular_rl.data import TDLossOutput, Transition, TransitionBatch
from modular_rl.learner import DQNLearner
from modular_rl.model import Model
from tests.test_dyna_dqn_composition import assert_tree_equal, one_cpu_thread


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]


def make_learner(dtype=torch.float32, device="cpu", optimizer="adam", stochastic=False):
    network = nn.Sequential(nn.Linear(2, 8), nn.BatchNorm1d(8) if stochastic else nn.Identity(),
                            nn.ReLU(), nn.Dropout(.25) if stochastic else nn.Identity(), nn.Linear(8, 3))
    online = Model(IdentityRepresentation(), DiscreteQEstimator(2, 3, network=network)).to(device=device, dtype=dtype)
    target = deepcopy(online)
    factory = torch.optim.Adam if optimizer == "adam" else torch.optim.SGD
    options = {} if optimizer == "adam" else {"momentum": .9}
    return DQNLearner(online, target, factory(online.parameters(), lr=.01, **options),
                      TD0TargetGenerator(.9), TDLossUpdateRule())


def make_batch(size, dtype=torch.float32):
    observation = torch.arange(size * 2, dtype=dtype).reshape(size, 2) / 10
    return TransitionBatch(observation, torch.arange(size) % 3, torch.linspace(-1, 2, size, dtype=dtype),
                           observation + .5, torch.arange(size) % 4 == 0, torch.arange(size) % 4 == 1)


def rng_state():
    return torch.get_rng_state().clone(), [state.clone() for state in torch.cuda.get_rng_state_all()]


def restore_rng(state):
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state_all(state[1])


def legacy_step(learner, prediction, target):
    """The five statements shared by both call sites at commit 9ccb7f7."""
    result = learner.update_rule.compute(prediction, target)
    learner.optimizer.zero_grad()
    result.loss.backward()
    learner.optimizer.step()
    return TDLossOutput(result.td_error.detach(), result.loss.detach())


def legacy_learn(learner, batch):
    """Frozen valid-input DQN evaluation order, independent of learn/optimize."""
    reference = next(learner.online_model.parameters())
    batch = batch.to(reference.device, dtype=reference.dtype)
    q, = learner.online_model(batch.observation).values.q
    prediction = q.gather(-1, batch.action.long().unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        learner.target_model.eval()
        next_q, = learner.target_model(batch.next_observation).values.q
        target = learner.target_generator.generate(batch.reward, next_q.max(-1).values, batch.terminated)
    return legacy_step(learner, prediction, target)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_size", [1, 4, 32])
@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
def test_optimize_math_in_place_storage_and_no_extra_operations(device, dtype, batch_size, optimizer, monkeypatch):
    torch.manual_seed(7)
    learner = make_learner(dtype, device, optimizer)
    batch = make_batch(batch_size, dtype).to(device)
    q, = learner.online_model(batch.observation).values.q
    prediction = q.gather(-1, batch.action[:, None]).squeeze(-1)
    prediction.retain_grad()
    target = torch.linspace(-2, 3, batch_size, dtype=dtype, device=device)
    expected_error = target - prediction.detach()
    expected_loss = expected_error.square().mean()
    target_before = deepcopy(learner.target_model.state_dict())
    before = [parameter.detach().clone() for parameter in learner.online_model.parameters()]
    pointers = [parameter.data_ptr() for parameter in learner.online_model.parameters()]
    same_optimizer = learner.optimizer
    calls = []
    compute, zero_grad, step = learner.update_rule.compute, learner.optimizer.zero_grad, learner.optimizer.step

    def loss(prediction, target):
        result = compute(prediction, target)
        result.loss.register_hook(lambda gradient: calls.append("backward"))
        calls.append("compute")
        return result

    def zero():
        calls.append("zero_grad")
        return zero_grad()

    def update():
        calls.append("step")
        return step()

    def unexpected(*args, **kwargs):
        pytest.fail("optimize must not evaluate a model, generate a target or synchronize")

    monkeypatch.setattr(learner.update_rule, "compute", loss)
    monkeypatch.setattr(learner.optimizer, "zero_grad", zero)
    monkeypatch.setattr(learner.optimizer, "step", update)
    for model in (learner.online_model, learner.target_model):
        monkeypatch.setattr(model, "forward", unexpected)
        monkeypatch.setattr(model, "act", unexpected)
    monkeypatch.setattr(learner.target_generator, "generate", unexpected)
    monkeypatch.setattr(learner, "sync_target", unexpected)
    random_before = rng_state()
    result = learner.optimize(prediction, target)
    assert calls == ["compute", "zero_grad", "backward", "step"]
    assert_tree_equal(random_before, rng_state())
    assert_tree_equal(expected_error, result.td_error)
    assert_tree_equal(expected_loss, result.loss)
    torch.testing.assert_close(prediction.grad, -2 * expected_error / batch_size)
    assert result.td_error.shape == (batch_size,) and result.loss.shape == ()
    assert result.loss.dtype == result.td_error.dtype == dtype
    assert result.loss.device == result.td_error.device == prediction.device
    assert result.loss.grad_fn is result.td_error.grad_fn is None
    assert not result.loss.requires_grad and not result.td_error.requires_grad
    assert learner.optimizer is same_optimizer and learner.optimizer.state
    assert pointers == [parameter.data_ptr() for parameter in learner.online_model.parameters()]
    assert any(not torch.equal(old, parameter) for old, parameter in zip(before, learner.online_model.parameters()))
    assert_tree_equal(target_before, learner.target_model.state_dict())
    assert all(parameter.grad is None for parameter in learner.target_model.parameters())


@pytest.mark.parametrize("problem,error,message", [
    ("prediction_type", TypeError, "Tensors"), ("target_type", TypeError, "Tensors"),
    ("scalar", ValueError, "shape"), ("matrix", ValueError, "shape"),
    ("shape", ValueError, "same shape"), ("empty", ValueError, "nonempty"),
    ("target_dtype", TypeError, "same dtype"), ("prediction_dtype", ValueError, "Online Model"),
    ("integer_target", TypeError, "floating"), ("target_device", ValueError, "same device"),
    ("prediction_device", ValueError, "Online Model"), ("target_graph", ValueError, "fixed"),
    ("prediction_graph", ValueError, "gradient path"), ("loss_shape", ValueError, "input shape"),
])
def test_invalid_optimize_inputs_preserve_parameters_gradients_and_existing_adam_state(problem, error, message, monkeypatch):
    learner = make_learner()
    learner.learn(make_batch(4))  # Validate failures against populated, not just empty, optimizer state.
    prediction = learner.online_model(make_batch(4).observation).values.q[0][:, 0]
    target = torch.ones(4)
    if problem == "prediction_type":
        prediction = [1., 2., 3., 4.]
    elif problem == "target_type":
        target = [1., 2., 3., 4.]
    elif problem == "scalar":
        prediction, target = prediction[0], target[0]
    elif problem == "matrix":
        prediction, target = prediction[:, None], target[:, None]
    elif problem == "shape":
        target = target[:, None]
    elif problem == "empty":
        prediction, target = prediction[:0], target[:0]
    elif problem == "target_dtype":
        target = target.double()
    elif problem == "prediction_dtype":
        prediction, target = prediction.double(), target.double()
    elif problem == "integer_target":
        target = target.long()
    elif problem == "target_device":
        target = target.to("meta")  # Real device mismatch, also testable on CPU-only hosts.
    elif problem == "prediction_device":
        prediction, target = prediction.to("meta"), target.to("meta")
    elif problem == "target_graph":
        target.requires_grad_(True)
    elif problem == "prediction_graph":
        prediction = prediction.detach()
    elif problem == "loss_shape":
        learner.update_rule = TDLossUpdateRule(nn.MSELoss())
    before = deepcopy((learner.online_model.state_dict(), learner.target_model.state_dict(),
                       learner.optimizer.state_dict(), [p.grad for p in learner.online_model.parameters()]))

    def unexpected(*args, **kwargs):
        pytest.fail("invalid inputs must fail before zero_grad/step")

    monkeypatch.setattr(learner.optimizer, "zero_grad", unexpected)
    monkeypatch.setattr(learner.optimizer, "step", unexpected)
    random_before = rng_state()
    with pytest.raises(error, match=message):
        learner.optimize(prediction, target)
    assert_tree_equal(before, (learner.online_model.state_dict(), learner.target_model.state_dict(),
                               learner.optimizer.state_dict(), [p.grad for p in learner.online_model.parameters()]))
    assert_tree_equal(random_before, rng_state())
    if problem == "target_graph":
        assert target.requires_grad and target.grad is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
@pytest.mark.parametrize("route", ["learn", "optimize", "stochastic_learn"])
def test_independent_initial_states_match_legacy_results_optimizer_buffers_and_rng(device, dtype, optimizer, route):
    torch.manual_seed(19)
    expected = make_learner(dtype, device, optimizer, stochastic=route == "stochastic_learn")
    actual = deepcopy(expected)
    for size in ([4, 32, 4] if route == "stochastic_learn" else [1, 4, 32]):
        batch = make_batch(size, dtype)
        before_rng = rng_state()
        reference = legacy_learn(expected, batch)
        after_rng = rng_state()
        restore_rng(before_rng)
        if route == "optimize":
            prepared = batch.to(device, dtype=dtype)
            q, = actual.online_model(prepared.observation).values.q
            prediction = q.gather(-1, prepared.action[:, None]).squeeze(-1)
            with torch.no_grad():
                next_q, = actual.target_model(prepared.next_observation).values.q
                target = actual.target_generator.generate(prepared.reward, next_q.max(-1).values, prepared.terminated)
            result = actual.optimize(prediction, target)
        else:
            result = actual.learn(batch)
        assert_tree_equal(reference, result)
        assert_tree_equal(expected.online_model.state_dict(), actual.online_model.state_dict())
        assert_tree_equal(expected.target_model.state_dict(), actual.target_model.state_dict())
        assert_tree_equal(expected.optimizer.state_dict(), actual.optimizer.state_dict())
        assert_tree_equal([p.grad for p in expected.online_model.parameters()],
                          [p.grad for p in actual.online_model.parameters()])
        assert_tree_equal(after_rng, rng_state())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n", [1, 3, 5])
def test_n_step_uses_shared_optimize_with_unchanged_prediction_and_current_target(n, dtype, monkeypatch):
    # ID inputs here use the existing example factory; no changes to metadata or sampling.
    from examples.gridworld_dyna_dqn import create_learner

    torch.manual_seed(7)
    actual = create_learner(n_states=3, n_actions=2, hidden_dims=(8,), dtype=dtype)
    expected = deepcopy(actual)
    window = [Transition(0, 1, 1., 1, False, False).to_batch()] + [
        Transition(1, 0, 2., 1, False, False).to_batch() for _ in range(n - 1)]
    record = make_n_step_experience(window, actual.target_generator, dtype=dtype)
    calls = []
    optimize = actual.optimize

    def recorded(prediction, target):
        calls.append((prediction.detach().clone(), target.clone()))
        return optimize(prediction, target)

    monkeypatch.setattr(actual, "optimize", recorded)
    # Keep an independent legacy optimizer tail while the helper prepares its unchanged n-step target.
    monkeypatch.setattr(expected, "optimize", lambda prediction, target: legacy_step(expected, prediction, target))
    for _ in range(2):
        before_rng = rng_state()
        reference = learn_real_n_step(expected, [record] * 4)
        after_rng = rng_state()
        restore_rng(before_rng)
        result = learn_real_n_step(actual, [record] * 4)
        assert_tree_equal(reference, result)
        assert_tree_equal(expected.online_model.state_dict(), actual.online_model.state_dict())
        assert_tree_equal(expected.target_model.state_dict(), actual.target_model.state_dict())
        assert_tree_equal(expected.optimizer.state_dict(), actual.optimizer.state_dict())
        assert_tree_equal(after_rng, rng_state())
    assert len(calls) == 2


def test_learn_delegates_its_computed_tensors_to_optimize(monkeypatch):
    learner = make_learner()
    batch = make_batch(4)
    expected = legacy_learn(deepcopy(learner), batch)
    calls = []
    optimize = learner.optimize

    def recorded(prediction, target):
        calls.append((prediction.shape, target.shape, prediction.requires_grad, target.requires_grad))
        return optimize(prediction, target)

    monkeypatch.setattr(learner, "optimize", recorded)
    assert_tree_equal(expected, learner.learn(batch))
    assert calls == [(torch.Size([4]), torch.Size([4]), True, False)]
