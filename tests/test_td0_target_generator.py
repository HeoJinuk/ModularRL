"""TD(0) arithmetic, explicit tensor contracts and caller-owned gradients."""

import numpy as np
import pytest
import torch

from modular_rl.component.buffer import ReplayBuffer
from modular_rl.component.estimator import StateValueEstimator, TabularQEstimator, TabularVEstimator
from modular_rl.component.representation import IdentityRepresentation, MLPRepresentation
from modular_rl.component.target_generator import BaseTargetGenerator, TD0TargetGenerator
from modular_rl.data import Transition
from modular_rl.model import Model


@pytest.fixture(autouse=True)
def fixed_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        yield


def test_minimal_base_contract():
    assert isinstance(TD0TargetGenerator(0.9), BaseTargetGenerator)
    with pytest.raises(TypeError, match="abstract"):
        BaseTargetGenerator()


@pytest.mark.parametrize("gamma,expected", [
    (0, [1, 2, 3, 4, 5, 6]),
    (0.9, [10, 2, 30, 4, 50, 6]),
    (1, [11, 2, 33, 4, 55, 6]),
])
@pytest.mark.parametrize("shape", [(), (6,), (2, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_td0_values_preserve_scalar_batch_and_leading_dimensions(gamma, expected, shape, dtype):
    if shape == ():
        reward = torch.tensor(1, dtype=dtype)
        next_value = torch.tensor(10, dtype=dtype)
        terminated = torch.tensor(False)
        expected_target = torch.tensor(expected[0], dtype=dtype)
    else:
        reward = torch.arange(1, 7, dtype=dtype).reshape(shape)
        next_value = torch.arange(10, 61, 10, dtype=dtype).reshape(shape)
        terminated = torch.tensor([False, True, False, True, False, True]).reshape(shape)
        expected_target = torch.tensor(expected, dtype=dtype).reshape(shape)
    target = TD0TargetGenerator(gamma).generate(reward, next_value, terminated)
    torch.testing.assert_close(target, expected_target)
    assert target.shape == shape and target.dtype == dtype and target.device == reward.device


def test_float64_gamma_is_not_rounded_through_a_default_float32_mask():
    gamma = 0.7312345678
    target = TD0TargetGenerator(gamma).generate(
        torch.tensor([1.1], dtype=torch.float64),
        torch.tensor([4.25], dtype=torch.float64),
        torch.tensor([False]),
    )
    torch.testing.assert_close(target, torch.tensor([4.20774691315], dtype=torch.float64), rtol=0, atol=1e-14)


def test_terminal_bootstrap_is_removed_even_when_unused_values_are_nonfinite():
    reward = torch.tensor([1.0, 2.0, 3.0])
    next_value = torch.tensor([float("nan"), float("inf"), -float("inf")])
    target = TD0TargetGenerator(0.9).generate(reward, next_value, torch.ones(3, dtype=torch.bool))
    torch.testing.assert_close(target, reward)


@pytest.mark.parametrize("gamma", [-0.1, 1.1, float("nan"), float("inf"), True, "0.9", torch.tensor(0.9)])
def test_invalid_gamma_fails_during_configuration(gamma):
    with pytest.raises(ValueError, match="gamma.*real number"):
        TD0TargetGenerator(gamma)


@pytest.mark.parametrize("field", ["reward", "next_value", "terminated"])
def test_non_tensor_inputs_are_not_converted(field):
    values = {"reward": torch.ones(2), "next_value": torch.ones(2), "terminated": torch.zeros(2, dtype=torch.bool)}
    values[field] = values[field].numpy()
    with pytest.raises(TypeError, match="must be Tensors"):
        TD0TargetGenerator(0.9).generate(**values)


@pytest.mark.parametrize("field,shape", [("reward", (2, 1)), ("next_value", (1,)), ("terminated", ())])
def test_broadcastable_but_different_shapes_fail(field, shape):
    values = {"reward": torch.ones(2), "next_value": torch.ones(2), "terminated": torch.zeros(2, dtype=torch.bool)}
    values[field] = torch.zeros(shape, dtype=values[field].dtype)
    with pytest.raises(ValueError, match="same shape"):
        TD0TargetGenerator(0.9).generate(**values)


@pytest.mark.parametrize("field,dtype,message", [
    ("reward", torch.long, "floating dtype"),
    ("next_value", torch.long, "floating dtype"),
    ("next_value", torch.complex64, "floating dtype"),
    ("next_value", torch.float64, "same dtype"),
    ("terminated", torch.long, "bool dtype"),
])
def test_numeric_contract_errors_are_clear(field, dtype, message):
    values = {"reward": torch.ones(2), "next_value": torch.ones(2), "terminated": torch.zeros(2, dtype=torch.bool)}
    values[field] = values[field].to(dtype=dtype)
    with pytest.raises(TypeError, match=message):
        TD0TargetGenerator(0.9).generate(**values)


def test_device_mismatch_is_rejected_without_moving_inputs():
    with pytest.raises(ValueError, match="same device"):
        TD0TargetGenerator(0.9).generate(torch.ones(2), torch.ones(2, device="meta"), torch.zeros(2, dtype=torch.bool))


def test_target_preserves_gradients_and_inputs_with_a_zero_terminal_derivative():
    reward = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    next_value = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64, requires_grad=True)
    terminated = torch.tensor([False, True, False])
    target = TD0TargetGenerator(0.9).generate(reward, next_value, terminated)
    assert target.requires_grad and target.grad_fn is not None
    target.sum().backward()
    torch.testing.assert_close(reward.grad, torch.ones(3, dtype=torch.float64))
    torch.testing.assert_close(next_value.grad, torch.tensor([0.9, 0.0, 0.9], dtype=torch.float64))
    torch.testing.assert_close(reward.detach(), torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    torch.testing.assert_close(next_value.detach(), torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64))
    assert torch.equal(terminated, torch.tensor([False, True, False]))


@pytest.mark.parametrize("policy", ["no_grad", "detach"])
def test_caller_can_disable_target_gradients(policy):
    generator = TD0TargetGenerator(0.9)
    reward = torch.tensor([1.0])
    next_value = torch.tensor([10.0], requires_grad=True)
    terminated = torch.tensor([False])
    if policy == "no_grad":
        with torch.no_grad():
            target = generator.generate(reward, next_value, terminated)
    else:
        target = generator.generate(reward, next_value.detach(), terminated)
    assert not target.requires_grad and target.grad_fn is None
    assert next_value.grad is None
    torch.testing.assert_close(target, torch.tensor([10.0]))


@pytest.mark.parametrize("bootstrap", ["V", "SARSA", "Q-learning"])
def test_tabular_bootstrap_values_connect_replay_to_td0(bootstrap):
    estimator = TabularVEstimator(4) if bootstrap == "V" else TabularQEstimator(4, 2)
    model = Model(IdentityRepresentation(), estimator)
    if bootstrap == "V":
        estimator.v_table.copy_(torch.tensor([0.0, 10.0, 20.0, 30.0]))
    else:
        estimator.q_table.copy_(torch.tensor([[0.0, 0.0], [10.0, 100.0], [20.0, 200.0], [30.0, 300.0]]))
    replay = ReplayBuffer(3, seed=8)
    for state in range(3):
        replay.add(Transition(np.int32(state), 0, float(state + 1), np.int32(state + 1), state == 1, state == 2))
    batch = replay.sample(3)
    estimate = model.forward(batch.next_observation)
    if bootstrap == "V":
        next_value = estimate.values.v
    elif bootstrap == "SARSA":
        next_actions = torch.zeros_like(batch.action)  # Supplied by the caller's next policy.
        next_value = estimate.values.q[0].gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_value = estimate.values.q[0].max(-1).values
    target = TD0TargetGenerator(0.5).generate(batch.reward, next_value, batch.terminated)
    expected = torch.tensor([51.0, 2.0, 153.0] if bootstrap == "Q-learning" else [6.0, 2.0, 18.0])
    torch.testing.assert_close(target, expected[batch.observation.long()])
    time_limit_row = (batch.observation == 2).nonzero().item()
    assert batch.truncated[time_limit_row] and not batch.terminated[time_limit_row]
    assert batch.next_observation[time_limit_row] == 3
    assert target[time_limit_row] > batch.reward[time_limit_row]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("freeze_bootstrap", [False, True])
def test_neural_bootstrap_gradient_policy_is_owned_by_the_caller(dtype, freeze_bootstrap):
    model = Model(MLPRepresentation(3, hidden_dims=(5,), activation=torch.nn.Tanh), StateValueEstimator(5)).to(dtype=dtype)
    replay = ReplayBuffer(3, seed=4)
    for state in range(3):
        observation = torch.full((3,), float(state), dtype=dtype)
        replay.add(Transition(observation, 0, float(state + 1), observation + 1, state == 1, state == 2))
    batch = replay.sample(3).to(dtype=dtype)
    if freeze_bootstrap:
        with torch.no_grad():
            next_value = model.forward(batch.next_observation).values.v
    else:
        next_value = model.forward(batch.next_observation).values.v
    target = TD0TargetGenerator(0.9).generate(batch.reward, next_value, batch.terminated)
    expected = []
    for index in range(3):
        reward = batch.reward[index].item()
        expected.append(reward if batch.terminated[index] else reward + 0.9 * next_value[index].item())
    torch.testing.assert_close(target, torch.tensor(expected, dtype=dtype))
    assert target.requires_grad == (not freeze_bootstrap)
    if freeze_bootstrap:
        assert target.grad_fn is None
        assert all(parameter.grad is None for parameter in model.parameters())
    else:
        target.sum().backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_td0_preserves_cuda_device_and_gradients():
    reward = torch.tensor([1.0, 2.0], device="cuda", requires_grad=True)
    next_value = torch.tensor([10.0, 20.0], device="cuda", requires_grad=True)
    terminated = torch.tensor([False, True], device="cuda")
    target = TD0TargetGenerator(0.9).generate(reward, next_value, terminated)
    assert target.device == reward.device
    torch.testing.assert_close(target, torch.tensor([10.0, 2.0], device="cuda"))
    target.sum().backward()
    torch.testing.assert_close(next_value.grad, torch.tensor([0.9, 0.0], device="cuda"))
