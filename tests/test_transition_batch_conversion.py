"""Owned B=1 conversion for online learning, without changing existing contracts."""

import numpy as np
import pytest
import torch

from modular_rl.component.estimator import TabularVEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularVUpdateRule
from modular_rl.data import Transition
from modular_rl.learner import TabularTD0PredictionLearner
from modular_rl.model import Model


def test_python_discrete_transition_converts_directly_for_online_td():
    batch = Transition(1, 0, 1., 2, False, True).to_batch()
    assert batch.observation.shape == batch.action.shape == batch.reward.shape == (1,)
    assert batch.observation.dtype == batch.action.dtype == torch.long
    assert batch.terminated.dtype == batch.truncated.dtype == torch.bool
    assert not batch.terminated.item() and batch.truncated.item() and batch.log_prob is None
    model = Model(IdentityRepresentation(), TabularVEstimator(3)).double()
    model.estimator.v_table.copy_(torch.tensor([0., 4., 10.]))
    learner = TabularTD0PredictionLearner(model, TD0TargetGenerator(.9), TabularVUpdateRule(.5))
    torch.testing.assert_close(learner.learn(batch), torch.tensor([6.], dtype=torch.float64))
    assert model.estimator.v_table[1].item() == 7.


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_continuous_transition_retains_shapes_dtypes_and_owned_values(dtype):
    observation = np.array([[1., 2.], [3., 4.]], dtype=dtype)
    action = np.array([.2, .8], dtype=dtype)
    next_observation = observation[:, ::-1]  # Negative strides must still be snapshotted.
    transition = Transition(observation, action, dtype(1.), next_observation, np.bool_(False), np.bool_(True))
    batch = transition.to_batch()
    expected = torch.from_numpy(observation.copy())
    assert batch.observation.shape == (1, 2, 2) and batch.action.shape == (1, 2)
    assert batch.observation.dtype == batch.action.dtype == batch.reward.dtype == expected.dtype
    observation[:] = -100
    action[:] = -100
    torch.testing.assert_close(batch.observation[0], expected)
    torch.testing.assert_close(batch.next_observation[0], expected.flip(-1))
    batch.action[0, 0] = 5
    assert action[0] == -100


def test_tensor_fields_and_behavior_log_prob_are_detached_and_do_not_alias():
    observation = torch.tensor([1., 2.], requires_grad=True)
    action = observation * 2
    reward, log_prob = observation.sum(), observation.prod()
    transition = Transition(observation, action, reward, action + 1, torch.tensor(False), torch.tensor(False), log_prob)
    batch = transition.to_batch()
    for name in ("observation", "action", "reward", "next_observation", "log_prob"):
        original, copied = getattr(transition, name), getattr(batch, name)
        assert not copied.requires_grad and copied.grad_fn is None
        assert copied.device == original.device and copied.dtype == original.dtype
        assert copied.data_ptr() != original.data_ptr()
        torch.testing.assert_close(copied[0], original.detach())
    batch.observation[0, 0] = 20
    assert observation[0].item() == 1


def test_integer_numpy_states_and_actions_are_not_cast_to_float():
    batch = Transition(np.int32(1), np.int16(2), np.float64(3), np.int32(2), False, False).to_batch()
    assert batch.observation.dtype == batch.next_observation.dtype == torch.int32
    assert batch.action.dtype == torch.int16 and batch.reward.dtype == torch.float64
    converted = batch.to(dtype=torch.float32)
    assert converted.reward.dtype == torch.float32 and converted.action.dtype == torch.int16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tensor_device_is_preserved_until_caller_moves_batch():
    value = torch.tensor([1., 2.], device="cuda", requires_grad=True)
    batch = Transition(value, value, value.sum(), value + 1, False, False).to_batch()
    assert batch.observation.device.type == "cuda" and not batch.observation.requires_grad
    assert batch.to("cpu").observation.device.type == "cpu"
