"""Deterministic Q-learning composition, device preparation and snapshots."""

import io
from dataclasses import replace

import numpy as np
import pytest
import torch

from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.buffer import ReplayBuffer
from modular_rl.component.estimator import ContinuousQEstimator, DiscreteQEstimator, TabularQEstimator, TabularVEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import Transition, TransitionBatch
from modular_rl.learner import TabularQLearner
from modular_rl.model import Model


def make_learner(dtype=torch.float32, selector=None, gamma=0.9):
    model = Model(IdentityRepresentation(), TabularQEstimator(3, 2), selector).to(dtype=dtype)
    model.estimator.q_table.copy_(torch.tensor([[2.0, 4.0], [10.0, 20.0], [30.0, 40.0]], dtype=dtype))
    return TabularQLearner(model, TD0TargetGenerator(gamma), TabularQUpdateRule(0.5))


def make_batch(states=(0,), actions=(0,), rewards=(1.0,), next_states=(1,),
               terminated=(False,), truncated=(False,), index_dtype=torch.int32):
    return TransitionBatch(torch.tensor(states, dtype=index_dtype), torch.tensor(actions, dtype=index_dtype),
                           torch.tensor(rewards, dtype=torch.float32), torch.tensor(next_states, dtype=index_dtype),
                           torch.tensor(terminated), torch.tensor(truncated))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_single_transition_from_replay_changes_q_table_with_the_bellman_target(dtype):
    learner = make_learner(dtype)
    replay = ReplayBuffer(1, seed=7)
    replay.add(Transition(np.int32(0), np.int16(0), 1.0, np.int32(1), False, False))
    batch = replay.sample(1)
    table = learner.model.estimator.q_table
    pointer = table.data_ptr()
    error = learner.learn(batch)
    torch.testing.assert_close(error, torch.tensor([17.0], dtype=dtype))
    torch.testing.assert_close(table, torch.tensor([[10.5, 4.0], [10.0, 20.0], [30.0, 40.0]], dtype=dtype))
    assert error.shape == (1,) and error.dtype == dtype and table.data_ptr() == pointer
    assert not error.requires_grad and error.grad_fn is None
    assert batch.observation.dtype == torch.int32 and batch.action.dtype == torch.int16
    assert batch.reward.dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("index_dtype", [torch.int16, torch.int32, torch.long])
def test_batch_errors_keep_row_order_and_indices_are_normalized(dtype, index_dtype):
    learner = make_learner(dtype)
    batch = make_batch(states=(2, 0), actions=(1, 0), rewards=(1.0, 2.0), next_states=(1, 2),
                       terminated=(False, True), truncated=(False, False), index_dtype=index_dtype)
    error = learner.learn(batch)
    torch.testing.assert_close(error, torch.tensor([-21.0, 0.0], dtype=dtype))
    assert learner.model.estimator.q_table[2, 1].item() == 29.5
    assert learner.model.estimator.q_table[0, 0].item() == 2.0


@pytest.mark.parametrize("terminated,truncated,expected_error,expected_value", [
    (False, False, 17.0, 10.5), (False, True, 17.0, 10.5),
    (True, False, -1.0, 1.5), (True, True, -1.0, 1.5),
])
def test_termination_mask_is_independent_of_truncation(terminated, truncated, expected_error, expected_value):
    learner = make_learner()
    batch = make_batch(terminated=(terminated,), truncated=(truncated,))
    error = learner.learn(batch)
    torch.testing.assert_close(error, torch.tensor([expected_error]))
    assert learner.model.estimator.q_table[0, 0].item() == expected_value
    assert batch.terminated.item() == terminated and batch.truncated.item() == truncated


def test_time_limit_uses_the_supplied_final_observation_without_inferring_a_reset():
    learner = make_learner()
    batch = make_batch(next_states=(2,), truncated=(True,))
    error = learner.learn(batch)
    torch.testing.assert_close(error, torch.tensor([35.0]))  # 1 + .9 * 40 - 2
    assert learner.model.estimator.q_table[0, 0].item() == 19.5


def test_duplicate_updates_use_targets_frozen_before_any_table_write():
    learner = make_learner()
    learner.model.estimator.q_table.zero_()
    batch = make_batch(states=(0, 0), actions=(0, 0), rewards=(1.0, 3.0), next_states=(0, 0),
                       terminated=(False, False), truncated=(False, False))
    error = learner.learn(batch)
    torch.testing.assert_close(error, torch.tensor([1.0, 2.5]))
    assert learner.model.estimator.q_table[0, 0].item() == 1.75


def test_bootstrap_values_for_later_rows_are_not_recomputed_after_earlier_updates():
    model = Model(IdentityRepresentation(), TabularQEstimator(2, 2))
    model.estimator.q_table.copy_(torch.tensor([[0.0, 0.0], [10.0, 20.0]]))
    learner = TabularQLearner(model, TD0TargetGenerator(0.5), TabularQUpdateRule(0.5))
    batch = make_batch(states=(1, 0), actions=(1, 0), rewards=(4.0, 1.0), next_states=(0, 1),
                       terminated=(False, False), truncated=(False, False))
    error = learner.learn(batch)
    torch.testing.assert_close(error, torch.tensor([-16.0, 11.0]))
    torch.testing.assert_close(model.estimator.q_table, torch.tensor([[5.5, 0.0], [10.0, 12.0]]))


def test_model_target_generator_and_update_rule_are_called_once_in_order(monkeypatch):
    learner = make_learner(torch.float64)
    initial = learner.model.estimator.q_table.clone()
    calls, returned = [], []
    forward, generate, update = learner.model.forward, learner.target_generator.generate, learner.update_rule.update

    def record_forward(observation):
        assert not torch.is_grad_enabled() and observation.dtype == torch.long
        calls.append("forward")
        return forward(observation)

    def record_generate(reward, next_value, terminated):
        assert not torch.is_grad_enabled() and reward.dtype == next_value.dtype == torch.float64
        torch.testing.assert_close(learner.model.estimator.q_table, initial)
        calls.append("target")
        return generate(reward, next_value, terminated)

    def record_update(table, states, actions, target):
        assert table is learner.model.estimator.q_table
        assert states.dtype == actions.dtype == torch.long
        calls.append("update")
        error = update(table, states, actions, target)
        returned.append(error)
        return error

    monkeypatch.setattr(learner.model, "forward", record_forward)
    monkeypatch.setattr(learner.target_generator, "generate", record_generate)
    monkeypatch.setattr(learner.update_rule, "update", record_update)
    error = learner.learn(make_batch())
    assert calls == ["forward", "target", "update"] and error is returned[0]
    assert torch.is_grad_enabled()


def test_action_selection_and_sampling_are_independent_of_learning(monkeypatch):
    learner = make_learner(selector=EpsilonGreedyActionSelector(0.3))
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Learning must not choose actions or sample a buffer")

    monkeypatch.setattr(learner.model.action_selector, "select", forbidden)
    monkeypatch.setattr(ReplayBuffer, "sample", forbidden)
    learner.learn(make_batch())
    assert torch.equal(torch.get_rng_state(), rng)


def test_preparation_removes_target_gradients_without_mutating_input_batch():
    learner = make_learner(torch.float64)
    reward = torch.tensor([1.0], requires_grad=True)
    batch = replace(make_batch(), reward=reward)
    error = learner.learn(batch)
    assert not error.requires_grad and error.grad_fn is None
    assert batch.reward is reward and reward.dtype == torch.float32 and reward.requires_grad
    assert reward.grad is None


def test_model_to_after_learner_construction_uses_the_current_table():
    learner = make_learner()
    old_table = learner.model.estimator.q_table
    learner.model.double()
    error = learner.learn(make_batch())
    assert error.dtype == torch.float64
    assert learner.model.estimator.q_table[0, 0].item() == 10.5
    assert old_table[0, 0].item() == 2.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_learned_model_state_dict_saves_and_restores(dtype):
    learner = make_learner(dtype)
    learner.learn(make_batch())
    checkpoint = io.BytesIO()
    torch.save(learner.model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = make_learner(dtype)
    restored.model.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    torch.testing.assert_close(restored.model.forward(torch.arange(3)).values.q[0],
                               learner.model.forward(torch.arange(3)).values.q[0])
    torch.testing.assert_close(restored.learn(make_batch()), learner.learn(make_batch()))


@pytest.mark.parametrize("model", [
    object(), Model(IdentityRepresentation(), TabularVEstimator(3)),
    Model(IdentityRepresentation(), DiscreteQEstimator(1, 2)),
    Model(IdentityRepresentation(), ContinuousQEstimator(1, 1)),
])
def test_incompatible_models_fail_at_construction(model):
    with pytest.raises(TypeError, match="Model with TabularQEstimator"):
        TabularQLearner(model, TD0TargetGenerator(0.9), TabularQUpdateRule(0.5))


@pytest.mark.parametrize("field,value,error,message", [
    ("observation", torch.tensor(0), ValueError, "batch shape"),
    ("observation", torch.tensor([[0]]), ValueError, "batch shape"),
    ("observation", torch.empty(0, dtype=torch.long), ValueError, "nonempty"),
    ("observation", torch.tensor([0.5]), TypeError, "integer index"),
    ("next_observation", torch.tensor([True]), TypeError, "integer index"),
    ("action", torch.tensor([0.0]), TypeError, "integer index"),
    ("action", torch.zeros(1, 1, dtype=torch.long), ValueError, "same.*shape"),
    ("next_observation", torch.tensor([0, 1]), ValueError, "same.*shape"),
    ("reward", torch.tensor([1]), TypeError, "floating dtype"),
    ("terminated", torch.tensor([1]), TypeError, "bool dtype"),
    ("truncated", torch.tensor([0, 0], dtype=torch.bool), ValueError, "same.*shape"),
    ("reward", np.array([1.0]), TypeError, "reward must be a Tensor"),
])
def test_invalid_batch_contracts_fail_before_writing_the_table(field, value, error, message):
    learner = make_learner()
    before = learner.model.estimator.q_table.clone()
    with pytest.raises(error, match=message):
        learner.learn(replace(make_batch(), **{field: value}))
    torch.testing.assert_close(learner.model.estimator.q_table, before)


def test_learn_requires_a_transition_batch():
    with pytest.raises(TypeError, match="TransitionBatch"):
        make_learner().learn({"observation": 0})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_replay_batch_is_prepared_for_a_cuda_model(dtype):
    learner = make_learner(dtype)
    learner.model.to("cuda")
    batch = make_batch()
    error = learner.learn(batch)
    assert error.device.type == "cuda" and error.dtype == dtype and not error.requires_grad
    torch.testing.assert_close(error.cpu(), torch.tensor([17.0], dtype=dtype))
    assert batch.observation.device.type == batch.reward.device.type == "cpu"
