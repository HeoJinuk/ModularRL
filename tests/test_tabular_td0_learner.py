"""Deterministic TD prediction, frozen batch targets and online composition."""

import io
from dataclasses import replace

import numpy as np
import pytest
import torch

from modular_rl.component.action_selector import GreedyActionSelector
from modular_rl.component.buffer import ReplayBuffer
from modular_rl.component.estimator import StateValueEstimator, TabularQEstimator, TabularVEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import MonteCarloReturnGenerator, TD0TargetGenerator
from modular_rl.component.update_rule import TabularVUpdateRule
from modular_rl.data import Episode, Transition, TransitionBatch
from modular_rl.learner import MCPredictionLearner, TabularTD0PredictionLearner
from modular_rl.model import Model


def make_learner(dtype=torch.float32, gamma=0.9, alpha=0.5, selector=None):
    model = Model(IdentityRepresentation(), TabularVEstimator(3), selector).to(dtype=dtype)
    model.estimator.v_table.copy_(torch.tensor([0., 4., 10.], dtype=dtype))
    return TabularTD0PredictionLearner(model, TD0TargetGenerator(gamma), TabularVUpdateRule(alpha))


def make_batch(states=(1,), rewards=(1.,), next_states=(2,), terminated=(False,), truncated=(False,),
               index_dtype=torch.int32):
    return TransitionBatch(
        torch.tensor(states, dtype=index_dtype), torch.zeros(len(states), dtype=torch.long),
        torch.tensor(rewards), torch.tensor(next_states, dtype=index_dtype),
        torch.tensor(terminated, dtype=torch.bool), torch.tensor(truncated, dtype=torch.bool),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("index_dtype", [torch.int16, torch.int32, torch.long])
def test_replay_single_transition_matches_bellman_example_and_updates_in_place(dtype, index_dtype):
    learner = make_learner(dtype)
    replay = ReplayBuffer(1, seed=7)
    replay.add(Transition(torch.tensor(1, dtype=index_dtype), 0, 1.,
                          torch.tensor(2, dtype=index_dtype), False, False))
    batch = replay.sample(1)
    table = learner.model.estimator.v_table
    pointer = table.data_ptr()
    error = learner.learn(batch)
    torch.testing.assert_close(error, torch.tensor([6.], dtype=dtype))  # target = 1 + .9 * 10 = 10
    torch.testing.assert_close(table, torch.tensor([0., 7., 10.], dtype=dtype))
    assert table.data_ptr() == pointer and error.shape == (1,) and error.device == table.device
    assert not error.requires_grad and error.grad_fn is None
    assert batch.observation.dtype == index_dtype and batch.reward.dtype == torch.float32
    torch.testing.assert_close(learner.model.forward(np.int32(1)).values.v, torch.tensor(7., dtype=dtype))


@pytest.mark.parametrize("terminated,truncated,error,value", [
    (False, False, 6., 7.), (False, True, 6., 7.),
    (True, False, -3., 2.5), (True, True, -3., 2.5),
])
def test_only_termination_masks_bootstrap(terminated, truncated, error, value):
    learner = make_learner()
    batch = make_batch(terminated=(terminated,), truncated=(truncated,))
    torch.testing.assert_close(learner.learn(batch), torch.tensor([error]))
    assert learner.model.estimator.v_table[1].item() == value
    assert batch.terminated.item() == terminated and batch.truncated.item() == truncated


def test_truncation_uses_supplied_final_state_rather_than_inferring_a_reset():
    learner = make_learner()
    batch = make_batch(next_states=(1,), truncated=(True,))
    torch.testing.assert_close(learner.learn(batch), torch.tensor([0.6]))  # 1 + .9 * 4 - 4


@pytest.mark.parametrize("gamma,alpha", [(0., 0.01), (0.9, 0.05), (1., 1.)])
def test_gamma_and_alpha_boundaries_reuse_existing_components(gamma, alpha):
    learner = make_learner(gamma=gamma, alpha=alpha)
    expected_error = 1 + gamma * 10 - 4
    torch.testing.assert_close(learner.learn(make_batch()), torch.tensor([expected_error]))
    torch.testing.assert_close(learner.model.estimator.v_table[1], torch.tensor(4 + alpha * expected_error))


def test_batch_freezes_all_bootstrap_targets_while_online_calls_use_updated_values():
    frozen, online = make_learner(torch.float64), make_learner(torch.float64)
    batch = make_batch(states=(2, 1), rewards=(0., 1.), next_states=(1, 2),
                       terminated=(False, False), truncated=(False, False))
    errors = frozen.learn(batch)
    torch.testing.assert_close(errors, torch.tensor([-6.4, 6.], dtype=torch.float64))
    torch.testing.assert_close(frozen.model.estimator.v_table, torch.tensor([0., 7., 6.8], dtype=torch.float64))
    first = online.learn(make_batch(states=(2,), rewards=(0.,), next_states=(1,)))
    second = online.learn(make_batch())
    torch.testing.assert_close(torch.cat((first, second)), torch.tensor([-6.4, 3.12], dtype=torch.float64))
    torch.testing.assert_close(online.model.estimator.v_table, torch.tensor([0., 5.56, 6.8], dtype=torch.float64))


def test_duplicate_state_rows_update_sequentially_with_frozen_targets():
    learner = make_learner(gamma=1)
    learner.model.estimator.v_table.zero_()
    batch = make_batch(states=(1, 1), rewards=(1., 3.), next_states=(1, 1),
                       terminated=(False, False), truncated=(False, False))
    torch.testing.assert_close(learner.learn(batch), torch.tensor([1., 2.5]))
    assert learner.model.estimator.v_table[1].item() == 1.75


def test_model_target_and_rule_are_called_once_in_order_under_the_existing_gradient_contract(monkeypatch):
    learner = make_learner(torch.float64)
    initial = learner.model.estimator.v_table.clone()
    calls, returned = [], []
    forward, generate, update = learner.model.forward, learner.target_generator.generate, learner.update_rule.update

    def record_forward(states):
        calls.append("forward")
        assert not torch.is_grad_enabled() and states.dtype == torch.long
        return forward(states)

    def record_generate(reward, next_value, terminated):
        calls.append("target")
        assert not torch.is_grad_enabled() and reward.dtype == next_value.dtype == torch.float64
        torch.testing.assert_close(learner.model.estimator.v_table, initial)
        target = generate(reward, next_value, terminated)
        assert not target.requires_grad
        return target

    def record_update(table, states, target):
        calls.append("update")
        assert table is learner.model.estimator.v_table and states.dtype == torch.long
        error = update(table, states, target)
        returned.append(error)
        return error

    monkeypatch.setattr(learner.model, "forward", record_forward)
    monkeypatch.setattr(learner.target_generator, "generate", record_generate)
    monkeypatch.setattr(learner.update_rule, "update", record_update)
    rewards = torch.tensor([1.], requires_grad=True)
    result = learner.learn(replace(make_batch(), reward=rewards))
    assert calls == ["forward", "target", "update"] and result is returned[0]
    assert rewards.grad is None and torch.is_grad_enabled()


def test_learning_does_not_sample_select_actions_or_use_action_values(monkeypatch):
    learner = make_learner(selector=GreedyActionSelector())
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Prediction must not select actions or sample replay")

    monkeypatch.setattr(learner.model.action_selector, "select", forbidden)
    monkeypatch.setattr(ReplayBuffer, "sample", forbidden)
    batch = replace(make_batch(), action=torch.tensor([[0.2, 0.8]]), log_prob=torch.tensor([-1.]))
    torch.testing.assert_close(learner.learn(batch), torch.tensor([6.]))
    assert torch.equal(rng, torch.get_rng_state())


def test_state_dict_restore_and_later_model_dtype_change_continue_identically():
    original = make_learner()
    original.learn(make_batch())
    checkpoint = io.BytesIO()
    torch.save(original.model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = make_learner()
    restored.model.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    for learner in (original, restored):
        learner.model.double()
    torch.testing.assert_close(original.learn(make_batch()), restored.learn(make_batch()))
    torch.testing.assert_close(original.model.estimator.v_table, restored.model.estimator.v_table)
    assert restored.model.estimator.v_table.dtype == torch.float64


def test_mc_and_td_have_independent_models_tables_and_learning_state():
    td = make_learner()
    mc = MCPredictionLearner(Model(IdentityRepresentation(), TabularVEstimator(3)), MonteCarloReturnGenerator(1))
    td_initial = td.model.estimator.v_table.clone()
    mc.learn(Episode((Transition(1, 0, 3., 0, True, False),)))
    torch.testing.assert_close(td.model.estimator.v_table, td_initial)
    mc_initial, counts = mc.model.estimator.v_table.clone(), mc.visit_counts.clone()
    td.learn(make_batch())
    torch.testing.assert_close(mc.model.estimator.v_table, mc_initial)
    torch.testing.assert_close(mc.visit_counts, counts)
    assert td.model.estimator.v_table.data_ptr() != mc.model.estimator.v_table.data_ptr()


@pytest.mark.parametrize("estimator", [TabularQEstimator(3, 2), StateValueEstimator(1)])
def test_inappropriate_model_connection_fails_clearly(estimator):
    with pytest.raises(TypeError, match="TabularVEstimator"):
        TabularTD0PredictionLearner(Model(IdentityRepresentation(), estimator), TD0TargetGenerator(1),
                                    TabularVUpdateRule(0.5))


@pytest.mark.parametrize("observation", [torch.tensor(1), torch.tensor([[1]]), torch.empty(0, dtype=torch.long)])
def test_learner_requires_batched_tabular_states(observation):
    with pytest.raises(ValueError, match="nonempty.*shape"):
        make_learner().learn(replace(make_batch(), observation=observation))


@pytest.mark.parametrize("field", ["next_observation", "reward", "terminated", "truncated"])
def test_mismatched_prediction_shapes_fail_before_any_write(field):
    learner = make_learner()
    original = learner.model.estimator.v_table.clone()
    batch = make_batch()
    with pytest.raises(ValueError, match="same.*shape"):
        learner.learn(replace(batch, **{field: getattr(batch, field).unsqueeze(0)}))
    torch.testing.assert_close(learner.model.estimator.v_table, original)


@pytest.mark.parametrize("field,dtype", [
    ("observation", torch.float32), ("next_observation", torch.bool),
    ("reward", torch.long), ("terminated", torch.long), ("truncated", torch.float32),
])
def test_semantically_wrong_input_dtypes_are_rejected(field, dtype):
    batch = make_batch()
    with pytest.raises(TypeError):
        make_learner().learn(replace(batch, **{field: getattr(batch, field).to(dtype=dtype)}))


def test_nonbatch_input_and_nontensor_used_field_fail_clearly():
    with pytest.raises(TypeError, match="TransitionBatch"):
        make_learner().learn(None)
    with pytest.raises(TypeError, match="reward must be a Tensor"):
        make_learner().learn(replace(make_batch(), reward=np.array([1.])))


@pytest.mark.parametrize("field", ["observation", "next_observation"])
def test_invalid_state_bounds_fail_without_changing_table(field):
    learner = make_learner()
    original = learner.model.estimator.v_table.clone()
    with pytest.raises(IndexError, match="State index out of bounds"):
        learner.learn(replace(make_batch(), **{field: torch.tensor([3])}))
    torch.testing.assert_close(learner.model.estimator.v_table, original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_batch_is_prepared_for_live_cuda_table(dtype):
    learner = make_learner(dtype)
    learner.model.cuda()
    error = learner.learn(make_batch())
    torch.testing.assert_close(error.cpu(), torch.tensor([6.], dtype=dtype))
    assert error.device == learner.model.estimator.v_table.device
    assert not error.requires_grad
