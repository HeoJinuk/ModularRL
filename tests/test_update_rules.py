"""Tabular sequential updates, neural loss contracts and learner-side flows."""

import copy
import io
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch
from torch import nn

from modular_rl.component.buffer import ReplayBuffer
from modular_rl.component.estimator import DiscreteQEstimator, TabularQEstimator, TabularVEstimator
from modular_rl.component.representation import IdentityRepresentation, MLPRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule, TabularVUpdateRule, TDLossUpdateRule
from modular_rl.data import TDLossOutput, Transition
from modular_rl.model import Model


@pytest.fixture(autouse=True)
def fixed_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)
        yield


def apply_tabular(kind, table, states, target, *, actions=None, alpha=0.5):
    if kind == "V":
        return TabularVUpdateRule(alpha).update(table, states, target)
    if actions is None:
        actions = torch.zeros_like(states)
    return TabularQUpdateRule(alpha).update(table, states, actions, target)


@pytest.mark.parametrize("kind", ["V", "Q"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batched", [False, True])
def test_tabular_updates_preserve_shape_dtype_and_original_storage(kind, dtype, batched):
    table = torch.tensor([10.0, 20.0, 30.0], dtype=dtype)
    if kind == "Q":
        table = table.unsqueeze(-1)
    original, pointer = table, table.data_ptr()
    states = torch.tensor([0, 2] if batched else 1)
    target = torch.tensor([2.0, 34.0] if batched else 4.0, dtype=dtype)
    error = apply_tabular(kind, table, states, target)
    torch.testing.assert_close(error, torch.tensor([-8.0, 4.0] if batched else -16.0, dtype=dtype))
    torch.testing.assert_close(table.reshape(-1), torch.tensor([6.0, 20.0, 32.0] if batched else [10.0, 12.0, 30.0], dtype=dtype))
    assert table is original and table.data_ptr() == pointer
    assert error.shape == target.shape and error.dtype == dtype and error.device == table.device


@pytest.mark.parametrize("rule", [TabularVUpdateRule, TabularQUpdateRule])
@pytest.mark.parametrize("alpha", [0, -0.1, 1.01, True, np.bool_(False), float("nan"), float("inf"),
                                  -float("inf"), np.float64("nan"), np.float32("inf"), "0.5", torch.tensor(0.5)])
def test_alpha_rejects_non_real_nonfinite_and_out_of_range_values(rule, alpha):
    with pytest.raises(ValueError, match="alpha.*finite real number.*0, 1"):
        rule(alpha)


@pytest.mark.parametrize("kind", ["V", "Q"])
@pytest.mark.parametrize("alpha", [0.25, 1, np.float16(0.5), np.float32(0.5), np.float64(0.5)])
def test_positive_alpha_and_numpy_real_scalars(kind, alpha):
    table = torch.zeros(1) if kind == "V" else torch.zeros(1, 1)
    error = apply_tabular(kind, table, torch.tensor(0), torch.tensor(2.0), alpha=alpha)
    assert error.item() == 2
    assert table.item() == 2 * float(alpha)


@pytest.mark.parametrize("kind", ["V", "Q"])
@pytest.mark.parametrize("targets,expected_errors,expected_value", [
    ([1.0, 3.0], [1.0, 2.5], 1.75),
    ([3.0, 1.0], [3.0, -0.5], 1.25),
])
def test_duplicate_indices_are_updated_in_input_order(kind, targets, expected_errors, expected_value):
    table = torch.zeros(1) if kind == "V" else torch.zeros(1, 1)
    error = apply_tabular(kind, table, torch.tensor([0, 0]), torch.tensor(targets))
    torch.testing.assert_close(error, torch.tensor(expected_errors))
    assert table.item() == expected_value


def test_q_duplicates_are_state_action_pairs_not_states_alone():
    table = torch.zeros(2, 2)
    error = TabularQUpdateRule(0.5).update(
        table, torch.tensor([0, 0, 1, 0, 0]), torch.tensor([0, 1, 0, 0, 1]),
        torch.tensor([1.0, 4.0, 2.0, 3.0, 0.0]),
    )
    torch.testing.assert_close(error, torch.tensor([1.0, 4.0, 2.0, 2.5, -2.0]))
    torch.testing.assert_close(table, torch.tensor([[1.75, 1.0], [1.0, 0.0]]))


@pytest.mark.parametrize("kind", ["V", "Q"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_targets_aliasing_table_storage_are_frozen_before_the_first_update(kind, dtype):
    values = torch.tensor([0.0, 10.0], dtype=dtype)
    table = values if kind == "V" else values.unsqueeze(0)
    target = table if kind == "V" else table[0]
    states = torch.tensor([1, 0] if kind == "V" else [0, 0])
    error = apply_tabular(kind, table, states, target, actions=torch.tensor([1, 0]), alpha=1)
    torch.testing.assert_close(error, torch.tensor([-10.0, 10.0], dtype=dtype))
    torch.testing.assert_close(table.reshape(-1), torch.tensor([10.0, 0.0], dtype=dtype))


@pytest.mark.parametrize("kind", ["V", "Q"])
def test_duplicate_updates_do_not_change_the_remaining_aliased_targets(kind):
    values = torch.tensor([1.0, 3.0])
    table = values if kind == "V" else values.unsqueeze(0)
    target = table if kind == "V" else table[0]
    states = torch.tensor([1, 1] if kind == "V" else [0, 0])
    error = apply_tabular(kind, table, states, target, actions=torch.tensor([1, 1]))
    torch.testing.assert_close(error, torch.tensor([-2.0, 1.0]))
    torch.testing.assert_close(table.reshape(-1), torch.tensor([1.0, 2.5]))


@pytest.mark.parametrize("kind", ["V", "Q"])
def test_tabular_no_grad_applies_to_tables_and_targets_without_changing_their_flags(kind):
    table = torch.zeros(1 if kind == "V" else (1, 1), requires_grad=True)
    target = torch.tensor([2.0, 4.0], requires_grad=True)
    error = apply_tabular(kind, table, torch.tensor([0, 0]), target)
    torch.testing.assert_close(error, torch.tensor([2.0, 3.0]))
    assert table.item() == 2.5
    assert not error.requires_grad and error.grad_fn is None
    assert table.requires_grad and target.requires_grad
    assert table.grad is None and target.grad is None
    torch.testing.assert_close(target.detach(), torch.tensor([2.0, 4.0]))
    assert torch.is_grad_enabled()


def test_noncontiguous_q_table_is_updated_through_its_original_view():
    storage = torch.zeros(2, 3)
    table = storage.T
    assert not table.is_contiguous()
    TabularQUpdateRule(0.5).update(table, torch.tensor(1), torch.tensor(0), torch.tensor(4.0))
    torch.testing.assert_close(storage, torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]))


@pytest.mark.parametrize("field,value,error,message", [
    ("v_table", np.zeros(3), TypeError, "must be Tensors"),
    ("states", np.array([0, 1]), TypeError, "must be Tensors"),
    ("target", np.ones(2), TypeError, "must be Tensors"),
    ("v_table", torch.zeros(3, 1), ValueError, "shape"),
    ("v_table", torch.zeros(3, dtype=torch.long), TypeError, "floating dtype"),
    ("states", torch.tensor([0, 1], dtype=torch.int32), TypeError, "torch.long"),
    ("states", torch.tensor([[0, 1]]), ValueError, "scalar.*batch"),
    ("target", torch.ones(1), ValueError, "same shape"),
    ("target", torch.ones(2, dtype=torch.float64), TypeError, "same floating dtype"),
    ("target", torch.ones(2, device="meta"), ValueError, "same device"),
    ("states", torch.tensor([0, -1]), IndexError, "State index"),
    ("states", torch.tensor([0, 3]), IndexError, "State index"),
])
def test_invalid_v_inputs_fail_before_any_table_write(field, value, error, message):
    values = {"v_table": torch.zeros(3), "states": torch.tensor([0, 1]), "target": torch.tensor([2.0, 4.0])}
    original = values["v_table"]
    values[field] = value
    with pytest.raises(error, match=message):
        TabularVUpdateRule(0.5).update(**values)
    torch.testing.assert_close(original, torch.zeros(3))


@pytest.mark.parametrize("field,value,error,message", [
    ("q_table", torch.zeros(3), ValueError, "shape"),
    ("actions", np.array([0, 1]), TypeError, "actions must be a Tensor"),
    ("actions", torch.tensor([0.0, 1.0]), TypeError, "torch.long"),
    ("actions", torch.tensor(0), ValueError, "same shape"),
    ("actions", torch.zeros(2, dtype=torch.long, device="meta"), ValueError, "same device"),
    ("actions", torch.tensor([0, -1]), IndexError, "Action index"),
    ("actions", torch.tensor([0, 2]), IndexError, "Action index"),
])
def test_invalid_q_actions_fail_before_any_table_write(field, value, error, message):
    values = {"q_table": torch.zeros(3, 2), "states": torch.tensor([0, 1]),
              "actions": torch.tensor([0, 1]), "target": torch.tensor([2.0, 4.0])}
    original = values["q_table"]
    values[field] = value
    with pytest.raises(error, match=message):
        TabularQUpdateRule(0.5).update(**values)
    torch.testing.assert_close(original, torch.zeros(3, 2))


@pytest.mark.parametrize("kind", ["V", "Q"])
def test_in_place_estimator_updates_remain_in_model_state_dict(kind):
    estimator = TabularVEstimator(3) if kind == "V" else TabularQEstimator(3, 2)
    model = Model(IdentityRepresentation(), estimator).double()
    table = estimator.v_table if kind == "V" else estimator.q_table
    original = table
    apply_tabular(kind, table, torch.tensor([1, 1]), torch.tensor([1.0, 3.0], dtype=torch.float64))
    assert table is original
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = Model(IdentityRepresentation(), TabularVEstimator(3) if kind == "V" else TabularQEstimator(3, 2)).double()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    expected, actual = model.forward(torch.arange(3)), restored.forward(torch.arange(3))
    torch.testing.assert_close(actual.values.v if kind == "V" else actual.values.q[0],
                               expected.values.v if kind == "V" else expected.values.q[0])
    assert table.reshape(-1)[1 if kind == "V" else 2].item() == 1.75


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(), (3,), (1, 3)])
def test_mse_signed_errors_and_scalar_mean_preserve_leading_dimensions(dtype, shape):
    prediction = torch.tensor(1.0 if shape == () else [1.0, 3.0, 5.0], dtype=dtype).reshape(shape)
    target = torch.tensor(3.0 if shape == () else [3.0, 1.0, 6.0], dtype=dtype).reshape(shape)
    result = TDLossUpdateRule().compute(prediction, target)
    torch.testing.assert_close(result.td_error, torch.tensor(2.0 if shape == () else [2.0, -2.0, 1.0], dtype=dtype).reshape(shape))
    torch.testing.assert_close(result.loss, torch.tensor(4.0 if shape == () else 3.0, dtype=dtype))
    assert result.loss.shape == () and result.loss.dtype == dtype
    with pytest.raises(FrozenInstanceError):
        result.loss = torch.tensor(0.0)


@pytest.mark.parametrize("loss_fn,expected", [
    (nn.SmoothL1Loss(reduction="none"), 3.5 / 3),
    (lambda prediction, target: (prediction - target).abs(), 5 / 3),
])
def test_smooth_l1_and_custom_callable_losses(loss_fn, expected):
    rule = TDLossUpdateRule(loss_fn)
    assert rule.loss_fn is loss_fn
    result = rule.compute(torch.tensor([1.0, 3.0, 5.0]), torch.tensor([3.0, 1.0, 6.0]))
    torch.testing.assert_close(result.loss, torch.tensor(expected))
    torch.testing.assert_close(result.td_error, torch.tensor([2.0, -2.0, 1.0]))


def test_weights_are_used_without_normalization_and_all_input_gradients_are_preserved():
    prediction = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([3.0, 1.0, 6.0], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([0.25, 1.0, 0.0], dtype=torch.float64, requires_grad=True)
    originals = [value.detach().clone() for value in (prediction, target, weights)]
    result = TDLossUpdateRule().compute(prediction, target, weights=weights)
    torch.testing.assert_close(result.loss, torch.tensor(5 / 3, dtype=torch.float64))
    assert result.td_error.requires_grad and result.loss.requires_grad
    assert prediction.grad is None and target.grad is None and weights.grad is None
    result.loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([-1 / 3, 4 / 3, 0], dtype=torch.float64))
    torch.testing.assert_close(target.grad, torch.tensor([1 / 3, -4 / 3, 0], dtype=torch.float64))
    torch.testing.assert_close(weights.grad, torch.tensor([4 / 3, 4 / 3, 1 / 3], dtype=torch.float64))
    for value, original in zip((prediction, target, weights), originals):
        torch.testing.assert_close(value.detach(), original)


def test_prediction_and_target_gradients_are_kept_without_weights():
    prediction = torch.tensor([1.0, 3.0], requires_grad=True)
    target = torch.tensor([3.0, 1.0], requires_grad=True)
    result = TDLossUpdateRule().compute(prediction, target)
    result.loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([-2.0, 2.0]))
    torch.testing.assert_close(target.grad, torch.tensor([2.0, -2.0]))


def test_target_detachment_and_priority_errors_are_controlled_by_the_caller():
    prediction = torch.tensor([1.0, 3.0], requires_grad=True)
    target = torch.tensor([3.0, 1.0], requires_grad=True)
    result = TDLossUpdateRule().compute(prediction, target.detach())
    priority_error = result.td_error.detach().abs()
    result.loss.backward()
    assert prediction.grad is not None and target.grad is None
    assert result.td_error.requires_grad and not priority_error.requires_grad
    torch.testing.assert_close(priority_error, torch.tensor([2.0, 2.0]))
    with torch.no_grad():
        no_grad_result = TDLossUpdateRule().compute(prediction, target)
    assert not no_grad_result.loss.requires_grad and not no_grad_result.td_error.requires_grad


def test_custom_loss_can_explicitly_use_higher_precision_without_implicit_rule_casts():
    prediction = torch.tensor([1.0], requires_grad=True)
    result = TDLossUpdateRule(lambda p, t: (p.double() - t.double()).square()).compute(prediction, torch.tensor([3.0]))
    assert result.td_error.dtype == torch.float32 and result.loss.dtype == torch.float64
    result.loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([-4.0]))


@pytest.mark.parametrize("loss_fn,error,message", [
    (3, TypeError, "callable"),
    (lambda p, t: 1.0, TypeError, "floating loss Tensor"),
    (lambda p, t: torch.ones_like(p, dtype=torch.long), TypeError, "floating loss Tensor"),
    (nn.MSELoss(reduction="mean"), ValueError, "reduction='none'"),
    (lambda p, t: (p - t).square().unsqueeze(-1), ValueError, "input shape"),
])
def test_invalid_loss_functions_fail_clearly(loss_fn, error, message):
    with pytest.raises(error, match=message):
        TDLossUpdateRule(loss_fn).compute(torch.ones(2), torch.zeros(2))


@pytest.mark.parametrize("field,value,error,message", [
    ("prediction", np.ones(2), TypeError, "must be Tensors"),
    ("target", np.zeros(2), TypeError, "must be Tensors"),
    ("target", torch.zeros(2, 1), ValueError, "same shape"),
    ("target", torch.zeros(1), ValueError, "same shape"),
    ("prediction", torch.ones(2, dtype=torch.long), TypeError, "floating dtype"),
    ("target", torch.zeros(2, dtype=torch.complex64), TypeError, "floating dtype"),
    ("target", torch.zeros(2, dtype=torch.float64), TypeError, "same dtype"),
    ("target", torch.zeros(2, device="meta"), ValueError, "same device"),
    ("weights", np.ones(2), TypeError, "weights must be a Tensor"),
    ("weights", torch.ones(2, 1), ValueError, "same shape"),
    ("weights", torch.tensor(1.0), ValueError, "same shape"),
    ("weights", torch.ones(2, dtype=torch.float64), TypeError, "same floating dtype"),
    ("weights", torch.ones(2, device="meta"), ValueError, "same device"),
])
def test_neural_tensor_contract_rejects_broadcasting_dtype_and_device_mismatches(field, value, error, message):
    values = {"prediction": torch.ones(2), "target": torch.zeros(2), "weights": None}
    values[field] = value
    with pytest.raises(error, match=message):
        TDLossUpdateRule().compute(**values)


def test_empty_inputs_do_not_silently_produce_a_nan_mean_loss():
    with pytest.raises(ValueError, match="nonempty"):
        TDLossUpdateRule().compute(torch.empty(0), torch.empty(0))


@pytest.mark.parametrize("kind", ["V", "Q"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_replay_model_td0_and_tabular_updates_with_duplicates_and_time_limit(kind, dtype):
    estimator = TabularVEstimator(4) if kind == "V" else TabularQEstimator(4, 2)
    model = Model(IdentityRepresentation(), estimator).to(dtype=dtype)
    table = estimator.v_table if kind == "V" else estimator.q_table
    if kind == "V":
        table.copy_(torch.tensor([0.0, 10.0, 20.0, 30.0], dtype=dtype))
    else:
        table.copy_(torch.tensor([[0.0, 10.0], [10.0, 20.0], [20.0, 30.0], [30.0, 40.0]], dtype=dtype))
    replay = ReplayBuffer(4, seed=6)
    for state, next_state, reward, terminal, truncated in [(0, 1, 1.0, False, False), (1, 2, 2.0, False, False),
                                                         (1, 0, 3.0, True, False), (2, 3, 4.0, False, True)]:
        replay.add(Transition(np.int32(state), 0, reward, np.int32(next_state), terminal, truncated))
    batch = replay.sample(4).to(dtype=dtype)
    states, actions = batch.observation.long(), batch.action.long()
    with torch.no_grad():
        estimate = model.forward(batch.next_observation)
        next_value = estimate.values.v if kind == "V" else estimate.values.q[0].max(-1).values
        target = TD0TargetGenerator(0.5).generate(batch.reward, next_value, batch.terminated)
    initial, expected_errors = table.tolist(), []
    for state, action, desired in zip(states.tolist(), actions.tolist(), target.tolist()):
        current = initial[state] if kind == "V" else initial[state][action]
        delta = desired - current
        expected_errors.append(delta)
        if kind == "V":
            initial[state] = current + 0.5 * delta
        else:
            initial[state][action] = current + 0.5 * delta
    error = apply_tabular(kind, table, states, target, actions=actions)
    torch.testing.assert_close(error, torch.tensor(expected_errors, dtype=dtype))
    torch.testing.assert_close(table, torch.tensor(initial, dtype=dtype))
    row = (states == 2).nonzero().item()
    assert batch.truncated[row] and not batch.terminated[row]
    assert target[row].item() == (19.0 if kind == "V" else 24.0)
    assert not error.requires_grad


@pytest.mark.parametrize("bootstrap", ["DQN", "Double DQN"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_neural_replay_model_td0_and_loss_preserve_learner_owned_optimization(bootstrap, dtype, batch_size):
    online = Model(MLPRepresentation(3, hidden_dims=(5,), activation=nn.Tanh), DiscreteQEstimator(5, 2)).to(dtype=dtype)
    with torch.no_grad():
        online.estimator.network.bias.copy_(torch.tensor([-20.0, 20.0], dtype=dtype))
    target_model = copy.deepcopy(online)
    with torch.no_grad():
        target_model.estimator.network.bias.copy_(torch.tensor([20.0, -20.0], dtype=dtype))
    replay = ReplayBuffer(4, seed=5)
    for state in range(4):
        observation = np.full(3, state, dtype=np.float32)
        replay.add(Transition(observation, state % 2, float(state + 1), observation + 1, state == 1, state == 3))
    batch = replay.sample(batch_size).to(dtype=dtype)
    prediction = online.forward(batch.observation).values.q[0].gather(-1, batch.action.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        target_q = target_model.forward(batch.next_observation).values.q[0]
        if bootstrap == "DQN":
            next_value = target_q.max(-1).values
        else:
            next_actions = online.forward(batch.next_observation).values.q[0].argmax(-1)
            next_value = target_q.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
            assert torch.all(next_value < target_q.max(-1).values)
        target = TD0TargetGenerator(0.9).generate(batch.reward, next_value, batch.terminated)
    result = TDLossUpdateRule().compute(prediction, target)
    assert isinstance(result, TDLossOutput) and result.loss.shape == ()
    torch.testing.assert_close(result.td_error, target - prediction)
    assert all(parameter.grad is None for parameter in online.parameters())
    assert all(parameter.grad is None for parameter in target_model.parameters())
    result.loss.backward()  # The future Learner owns backward and optimizer.step.
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in online.parameters())
    assert all(parameter.grad is None for parameter in target_model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("kind", ["V", "Q", "loss"])
def test_update_rules_preserve_cuda_device(kind):
    if kind == "loss":
        prediction = torch.tensor([1.0, 3.0], device="cuda", requires_grad=True)
        result = TDLossUpdateRule().compute(prediction, torch.tensor([3.0, 1.0], device="cuda"))
        assert result.loss.device == result.td_error.device == prediction.device
        result.loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([-2.0, 2.0], device="cuda"))
    else:
        table = torch.zeros(1 if kind == "V" else (1, 1), device="cuda")
        error = apply_tabular(kind, table, torch.tensor([0, 0], device="cuda"), torch.tensor([1.0, 3.0], device="cuda"))
        torch.testing.assert_close(error, torch.tensor([1.0, 2.5], device="cuda"))
        assert error.device == table.device and table.item() == 1.75
