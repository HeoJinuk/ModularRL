"""Deterministic reverse weighted IS, post-update greediness and atomic state."""

import io
import math

import numpy as np
import pytest
import torch

from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import DiscreteQEstimator, TabularQEstimator, TabularVEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import MonteCarloReturnGenerator
from modular_rl.data import Episode, Transition
from modular_rl.learner import OffPolicyMCControlLearner
from modular_rl.model import Model


def make_learner(dtype=torch.float64, gamma=1, selector=None):
    model = Model(IdentityRepresentation(), TabularQEstimator(4, 2), selector).to(dtype=dtype)
    return OffPolicyMCControlLearner(model, MonteCarloReturnGenerator(gamma))


def episode(states=(0,), actions=(0,), rewards=(4.,), probabilities=None, terminated=True, truncated=False):
    probabilities = [.25] * len(states) if probabilities is None else probabilities
    return Episode(tuple(Transition(np.int32(state), np.int32(action), np.float64(reward), np.int32(next_state),
                                    terminated if time == len(states) - 1 else False,
                                    truncated if time == len(states) - 1 else False,
                                    torch.tensor(math.log(probability), dtype=torch.float64))
                         for time, (state, action, reward, next_state, probability) in enumerate(zip(
                             states, actions, rewards, list(states[1:]) + [3], probabilities))))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_one_step_starts_with_unit_weight_and_updates_the_existing_table(dtype):
    learner = make_learner(dtype)
    table = learner.model.estimator.q_table
    table.fill_(99)
    pointer = table.data_ptr()
    sample = episode()
    before = [t.log_prob.clone() for t in sample.transitions]
    returns = learner.learn(sample)
    torch.testing.assert_close(returns, torch.tensor([4.], dtype=dtype))
    assert table[0, 0] == 4 and learner.weight_sums[0, 0] == 1  # Not 4*4 or C=4.
    assert learner.model.estimator.q_table is table and table.data_ptr() == pointer
    assert table[0, 1] == 99 and learner.weight_sums.sum() == 1
    assert learner.weight_sums.dtype == torch.float64 and learner.weight_sums.shape == (4, 2)
    assert returns.shape == (1,) and returns.grad_fn is None and not returns.requires_grad
    for record, saved in zip(sample.transitions, before):
        torch.testing.assert_close(record.log_prob, saved, rtol=0, atol=0)
        assert record.observation.dtype == record.action.dtype == torch.int32


@pytest.mark.parametrize("gamma,expected", [(0, [1., 3.]), (.5, [2.5, 3.]), (1, [4., 3.])])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_two_steps_use_next_action_ratio_after_its_update(gamma, expected, dtype):
    learner = make_learner(dtype, gamma)
    result = learner.learn(episode((0, 1), (0, 1), (1., 3.)))
    torch.testing.assert_close(result, torch.tensor(expected, dtype=dtype))
    assert learner.model.estimator.q_table[1, 1] == 3
    assert learner.model.estimator.q_table[0, 0] == expected[0]
    assert learner.weight_sums[1, 1] == 1 and learner.weight_sums[0, 0] == 4


def test_repeated_episodes_form_the_weighted_average_with_different_behavior_probabilities():
    learner = make_learner()
    learner.learn(episode((0, 1), (0, 1), (1., 3.)))
    learner.learn(episode((0, 1), (0, 1), (2., 4.), probabilities=(.5, .5)))
    assert learner.weight_sums[0, 0] == 6 and learner.weight_sums[1, 1] == 2
    assert learner.model.estimator.q_table[0, 0].item() == pytest.approx(14 / 3)
    assert learner.model.estimator.q_table[1, 1] == 3.5


def test_repeated_pair_is_updated_in_reverse_order_without_visit_filtering():
    learner = make_learner()
    result = learner.learn(episode((1, 2, 1), (0, 1, 0), (1., 2., 3.)))
    torch.testing.assert_close(result, torch.tensor([6., 5., 3.], dtype=torch.float64))
    assert learner.weight_sums[1, 0] == 17 and learner.weight_sums[2, 1] == 4
    assert learner.model.estimator.q_table[1, 0].item() == pytest.approx(99 / 17)
    assert learner.model.estimator.q_table[2, 1] == 5


def test_post_update_greedy_action_can_change_from_the_previous_target():
    learner = make_learner()
    learner.model.estimator.q_table[1, 0] = 2  # Before the terminal update, action 1 is not greedy.
    learner.learn(episode((0, 1), (0, 1), (1., 3.)))
    assert learner.model.estimator.q_table[1].argmax() == 1
    assert learner.weight_sums[0, 0] == 4  # The earlier step is updated after improvement.


def test_early_break_updates_the_mismatch_step_but_leaves_earlier_steps_untouched():
    learner = make_learner()
    learner.model.estimator.q_table[1] = torch.tensor([8., 3.])  # Action 0 starts greedy.
    returns = learner.learn(episode((0, 1), (0, 0), (1., 0.)))
    torch.testing.assert_close(returns, torch.tensor([1., 0.], dtype=torch.float64))
    assert learner.model.estimator.q_table[1, 0] == 0 and learner.weight_sums[1, 0] == 1
    assert learner.model.estimator.q_table[1].argmax() == 1
    assert learner.weight_sums[0, 0] == learner.model.estimator.q_table[0, 0] == 0


@pytest.mark.parametrize("action,earlier_weight", [(0, 4.), (1, 0.)])
def test_exact_ties_choose_smallest_action_id_without_sampling(action, earlier_weight):
    learner = make_learner()
    rng_state = torch.get_rng_state().clone()
    learner.learn(episode((0, 1), (0, action), (1., 0.)))
    assert learner.weight_sums[1, action] == 1
    assert learner.weight_sums[0, 0] == earlier_weight
    torch.testing.assert_close(torch.get_rng_state(), rng_state, rtol=0, atol=0)


def test_same_episode_logs_are_used_after_q_and_epsilon_have_changed():
    learner = make_learner(selector=EpsilonGreedyActionSelector(.1))
    sample = episode((0, 1), (0, 1), (1., 3.))
    learner.model.action_selector.epsilon = 0
    learner.model.estimator.q_table[1, 0] = 2
    assert learner.model.action_selector.probabilities(learner.model.forward(1))[1] == 0
    learner.learn(sample)
    assert learner.weight_sums[0, 0] == 4
    assert sample.transitions[-1].log_prob.exp().item() == pytest.approx(.25)


def test_no_grad_no_model_queries_and_generator_reuse(monkeypatch):
    learner = make_learner()
    learner.model.estimator.q_table.requires_grad_(True)
    monkeypatch.setattr(learner.model, "forward", lambda *args: pytest.fail("no bootstrap/model query"))
    monkeypatch.setattr(learner.model, "act", lambda *args: pytest.fail("no action selection"))
    generate = learner.return_generator.generate
    calls = []

    def returns(rewards):
        calls.append(rewards.clone())
        assert not torch.is_grad_enabled()
        return generate(rewards)

    monkeypatch.setattr(learner.return_generator, "generate", returns)
    result = learner.learn(episode((0, 1), (0, 1), (1., 3.)))
    assert len(calls) == 1 and not result.requires_grad
    assert learner.model.estimator.q_table.grad is None


@pytest.mark.parametrize("terminated,truncated", [(False, False), (False, True)])
def test_incomplete_episode_is_rejected_before_any_update(terminated, truncated):
    learner = make_learner()
    with pytest.raises(ValueError, match="incomplete"):
        learner.learn(episode(terminated=terminated, truncated=truncated))
    assert learner.weight_sums.sum() == learner.model.estimator.q_table.sum() == 0


def test_natural_termination_exactly_on_time_limit_is_complete():
    learner = make_learner()
    learner.learn(episode(truncated=True))
    assert learner.model.estimator.q_table[0, 0] == 4


@pytest.mark.parametrize("log_b,error", [
    (None, ValueError), (torch.tensor(float('-inf')), ValueError), (torch.tensor(float('inf')), ValueError),
    (torch.tensor(float('nan')), ValueError), (torch.tensor(.1), ValueError), (torch.tensor(0), TypeError),
    (torch.tensor(True), TypeError), (torch.tensor([-.5]), TypeError),
])
def test_missing_zero_or_invalid_behavior_probabilities_fail_before_learning(log_b, error):
    learner = make_learner()
    sample = Episode((Transition(0, 0, 4., 3, True, False, log_b),))
    with pytest.raises(error, match="log_prob"):
        learner.learn(sample)
    assert learner.weight_sums.sum() == learner.model.estimator.q_table.sum() == 0


@pytest.mark.parametrize("field,value,error", [
    ("state", 0., TypeError), ("state", True, TypeError), ("state", np.array([0]), TypeError),
    ("state", -1, IndexError), ("state", 4, IndexError), ("action", 0., TypeError),
    ("action", True, TypeError), ("action", 1j, TypeError), ("action", np.array([0]), TypeError),
    ("action", -1, IndexError), ("action", 2, IndexError),
])
def test_invalid_state_action_indices_fail_clearly(field, value, error):
    learner = make_learner()
    state, action = (value, 0) if field == "state" else (0, value)
    next_state = np.zeros_like(state) if isinstance(state, np.ndarray) else type(state)(3)
    sample = Episode((Transition(state, action, 4., next_state, True, False, torch.tensor(-math.log(4.))),))
    with pytest.raises(error, match=field):
        learner.learn(sample)
    assert learner.weight_sums.sum() == 0


@pytest.mark.parametrize("estimator", [TabularVEstimator(4), DiscreteQEstimator(1, 2)])
def test_wrong_model_connection_is_rejected(estimator):
    with pytest.raises(TypeError, match="TabularQEstimator"):
        OffPolicyMCControlLearner(Model(IdentityRepresentation(), estimator), MonteCarloReturnGenerator(1))


def test_non_episode_input_is_rejected():
    with pytest.raises(TypeError, match="Episode"):
        make_learner().learn(episode().transitions)


def snapshot(learner):
    return learner.model.estimator.q_table.clone(), learner.weight_sums.clone()


def assert_unchanged(learner, saved):
    for value, original in zip(snapshot(learner), saved):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


def test_weight_overflow_rolls_back_even_the_already_computed_terminal_update():
    learner = make_learner()
    saved = snapshot(learner)
    sample = Episode((Transition(0, 0, 1., 1, False, False, torch.tensor(0.)),
                      Transition(1, 1, 3., 3, True, False, torch.tensor(-1000., dtype=torch.float64))))
    with pytest.raises(FloatingPointError, match="weight overflowed"):
        learner.learn(sample)
    assert_unchanged(learner, saved)


def test_unused_weight_after_break_or_first_transition_is_not_computed():
    for states, actions, rewards in (((0,), (0,), (4.,)), ((0, 1), (0, 1), (1., 0.))):
        learner = make_learner()
        sample = episode(states, actions, rewards)
        sample.transitions[-1].log_prob.fill_(-1000)
        learner.learn(sample)  # Last action 1 ties action 0 and breaks before dividing by b.
        assert learner.weight_sums.sum() == 1


def test_cumulative_weight_overflow_preserves_the_previous_valid_learning_state():
    learner = make_learner()
    learner.learn(episode())
    learner.weight_sums[0, 0] = 1e308
    saved = snapshot(learner)
    sample = episode((0, 1), (0, 1), (1., 3.), probabilities=(1., 1e-308))
    with pytest.raises(FloatingPointError, match="cumulative weight"):
        learner.learn(sample)
    assert_unchanged(learner, saved)


def test_q_arithmetic_overflow_is_atomic():
    learner = make_learner()
    learner.model.estimator.q_table[0, 0] = -1e308
    saved = snapshot(learner)
    with pytest.raises(FloatingPointError, match="Q or cumulative"):
        learner.learn(episode(rewards=(1e308,)))
    assert_unchanged(learner, saved)


def test_return_overflow_is_atomic():
    learner = make_learner()
    saved = snapshot(learner)
    with pytest.raises(FloatingPointError, match="returns became non-finite"):
        learner.learn(episode((0, 1), (0, 1), (1e308, 1e308)))
    assert_unchanged(learner, saved)


def test_model_and_c_checkpoint_resume_exactly():
    learner = make_learner(gamma=.9, selector=EpsilonGreedyActionSelector(.1))
    sample = episode((0, 1), (0, 1), (1., 3.))
    learner.learn(sample)
    checkpoint = io.BytesIO()
    torch.save({"model": learner.model.state_dict(), "weight_sums": learner.weight_sums,
                "gamma": learner.return_generator.gamma}, checkpoint)
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    restored = make_learner(gamma=saved["gamma"], selector=EpsilonGreedyActionSelector(.1))
    restored.model.load_state_dict(saved["model"], strict=True)
    restored.weight_sums.copy_(saved["weight_sums"])
    assert "weight_sums" not in restored.model.state_dict()
    for item in (learner, restored):
        item.learn(sample)
        item.learn(episode((0, 1), (0, 1), (2., 4.), probabilities=(.5, .5)))
    assert_unchanged(restored, snapshot(learner))


def test_float64_weight_sums_survive_model_dtype_change():
    learner = make_learner(torch.float32)
    learner.learn(episode())
    learner.model.double()
    returns = learner.learn(episode())
    assert returns.dtype == torch.float64 and learner.weight_sums.dtype == torch.float64
    assert learner.weight_sums[0, 0] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_episode_updates_cuda_model_and_weight_sums(dtype):
    learner = make_learner(dtype)
    learner.model.cuda()
    result = learner.learn(episode((0, 1), (0, 1), (1., 3.)))
    assert result.device.type == learner.weight_sums.device.type == "cuda" and result.dtype == dtype
    assert learner.model.estimator.q_table[0, 0] == 4 and learner.weight_sums[0, 0] == 4
