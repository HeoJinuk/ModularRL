"""Reverse returns, chronological visits, exact averages and separate count state."""

import io

import numpy as np
import pytest
import torch

from modular_rl.component.estimator import DiscreteQEstimator, TabularQEstimator, TabularVEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import BaseTargetGenerator, MonteCarloReturnGenerator
from modular_rl.data import Episode, Transition
from modular_rl.learner import MCPredictionLearner
from modular_rl.model import Model


def repeated_episode(index_dtype=torch.int32):
    steps = ((1, 1., 2, False), (2, 2., 1, False), (1, 3., 0, True))
    return Episode(tuple(Transition(torch.tensor(s, dtype=index_dtype), 0, r,
                                    torch.tensor(n, dtype=index_dtype), ended, False)
                         for s, r, n, ended in steps))


def make_learner(mode="first", dtype=torch.float64, gamma=1):
    model = Model(IdentityRepresentation(), TabularVEstimator(4)).to(dtype=dtype)
    return MCPredictionLearner(model, MonteCarloReturnGenerator(gamma), mode)


@pytest.mark.parametrize("gamma,expected", [(0, [1, 2, 3]), (.5, [2.75, 3.5, 3]), (1, [6, 5, 3])])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_discounted_returns_are_computed_backwards_with_correct_shape_and_dtype(gamma, expected, dtype):
    rewards = torch.tensor([1., 2., 3.], dtype=dtype)
    generator = MonteCarloReturnGenerator(gamma)
    result = generator.generate(rewards)
    torch.testing.assert_close(result, torch.tensor(expected, dtype=dtype))
    assert result.shape == (3,) and result.dtype == dtype and result.device == rewards.device
    assert not isinstance(generator, BaseTargetGenerator)
    torch.testing.assert_close(rewards, torch.tensor([1., 2., 3.], dtype=dtype))


def test_return_generator_arithmetic_preserves_gradients_but_does_not_run_backward():
    rewards = torch.tensor([1., 2., 3.], dtype=torch.float64, requires_grad=True)
    returns = MonteCarloReturnGenerator(.5).generate(rewards)
    assert returns.requires_grad and rewards.grad is None
    returns.sum().backward()
    torch.testing.assert_close(rewards.grad, torch.tensor([1., 1.5, 1.75], dtype=torch.float64))


def test_finite_rewards_with_overflowing_total_return_fail_explicitly():
    with pytest.raises(FloatingPointError, match="non-finite"):
        MonteCarloReturnGenerator(1).generate(torch.tensor([1e308, 1e308], dtype=torch.float64))


def test_overflow_is_detected_before_any_sample_average_or_count_update():
    learner = make_learner()
    episode = Episode((Transition(1, 0, np.float64(1e308), 2, False, False),
                       Transition(2, 0, np.float64(1e308), 0, True, False)))
    with pytest.raises(FloatingPointError, match="non-finite"):
        learner.learn(episode)
    assert learner.visit_counts.sum().item() == learner.model.estimator.v_table.sum().item() == 0


@pytest.mark.parametrize("mode,state_one,count_one", [("first", 6., 1), ("every", 4.5, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_repeated_state_example_uses_first_chronological_visit_or_every_visit(mode, state_one, count_one, dtype):
    learner = make_learner(mode, dtype)
    episode = repeated_episode()
    table = learner.model.estimator.v_table
    pointer = table.data_ptr()
    returns = learner.learn(episode)
    torch.testing.assert_close(returns, torch.tensor([6., 5., 3.], dtype=dtype))
    torch.testing.assert_close(table, torch.tensor([0., state_one, 5., 0.], dtype=dtype))
    torch.testing.assert_close(learner.visit_counts, torch.tensor([0, count_one, 1, 0]))
    assert table.data_ptr() == pointer and not returns.requires_grad and returns.grad_fn is None
    assert episode.transitions[0].observation.dtype == torch.int32


@pytest.mark.parametrize("mode,expected,count", [("first", 4., 2), ("every", 11 / 3, 3)])
def test_incremental_sample_average_accumulates_across_episodes(mode, expected, count):
    learner = make_learner(mode)
    learner.learn(repeated_episode())
    learner.learn(Episode((Transition(1, 0, 2., 0, True, False),)))
    assert learner.visit_counts[1].item() == count
    assert learner.model.estimator.v_table[1].item() == pytest.approx(expected)
    assert learner.model.estimator.v_table[2].item() == 5 and learner.visit_counts[2].item() == 1


def test_first_visit_means_earliest_time_not_first_encountered_during_reverse_return_calculation():
    episode = Episode((Transition(1, 0, 1., 1, False, False),
                       Transition(1, 0, 2., 1, False, False), Transition(1, 0, 3., 0, True, False)))
    learner = make_learner()
    learner.learn(episode)
    assert learner.model.estimator.v_table[1].item() == 6 and learner.visit_counts[1].item() == 1


def test_next_observation_is_not_a_visit_and_initial_table_values_are_not_pseudo_samples():
    learner = make_learner()
    learner.model.estimator.v_table.fill_(99)
    learner.learn(Episode((Transition(1, 0, 2., 0, True, False),)))
    assert learner.model.estimator.v_table[1].item() == 2 and learner.visit_counts[1].item() == 1
    assert learner.model.estimator.v_table[0].item() == 99 and learner.visit_counts[0].item() == 0


@pytest.mark.parametrize("terminated,truncated", [(False, False), (False, True)])
def test_incomplete_episode_is_rejected_before_returns_counts_or_values_change(terminated, truncated, monkeypatch):
    learner = make_learner()

    def forbidden(*args, **kwargs):
        raise AssertionError("Incomplete episodes must not generate returns")

    monkeypatch.setattr(learner.return_generator, "generate", forbidden)
    with pytest.raises(ValueError, match="incomplete"):
        learner.learn(Episode((Transition(1, 0, 2., 0, terminated, truncated),)))
    assert torch.count_nonzero(learner.visit_counts) == torch.count_nonzero(learner.model.estimator.v_table) == 0


def test_actual_termination_is_complete_even_when_the_time_limit_flag_is_also_true():
    learner = make_learner()
    learner.learn(Episode((Transition(1, 0, 2., 0, True, True),)))
    assert learner.model.estimator.v_table[1].item() == 2


def test_learning_uses_no_model_forward_sampling_or_bootstrap(monkeypatch):
    learner = make_learner()

    def forbidden(*args, **kwargs):
        raise AssertionError("MC prediction does not query a model or select behavior actions")

    monkeypatch.setattr(learner.model, "forward", forbidden)
    monkeypatch.setattr(learner.model, "act", forbidden)
    learner.learn(repeated_episode())
    assert learner.model.estimator.v_table[1].item() == 6


def test_model_dtype_changes_use_the_live_table_without_changing_count_dtype():
    learner = make_learner(dtype=torch.float32)
    learner.learn(repeated_episode())
    learner.model.double()
    returns = learner.learn(repeated_episode())
    assert returns.dtype == torch.float64 and learner.visit_counts.dtype == torch.long
    assert learner.visit_counts[1].item() == 2 and learner.model.estimator.v_table[1].item() == 6


@pytest.mark.parametrize("mode", ["first", "every"])
def test_model_and_separately_saved_visit_counts_resume_identical_sample_averages(mode):
    learner = make_learner(mode)
    learner.learn(repeated_episode())
    assert not any("visit_counts" in key for key in learner.model.state_dict())
    checkpoint = io.BytesIO()
    torch.save({"model": learner.model.state_dict(), "visit_counts": learner.visit_counts.clone(),
                "visit_mode": learner.visit_mode, "gamma": learner.return_generator.gamma}, checkpoint)
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    restored = make_learner(saved["visit_mode"], gamma=saved["gamma"])
    restored.model.load_state_dict(saved["model"], strict=True)
    restored.visit_counts.copy_(saved["visit_counts"])
    following = Episode((Transition(1, 0, 2., 0, True, False),))
    torch.testing.assert_close(restored.learn(following), learner.learn(following))
    torch.testing.assert_close(restored.model.estimator.v_table, learner.model.estimator.v_table)
    torch.testing.assert_close(restored.visit_counts, learner.visit_counts)


@pytest.mark.parametrize("gamma", [-.1, 1.1, True, np.nan, np.inf, "1"])
def test_invalid_discount_is_rejected(gamma):
    with pytest.raises(ValueError, match="gamma"):
        MonteCarloReturnGenerator(gamma)


@pytest.mark.parametrize("reward,error,message", [
    (np.ones(3), TypeError, "Tensor"), (torch.tensor(1.), ValueError, "shape"),
    (torch.zeros(2, 2), ValueError, "shape"), (torch.empty(0), ValueError, "nonempty"),
    (torch.ones(2, dtype=torch.long), TypeError, "floating"),
    (torch.ones(2, dtype=torch.bool), TypeError, "floating"),
    (torch.tensor([np.inf]), ValueError, "finite"), (torch.tensor([np.nan]), ValueError, "finite"),
])
def test_return_input_contracts_fail_explicitly(reward, error, message):
    with pytest.raises(error, match=message):
        MonteCarloReturnGenerator(1).generate(reward)


@pytest.mark.parametrize("state,error,message", [
    (1., TypeError, "integer state"), (True, TypeError, "integer state"),
    (np.array([1]), TypeError, "scalar integer"), (-1, IndexError, "bounds"), (4, IndexError, "bounds"),
])
def test_bad_state_inputs_are_rejected_before_any_learning(state, error, message):
    learner = make_learner()
    next_state = np.zeros_like(state) if isinstance(state, np.ndarray) else type(state)(0)
    episode = Episode((Transition(state, 0, 1., next_state, True, False),))
    with pytest.raises(error, match=message):
        learner.learn(episode)
    assert learner.visit_counts.sum().item() == 0


@pytest.mark.parametrize("estimator", [TabularQEstimator(4, 2), DiscreteQEstimator(1, 2)])
def test_only_tabular_v_models_are_supported(estimator):
    with pytest.raises(TypeError, match="TabularVEstimator"):
        MCPredictionLearner(Model(IdentityRepresentation(), estimator), MonteCarloReturnGenerator(1))


def test_invalid_visit_mode_and_non_episode_learning_input_are_rejected():
    with pytest.raises(ValueError, match="visit_mode"):
        make_learner("last")
    with pytest.raises(TypeError, match="Episode"):
        make_learner().learn(repeated_episode().transitions)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_episode_can_update_cuda_table_and_count_state(dtype):
    learner = make_learner("every", dtype)
    learner.model.cuda()
    returns = learner.learn(repeated_episode())
    assert returns.device.type == learner.visit_counts.device.type == "cuda"
    assert learner.visit_counts.dtype == torch.long and returns.dtype == dtype
    torch.testing.assert_close(learner.model.estimator.v_table.cpu(), torch.tensor([0, 4.5, 5, 0], dtype=dtype))
