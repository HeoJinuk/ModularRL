"""Chronological state-action visits, sample averages and existing Model policy."""

import io

import numpy as np
import pytest
import torch

from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import DiscreteQEstimator, TabularQEstimator, TabularVEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import MonteCarloReturnGenerator
from modular_rl.data import Episode, Transition
from modular_rl.learner import TabularMCControlLearner
from modular_rl.model import Model


def make_learner(mode="first", dtype=torch.float64, gamma=1, selector=None):
    model = Model(IdentityRepresentation(), TabularQEstimator(4, 2), selector).to(dtype=dtype)
    return TabularMCControlLearner(model, MonteCarloReturnGenerator(gamma), mode)


def repeated_episode():
    return Episode(tuple(Transition(torch.tensor(s, dtype=torch.int32), torch.tensor(a, dtype=torch.int32),
                                    r, torch.tensor(n, dtype=torch.int32), ended, False)
                         for s, a, r, n, ended in ((1, 0, 1., 2, False), (2, 1, 2., 1, False),
                                                   (1, 0, 3., 0, True))))


@pytest.mark.parametrize("mode,q_one,count_one", [("first", 6., 1), ("every", 4.5, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_repeated_pair_uses_first_chronological_visit_or_all_visits(mode, q_one, count_one, dtype):
    learner = make_learner(mode, dtype)
    table = learner.model.estimator.q_table
    pointer = table.data_ptr()
    episode = repeated_episode()
    snapshot = [tuple(value.clone() for value in (t.observation, t.action, t.reward, t.next_observation))
                for t in episode.transitions]
    returns = learner.learn(episode)
    torch.testing.assert_close(returns, torch.tensor([6., 5., 3.], dtype=dtype))
    torch.testing.assert_close(table, torch.tensor([[0, 0], [q_one, 0], [0, 5], [0, 0]], dtype=dtype))
    torch.testing.assert_close(learner.visit_counts, torch.tensor([[0, 0], [count_one, 0], [0, 1], [0, 0]]))
    assert learner.model.estimator.q_table is table and table.data_ptr() == pointer
    assert returns.shape == (3,) and not returns.requires_grad and returns.grad_fn is None
    for transition, original in zip(episode.transitions, snapshot):
        for current, saved in zip((transition.observation, transition.action, transition.reward,
                                   transition.next_observation), original):
            torch.testing.assert_close(current, saved)


@pytest.mark.parametrize("gamma,expected", [(0, [1., 2., 3.]), (.5, [2.75, 3.5, 3.]), (1, [6., 5., 3.])])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_control_reuses_generator_discount_and_live_table_dtype(gamma, expected, dtype):
    learner = make_learner(dtype=dtype, gamma=gamma)
    returns = learner.learn(repeated_episode())
    torch.testing.assert_close(returns, torch.tensor(expected, dtype=dtype))
    assert returns.device == learner.model.estimator.q_table.device
    assert learner.model.estimator.q_table[1, 0] == expected[0]
    assert learner.model.estimator.q_table[2, 1] == expected[1]


@pytest.mark.parametrize("mode,expected", [("first", 6.), ("every", 4.5)])
def test_different_actions_at_one_state_are_distinct_first_visits(mode, expected):
    episode = Episode((Transition(1, 0, 1., 1, False, False), Transition(1, 1, 2., 1, False, False),
                       Transition(1, 0, 3., 0, True, False)))
    learner = make_learner(mode)
    learner.learn(episode)
    torch.testing.assert_close(learner.model.estimator.q_table[1], torch.tensor([expected, 5.], dtype=torch.float64))
    assert learner.visit_counts[1, 1] == 1
    assert learner.visit_counts[1, 0] == (1 if mode == "first" else 2)


@pytest.mark.parametrize("mode,expected,count", [("first", 4., 2), ("every", 11 / 3, 3)])
def test_sample_average_accumulates_across_episodes(mode, expected, count):
    learner = make_learner(mode)
    learner.learn(repeated_episode())
    learner.learn(Episode((Transition(1, 0, 2., 0, True, False),)))
    assert learner.model.estimator.q_table[1, 0].item() == pytest.approx(expected)
    assert learner.visit_counts[1, 0].item() == count
    assert learner.model.estimator.q_table[2, 1] == 5 and learner.visit_counts[2, 1] == 1


def test_final_next_observation_is_not_an_extra_visit_and_initial_q_is_not_a_sample():
    learner = make_learner()
    learner.model.estimator.q_table.fill_(99)
    learner.learn(Episode((Transition(1, 0, 2., 0, True, False),)))
    assert learner.model.estimator.q_table[1, 0] == 2
    assert learner.visit_counts.sum() == learner.visit_counts[1, 0] == 1
    torch.testing.assert_close(learner.model.estimator.q_table[0], torch.full((2,), 99., dtype=torch.float64))


@pytest.mark.parametrize("terminated,truncated", [(False, False), (False, True)])
def test_incomplete_episode_fails_before_returns_or_any_update(terminated, truncated, monkeypatch):
    learner = make_learner()
    monkeypatch.setattr(learner.return_generator, "generate", lambda rewards: pytest.fail("must not generate returns"))
    with pytest.raises(ValueError, match="incomplete"):
        learner.learn(Episode((Transition(1, 0, 2., 0, terminated, truncated),)))
    assert learner.visit_counts.sum() == learner.model.estimator.q_table.sum() == 0


def test_natural_termination_at_time_limit_is_complete():
    learner = make_learner()
    learner.learn(Episode((Transition(1, 0, 2., 0, True, True),)))
    assert learner.model.estimator.q_table[1, 0] == 2 and learner.visit_counts[1, 0] == 1


def test_no_grad_no_bootstrap_no_action_selection_during_learning(monkeypatch):
    learner = make_learner()
    table = learner.model.estimator.q_table
    table.requires_grad_(True)
    monkeypatch.setattr(learner.model, "forward", lambda *args: pytest.fail("no model query in MC update"))
    monkeypatch.setattr(learner.model, "act", lambda *args: pytest.fail("no policy sampling in MC update"))
    original = learner.return_generator.generate

    def generate(rewards):
        assert not torch.is_grad_enabled()
        return original(rewards)

    monkeypatch.setattr(learner.return_generator, "generate", generate)
    returns = learner.learn(repeated_episode())
    assert not returns.requires_grad and table.grad is None


def test_return_overflow_fails_before_table_or_counts_change():
    learner = make_learner()
    episode = Episode((Transition(1, 0, np.float64(1e308), 2, False, False),
                       Transition(2, 1, np.float64(1e308), 0, True, False)))
    with pytest.raises(FloatingPointError, match="non-finite"):
        learner.learn(episode)
    assert learner.visit_counts.sum() == learner.model.estimator.q_table.sum() == 0


@pytest.mark.parametrize("field,value,error", [
    ("state", 1., TypeError), ("state", True, TypeError), ("state", np.array([1]), TypeError),
    ("state", -1, IndexError), ("state", 4, IndexError),
    ("action", 1., TypeError), ("action", True, TypeError), ("action", 1j, TypeError),
    ("action", np.array([0]), TypeError), ("action", -1, IndexError), ("action", 2, IndexError),
])
def test_invalid_state_action_indices_fail_clearly(field, value, error):
    learner = make_learner()
    state = value if field == "state" else 1
    action = value if field == "action" else 0
    next_state = np.zeros_like(state) if isinstance(state, np.ndarray) else type(state)(0)
    episode = Episode((Transition(state, action, 1., next_state, True, False),))
    with pytest.raises(error, match=f"MC {field}"):
        learner.learn(episode)
    assert learner.visit_counts.sum() == learner.model.estimator.q_table.sum() == 0


def test_later_invalid_index_does_not_leave_partial_updates():
    learner = make_learner()
    episode = Episode((Transition(1, 0, 1., 2, False, False), Transition(2, 3, 2., 0, True, False)))
    with pytest.raises(IndexError, match="action"):
        learner.learn(episode)
    assert learner.visit_counts.sum() == learner.model.estimator.q_table.sum() == 0


@pytest.mark.parametrize("estimator", [TabularVEstimator(4), DiscreteQEstimator(1, 2)])
def test_wrong_model_connection_is_rejected(estimator):
    with pytest.raises(TypeError, match="TabularQEstimator"):
        TabularMCControlLearner(Model(IdentityRepresentation(), estimator), MonteCarloReturnGenerator(1))


def test_invalid_mode_and_non_episode_input():
    with pytest.raises(ValueError, match="visit_mode"):
        make_learner("last")
    with pytest.raises(TypeError, match="Episode"):
        make_learner().learn(repeated_episode().transitions)


def test_updated_q_immediately_changes_existing_epsilon_greedy_policy_and_external_epsilon():
    learner = make_learner(selector=EpsilonGreedyActionSelector(.2))
    model = learner.model
    torch.testing.assert_close(model.action_selector.probabilities(model.forward(1)), torch.full((2,), .5, dtype=torch.float64))
    learner.learn(Episode((Transition(1, 1, 4., 0, True, False),)))
    torch.testing.assert_close(model.action_selector.probabilities(model.forward(1)), torch.tensor([.1, .9], dtype=torch.float64))
    selected = model.act(1, deterministic=True)
    assert selected.action.item() == 1 and selected.log_prob is None
    model.action_selector.epsilon = 1
    torch.testing.assert_close(model.action_selector.probabilities(model.forward(1)), torch.full((2,), .5, dtype=torch.float64))
    assert model.act(1, deterministic=True).action.item() == 1


@pytest.mark.parametrize("mode", ["first", "every"])
def test_model_and_separately_saved_counts_resume_exact_sample_averages(mode):
    learner = make_learner(mode, selector=EpsilonGreedyActionSelector(.1))
    learner.learn(repeated_episode())
    checkpoint = io.BytesIO()
    torch.save({"model": learner.model.state_dict(), "visit_counts": learner.visit_counts.clone(),
                "gamma": learner.return_generator.gamma, "visit_mode": learner.visit_mode}, checkpoint)
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    restored = make_learner(saved["visit_mode"], gamma=saved["gamma"], selector=EpsilonGreedyActionSelector(1))
    restored.model.load_state_dict(saved["model"], strict=True)
    restored.visit_counts.copy_(saved["visit_counts"])
    assert "visit_counts" not in restored.model.state_dict()
    assert restored.model.action_selector.epsilon == pytest.approx(.1)
    for candidate in (learner, restored):
        candidate.learn(repeated_episode())
        candidate.learn(Episode((Transition(1, 0, 2., 0, True, False),)))
    torch.testing.assert_close(restored.model.estimator.q_table, learner.model.estimator.q_table, rtol=0, atol=0)
    torch.testing.assert_close(restored.visit_counts, learner.visit_counts)


def test_model_dtype_change_uses_live_table_and_preserves_integer_counts():
    learner = make_learner(dtype=torch.float32)
    learner.learn(repeated_episode())
    learner.model.double()
    returns = learner.learn(repeated_episode())
    assert returns.dtype == torch.float64 and learner.visit_counts.dtype == torch.long
    assert learner.visit_counts[1, 0] == 2 and learner.model.estimator.q_table[1, 0] == 6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_episode_updates_cuda_table_and_count_state(dtype):
    learner = make_learner("every", dtype)
    learner.model.cuda()
    returns = learner.learn(repeated_episode())
    assert returns.device.type == learner.visit_counts.device.type == "cuda"
    assert returns.dtype == dtype and learner.visit_counts.dtype == torch.long
    torch.testing.assert_close(learner.model.estimator.q_table[1].cpu(), torch.tensor([4.5, 0], dtype=dtype))
