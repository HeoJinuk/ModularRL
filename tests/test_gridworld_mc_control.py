"""Short on-policy interaction smoke tests, without stochastic score thresholds."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from examples import gridworld_mc_control as example
from modular_rl.learner import TabularMCControlLearner


def test_short_example_uses_existing_model_table_counts_and_separate_evaluation():
    learner, result = example.train_and_evaluate(episodes=20, log_every=0)
    assert isinstance(learner, TabularMCControlLearner)
    assert result["episodes"] == 20 and result["steps"] >= 20 and result["truncated_episodes"] == 0
    assert result["training_rewards"].shape == (20,)
    assert np.isfinite(result["q_values"]).all()
    np.testing.assert_array_equal(result["q_values"], learner.model.estimator.q_table.numpy())
    np.testing.assert_array_equal(result["visit_counts"], learner.visit_counts.numpy())
    assert result["q_values"].shape == result["visit_counts"].shape == (11, 4)
    assert result["visit_counts"].sum() > 0
    assert result["visit_counts"].max() <= 20
    assert result["visit_counts"][3].sum() == 0  # Goal placeholder is not a visit.
    np.testing.assert_allclose(result["greedy_policy"].sum(axis=1), 1)
    assert result["history"][0]["greedy"] == result["before"]
    assert result["history"][-1]["greedy"] == result["after"]
    assert result["history"][-1]["steps"] == result["steps"]
    assert result["dp_steps"] == 5  # Deterministic DP reference, not an MC performance threshold.
    assert result["dp_values"][7] == pytest.approx(.6561)
    assert result["after"]["terminated"] or result["after"]["truncated"]


def test_seed_reproduces_tables_counts_episodes_rewards_and_evaluation():
    _, first = example.train_and_evaluate(episodes=20, seed=11, log_every=10)
    _, second = example.train_and_evaluate(episodes=20, seed=11, log_every=10)
    assert first["steps"] == second["steps"] and first["history"] == second["history"]
    for name in ("q_values", "visit_counts", "training_rewards", "greedy_policy"):
        np.testing.assert_array_equal(first[name], second[name])


def test_q_is_fixed_through_each_collection_and_only_then_updated(monkeypatch):
    original_collect, original_learn = example.collect_episode, TabularMCControlLearner.learn
    models, calls = [], []

    def learn(learner, episode):
        assert episode.terminated
        calls.append(len(episode))
        return original_learn(learner, episode)

    original_model = example.Model

    def model(*args, **kwargs):
        result = original_model(*args, **kwargs)
        models.append(result)
        return result

    def collect(env, select_action, **kwargs):
        table = models[0].estimator.q_table
        fixed_q = table.clone()

        def action(state):
            torch.testing.assert_close(table, fixed_q, rtol=0, atol=0)
            return select_action(state)

        episode = original_collect(env, action, **kwargs)
        torch.testing.assert_close(table, fixed_q, rtol=0, atol=0)
        return episode

    monkeypatch.setattr(example, "Model", model)
    monkeypatch.setattr(example, "collect_episode", collect)
    monkeypatch.setattr(TabularMCControlLearner, "learn", learn)
    learner, _ = example.train_and_evaluate(episodes=5, log_every=1)
    assert len(calls) == 5 and learner.model is models[0]
    assert torch.count_nonzero(learner.model.estimator.q_table) > 0


def test_real_time_limit_aborts_before_learning_and_closes_both_environments(monkeypatch):
    closed = []
    original = example.make_dlfs4_gridworld_env

    def make(**kwargs):
        env = original(**kwargs)
        close = env.close

        def finish():
            closed.append(env)
            close()

        monkeypatch.setattr(env, "close", finish)
        return env

    monkeypatch.setattr(example, "make_dlfs4_gridworld_env", make)
    monkeypatch.setattr(TabularMCControlLearner, "learn", lambda *args: pytest.fail("must not learn a cutoff"))
    with pytest.raises(RuntimeError, match="truncated.*MC Control did not learn"):
        example.train_and_evaluate(episodes=1, max_steps=1, log_every=0)
    assert len(closed) == 2 and closed[0] is not closed[1]


def test_example_cli_reports_behavior_and_greedy_results_and_dp_reference():
    result = subprocess.run([sys.executable, "-m", "examples.gridworld_mc_control", "--episodes", "10"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                            check=True, timeout=30)
    for label in ("DLFS4 on-policy MC Control:", "before greedy:", "after greedy:", "train_mean_reward=",
                  "DP optimal: steps=5", "State-action visit counts:", "Greedy policy probabilities"):
        assert label in result.stdout
