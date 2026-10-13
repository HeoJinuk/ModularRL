"""Short MC experiment smoke checks, with deterministic correctness tested separately."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from examples.gridworld_mc import train_and_compare
from modular_rl.data import Episode, Transition
from modular_rl.environment import TabularEpisodeSampler
from modular_rl.learner import MCPredictionLearner


def test_short_mc_experiment_uses_both_modes_shared_episodes_and_only_nonterminal_starts():
    learners, results = train_and_compare(episodes=30, seed=7, log_every=0)
    assert results["episodes"] == 30 and results["steps"] > 30
    assert results["start_counts"].sum() == 30 and results["start_counts"][[0, 15]].sum() == 0
    for mode in ("first", "every"):
        values, counts = results[mode]["values"], results[mode]["visit_counts"]
        assert values.shape == counts.shape == (16,)
        np.testing.assert_array_equal(values[[0, 15]], [0, 0])
        np.testing.assert_array_equal(counts[[0, 15]], [0, 0])
        assert np.isfinite(results[mode]["rmse"])
        nonterminal = np.arange(1, 15)
        expected_rmse = np.sqrt(np.mean((values[nonterminal] - results["reference"][nonterminal]) ** 2))
        assert results[mode]["rmse"] == expected_rmse
    assert results["every"]["visit_counts"].sum() == results["steps"]
    assert np.all(results["first"]["visit_counts"] <= results["every"]["visit_counts"])
    assert results["first"]["visit_counts"].max() <= results["episodes"]
    assert learners["first"].model is not learners["every"].model


def test_seed_reproduces_sampling_visit_counts_tables_and_rmse():
    _, first = train_and_compare(episodes=20, seed=11, log_every=0)
    _, second = train_and_compare(episodes=20, seed=11, log_every=0)
    assert first["steps"] == second["steps"] and first["initial_rmse"] == second["initial_rmse"]
    np.testing.assert_array_equal(first["start_counts"], second["start_counts"])
    for mode in ("first", "every"):
        np.testing.assert_array_equal(first[mode]["values"], second[mode]["values"])
        np.testing.assert_array_equal(first[mode]["visit_counts"], second[mode]["visit_counts"])
        assert first[mode]["rmse"] == second[mode]["rmse"]


def test_experiment_aborts_on_a_truncated_episode_before_either_learner_updates(monkeypatch):
    def truncate(*args, **kwargs):
        return Episode((Transition(1, 0, -1., 1, False, True),))

    def forbidden(*args, **kwargs):
        raise AssertionError("Truncation must not be learned or silently discarded")

    monkeypatch.setattr(TabularEpisodeSampler, "sample", truncate)
    monkeypatch.setattr(MCPredictionLearner, "learn", forbidden)
    with pytest.raises(RuntimeError, match="truncated.*MC did not learn"):
        train_and_compare(episodes=1, log_every=0)


def test_mc_example_module_reports_values_counts_steps_and_dp_rmse():
    result = subprocess.run([sys.executable, "-m", "examples.gridworld_mc", "--episodes", "20"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                            check=True, timeout=30)
    assert "Sutton 4x4 MC Prediction: gamma=1 seed=7 episodes=20 steps=" in result.stdout
    assert "truncated=0" in result.stdout and "14 nonterminal states" in result.stdout
    assert "DP uniform-policy V:" in result.stdout and "Start counts:" in result.stdout
    for mode in ("first", "every"):
        assert f"{mode}-visit V (RMSE=" in result.stdout and f"{mode}-visit update counts:" in result.stdout
