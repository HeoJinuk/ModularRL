"""Seeded Gymnasium runs and n=1 compatibility with the established example."""

import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from examples import gridworld_n_step_sarsa as example
from examples import gridworld_sarsa as previous


@pytest.mark.parametrize("n", [1, 3, 5])
def test_short_online_runs_record_one_update_per_step_and_independent_evaluation(n):
    model, result = example.train_and_evaluate(n=n, episodes=20, seed=11, log_every=0)
    assert result["n"] == n and result["episodes"] == 20
    assert result["steps"] == result["updates"] == result["episode_lengths"].sum()
    assert result["training_rewards"].shape == result["episode_lengths"].shape == (20,)
    assert result["terminated_episodes"] + result["truncated_episodes"] == 20
    assert result["q_values"].shape == (11, 4) and np.isfinite(result["q_values"]).all()
    np.testing.assert_array_equal(result["q_values"], model.estimator.q_table.numpy())
    assert result["dp_steps"] == len(result["dp_path"])-1 == 5
    assert result["before"]["truncated"] and result["before"]["steps"] == 20
    assert np.isfinite(result["max_q_rmse"])
    result["q_values"].fill(123)
    assert not np.all(model.estimator.q_table.numpy() == 123)


@pytest.mark.parametrize("seed,cap", [(7, 2000), (11, 2000), (19, 2000), (11, 1)])
def test_one_step_gridworld_matches_previous_sarsa_rewards_lengths_flags_and_q(seed, cap):
    _, current = example.train_and_evaluate(n=1, episodes=20, seed=seed, max_steps=cap, log_every=0)
    _, reference = previous.train_and_evaluate("sarsa", episodes=20, seed=seed, max_steps=cap, log_every=0)
    for name in ("steps", "terminated_episodes", "truncated_episodes", "before", "after", "max_q_rmse"):
        assert current[name] == reference[name]
    for name in ("q_values", "training_rewards", "episode_lengths"):
        np.testing.assert_array_equal(current[name], reference[name])


@pytest.mark.parametrize("n", [3, 5])
def test_rng_seed_reproduces_n_step_table_and_online_counts(n):
    _, first = example.train_and_evaluate(n=n, episodes=20, seed=11, log_every=10)
    _, second = example.train_and_evaluate(n=n, episodes=20, seed=11, log_every=10)
    assert first["history"] == second["history"] and first["steps"] == second["steps"]
    for name in ("q_values", "training_rewards", "episode_lengths"):
        np.testing.assert_array_equal(first[name], second[name])


def test_real_time_limit_flushes_all_pending_updates_and_closes_both_environments(monkeypatch):
    closed, original_make = [], example.make_dlfs4_gridworld_env

    def make(**kwargs):
        env = original_make(**kwargs)
        close = env.close

        def finish():
            closed.append(env)
            close()

        monkeypatch.setattr(env, "close", finish)
        return env

    monkeypatch.setattr(example, "make_dlfs4_gridworld_env", make)
    _, result = example.train_and_evaluate(n=5, episodes=3, max_steps=2, log_every=0)
    assert result["steps"] == result["updates"] == 6 and result["truncated_episodes"] == 3
    assert len(closed) == 2 and closed[0] is not closed[1]


def test_example_functions_load_in_notebook_directory_without_installed_examples(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root / "notebooks" / "dlfs4")
    api = runpy.run_path(str(root / "examples" / "gridworld_n_step_sarsa.py"))
    assert all(callable(api[name]) for name in ("learn_n_step_sarsa", "run_episode", "train_and_evaluate"))


def test_cli_reports_horizons_actual_updates_and_existing_comparators():
    result = subprocess.run([sys.executable, "-m", "examples.gridworld_n_step_sarsa", "--episodes", "5", "--seeds", "11"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True, timeout=30)
    for label in ("DLFS4 n-step SARSA: n=1", "DLFS4 n-step SARSA: n=3", "DLFS4 n-step SARSA: n=5",
                  "updates=", "Reference sarsa:", "Reference q_learning:", "DP optimal path:", "Q (up/down/left/right):"):
        assert label in result.stdout
