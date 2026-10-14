"""Fixed-policy DP comparison, reproducibility and notebook file loading."""

from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

from examples import gridworld_off_policy_n_step_sarsa as example


@pytest.mark.parametrize("n", [1, 3, 5])
def test_same_behavior_trajectories_and_exact_matched_policy_equivalence(n):
    results = example.run_comparison(seed=11, episodes=4, ns=(n,), log_every=0)
    for result in results:
        assert result["updates"] == result["steps"] == result["episode_lengths"].sum()
        assert result["q_values"].shape == result["target_policy"].shape == result["dp_q_values"].shape == (11, 4)
        assert np.isfinite(result["q_values"]).all() and np.isfinite(result["q_rmse"])
        assert result["rho_zero_fraction"] == 0 and 0 < result["ess"] <= result["updates"]+1e-8
        assert result["terminated_episodes"]+result["truncated_episodes"] == 4
    for key in ("training_rewards", "episode_lengths"):
        np.testing.assert_array_equal(results[0][key], results[2][key])
    for result in results[:2]:
        np.testing.assert_array_equal(result["importance_ratios"], np.ones(result["updates"]))
    assert not np.array_equal(results[0]["target_policy"], results[2]["target_policy"])
    assert not np.array_equal(results[0]["dp_q_values"], results[2]["dp_q_values"])
    assert results[2]["rho_max"] <= 1.2**n+1e-12


def test_seed_reproduces_tables_metrics_and_owned_fixed_policy():
    _, first = example.train_and_evaluate(episodes=5, seed=7, log_every=2)
    model, second = example.train_and_evaluate(episodes=5, seed=7, log_every=2)
    for key in ("q_values", "importance_ratios", "training_rewards", "episode_lengths", "target_policy"):
        np.testing.assert_array_equal(first[key], second[key])
    assert first["history"] == second["history"] and first["ess"] == second["ess"]
    second["q_values"].fill(100)
    assert not np.all(model.estimator.q_table.numpy() == 100)
    np.testing.assert_array_equal(first["target_policy"][0], [.3, .2, .2, .3])


def test_time_limit_updates_every_step_and_reuses_final_action():
    _, result = example.train_and_evaluate(n=5, episodes=3, max_steps=2, log_every=0)
    assert result["steps"] == result["updates"] == 6 and result["truncated_episodes"] == 3


def test_example_loads_from_notebook_directory(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root / "notebooks" / "dlfs4")
    api = runpy.run_path(str(root / "examples" / "gridworld_off_policy_n_step_sarsa.py"))
    assert all(callable(api[key]) for key in ("learn_off_policy_n_step_sarsa", "run_episode", "run_comparison"))


def test_cli_reports_policy_evaluation_not_optimal_control():
    result = subprocess.run([sys.executable, "-m", "examples.gridworld_off_policy_n_step_sarsa", "--episodes", "1",
                             "--seeds", "7", "--ns", "1", "--log-every", "0"], capture_output=True,
                            text=True, check=True, timeout=30, cwd=Path(__file__).resolve().parents[1])
    for label in ("mode=on_policy", "mode=matched", "mode=different", "Qpi_RMSE", "rho_mean", "ESS=", "Target pi:"):
        assert label in result.stdout
