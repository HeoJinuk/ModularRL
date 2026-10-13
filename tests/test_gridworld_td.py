"""Shared sampling, chronological B=1 TD and reproducible comparison metrics."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from examples.gridworld_mc import train_and_compare as mc_comparison
from examples.gridworld_td import train_and_compare
from modular_rl.data import Episode, Transition
from modular_rl.learner import MCPredictionLearner, TabularTD0PredictionLearner


def test_shared_episode_comparison_matches_existing_mc_and_has_independent_tables():
    learners, results = train_and_compare(episodes=20, seed=7, log_every=0)
    _, mc = mc_comparison(episodes=20, seed=7, log_every=0)
    assert results["steps"] == mc["steps"] and results["episodes"] == 20
    np.testing.assert_array_equal(results["start_counts"], mc["start_counts"])
    np.testing.assert_array_equal(results["reference"], mc["reference"])
    for mode in ("first", "every"):
        for field in ("values", "visit_counts"):
            np.testing.assert_array_equal(results["methods"][mode][field], mc[mode][field])
        assert results["methods"][mode]["rmse"] == mc[mode]["rmse"]
    assert len({learner.model.estimator.v_table.data_ptr() for learner in learners.values()}) == 5
    assert learners["first"].visit_counts is not learners["every"].visit_counts
    for metrics in results["methods"].values():
        assert metrics["values"].shape == (16,)
        np.testing.assert_array_equal(metrics["values"][[0, 15]], [0, 0])
        expected = np.sqrt(np.mean((metrics["values"][1:15] - results["reference"][1:15]) ** 2))
        assert metrics["rmse"] == expected
    assert results["history"][0]["episodes"] == results["history"][0]["steps"] == 0
    assert results["history"][-1]["steps"] == results["steps"]


def test_each_td_learner_gets_same_episode_in_chronological_single_transition_calls(monkeypatch):
    episode = Episode((Transition(1, 0, 1., 2, False, False),
                       Transition(2, 0, 2., 1, False, False), Transition(1, 0, 3., 0, True, False)))
    mc_episodes, td_rows = [], {}
    mc_learn, td_learn = MCPredictionLearner.learn, TabularTD0PredictionLearner.learn

    def record_mc(self, received):
        mc_episodes.append(received)
        return mc_learn(self, received)

    def record_td(self, batch):
        assert batch.observation.shape == (1,)
        td_rows.setdefault(self.update_rule.alpha, []).append((batch.observation.item(), batch.reward.item(),
                                                              batch.next_observation.item(), batch.terminated.item()))
        return td_learn(self, batch)

    monkeypatch.setattr("examples.gridworld_td.collect_episode", lambda *args, **kwargs: episode)
    monkeypatch.setattr(MCPredictionLearner, "learn", record_mc)
    monkeypatch.setattr(TabularTD0PredictionLearner, "learn", record_td)
    _, results = train_and_compare(episodes=1, log_every=0)
    assert len(mc_episodes) == 2 and all(received is episode for received in mc_episodes)
    for rows in td_rows.values():
        assert rows == [(1, 1., 2, False), (2, 2., 1, False), (1, 3., 0, True)]
    assert set(td_rows) == {0.01, 0.05, 0.1} and results["steps"] == 3
    assert results["methods"]["first"]["values"][1] == 6.
    assert results["methods"]["every"]["values"][1] == 4.5


def test_seed_reproduces_experience_td_tables_and_episode_step_rmse_history():
    _, first = train_and_compare(episodes=20, seed=11, log_every=5)
    _, second = train_and_compare(episodes=20, seed=11, log_every=5)
    assert first["history"] == second["history"]
    assert [row["episodes"] for row in first["history"]] == [0, 5, 10, 15, 20]
    assert first["steps"] == second["steps"]
    np.testing.assert_array_equal(first["start_counts"], second["start_counts"])
    for name in first["methods"]:
        np.testing.assert_array_equal(first["methods"][name]["values"], second["methods"][name]["values"])


def test_distinct_custom_alphas_keep_separate_models_and_metrics():
    alphas = (0.010000001, 0.010000002)
    learners, results = train_and_compare(episodes=1, alphas=alphas, log_every=0)
    td = [learner for learner in learners.values() if isinstance(learner, TabularTD0PredictionLearner)]
    assert len(td) == 2 and {learner.update_rule.alpha for learner in td} == set(alphas)
    assert len(results["methods"]) == 4


def test_truncated_episode_aborts_before_any_mc_or_td_update(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No learner may process a censored comparison episode")

    episode = Episode((Transition(1, 0, -1., 1, False, True),))
    monkeypatch.setattr("examples.gridworld_td.collect_episode", lambda *args, **kwargs: episode)
    monkeypatch.setattr(MCPredictionLearner, "learn", forbidden)
    monkeypatch.setattr(TabularTD0PredictionLearner, "learn", forbidden)
    with pytest.raises(RuntimeError, match="truncated.*no learner processed"):
        train_and_compare(episodes=1, log_every=0)


def test_comparison_cli_reports_values_alpha_seed_and_multiseed_rmse():
    result = subprocess.run(
        [sys.executable, "-m", "examples.gridworld_td", "--episodes", "5", "--seeds", "7", "11", "--log-every", "0"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True, timeout=30,
    )
    for seed in (7, 11):
        assert f"gamma=1 seed={seed} episodes=5 steps=" in result.stdout
    assert "DP uniform-policy V:" in result.stdout and "14 nonterminal states" in result.stdout
    for name in ("first", "every", "td(alpha=0.01)", "td(alpha=0.05)", "td(alpha=0.1)"):
        assert f"{name} V (RMSE=" in result.stdout
    assert "mean +/- std (ddof=0)" in result.stdout
