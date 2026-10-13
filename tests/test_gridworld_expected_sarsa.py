"""Short Gymnasium runs verify composition, timing and unchanged comparators."""

import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from examples import gridworld_expected_sarsa as example
from examples import gridworld_sarsa as previous


@pytest.mark.parametrize("algorithm", ["expected_sarsa", "q_learning", "sarsa"])
def test_short_runs_report_same_budget_and_independent_training_and_evaluation(algorithm):
    model, result = example.train_and_evaluate(algorithm, episodes=20, seed=11, log_every=0)
    assert result["episodes"] == 20 and result["algorithm"] == algorithm
    assert result["episode_lengths"].sum() == result["steps"] >= 20
    assert result["training_rewards"].shape == result["episode_lengths"].shape == (20,)
    assert result["terminated_episodes"] + result["truncated_episodes"] == 20
    assert result["q_values"].shape == (11, 4) and np.isfinite(result["q_values"]).all()
    np.testing.assert_array_equal(result["q_values"], model.estimator.q_table.numpy())
    assert result["dp_steps"] == 5 and result["dp_values"][example.DLFS4_START_STATE] == pytest.approx(.9**4)
    assert (result["gamma"], result["alpha"], result["epsilon"]) == (.9, .8, .1)
    assert result["before"]["truncated"] and result["before"]["steps"] == 20
    assert np.isfinite(result["max_q_rmse"])
    result["q_values"].fill(123.)
    assert not torch.all(model.estimator.q_table == 123.)  # Report owns its snapshot.


def test_expected_sarsa_needs_no_learner_and_is_reproducible(monkeypatch):
    def unexpected_learner(*args, **kwargs):
        pytest.fail("Expected SARSA must compose components without constructing a learner")

    monkeypatch.setattr(example, "TabularQLearner", unexpected_learner)
    monkeypatch.setattr(example, "TabularSARSALearner", unexpected_learner)
    _, first = example.train_and_evaluate(episodes=20, seed=11, log_every=10)
    _, second = example.train_and_evaluate(episodes=20, seed=11, log_every=10)
    assert first["steps"] == second["steps"] and first["history"] == second["history"]
    for name in ("q_values", "training_rewards", "episode_lengths"):
        np.testing.assert_array_equal(first[name], second[name])


@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa"])
def test_comparators_preserve_previous_example_learning_and_sampling(algorithm):
    _, result = example.train_and_evaluate(algorithm, episodes=20, seed=11, log_every=0)
    _, old = previous.train_and_evaluate(algorithm, episodes=20, seed=11, log_every=0)
    assert result["steps"] == old["steps"] and result["after"] == old["after"]
    for name in ("q_values", "training_rewards", "episode_lengths"):
        np.testing.assert_array_equal(result[name], old[name])


def test_time_limit_bootstraps_final_observation_without_sampling_next_action(monkeypatch):
    original_make, original_learn, original_act = example.make_dlfs4_gridworld_env, example.learn_expected_sarsa, example.Model.act
    closed, batches, selectors, sampled = [], [], [], []

    def make(**kwargs):
        env = original_make(**kwargs)
        close = env.close

        def finish():
            closed.append(env)
            close()

        monkeypatch.setattr(env, "close", finish)
        return env

    def act(model, observation, *, deterministic=False):
        if not deterministic:
            sampled.append(model.action_selector)
        return original_act(model, observation, deterministic=deterministic)

    def learn(model, batch, target, rule, *, policy_selector):
        assert batch.truncated.item() and not batch.terminated.item()
        assert policy_selector is model.action_selector is sampled[-1]
        model.estimator.q_table.fill_(1.)
        rng = torch.random.get_rng_state().clone()
        error = original_learn(model, batch, target, rule, policy_selector=policy_selector)
        torch.testing.assert_close(error, batch.reward.double() + target.gamma - 1.)
        assert torch.equal(rng, torch.random.get_rng_state())
        selectors.append(policy_selector)
        batches.append(batch)
        return error

    monkeypatch.setattr(example, "make_dlfs4_gridworld_env", make)
    monkeypatch.setattr(example, "learn_expected_sarsa", learn)
    monkeypatch.setattr(example.Model, "act", act)
    _, result = example.train_and_evaluate(episodes=3, max_steps=1, log_every=0)
    assert len(batches) == len(sampled) == len(selectors) == 3
    assert result["truncated_episodes"] == result["steps"] == 3
    assert len(closed) == 2 and closed[0] is not closed[1]


def test_example_loads_from_notebook_directory_without_an_installed_examples_package(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root / "notebooks" / "dlfs4")
    api = runpy.run_path(str(root / "examples" / "gridworld_expected_sarsa.py"))
    assert callable(api["learn_expected_sarsa"]) and callable(api["train_and_evaluate"])


def test_cli_reports_all_algorithms_q_tables_and_dp_reference():
    result = subprocess.run([sys.executable, "-m", "examples.gridworld_expected_sarsa", "--episodes", "5", "--seeds", "11"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                            check=True, timeout=30)
    for label in ("DLFS4 expected_sarsa:", "DLFS4 q_learning:", "DLFS4 sarsa:", "before greedy:", "after greedy:",
                  "alpha=0.800", "length=", "DP optimal: steps=5", "Q (state rows;"):
        assert label in result.stdout
