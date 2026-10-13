"""Online collection, fair per-step selection, evaluation and example reuse."""

import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from examples import gridworld_double_q_learning as example


def test_short_gridworld_run_reports_independent_tables_policy_and_dp_reference():
    (model1, model2, selector), result = example.train_and_evaluate(episodes=20, seed=11, log_every=0)
    assert model1.action_selector is model2.action_selector is None
    assert model1.estimator.q_table.untyped_storage().data_ptr() != model2.estimator.q_table.untyped_storage().data_ptr()
    assert result["episodes"] == 20 and result["algorithm"] == "double_q"
    assert result["training_rewards"].shape == result["episode_lengths"].shape == (20,)
    assert result["episode_lengths"].sum() == result["update_counts"].sum() == result["steps"]
    assert np.all(result["update_counts"] > 0)
    assert result["terminated_episodes"] + result["truncated_episodes"] == 20
    assert selector.epsilon == pytest.approx(.1) and result["alpha"] == .8 and result["gamma"] == .9
    assert result["before"]["truncated"] and result["before"]["steps"] == 20
    assert result["dp_steps"] == len(result["dp_path"]) - 1 == 5
    assert result["dp_values"][example.DLFS4_START_STATE] == pytest.approx(.9**4)
    for model, name in ((model1, "q1_values"), (model2, "q2_values")):
        assert result[name].shape == (11, 4) and np.isfinite(result[name]).all()
        np.testing.assert_array_equal(result[name], model.estimator.q_table.numpy())
    average = (result["q1_values"] + result["q2_values"]) / 2
    np.testing.assert_array_equal(result["q_values"], average)
    nonterminal = np.setdiff1d(np.arange(11), [3])
    assert result["max_q_rmse"] == pytest.approx(np.sqrt(np.mean((average.max(axis=1)[nonterminal] - result["dp_values"][nonterminal])**2)))
    result["q1_values"].fill(123)
    assert not torch.all(model1.estimator.q_table == 123)


def test_same_seed_reproduces_both_tables_update_counts_and_behavior_trajectory():
    _, first = example.train_and_evaluate(episodes=20, seed=11, log_every=10)
    _, second = example.train_and_evaluate(episodes=20, seed=11, log_every=10)
    assert first["steps"] == second["steps"] and first["history"] == second["history"]
    for name in ("q1_values", "q2_values", "training_rewards", "episode_lengths", "update_counts"):
        np.testing.assert_array_equal(first[name], second[name])


def test_online_collection_uses_q_sum_log_prob_and_one_coin_per_step(monkeypatch):
    original_act, original_choose, original_learn = example.act_double_q, example.choose_update_first, example.learn_double_q
    sampled, choices, learned, rngs = [], [], [], []

    def act(model1, model2, observation, selector, *, deterministic=False):
        output = original_act(model1, model2, observation, selector, deterministic=deterministic)
        if not deterministic:
            pi = selector.probabilities(example.combined_estimate(model1, model2, observation))
            torch.testing.assert_close(output.log_prob, pi[output.action].log())
            sampled.append(output)
        return output

    def coin(rng):
        value = original_choose(rng)
        choices.append(value)
        rngs.append(rng)
        return value

    def learn(model1, model2, batch, target, rule, *, update_first):
        assert len(sampled) == len(choices) == len(learned) + 1
        assert batch.observation.shape == batch.action.shape == (1,)
        torch.testing.assert_close(batch.log_prob[0], sampled[-1].log_prob)
        other = model2 if update_first else model1
        unchanged = other.estimator.q_table.clone()
        error = original_learn(model1, model2, batch, target, rule, update_first=update_first)
        torch.testing.assert_close(other.estimator.q_table, unchanged, rtol=0, atol=0)
        learned.append(update_first)
        return error

    monkeypatch.setattr(example, "act_double_q", act)
    monkeypatch.setattr(example, "choose_update_first", coin)
    monkeypatch.setattr(example, "learn_double_q", learn)
    _, result = example.train_and_evaluate(episodes=3, seed=11, log_every=1)
    assert len(sampled) == len(choices) == len(learned) == result["steps"]
    assert all(rng is rngs[0] for rng in rngs)
    np.testing.assert_array_equal(result["update_counts"], [sum(choices), len(choices)-sum(choices)])


def test_real_time_limit_keeps_bootstrap_uses_final_observation_and_closes_environments(monkeypatch):
    original_make, original_learn = example.make_dlfs4_gridworld_env, example.learn_double_q
    closed, observations, learned = [], [], []

    def make(**kwargs):
        env = original_make(**kwargs)
        close, step = env.close, env.step

        def finish():
            closed.append(env)
            close()

        def record(action):
            result = step(action)
            if kwargs["max_episode_steps"] == 1:
                observations.append(result[0])
            return result

        monkeypatch.setattr(env, "close", finish)
        monkeypatch.setattr(env, "step", record)
        return env

    def learn(model1, model2, batch, target, rule, *, update_first):
        assert batch.truncated.item() and not batch.terminated.item()
        assert batch.next_observation.item() == observations[-1]
        for model in (model1, model2):
            model.estimator.q_table.fill_(1.)
        error = original_learn(model1, model2, batch, target, rule, update_first=update_first)
        torch.testing.assert_close(error, batch.reward.double() + target.gamma - 1.)
        learned.append(update_first)
        return error

    monkeypatch.setattr(example, "make_dlfs4_gridworld_env", make)
    monkeypatch.setattr(example, "learn_double_q", learn)
    _, result = example.train_and_evaluate(episodes=3, max_steps=1, log_every=0)
    assert len(learned) == len(observations) == result["steps"] == result["truncated_episodes"] == 3
    assert len(closed) == 2 and closed[0] is not closed[1]


def test_module_loads_for_notebook_without_importing_an_examples_package(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root / "notebooks" / "dlfs4")
    api = runpy.run_path(str(root / "examples" / "gridworld_double_q_learning.py"))
    for name in ("combined_estimate", "act_double_q", "learn_double_q", "train_and_evaluate"):
        assert callable(api[name])


def test_cli_reuses_existing_comparators_and_reports_both_q_tables_and_paths():
    result = subprocess.run([sys.executable, "-m", "examples.gridworld_double_q_learning", "--episodes", "5",
                             "--seeds", "11", "--compare-all"], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, check=True, timeout=30)
    for label in ("DLFS4 double_q:", "DLFS4 q_learning:", "DLFS4 sarsa:", "DLFS4 expected_sarsa:",
                  "before greedy:", "after greedy:", "DP optimal path:", "DP optimal: steps=5", "Q1/Q2 update counts:",
                  "q1_values (", "q2_values (", "q_values ("):
        assert label in result.stdout
