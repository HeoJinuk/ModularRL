"""Gymnasium collection compatibility and reproducible control smoke tests."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from examples import gridworld_off_policy_mc_control as example
from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import TabularQEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import MonteCarloReturnGenerator
from modular_rl.data import Episode, Transition
from modular_rl.environment import collect_episode, make_dlfs4_gridworld_env
from modular_rl.learner import OffPolicyMCControlLearner
from modular_rl.model import Model


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_step_by_step_and_helper_record_the_same_behavior_and_control_updates(dtype):
    env = make_dlfs4_gridworld_env()
    models = [Model(IdentityRepresentation(), TabularQEstimator(11, 4),
                    EpsilonGreedyActionSelector(.1)).to(dtype=dtype) for _ in range(2)]
    try:
        torch.manual_seed(7)
        observation, _ = env.reset(seed=7)
        transitions = []
        while True:
            output = models[0].act(observation)
            action = output.action.item()
            log_prob = output.log_prob.detach().cpu().clone()
            next_observation, reward, terminated, truncated, _ = env.step(action)
            transitions.append(Transition(observation, action, reward, next_observation,
                                          terminated, truncated, log_prob))
            if terminated or truncated:
                break
            observation = next_observation
        manual = Episode(tuple(transitions))
        torch.manual_seed(7)
        collected = collect_episode(env, models[1].act, seed=7)
        assert manual.terminated and collected.terminated and len(manual) == len(collected)
        for left, right in zip(manual.transitions, collected.transitions):
            for name in ("observation", "action", "reward", "next_observation", "terminated", "truncated", "log_prob"):
                torch.testing.assert_close(getattr(left, name), getattr(right, name), rtol=0, atol=0)
            assert left.log_prob.exp().item() == pytest.approx(.25)  # All initial Q actions tie.
        learners = [OffPolicyMCControlLearner(model, MonteCarloReturnGenerator(.9)) for model in models]
        torch.testing.assert_close(learners[0].learn(manual), learners[1].learn(collected), rtol=0, atol=0)
        torch.testing.assert_close(models[0].estimator.q_table, models[1].estimator.q_table, rtol=0, atol=0)
        torch.testing.assert_close(learners[0].weight_sums, learners[1].weight_sums, rtol=0, atol=0)
    finally:
        env.close()


def test_short_example_uses_one_model_c_and_separate_deterministic_evaluation():
    learner, result = example.train_and_evaluate(episodes=20, log_every=0)
    assert isinstance(learner, OffPolicyMCControlLearner)
    assert result["episodes"] == 20 and result["steps"] >= 20 and result["truncated_episodes"] == 0
    assert result["training_rewards"].shape == (20,)
    assert result["q_values"].shape == result["weight_sums"].shape == (11, 4)
    assert result["greedy_actions"].shape == (11,) and result["epsilon"] == .1
    np.testing.assert_array_equal(result["q_values"], learner.model.estimator.q_table.numpy())
    np.testing.assert_array_equal(result["weight_sums"], learner.weight_sums.numpy())
    np.testing.assert_array_equal(result["greedy_actions"], result["q_values"].argmax(axis=1))
    assert np.isfinite(result["q_values"]).all() and np.isfinite(result["weight_sums"]).all()
    assert (result["weight_sums"] >= 0).all() and result["weight_sums"][3].sum() == 0
    assert result["dp_steps"] == 5 and result["dp_values"][7] == pytest.approx(.9 ** 4)
    assert np.isfinite(result["max_q_rmse"])


def test_same_seed_reproduces_q_c_rewards_and_evaluation():
    _, first = example.train_and_evaluate(episodes=20, seed=11, log_every=10)
    _, second = example.train_and_evaluate(episodes=20, seed=11, log_every=10)
    assert first["steps"] == second["steps"] and first["history"] == second["history"]
    for name in ("q_values", "weight_sums", "training_rewards", "greedy_actions"):
        np.testing.assert_array_equal(first[name], second[name])


def test_q_is_fixed_during_collection_and_log_prob_is_present(monkeypatch):
    original_collect, original_learn = example.collect_episode, OffPolicyMCControlLearner.learn
    models, lengths = [], []
    original_model = example.Model

    def model(*args, **kwargs):
        result = original_model(*args, **kwargs)
        models.append(result)
        return result

    def collect(env, callback, **kwargs):
        table = models[0].estimator.q_table
        fixed_q = table.clone()

        def action(state):
            torch.testing.assert_close(table, fixed_q, rtol=0, atol=0)
            return callback(state)

        sample = original_collect(env, action, **kwargs)
        torch.testing.assert_close(table, fixed_q, rtol=0, atol=0)
        return sample

    def learn(learner, sample):
        assert sample.terminated and all(t.log_prob is not None for t in sample.transitions)
        lengths.append(len(sample))
        return original_learn(learner, sample)

    monkeypatch.setattr(example, "Model", model)
    monkeypatch.setattr(example, "collect_episode", collect)
    monkeypatch.setattr(OffPolicyMCControlLearner, "learn", learn)
    learner, _ = example.train_and_evaluate(episodes=5, log_every=1)
    assert len(lengths) == 5 and len(models) == 1 and learner.model is models[0]
    assert torch.count_nonzero(learner.weight_sums) > 0


def test_real_truncation_aborts_before_learning_and_closes_both_environments(monkeypatch):
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
    monkeypatch.setattr(OffPolicyMCControlLearner, "learn", lambda *args: pytest.fail("no truncated MC update"))
    with pytest.raises(RuntimeError, match="truncated.*did not learn"):
        example.train_and_evaluate(episodes=1, max_steps=1, log_every=0)
    assert len(closed) == 2 and closed[0] is not closed[1]


def test_example_requires_positive_behavior_exploration():
    with pytest.raises(ValueError, match="epsilon.*cover"):
        example.train_and_evaluate(epsilon=0)


def test_cli_reports_behavior_greedy_dp_and_weights():
    result = subprocess.run([sys.executable, "-m", "examples.gridworld_off_policy_mc_control", "--episodes", "10"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                            check=True, timeout=30)
    for label in ("DLFS4 Off-policy MC Control:", "before greedy:", "after greedy:", "epsilon=0.100",
                  "DP optimal: steps=5", "C cumulative weights:", "Deterministic greedy action IDs:"):
        assert label in result.stdout
