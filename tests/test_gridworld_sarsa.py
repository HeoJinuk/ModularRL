"""Online next-action timing, reuse and separate seeded comparison runs."""

import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from examples import gridworld_sarsa as example
from modular_rl.data import ActionOutput, TabularMDP, TransitionOutcome as Outcome
from modular_rl.environment import TabularEnv
from modular_rl.learner import TabularSARSALearner


@pytest.mark.parametrize("terminated,cap,expected", [
    (False, 5, [("select", 0), ("step", 0), ("select", 1), ("learn", 1), ("step", 1), ("learn", None)]),
    (False, 1, [("select", 0), ("step", 0), ("select", 1), ("learn", 1)]),
    (True, 1, [("select", 0), ("step", 0), ("learn", None)]),
])
def test_actual_next_action_is_selected_once_before_learning_and_carried_to_next_step(terminated, cap, expected, monkeypatch):
    mdp = TabularMDP(1, 4, (((Outcome(1., 0, 2., terminated),), (Outcome(1., 0, 1., True),),
                            (Outcome(1., 0, 0., True),), (Outcome(1., 0, 0., True),)),))
    events, envs, models = [], [], []
    original_model, original_learn = example.Model, TabularSARSALearner.learn

    def make(**kwargs):
        env = gym.wrappers.TimeLimit(TabularEnv(mdp, start_state=0), max_episode_steps=kwargs["max_episode_steps"])
        if not envs:
            step = env.step

            def record(action):
                events.append(("step", action))
                return step(action)

            monkeypatch.setattr(env, "step", record)
        envs.append(env)
        return env

    def model(*args, **kwargs):
        result = original_model(*args, **kwargs)
        result.estimator.q_table[0] = torch.tensor([4., 2., 0., 0.])
        act = result.act
        actions = iter((0, 1))

        def select(observation, *, deterministic=False):
            if deterministic:
                return act(observation, deterministic=True)
            action = next(actions)
            events.append(("select", action))
            assert result.estimator.q_table[0, 0] == 4  # Both samples precede the first update.
            return ActionOutput(torch.tensor(action), torch.tensor(np.log(.5)))

        monkeypatch.setattr(result, "act", select)
        models.append(result)
        return result

    def learn(learner, batch, *, next_action=None):
        events.append(("learn", next_action))
        return original_learn(learner, batch, next_action=next_action)

    monkeypatch.setattr(example, "Model", model)
    monkeypatch.setattr(example, "DLFS4_STATES", ((0, 0),))
    monkeypatch.setattr(example, "make_dlfs4_gridworld_env", make)
    monkeypatch.setattr(TabularSARSALearner, "learn", learn)
    learner, result = example.train_and_evaluate(episodes=1, alpha=.5, max_steps=cap, log_every=0)
    assert events == expected and learner.model is models[0]
    assert result["steps"] == (2 if cap == 5 else 1)
    assert learner.model.estimator.q_table[0, 0].item() == pytest.approx(3. if terminated else 3.9)
    if cap == 1:
        assert result["truncated_episodes"] == 1  # Final next action is never executed after truncation.
        assert result["terminated_episodes"] == int(terminated)


@pytest.mark.parametrize("algorithm", ["sarsa", "q_learning"])
def test_short_gridworld_comparison_reports_exploration_and_separate_evaluation(algorithm):
    learner, result = example.train_and_evaluate(algorithm, episodes=20, log_every=0)
    assert result["algorithm"] == algorithm and result["episodes"] == 20 and result["steps"] >= 20
    assert result["episode_lengths"].sum() == result["steps"]
    assert result["training_rewards"].shape == result["episode_lengths"].shape == (20,)
    assert result["q_values"].shape == (11, 4) and np.isfinite(result["q_values"]).all()
    np.testing.assert_array_equal(result["q_values"], learner.model.estimator.q_table.numpy())
    assert result["epsilon"] == .1 and result["alpha"] == .8 and result["dp_steps"] == 5
    assert result["terminated_episodes"] == 20 and result["truncated_episodes"] == 0
    assert result["before"]["truncated"] and result["before"]["steps"] == 20
    assert np.isfinite(result["max_q_rmse"])


@pytest.mark.parametrize("algorithm", ["sarsa", "q_learning"])
def test_rng_seed_reproduces_the_same_algorithm_run(algorithm):
    _, first = example.train_and_evaluate(algorithm, episodes=20, seed=11, log_every=10)
    _, second = example.train_and_evaluate(algorithm, episodes=20, seed=11, log_every=10)
    assert first["steps"] == second["steps"] and first["history"] == second["history"]
    for name in ("q_values", "training_rewards", "episode_lengths"):
        np.testing.assert_array_equal(first[name], second[name])


def test_real_time_limit_keeps_bootstrap_and_closes_environments(monkeypatch):
    closed, calls = [], []
    original_make, original_learn = example.make_dlfs4_gridworld_env, TabularSARSALearner.learn

    def make(**kwargs):
        env = original_make(**kwargs)
        close = env.close

        def finish():
            closed.append(env)
            close()

        monkeypatch.setattr(env, "close", finish)
        return env

    def learn(learner, batch, *, next_action=None):
        assert batch.truncated.item() and not batch.terminated.item() and next_action is not None
        calls.append(next_action)
        return original_learn(learner, batch, next_action=next_action)

    monkeypatch.setattr(example, "make_dlfs4_gridworld_env", make)
    monkeypatch.setattr(TabularSARSALearner, "learn", learn)
    _, result = example.train_and_evaluate(episodes=3, max_steps=1, log_every=0)
    assert len(calls) == 3 and len(closed) == 2 and closed[0] is not closed[1]
    assert result["truncated_episodes"] == 3 and result["steps"] == 3


def test_cli_reports_both_algorithms_and_dp_reference():
    result = subprocess.run([sys.executable, "-m", "examples.gridworld_sarsa", "--episodes", "10", "--seeds", "7"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                            check=True, timeout=30)
    for label in ("DLFS4 sarsa:", "DLFS4 q_learning:", "before greedy:", "after greedy:", "alpha=0.800",
                  "length=", "DP optimal: steps=5", "Q (state rows;"):
        assert label in result.stdout
