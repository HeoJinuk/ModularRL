"""Real environment smoke checks; sparse-reward success is not a test threshold."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

gym = pytest.importorskip("gymnasium", reason="Install the examples extra for FrozenLake smoke tests")

from examples.frozenlake_tabular_q import train_and_evaluate
from modular_rl.component.buffer import ReplayBuffer
from modular_rl.learner import TabularQLearner


@pytest.fixture(autouse=True)
def preserve_rng():
    with torch.random.fork_rng(devices=[]):
        yield


def test_time_limit_training_loop_records_boundaries_and_closes_the_environment(monkeypatch):
    transitions, closed = [], []
    add, make = ReplayBuffer.add, gym.make

    def record_add(buffer, transition):
        transitions.append(transition)
        return add(buffer, transition)

    def record_make(*args, **kwargs):
        env = make(*args, **kwargs)
        close = env.close

        def record_close():
            closed.append(True)
            close()

        env.close = record_close
        return env

    monkeypatch.setattr(ReplayBuffer, "add", record_add)
    monkeypatch.setattr(gym, "make", record_make)
    model, results = train_and_evaluate(episodes=6, evaluation_episodes=3, max_episode_steps=1, log_every=0)
    assert model.estimator.q_table.shape == (16, 4)
    assert len(transitions) == results["training"]["steps"] == 6
    assert all(transition.truncated and not transition.terminated for transition in transitions)
    assert all(isinstance(transition.reward, float) for transition in transitions)
    assert all(transition.log_prob is not None for transition in transitions)
    assert results["training"]["truncated"] == 6 and results["training"]["terminated"] == 0
    assert results["before"]["episodes"] == results["after"]["episodes"] == 3
    assert results["before"]["truncated"] == results["after"]["truncated"] == 3
    assert closed == [True]


def test_seed_reproduces_actions_transitions_sampling_and_resulting_tables(monkeypatch):
    trace = []
    add = ReplayBuffer.add

    def record_add(buffer, transition):
        trace.append((transition.observation, transition.action.item(), transition.reward,
                      transition.next_observation, transition.terminated, transition.truncated))
        return add(buffer, transition)

    monkeypatch.setattr(ReplayBuffer, "add", record_add)
    first, results = train_and_evaluate(episodes=6, evaluation_episodes=2, seed=11, max_episode_steps=4, log_every=0)
    first_trace = trace.copy()
    trace.clear()
    second, repeated = train_and_evaluate(episodes=6, evaluation_episodes=2, seed=11, max_episode_steps=4, log_every=0)
    assert first_trace == trace and results == repeated
    torch.testing.assert_close(first.estimator.q_table, second.estimator.q_table)


def test_environment_is_closed_when_learning_raises(monkeypatch):
    closed = []
    make = gym.make

    def record_make(*args, **kwargs):
        env = make(*args, **kwargs)
        close = env.close

        def record_close():
            closed.append(True)
            close()

        env.close = record_close
        return env

    def fail(*args, **kwargs):
        raise RuntimeError("learning failed")

    monkeypatch.setattr(gym, "make", record_make)
    monkeypatch.setattr(TabularQLearner, "learn", fail)
    with pytest.raises(RuntimeError, match="learning failed"):
        train_and_evaluate(episodes=1, evaluation_episodes=1, max_episode_steps=1, log_every=0)
    assert closed == [True]


def test_frozenlake_example_runs_as_a_module_and_reports_episode_information():
    result = subprocess.run([sys.executable, "-m", "examples.frozenlake_tabular_q", "--episodes", "4",
                             "--eval-episodes", "2", "--max-steps", "1"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                            check=True, timeout=30)
    assert "FrozenLake-v1 is_slippery=False" in result.stdout
    assert "before greedy:" in result.stdout and "after greedy:" in result.stdout
    assert "training: episodes=4 steps=4" in result.stdout and "truncated=4" in result.stdout
    assert "Q-table: changed_entries=" in result.stdout
