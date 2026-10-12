"""Short real CartPole loops; full learning performance is reported separately."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

gym = pytest.importorskip("gymnasium", reason="Install the examples extra for CartPole smoke tests")

from examples.cartpole_dqn import train_and_evaluate
from modular_rl.component.buffer import ReplayBuffer
from modular_rl.learner import DQNLearner


@pytest.fixture(autouse=True)
def preserve_rng():
    with torch.random.fork_rng(devices=[]):
        yield


def test_real_time_limit_loop_keeps_final_observation_and_external_training_schedule(monkeypatch):
    transitions, update_steps, sync_steps, closed = [], [], [], []
    last_observation = []
    add, learn, sync, make = ReplayBuffer.add, DQNLearner.learn, DQNLearner.sync_target, gym.make

    def record_make(*args, **kwargs):
        env = make(*args, **kwargs)
        step, close = env.step, env.close

        def record_step(action):
            result = step(action)
            last_observation[:] = [result[0].copy()]
            return result

        def record_close():
            closed.append(True)
            close()

        env.step, env.close = record_step, record_close
        return env

    def record_add(buffer, transition):
        np.testing.assert_array_equal(transition.next_observation, last_observation[0])
        transitions.append(transition)
        return add(buffer, transition)

    def record_learn(learner, batch):
        update_steps.append(len(transitions))
        return learn(learner, batch)

    def record_sync(learner):
        sync_steps.append(len(transitions))
        return sync(learner)

    monkeypatch.setattr(gym, "make", record_make)
    monkeypatch.setattr(ReplayBuffer, "add", record_add)
    monkeypatch.setattr(DQNLearner, "learn", record_learn)
    monkeypatch.setattr(DQNLearner, "sync_target", record_sync)
    threads = torch.get_num_threads()
    model, results = train_and_evaluate(episodes=6, evaluation_episodes=3, max_episode_steps=1,
                                        batch_size=2, warmup_steps=3, target_sync_steps=2, log_every=0)
    assert len(transitions) == results["training"]["steps"] == 6
    assert all(transition.truncated and not transition.terminated for transition in transitions)
    assert all(transition.observation.shape == transition.next_observation.shape == (4,) for transition in transitions)
    assert all(transition.observation.dtype == np.float32 and isinstance(transition.reward, float)
               for transition in transitions)
    assert all(transition.action.shape == () and transition.action.dtype == torch.long for transition in transitions)
    assert all(transition.log_prob is not None and not transition.log_prob.requires_grad for transition in transitions)
    assert update_steps == [3, 4, 5, 6] and sync_steps == [0, 2, 4, 6]
    assert results["training"]["updates"] == 4 and results["training"]["target_syncs"] == 3
    assert results["training"]["max_parameter_change"] > 0
    assert results["before"]["truncated"] == results["after"]["truncated"] == 3
    assert model.training and closed == [True] and torch.get_num_threads() == threads


def test_seed_reproduces_short_training_and_evaluation():
    arguments = dict(episodes=5, evaluation_episodes=2, seed=11, max_episode_steps=3,
                     batch_size=2, warmup_steps=2, target_sync_steps=3, epsilon_decay_steps=8, log_every=0)
    first, results = train_and_evaluate(**arguments)
    second, repeated = train_and_evaluate(**arguments)
    assert results == repeated
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key])


def test_environment_and_thread_setting_are_restored_when_learning_raises(monkeypatch):
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
    monkeypatch.setattr(DQNLearner, "learn", fail)
    threads = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="learning failed"):
        train_and_evaluate(episodes=1, evaluation_episodes=1, max_episode_steps=1,
                           batch_size=1, warmup_steps=1, log_every=0)
    assert closed == [True] and torch.get_num_threads() == threads


def test_cartpole_module_reports_rewards_loss_td_errors_and_evaluation():
    result = subprocess.run([sys.executable, "-m", "examples.cartpole_dqn", "--episodes", "2",
                             "--eval-episodes", "2", "--max-steps", "2", "--batch-size", "1",
                             "--warmup-steps", "1", "--target-sync-steps", "2"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                            check=True, timeout=30)
    assert "CartPole-v1 seed=" in result.stdout
    assert "before greedy: evaluation_mean_reward=" in result.stdout
    assert "after greedy: evaluation_mean_reward=" in result.stdout
    assert "episode=2/2 reward=" in result.stdout and "epsilon=" in result.stdout
    assert "loss=" in result.stdout and "td_mean=" in result.stdout and "td_abs_mean=" in result.stdout
    assert "training: episodes=2 steps=4 updates=4 target_syncs=2" in result.stdout
