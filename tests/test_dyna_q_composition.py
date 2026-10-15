"""Deterministic contracts for example-local model learning and Q-planning."""

import copy
from collections import deque

import numpy as np
import pytest
import torch

from examples.gridworld_dyna_q import TabularWorldModel, learn_real_n_step, plan, run_steps
from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import TabularQEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import Transition
from modular_rl.environment import make_dlfs4_gridworld_env
from modular_rl.learner import TabularQLearner
from modular_rl.model import Model


def make_learner(dtype=torch.float64, device="cpu", *, gamma=.9, alpha=.5):
    model = Model(IdentityRepresentation(), TabularQEstimator(11, 4),
                  EpsilonGreedyActionSelector(.1)).to(device=device, dtype=dtype)
    return TabularQLearner(model, TD0TargetGenerator(gamma), TabularQUpdateRule(alpha))


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
def test_real_then_planning_math_dtype_device_in_place(dtype, device):
    learner = make_learner(dtype, device)
    table = learner.model.estimator.q_table
    pointer = table.data_ptr()
    table[0, 1], table[2] = 2, table.new_tensor([4, 6, 0, 0])
    transition = Transition(0, 1, 1., 2, False, False)
    error = learner.learn(transition.to_batch())
    assert error.item() == pytest.approx(4.4)
    assert table[0, 1].item() == pytest.approx(4.2)
    world = TabularWorldModel()
    world.observe(transition)
    error = plan(world, learner, steps=1, rng=np.random.default_rng(7))
    assert error.item() == pytest.approx(2.2)
    assert table[0, 1].item() == pytest.approx(5.3)
    assert error.dtype == dtype and error.device.type == device
    assert not error.requires_grad and error.grad_fn is None
    assert pointer == table.data_ptr()


@pytest.mark.parametrize("terminated,truncated,expected", [(True, False, 2.5), (True, True, 2.5),
                                                           (False, True, 4.2)])
def test_termination_and_external_truncation(terminated, truncated, expected):
    learner = make_learner()
    table = learner.model.estimator.q_table
    table[0, 1], table[2] = 2, torch.tensor([4, 6, 0, 0])
    if terminated:
        table[2].fill_(1e20)
    world = TabularWorldModel()
    world.observe(Transition(0, 1, 3. if terminated else 1., 2, terminated, truncated))
    predicted = world.sample(np.random.default_rng(1))
    assert predicted.terminated is terminated and predicted.truncated is False
    plan(world, learner, steps=1, rng=np.random.default_rng(2))
    assert table[0, 1].item() == pytest.approx(expected)


@pytest.mark.parametrize("steps", [0, 1, 5, 10])
def test_planning_budget_and_empty_model(steps):
    learner, world, rng = make_learner(), TabularWorldModel(), np.random.default_rng(7)
    before_rng = copy.deepcopy(rng.bit_generator.state)
    assert plan(world, learner, steps=steps, rng=rng).shape == (0,)
    assert before_rng == rng.bit_generator.state
    with pytest.raises(ValueError, match="empty"):
        world.sample(rng)
    world.observe(Transition(0, 1, 1., 2, False, False))
    errors = plan(world, learner, steps=steps, rng=rng)
    assert errors.shape == (steps,)
    assert learner.model.estimator.q_table[0, 1].item() == pytest.approx(1 - .5 ** steps)
    if steps == 0:
        assert before_rng == rng.bit_generator.state


def test_model_owns_values_returns_copies_and_latest_outcome():
    world = TabularWorldModel()
    state = np.array(0, dtype=np.int32)
    reward = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    world.observe(Transition(state, np.int16(1), reward, np.int64(2), False, True))
    state[...] = 9
    with torch.no_grad():
        reward.fill_(99)
    predicted = world.sample(np.random.default_rng(1))
    assert predicted.observation == 0 and predicted.reward.item() == 1
    assert predicted.reward.dtype == torch.float64 and predicted.reward.device.type == "cpu"
    assert not predicted.reward.requires_grad
    predicted.reward.fill_(100)
    state_dict = world.state_dict()
    state_dict["outcomes"][0].reward.fill_(200)
    assert world.sample(np.random.default_rng(1)).reward.item() == 1
    world.observe(Transition(0, 1, 7., 4, True, False))
    assert len(world) == 1
    predicted = world.sample(np.random.default_rng(1))
    assert predicted.next_observation == 4 and predicted.reward.item() == 7 and predicted.terminated
    assert predicted.log_prob is None
    world.reset()
    assert len(world) == 0


def test_uniform_pairs_not_uniform_states_and_rng_reproducibility():
    world = TabularWorldModel()
    for state, action in [(0, 0), (1, 0), (1, 1), (1, 2)]:
        world.observe(Transition(state, action, float(action), 2, False, False))
    rng1, rng2 = np.random.default_rng(7), np.random.default_rng(7)
    draws = [(world.sample(rng1).observation, world.sample(rng2).observation) for _ in range(12000)]
    assert all(a == b for a, b in draws)
    assert np.mean([a == 0 for a, _ in draws]) == pytest.approx(.25, abs=.02)


@pytest.mark.parametrize("field,value,error", [
    ("observation", .5, TypeError), ("action", True, TypeError),
    ("next_observation", -1, ValueError), ("observation", [1], ValueError),
    ("reward", 1, TypeError), ("reward", float("nan"), ValueError),
    ("reward", float("inf"), ValueError), ("terminated", 1, TypeError),
    ("truncated", 0, TypeError),
])
def test_model_input_failure_preserves_state(field, value, error):
    world = TabularWorldModel()
    values = dict(observation=0, action=1, reward=1., next_observation=2, terminated=False, truncated=False)
    world.observe(Transition(**values))
    values[field] = value
    with pytest.raises(error):
        world.observe(Transition(**values))
    assert len(world) == 1
    assert world.sample(np.random.default_rng(1)).reward.item() == 1


def test_offline_pretraining_does_not_change_q_and_invalid_restore_is_atomic():
    learner, world = make_learner(), TabularWorldModel()
    before = learner.model.estimator.q_table.clone()
    world.observe(Transition(0, 1, 1., 2, False, False))
    assert torch.equal(before, learner.model.estimator.q_table)
    with pytest.raises(TypeError):
        world.load_state_dict({"outcomes": [Transition(1, 0, 2., 2, False, False),
                                            Transition(2, .5, 1., 2, False, False)]})
    assert len(world) == 1
    assert world.sample(np.random.default_rng(1)).observation == 0


def test_idle_planning_changes_only_q_not_environment_world_or_behavior_rng():
    learner, world = make_learner(), TabularWorldModel()
    env = make_dlfs4_gridworld_env(max_episode_steps=20)
    try:
        env.reset(seed=7)
        world.observe(Transition(0, 1, 1., 2, False, False))
        before = (env.unwrapped._state, env._elapsed_steps, copy.deepcopy(env.unwrapped.np_random.bit_generator.state))
        torch.manual_seed(99)
        behavior_rng = torch.get_rng_state().clone()
        outcomes = world.state_dict()
        rng = np.random.default_rng(11)
        first = plan(world, learner, steps=5, rng=rng)
        second = plan(world, learner, steps=10, rng=rng)
        assert first.numel() + second.numel() == 15
        assert learner.model.estimator.q_table[0, 1] > 0
        assert before == (env.unwrapped._state, env._elapsed_steps, env.unwrapped.np_random.bit_generator.state)
        assert torch.equal(behavior_rng, torch.get_rng_state())
        assert torch.equal(outcomes["outcomes"][0].reward, world.state_dict()["outcomes"][0].reward)
        assert len(world) == 1
    finally:
        env.close()


@pytest.mark.parametrize("gamma", [0., .9, 1.])
@pytest.mark.parametrize("terminated", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
def test_real_n_step_reverse_composition(gamma, terminated, dtype, device):
    learner = make_learner(dtype, device, gamma=gamma)
    table = learner.model.estimator.q_table
    table[0, 1], table[3, 0] = 2, 8
    window = deque(Transition(s, 1, float(s + 1), s + 1, terminated and s == 2,
                              not terminated and s == 2).to_batch() for s in range(3))
    expected = 1 + gamma * 2 + gamma ** 2 * 3 + (0 if terminated else gamma ** 3 * 8)
    error = learn_real_n_step(learner, window)
    assert error.dtype == dtype and error.device.type == device
    assert not error.requires_grad
    assert error.item() == pytest.approx(expected - 2)
    assert table[0, 1].item() == pytest.approx(2 + .5 * (expected - 2))
    assert table[3, 0] == 8
    assert len(window) == 3  # Caller removes head; no internal storage ownership.


class ChainEnv:
    """Four real steps and one scalar action, with explicit boundary selection."""

    def __init__(self, *, terminal=True):
        self.terminal = terminal
        self.calls = 0

    def reset(self, seed=None):
        self.state = 0
        return 0, {}

    def step(self, action):
        self.calls += 1
        self.state += 1
        return self.state, 1., self.terminal and self.state == 4, not self.terminal and self.state == 4, {}


@pytest.mark.parametrize("n", [1, 3, 5])
@pytest.mark.parametrize("terminal", [False, True])
def test_delayed_real_updates_model_and_planning_immediate_flush(n, terminal):
    learner, world, env = make_learner(), TabularWorldModel(), ChainEnv(terminal=terminal)
    result = run_steps(env, learner, steps=4, world_model=world, planning_steps=5,
                       rng=np.random.default_rng(8), n=n)
    assert result["steps"] == env.calls == result["model_updates"] == result["real_updates"] == 4
    assert result["planning_updates"] == 20
    assert result["flush_updates"] == min(n - 1, 4)
    assert result["partial_episode"] is None
    for row in result["trace"][:min(n - 1, 3)]:
        assert row["real_updates"] == 0 and row["planning_updates"] == 5
    assert sum(row["real_updates"] for row in result["trace"]) == 4
    assert len(world) == 4


def test_exact_step_budget_flushes_partial_episode_without_terminal_learning():
    learner, world, env = make_learner(), TabularWorldModel(), ChainEnv()
    learner.model.estimator.q_table[2].fill_(10)
    result = run_steps(env, learner, steps=2, world_model=world, n=5)
    assert result["partial_episode"] == {"steps": 2, "reward": 2.}
    assert result["episodes"] == 0 and result["flush_updates"] == 2
    assert result["trace"][-1]["targets"] == pytest.approx([10., 10.])
    assert all(not record.terminated and not record.truncated for record in world.state_dict()["outcomes"])


def test_real_and_planning_may_use_different_target_generators_same_q():
    real = make_learner(gamma=.9)
    planning = TabularQLearner(real.model, TD0TargetGenerator(0.), TabularQUpdateRule(.5))
    table = real.model.estimator.q_table
    table[0, 1], table[2, 0] = 2, 6
    transition = Transition(0, 1, 1., 2, False, False)
    real.learn(transition.to_batch())
    world = TabularWorldModel()
    world.observe(transition)
    error = plan(world, planning, steps=1, rng=np.random.default_rng(1))
    assert error.item() == pytest.approx(1 - 4.2)
    assert table[0, 1].item() == pytest.approx(2.6)


def test_duplicate_real_pairs_are_updated_in_order_and_current_targets_read_each_call():
    learner = make_learner(gamma=0.)
    window = deque([Transition(0, 1, 1., 0, False, False).to_batch(),
                    Transition(0, 1, 3., 0, True, False).to_batch()])
    assert learn_real_n_step(learner, window).item() == 1
    window.popleft()
    assert learn_real_n_step(learner, window).item() == 2.5
    assert learner.model.estimator.q_table[0, 1].item() == 1.75


@pytest.mark.parametrize("invalid", [-1, .5, True])
def test_invalid_planning_budget(invalid):
    with pytest.raises(ValueError):
        plan(TabularWorldModel(), make_learner(), steps=invalid, rng=np.random.default_rng(1))


def test_invalid_window_and_shared_q_contract():
    learner = make_learner()
    with pytest.raises(ValueError, match="at least one"):
        learn_real_n_step(learner, [])
    window = [Transition(0, 1, 1., 2, True, False).to_batch(),
              Transition(2, 1, 1., 3, False, False).to_batch()]
    with pytest.raises(ValueError, match="boundary"):
        learn_real_n_step(learner, window)
    with pytest.raises(ValueError, match="share"):
        run_steps(ChainEnv(), learner, steps=1, planning_learner=make_learner())


def test_checkpoint_q_model_world_rng_settings_pending_window(tmp_path):
    """Resume explicit window/plan composition; not a generic environment checkpoint."""
    torch.manual_seed(7)
    learner, world, rng = make_learner(), TabularWorldModel(), np.random.default_rng(11)
    pending = deque([Transition(0, 1, 1., 1, False, False).to_batch(),
                     Transition(1, 2, 2., 2, False, False).to_batch()])
    for row in (Transition(0, 1, 1., 1, False, False), Transition(1, 2, 2., 2, False, False)):
        world.observe(row)
    plan(world, learner, steps=3, rng=rng)
    settings = {"n": 3, "gamma": .9, "alpha": .5, "planning_steps": 5, "epsilon": .1}
    checkpoint = copy.deepcopy({"q_model": learner.model.state_dict(), "world": world.state_dict(),
                                "planning_rng": rng.bit_generator.state, "behavior_rng": torch.get_rng_state(),
                                "settings": settings, "pending": pending})
    path = tmp_path / "dyna.pt"
    torch.save(checkpoint, path)

    def continue_run(learner, world, rng, pending):
        selected = learner.model.act(2)
        row = Transition(2, selected.action.item(), 3., 3, True, False, selected.log_prob)
        pending.append(row.to_batch())
        real_errors = [learn_real_n_step(learner, pending)]
        pending.popleft()
        world.observe(row)
        planned = plan(world, learner, steps=5, rng=rng)
        while pending:
            real_errors.append(learn_real_n_step(learner, pending))
            pending.popleft()
        return selected.action, torch.cat(real_errors), planned

    expected = continue_run(learner, world, rng, pending)
    expected_behavior = torch.get_rng_state().clone()
    restored = torch.load(path, weights_only=False)  # Locally created, trusted example checkpoint.
    resumed, world2, rng2 = make_learner(), TabularWorldModel(), np.random.default_rng()
    resumed.model.load_state_dict(restored["q_model"])
    world2.load_state_dict(restored["world"])
    rng2.bit_generator.state = restored["planning_rng"]
    torch.set_rng_state(restored["behavior_rng"])
    actual = continue_run(resumed, world2, rng2, restored["pending"])
    assert restored["settings"] == settings
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert torch.equal(learner.model.estimator.q_table, resumed.model.estimator.q_table)
    assert torch.equal(expected_behavior, torch.get_rng_state())
    assert rng.bit_generator.state == rng2.bit_generator.state
