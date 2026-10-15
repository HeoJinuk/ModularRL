"""Dyna-DQN arithmetic, shared optimizer, sampling and n-step metadata contracts."""

from collections import deque
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from examples.gridworld_dyna_dqn import (NStepExperience, create_learner, learn_real_n_step,
                                       make_n_step_experience, plan, run_steps, synthetic_batch)
from examples.gridworld_dyna_q import TabularWorldModel
from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.buffer import ReplayBuffer
from modular_rl.component.estimator import DiscreteQEstimator
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TDLossUpdateRule
from modular_rl.data import TDLossOutput, Transition, TransitionBatch
from modular_rl.environment import make_dlfs4_gridworld_env
from modular_rl.learner import DQNLearner
from modular_rl.model import Model


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def linear_learner(dtype=torch.float64, device="cpu", gamma=.9):
    model = Model(nn.Embedding.from_pretrained(torch.eye(3), freeze=True),
                  DiscreteQEstimator(3, 2, network=nn.Linear(3, 2, bias=False)),
                  EpsilonGreedyActionSelector(.1)).to(device=device, dtype=dtype)
    target = deepcopy(model)
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.1, momentum=.9)
    learner = DQNLearner(model, target, optimizer, TD0TargetGenerator(gamma),
                         TDLossUpdateRule(nn.SmoothL1Loss(reduction="none")))
    with torch.no_grad():
        model.estimator.network.weight.copy_(torch.tensor([[0., 0., 44.], [2., 0., 66.]]))
        target.estimator.network.weight.copy_(torch.tensor([[0., 0., 4.], [0., 0., 6.]]))
    return learner


def assert_tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_tree_equal(first[key], second[key])
    elif isinstance(first, (tuple, list, deque)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_tree_equal(a, b)
    elif isinstance(first, (TransitionBatch, NStepExperience, TDLossOutput)):
        for field in first.__dataclass_fields__:
            assert_tree_equal(getattr(first, field), getattr(second, field))
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    else:
        assert first == second


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_size", [1, 4, 32])
@pytest.mark.parametrize("terminated,truncated,target", [(False, False, 6.4), (False, True, 6.4),
                                                        (True, False, 3.), (True, True, 3.)])
def test_target_from_target_network_shape_dtype_device(device, dtype, batch_size, terminated, truncated, target):
    learner, world = linear_learner(dtype, device), TabularWorldModel()
    reward = 3. if terminated else 1.
    if terminated:
        with torch.no_grad():
            learner.target_model.estimator.network.weight[:, 2] = torch.tensor([100., 200.], dtype=dtype, device=device)
    world.observe(Transition(0, 1, reward, 2, terminated, truncated))
    batch = synthetic_batch(world, batch_size=batch_size, rng=np.random.default_rng(7))
    assert batch.observation.shape == batch.action.shape == batch.reward.shape == (batch_size,)
    assert batch.observation.dtype == batch.action.dtype == torch.long
    assert batch.terminated.dtype == batch.truncated.dtype == torch.bool
    assert batch.log_prob is None and not batch.truncated.any()
    # Real batches retain the original boundary; synthetic batches omit external truncation.
    batch = replace(batch, truncated=torch.full_like(batch.truncated, truncated))
    output = learner.learn(batch)
    torch.testing.assert_close(output.td_error, torch.full((batch_size,), target - 2, dtype=dtype, device=device))
    expected_loss = nn.SmoothL1Loss()(torch.tensor(2.), torch.tensor(target)).item()
    assert output.loss.item() == pytest.approx(expected_loss, rel=1e-6)
    assert output.loss.shape == () and not output.loss.requires_grad and not output.td_error.requires_grad
    assert all(p.grad is None for p in learner.target_model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in learner.online_model.parameters())


def test_real_then_planning_same_sgd_momentum_parameter_and_target():
    learner, world = linear_learner(), TabularWorldModel()
    weight = learner.online_model.estimator.network.weight
    optimizer = learner.optimizer
    target = learner.target_model
    pointer = weight.data_ptr()
    target_before = deepcopy(target.state_dict())
    transition = Transition(0, 1, 1., 2, False, False)
    real = learner.learn(transition.to_batch())
    assert real.loss.item() == pytest.approx(3.9)
    assert weight[1, 0].item() == pytest.approx(2.1)
    assert optimizer.state[weight]["momentum_buffer"][1, 0].item() == -1
    world.observe(transition)
    planned = plan(world, learner, steps=1, batch_size=1, rng=np.random.default_rng(7))
    assert planned["updates"] == 1
    assert planned["results"][0].loss.item() == pytest.approx(3.8)
    assert weight[1, 0].item() == pytest.approx(2.29)
    assert optimizer.state[weight]["momentum_buffer"][1, 0].item() == pytest.approx(-1.9)
    assert weight.data_ptr() == pointer and learner.optimizer is optimizer and learner.target_model is target
    assert_tree_equal(target_before, target.state_dict())


@pytest.mark.parametrize("budget", [0, 1, 5, 10])
def test_planning_budget_empty_model_and_torch_rng(budget):
    learner, world, rng = linear_learner(), TabularWorldModel(), np.random.default_rng(7)
    sampling_state, behavior_state = deepcopy(rng.bit_generator.state), torch.get_rng_state().clone()
    empty = plan(world, learner, steps=budget, batch_size=32, rng=rng)
    assert empty["requested_updates"] == budget and empty["updates"] == 0
    assert rng.bit_generator.state == sampling_state
    world.observe(Transition(0, 1, 1., 2, False, False))
    actual = plan(world, learner, steps=budget, batch_size=32, rng=rng)
    assert actual["updates"] == len(actual["results"]) == budget
    assert torch.equal(behavior_state, torch.get_rng_state())
    if budget == 0:
        assert rng.bit_generator.state == sampling_state


@pytest.mark.parametrize("shape", [(), (4,), (2, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
def test_one_hot_boundary_and_model_shapes(shape, dtype, device):
    learner = create_learner(hidden_dims=(8,), dtype=dtype, device=device)
    ids = torch.zeros(shape, dtype=torch.int32)
    q, = learner.online_model(ids).values.q
    assert q.shape == (*shape, 4) and q.dtype == dtype and q.device.type == device
    embedded = learner.online_model.representation[0](ids.long().to(device))
    assert embedded.shape == (*shape, 11)
    assert torch.equal(embedded[..., 0], torch.ones(shape, dtype=dtype, device=device))
    assert not embedded.requires_grad
    assert learner.online_model.act(ids).action.shape == shape
    assert not learner.online_model.representation[0].weight.requires_grad


def test_synthetic_batch_mixed_pairs_latest_prediction_owned_and_reproducible():
    world = TabularWorldModel()
    world.observe(Transition(0, 1, 1., 2, False, False))
    world.observe(Transition(1, 0, 2., 2, True, False))
    world.observe(Transition(0, 1, 5., 1, False, True))
    a = synthetic_batch(world, batch_size=32, rng=np.random.default_rng(7))
    b = synthetic_batch(world, batch_size=32, rng=np.random.default_rng(7))
    assert_tree_equal(a, b)
    assert set(a.observation.tolist()) == {0, 1}
    assert torch.all(a.reward[a.observation == 0] == 5)
    a.reward.fill_(99)
    assert world.sample(np.random.default_rng(1)).reward.item() in (2, 5)


@pytest.mark.parametrize("gamma", [0., .9, 1.])
@pytest.mark.parametrize("terminated", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
def test_n_step_reward_sum_horizon_current_target_and_gradient(gamma, terminated, dtype, device):
    learner = linear_learner(dtype, device, gamma)
    window = [Transition(0, 1, 1., 1, False, False).to_batch(),
              Transition(1, 0, 2., 1, False, False).to_batch(),
              Transition(1, 0, 3., 2, terminated, not terminated).to_batch()]
    experience = make_n_step_experience(window, learner.target_generator, dtype=dtype)
    assert experience.horizon == 3 and experience.gamma == gamma
    assert experience.reward_sum.item() == pytest.approx(1 + gamma * 2 + gamma ** 2 * 3)
    assert experience.reward_sum.device.type == "cpu" and not experience.reward_sum.requires_grad
    # Metadata was collected before this target change; it must not cache Q=6.
    with torch.no_grad():
        learner.target_model.estimator.network.weight[1, 2] = 10
    result = learn_real_n_step(learner, [experience] * 4)
    expected = 1 + gamma * 2 + gamma ** 2 * 3 + (0 if terminated else gamma ** 3 * 10)
    torch.testing.assert_close(result.td_error, torch.full((4,), expected - 2, dtype=dtype, device=device))
    assert not result.loss.requires_grad and not result.td_error.requires_grad
    assert learner.online_model.estimator.network.weight.grad is not None
    assert all(p.grad is None for p in learner.target_model.parameters())


def test_n_step_one_matches_existing_dqn_and_uses_same_adam_state():
    torch.manual_seed(7)
    first = create_learner(n_states=3, n_actions=2, hidden_dims=(8,))
    second = create_learner(n_states=3, n_actions=2, hidden_dims=(8,))
    second.online_model.load_state_dict(first.online_model.state_dict())
    second.target_model.load_state_dict(first.target_model.state_dict())
    row = Transition(0, 1, 1., 2, False, False).to_batch()
    a = first.learn(row)
    experience = make_n_step_experience([row], second.target_generator)
    b = learn_real_n_step(second, [experience])
    assert_tree_equal(a.td_error, b.td_error)
    assert_tree_equal(first.online_model.state_dict(), second.online_model.state_dict())
    assert_tree_equal(first.optimizer.state_dict(), second.optimizer.state_dict())
    world = TabularWorldModel()
    world.observe(Transition(0, 1, 1., 2, False, False))
    plan(world, second, steps=1, batch_size=4, rng=np.random.default_rng(7))
    assert all(value["step"].item() == 2 for value in second.optimizer.state.values())


class ChainEnv:
    def __init__(self, terminal=True):
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
def test_real_delay_world_immediate_planning_flush_and_env_step_sync(n, terminal):
    learner, world, replay, env = create_learner(hidden_dims=(8,)), TabularWorldModel(), ReplayBuffer(20, seed=8), ChainEnv(terminal)
    out = run_steps(env, learner, replay, steps=4, n=n, world_model=world, planning_steps=5,
                    planning_rng=np.random.default_rng(9), batch_size=1, minimum_buffer_size=1,
                    target_sync_steps=2, record=True)
    assert out["environment_steps"] == env.calls == out["real_transitions"] == out["world_observations"] == 4
    assert out["planning_updates"] == out["requested_planning_updates"] == 20
    assert out["real_updates"] == max(0, 5 - n)
    assert out["target_syncs"] == 2
    assert [r["sync"] for r in out["trace"]] == [False, True, False, True]
    assert len(replay) == 4 and len(world) == 4 and not out["pending"]
    if n > 1:
        assert out["eligible_records"] == len(out["n_step_replay"]) == 4
        assert [r.horizon for r in out["n_step_replay"]] == [min(n, 4 - i) for i in range(4)]
        assert out["flush_records"] == min(n - 1, 4)
        assert all(row["real_updates"] == 0 and row["planning_updates"] == 5 for row in out["trace"][:n - 1])


def test_warmup_frequency_gradient_steps_and_target_sync_counts():
    learner, world, replay = create_learner(hidden_dims=(8,)), TabularWorldModel(), ReplayBuffer(20, seed=8)
    out = run_steps(ChainEnv(), learner, replay, steps=10, world_model=world, planning_steps=1,
                    planning_rng=np.random.default_rng(9), batch_size=4, minimum_buffer_size=5,
                    train_frequency=2, gradient_steps=3, target_sync_steps=3, record=True)
    assert out["real_updates"] == 9  # steps 6,8,10; three gradient steps each
    assert out["planning_updates"] == 10 and out["total_updates"] == 19
    assert out["target_syncs"] == 3
    assert len(replay) == 10  # Synthetic B=4 updates never enter real replay.
    assert all(state["step"].item() == 19 for state in learner.optimizer.state.values())


def test_idle_planning_keeps_environment_world_target_mode_and_behavior_rng():
    learner, world = create_learner(hidden_dims=(8,)), TabularWorldModel()
    env = make_dlfs4_gridworld_env(max_episode_steps=20)
    try:
        env.reset(seed=7)
        world.observe(Transition(0, 1, 1., 2, False, False))
        environment = (env.unwrapped._state, env._elapsed_steps, deepcopy(env.unwrapped.np_random.bit_generator.state))
        behavior = torch.get_rng_state().clone()
        target = deepcopy(learner.target_model.state_dict())
        online = deepcopy(learner.online_model.state_dict())
        world_before = world.state_dict()
        mode = learner.online_model.training
        rng = np.random.default_rng(9)
        for steps in (5, 10):
            assert plan(world, learner, steps=steps, batch_size=4, rng=rng)["updates"] == steps
        assert environment == (env.unwrapped._state, env._elapsed_steps, env.unwrapped.np_random.bit_generator.state)
        assert torch.equal(behavior, torch.get_rng_state())
        assert_tree_equal(world_before["outcomes"][0].reward, world.state_dict()["outcomes"][0].reward)
        assert_tree_equal(target, learner.target_model.state_dict())
        assert any(not torch.equal(value, learner.online_model.state_dict()[name]) for name, value in online.items())
        assert all(state["step"].item() == 15 for state in learner.optimizer.state.values())
        assert learner.online_model.training == mode and not learner.target_model.training
    finally:
        env.close()


@pytest.mark.parametrize("budget", [-1, True, .5])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        plan(TabularWorldModel(), linear_learner(), steps=budget, batch_size=1, rng=np.random.default_rng(7))


def test_invalid_window_metadata_and_changed_gamma_fail_before_updates():
    learner = linear_learner()
    with pytest.raises(ValueError, match="nonempty"):
        make_n_step_experience([], learner.target_generator)
    with pytest.raises(ValueError, match="boundary"):
        make_n_step_experience([Transition(0, 1, 1., 1, True, False).to_batch(),
                               Transition(1, 1, 2., 2, False, False).to_batch()], learner.target_generator)
    record = make_n_step_experience([Transition(0, 1, 1., 2, False, False).to_batch()], learner.target_generator)
    for invalid in (replace(record, horizon=0), replace(record, gamma=.5), replace(record, action=torch.tensor(1.))):
        with pytest.raises((ValueError, TypeError)):
            learn_real_n_step(learner, [invalid])
