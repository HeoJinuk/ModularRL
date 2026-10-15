"""Independent no-planning reference, trusted checkpoint and environment smoke tests."""

from collections import deque
from copy import deepcopy

import numpy as np
import pytest
import torch

from examples.gridworld_dyna_dqn import (create_learner, learn_real_n_step, make_n_step_experience,
                                       plan, run_steps, train_and_evaluate)
from examples.gridworld_dyna_q import TabularWorldModel
from modular_rl.component.buffer import ReplayBuffer
from modular_rl.data import Transition
from modular_rl.environment import make_dlfs4_gridworld_env
from tests.test_dyna_dqn_composition import assert_tree_equal, one_cpu_thread


def reference_dqn(env, learner, replay, *, steps, seed):
    """CartPole example's DQN update order, adapted to an ID-input step budget.

    No call to Dyna run_steps, plan or n-step helpers. Config matches the
    equality test: warmup=8, B=4, train every 2 steps, 2 gradient steps,
    target sync every 7 ENVIRONMENT steps, epsilon 1 -> .05 over 30 steps.
    """
    observation, _ = env.reset(seed=seed)
    trace = []
    for time in range(1, steps + 1):
        learner.online_model.action_selector.epsilon = 1 - .95 * min(1., (time - 1) / 30)
        selected = learner.online_model.act(observation)
        next_state, reward, terminated, truncated, _ = env.step(selected.action.item())
        cutoff = time == steps and not (terminated or truncated)
        transition = Transition(observation, selected.action, float(reward), next_state,
                                terminated, truncated or cutoff, selected.log_prob)
        replay.add(transition)
        batches, outputs = [], []
        if len(replay) >= 8 and time % 2 == 0:
            for _ in range(2):
                batch = replay.sample(4)
                batches.append(batch)
                outputs.append(learner.learn(batch))
        if time % 7 == 0:
            learner.sync_target()
        trace.append({"transition": transition.to_batch(), "real_batches": batches, "real_results": outputs,
                      "real_updates": len(outputs), "planning_updates": 0, "sync": time % 7 == 0,
                      "cutoff": cutoff, "flush_records": 0, "eligible_records": 0})
        if terminated or truncated or cutoff:
            if time < steps:
                observation, _ = env.reset()
        else:
            observation = next_state
    return trace


def record_targets(learner):
    targets = []
    generate = learner.target_generator.generate

    def recorded(reward, next_value, terminated):
        target = generate(reward, next_value, terminated)
        targets.append(target.detach().cpu().clone())
        return target

    learner.target_generator.generate = recorded
    return targets


@pytest.mark.parametrize("seed", [7, 11, 19])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_planning_zero_bit_exact_independent_dqn_reference(seed, dtype):
    env_a = make_dlfs4_gridworld_env(max_episode_steps=17)
    env_b = make_dlfs4_gridworld_env(max_episode_steps=17)
    try:
        torch.manual_seed(seed)
        reference = create_learner(hidden_dims=(8,), dtype=dtype)
        replay_a = ReplayBuffer(40, seed=seed + 1)
        targets_a = record_targets(reference)
        expected = reference_dqn(env_a, reference, replay_a, steps=80, seed=seed)
        behavior = torch.get_rng_state().clone()

        torch.manual_seed(seed)
        dyna = create_learner(hidden_dims=(8,), dtype=dtype)
        replay_b = ReplayBuffer(40, seed=seed + 1)
        targets_b = record_targets(dyna)
        planning_rng = np.random.default_rng(seed + 2)
        planning_state = deepcopy(planning_rng.bit_generator.state)
        actual = run_steps(env_b, dyna, replay_b, steps=80, seed=seed, world_model=TabularWorldModel(),
                           planning_steps=0, planning_rng=planning_rng, batch_size=4, minimum_buffer_size=8,
                           train_frequency=2, gradient_steps=2, target_sync_steps=7,
                           epsilon_decay_steps=30, record=True)
        assert_tree_equal(expected, actual["trace"])
        assert_tree_equal(targets_a, targets_b)
        assert_tree_equal(reference.online_model.state_dict(), dyna.online_model.state_dict())
        assert_tree_equal(reference.target_model.state_dict(), dyna.target_model.state_dict())
        assert_tree_equal(reference.optimizer.state_dict(), dyna.optimizer.state_dict())
        assert torch.equal(behavior, torch.get_rng_state())
        assert planning_rng.bit_generator.state == planning_state
        assert env_a.unwrapped.np_random.bit_generator.state == env_b.unwrapped.np_random.bit_generator.state
        assert actual["target_syncs"] == 80 // 7
        assert actual["real_updates"] == actual["total_updates"] == 74
        assert actual["planning_updates"] == 0 and actual["world_observations"] == 80
        # Contents plus multiple following samples verify the private replay RNG
        # behavior without reading/mutating its implementation attributes.
        a, b = deepcopy(replay_a), deepcopy(replay_b)
        for size in (len(a), 1, 4, 4, 4):
            assert_tree_equal(a.sample(size), b.sample(size))
    finally:
        env_a.close()
        env_b.close()


@pytest.mark.parametrize("n", [1, 3, 5])
@pytest.mark.parametrize("budget", [0, 1, 5])
def test_short_gridworld_training_reproducibility_sources_and_counts(n, budget):
    first_learner, a = train_and_evaluate(steps=80, seed=7, n=n, planning_steps=budget)
    second_learner, b = train_and_evaluate(steps=80, seed=7, n=n, planning_steps=budget)
    assert a["environment_steps"] == a["real_transitions"] == a["world_observations"] == 80
    assert a["real_updates"] == 17  # 64..80 inclusive
    assert a["planning_updates"] == a["requested_planning_updates"] == 80 * budget
    assert a["total_updates"] == 17 + 80 * budget
    assert a["target_syncs"] == 0
    assert len(a["replay"]) == 80 and 0 < a["world_pairs"] <= 40
    if n > 1:
        assert a["eligible_records"] == 80
        assert len(a["n_step_replay"]) == 80 and not a["pending"]
    for field in ("q_values", "real_losses", "planning_losses", "training_rewards", "episode_lengths", "behavior_rng"):
        assert_tree_equal(a[field], b[field])
    assert a["after"] == b["after"] and a["dp_steps"] == 5
    assert np.isfinite(a["real_losses"]).all() and np.isfinite(a["planning_losses"]).all()
    assert np.isfinite(a["max_q_rmse"]) and a["elapsed_seconds"] > 0
    assert_tree_equal(first_learner.optimizer.state_dict(), second_learner.optimizer.state_dict())


def test_checkpoint_restores_models_optimizer_replay_rng_pending_metadata_and_counters(tmp_path):
    """Same-version trusted snapshot, fixed next stream; no environment resume claim."""
    torch.manual_seed(7)
    settings = {"n": 3, "gamma": .9, "hidden_dims": (8,), "learning_rate": 1e-4,
                "dtype": torch.float64, "epsilon": .2, "planning_steps": 1, "sync_steps": 3}
    learner = create_learner(hidden_dims=settings["hidden_dims"], dtype=settings["dtype"])
    learner.online_model.action_selector.epsilon = settings["epsilon"]
    replay, world = ReplayBuffer(12, seed=8), TabularWorldModel()
    planning_rng, replay_rng = np.random.default_rng(9), np.random.default_rng(10)
    pending, metadata = deque(), deque(maxlen=12)
    counters = {"environment_steps": 0, "real_updates": 0, "planning_updates": 0, "syncs": 0}
    for state in (0, 1):
        selected = learner.online_model.act(state)
        row = Transition(state, selected.action, float(state + 1), state + 1, False, False, selected.log_prob)
        replay.add(row)
        pending.append(row.to_batch())
        world.observe(row)
        plan(world, learner, steps=1, batch_size=2, rng=planning_rng)
        counters["environment_steps"] += 1
        counters["planning_updates"] += 1
    replay.sample(1)
    replay_rng.integers(100)
    path = tmp_path / "dyna_dqn.pt"
    torch.save({"online": learner.online_model.state_dict(), "target": learner.target_model.state_dict(),
                "optimizer": learner.optimizer.state_dict(), "replay": replay, "world": world.state_dict(),
                "planning_rng": planning_rng.bit_generator.state, "metadata_rng": replay_rng.bit_generator.state,
                "behavior_rng": torch.get_rng_state(), "pending": pending, "metadata": metadata,
                "counters": counters, "settings": settings}, path)

    def continue_stream(learner, replay, world, planning_rng, replay_rng, pending, metadata, counters):
        selected = learner.online_model.act(2)
        row = Transition(2, selected.action, 3., 0, True, False, selected.log_prob)
        replay.add(row)
        pending.append(row.to_batch())
        metadata.append(make_n_step_experience(pending, learner.target_generator, dtype=torch.float64))
        pending.popleft()
        real = learn_real_n_step(learner, [metadata[0]])
        counters["real_updates"] += 1
        world.observe(row)
        planned = plan(world, learner, steps=1, batch_size=2, rng=planning_rng)
        counters["planning_updates"] += planned["updates"]
        counters["environment_steps"] += 1
        while pending:
            metadata.append(make_n_step_experience(pending, learner.target_generator, dtype=torch.float64))
            pending.popleft()
        if counters["environment_steps"] % settings["sync_steps"] == 0:
            learner.sync_target()
            counters["syncs"] += 1
        indices = replay_rng.choice(len(metadata), size=2, replace=False)
        extra = learn_real_n_step(learner, [metadata[int(i)] for i in indices])
        counters["real_updates"] += 1
        return real, planned["results"], extra, replay.sample(2), learner.online_model.act(1)

    expected = continue_stream(learner, replay, world, planning_rng, replay_rng, pending, metadata, counters)
    behavior = torch.get_rng_state().clone()
    state = torch.load(path, weights_only=False)  # Only this locally created, trusted file.
    resumed = create_learner(hidden_dims=state["settings"]["hidden_dims"], dtype=state["settings"]["dtype"])
    resumed.online_model.load_state_dict(state["online"])
    resumed.target_model.load_state_dict(state["target"])
    resumed.optimizer.load_state_dict(state["optimizer"])
    world2, plan_rng2, replay_rng2 = TabularWorldModel(), np.random.default_rng(), np.random.default_rng()
    world2.load_state_dict(state["world"])
    plan_rng2.bit_generator.state = state["planning_rng"]
    replay_rng2.bit_generator.state = state["metadata_rng"]
    torch.set_rng_state(state["behavior_rng"])
    actual = continue_stream(resumed, state["replay"], world2, plan_rng2, replay_rng2,
                             state["pending"], state["metadata"], state["counters"])
    for a, b in zip(expected[:-1], actual[:-1]):
        assert_tree_equal(a, b)
    assert_tree_equal(expected[-1].action, actual[-1].action)
    assert_tree_equal(expected[-1].log_prob, actual[-1].log_prob)
    assert_tree_equal(learner.online_model.state_dict(), resumed.online_model.state_dict())
    assert_tree_equal(learner.target_model.state_dict(), resumed.target_model.state_dict())
    assert_tree_equal(learner.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert counters == state["counters"] and counters["syncs"] == 1
    assert_tree_equal(list(metadata), list(state["metadata"]))
    assert torch.equal(behavior, torch.get_rng_state())
    assert planning_rng.bit_generator.state == plan_rng2.bit_generator.state
    assert replay_rng.bit_generator.state == replay_rng2.bit_generator.state
    assert_tree_equal(replay.sample(2), state["replay"].sample(2))
