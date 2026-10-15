"""Fixed real-step experiments, exact no-planning equivalence and reproducibility."""

import numpy as np
import pytest
import torch

from examples.gridworld_dyna_q import TabularWorldModel, run_steps, train_and_evaluate
from modular_rl.data import Transition
from modular_rl.environment import make_dlfs4_gridworld_env
from tests.test_dyna_q_composition import make_learner


def reference_q_learning(env, learner, steps, seed):
    """Original gridworld_sarsa Q branch/action timing, with a fixed step budget.

    No world model, planning or shared run_steps implementation. Episode resets
    follow the existing example; only the stopping budget differs.
    """
    observation, _ = env.reset(seed=seed)
    selected = learner.model.act(observation)
    trace, rewards, lengths = [], [], []
    reward_sum, length = 0., 0
    for time in range(steps):
        action = selected.action.item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        budget_cutoff = time == steps - 1 and not (terminated or truncated)
        table = learner.model.estimator.q_table
        old_q = table[observation, action].clone()
        error = learner.learn(Transition(observation, action, reward, next_state, terminated,
                                         truncated or budget_cutoff, selected.log_prob).to_batch())
        trace.append((observation, action, reward, next_state, terminated, truncated,
                      (old_q + error[0]).item(), error.item()))
        reward_sum += reward
        length += 1
        if terminated or truncated:
            rewards.append(reward_sum)
            lengths.append(length)
            if time < steps - 1:
                observation, _ = env.reset()
                selected = learner.model.act(observation)
                reward_sum, length = 0., 0
        elif time < steps - 1:
            observation = next_state
            selected = learner.model.act(observation)  # AFTER Q update, as in existing Q-learning.
    return trace, rewards, lengths


@pytest.mark.parametrize("seed", [7, 11, 19])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_planning_zero_bit_exact_against_existing_q_learning_flow(seed, dtype):
    steps = 100
    reference, dyna = make_learner(dtype), make_learner(dtype)
    env1, env2 = make_dlfs4_gridworld_env(max_episode_steps=17), make_dlfs4_gridworld_env(max_episode_steps=17)
    try:
        torch.manual_seed(seed)
        trace, rewards, lengths = reference_q_learning(env1, reference, steps, seed)
        behavior_rng = torch.get_rng_state().clone()
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed + 1)
        planning_rng = rng.bit_generator.state
        result = run_steps(env2, dyna, steps=steps, world_model=TabularWorldModel(),
                           planning_steps=0, rng=rng, seed=seed)
        actual = [(row["state"], row["action"], row["reward"], row["next_state"], row["terminated"],
                   row["truncated"], row["targets"][0], row["td_errors"][0]) for row in result["trace"]]
        assert trace == actual
        assert rewards == result["training_rewards"].tolist()
        assert lengths == result["episode_lengths"].tolist()
        assert torch.equal(reference.model.estimator.q_table, dyna.model.estimator.q_table)
        assert torch.equal(behavior_rng, torch.get_rng_state())
        assert rng.bit_generator.state == planning_rng
        assert result["steps"] == result["real_updates"] == result["model_updates"] == steps
        assert result["planning_updates"] == 0
    finally:
        env1.close()
        env2.close()


@pytest.mark.parametrize("n", [1, 3, 5])
@pytest.mark.parametrize("budget", [0, 1, 5, 10])
def test_real_and_planning_counts_reproducible_short_gridworld(n, budget):
    learner1, first = train_and_evaluate(steps=100, seed=7, n=n, planning_steps=budget)
    learner2, second = train_and_evaluate(steps=100, seed=7, n=n, planning_steps=budget)
    assert first["steps"] == first["real_updates"] == first["model_updates"] == 100
    assert first["planning_updates"] == 100 * budget
    assert 1 <= first["world_pairs"] <= 40
    assert first["model_correct_pairs"] == first["world_pairs"]
    assert first["dp_steps"] == 5
    assert np.isfinite(first["max_q_rmse"])
    assert first["trace"] == second["trace"]
    assert first["after"] == second["after"]
    assert np.array_equal(first["training_rewards"], second["training_rewards"])
    assert torch.equal(learner1.model.estimator.q_table, learner2.model.estimator.q_table)


def test_baseline_and_dyna_zero_experiment_results_identical():
    _, baseline = train_and_evaluate(steps=100, seed=11, planning_steps=0, baseline=True)
    _, dyna = train_and_evaluate(steps=100, seed=11, planning_steps=0)
    for name in ("trace", "after", "max_q_rmse", "partial_episode", "steps", "real_updates"):
        assert baseline[name] == dyna[name]
    assert np.array_equal(baseline["q_values"], dyna["q_values"])
    assert baseline["model_updates"] == baseline["world_pairs"] == 0
    assert dyna["model_updates"] == 100


@pytest.mark.parametrize("kwargs", [{"baseline": True}, {"baseline": True, "n": 3, "planning_steps": 0},
                                    {"steps": 0}, {"n": 0}, {"planning_steps": -1}])
def test_invalid_experiment_settings(kwargs):
    with pytest.raises(ValueError):
        train_and_evaluate(**kwargs)
