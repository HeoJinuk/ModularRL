"""Official planning matches the frozen example algorithms and their RNG streams."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from modular_rl import world_models as W, planners as P
from modular_rl.data import Transition
from examples import gridworld_dyna_q as tabular, gridworld_dyna_dqn as neural
from modular_rl.environment import make_dlfs4_gridworld_env
from tests import _legacy_planning as old
from tests.test_dyna_q_composition import make_learner
from tests.test_dyna_dqn_composition import linear_learner, assert_tree_equal


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def populate(world):
    for transition in (Transition(0, 1, 1., 2, False, True),
                       Transition(1, 0, 3., 2, True, False),
                       Transition(0, 1, 2., 1, False, False)):
        world.observe(transition)
    return world


def test_world_alias_checkpoint_order_and_owned_samples_match_original():
    assert W.Tabular is tabular.TabularWorldModel is neural.TabularWorldModel
    world, reference = populate(W.Tabular()), populate(old.TabularWorldModel())
    assert_tree_equal(world.state_dict(), reference.state_dict())
    restored = W.Tabular()
    restored.load_state_dict(reference.state_dict())
    first, second = np.random.default_rng(7), np.random.default_rng(7)
    for _ in range(50):
        a, b = restored.sample(first).to_batch(), reference.sample(second).to_batch()
        assert_tree_equal(a, b)
        a.reward.fill_(99)
    assert first.bit_generator.state == second.bit_generator.state
    assert_tree_equal(world.state_dict(), restored.state_dict())


@pytest.mark.parametrize("budget", [0, 1, 5, 10])
def test_tabular_planner_errors_table_and_rng_equal_frozen_reference(budget):
    learner, reference = make_learner(), make_learner()
    world, reference_world = populate(W.Tabular()), populate(old.TabularWorldModel())
    rng, reference_rng = np.random.default_rng(7), np.random.default_rng(7)
    behavior = torch.get_rng_state().clone()
    expected = old.tabular_plan(reference_world, reference, steps=budget, rng=reference_rng)
    result = P.Dyna(world, learner, rng=rng).plan(budget, record_batches=True)
    assert result['updates'] == result['requested_updates'] == budget
    actual = torch.cat(result['results']) if budget else learner.model.estimator.q_table.new_empty(0)
    assert torch.equal(actual, expected)
    assert_tree_equal(learner.model.state_dict(), reference.model.state_dict())
    assert rng.bit_generator.state == reference_rng.bit_generator.state
    assert torch.equal(behavior, torch.get_rng_state())
    assert_tree_equal(world.state_dict(), reference_world.state_dict())


@pytest.mark.parametrize("budget", [0, 1, 5, 10])
@pytest.mark.parametrize("batch_size", [1, 8])
def test_neural_planner_matches_losses_parameters_optimizer_and_target(budget, batch_size):
    learner, reference = linear_learner(), linear_learner()
    world, reference_world = populate(W.Tabular()), populate(old.TabularWorldModel())
    rng, reference_rng = np.random.default_rng(11), np.random.default_rng(11)
    expected = old.neural_plan(reference_world, reference, steps=budget, batch_size=batch_size,
                               rng=reference_rng, record_batches=True)
    result = P.Dyna(world, learner, batch_size=batch_size, rng=rng).plan(budget, record_batches=True)
    assert_tree_equal(result, expected)
    for name in ('online_model', 'target_model', 'optimizer'):
        assert_tree_equal(getattr(learner, name).state_dict(), getattr(reference, name).state_dict())
    assert rng.bit_generator.state == reference_rng.bit_generator.state


def test_empty_and_zero_budget_plans_do_not_call_sample_learn_or_consume_rng(monkeypatch):
    planner = P.Dyna(W.Tabular(), make_learner(), seed=19)
    before = deepcopy(planner.rng.bit_generator.state)
    def forbidden(*args, **kwargs):
        pytest.fail('idle planning performed work')
    monkeypatch.setattr(planner.world_model, 'sample', forbidden)
    monkeypatch.setattr(planner.learner, 'learn', forbidden)
    assert planner.plan(5)['updates'] == 0
    planner.world_model.observe(Transition(0, 1, 1., 2, False, False))
    assert planner.plan(0)['updates'] == 0
    assert before == planner.rng.bit_generator.state


@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("budget", [0, 1, 5, 10])
def test_dyna_q_run_preserves_environment_trajectory_targets_and_flush(monkeypatch, n, budget):
    def run(plan):
        monkeypatch.setattr(tabular, 'plan', plan)
        torch.manual_seed(7)
        learner, world = make_learner(), W.Tabular()
        env = make_dlfs4_gridworld_env(max_episode_steps=12)
        rng = np.random.default_rng(8)
        try:
            result = tabular.run_steps(env, learner, steps=24, world_model=world,
                planning_steps=budget, rng=rng, n=n, seed=7)
            return result, learner.model.state_dict(), torch.get_rng_state(), rng.bit_generator.state, env.np_random.bit_generator.state
        finally:
            env.close()
    original = tabular.plan
    assert_tree_equal(run(old.tabular_plan), run(original))


@pytest.mark.parametrize("n", [1, 3])
def test_dyna_dqn_schedule_optimizer_and_independent_real_horizon_match(monkeypatch, n):
    def run(plan):
        monkeypatch.setattr(neural, 'plan', plan)
        torch.manual_seed(11)
        learner = neural.create_learner(hidden_dims=(8,))
        replay = neural.ReplayBuffer(40, seed=12)
        world = W.Tabular()
        rng = np.random.default_rng(13)
        env = make_dlfs4_gridworld_env(max_episode_steps=8)
        try:
            result = neural.run_steps(env, learner, replay, steps=16, world_model=world,
                planning_steps=1, planning_rng=rng, batch_size=2, minimum_buffer_size=4,
                target_sync_steps=5, n=n, seed=11, record=True)
            result.pop('n_step_rng')
            return (result, learner.online_model.state_dict(), learner.target_model.state_dict(),
                    learner.optimizer.state_dict(), torch.get_rng_state(), rng.bit_generator.state,
                    replay._rng.bit_generator.state, env.np_random.bit_generator.state)
        finally:
            env.close()
    original = neural.plan
    assert_tree_equal(run(old.neural_plan), run(original))


@pytest.mark.parametrize("kwargs", [{'batch_size': 0}, {'seed':7,'rng':np.random.default_rng(7)}, {'rng':7}])
def test_planner_invalid_configuration(kwargs):
    with pytest.raises((ValueError, TypeError)):
        P.Dyna(W.Tabular(), make_learner(), **kwargs)
