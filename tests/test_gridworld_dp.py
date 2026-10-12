"""Independent Sutton reference values and reusable Gridworld dynamics."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from examples.gridworld_dp import solve_gridworld
from modular_rl.component.buffer import ReplayBuffer
from modular_rl.data import Transition
from modular_rl.dp import action_values, policy_evaluation, policy_iteration, uniform_policy, value_iteration
from modular_rl.environment import GRIDWORLD_ACTIONS, sutton_gridworld

RANDOM_VALUES = np.array([0, -14, -20, -22, -14, -18, -20, -20,
                          -20, -20, -18, -14, -22, -20, -14, 0])
OPTIMAL_VALUES = np.array([0, -1, -2, -3, -1, -2, -3, -2,
                           -2, -3, -2, -1, -3, -2, -1, 0])


def test_reusable_gridworld_has_the_requested_states_actions_and_boundary_transitions():
    mdp = sutton_gridworld()
    assert mdp.n_states == 16 and mdp.n_actions == 4 and mdp.terminal_states == (0, 15)
    assert GRIDWORLD_ACTIONS == ("up", "down", "left", "right")
    for state, action in ((1, 0), (4, 2), (11, 3), (14, 1)):
        outcome, = mdp.transitions[state][action]
        assert outcome.next_state == state and outcome.reward == -1 and not outcome.terminated
    assert [mdp.transitions[6][a][0].next_state for a in range(4)] == [2, 10, 5, 7]
    for state, action, terminal in ((1, 2, 0), (4, 0, 0), (11, 1, 15), (14, 3, 15)):
        outcome, = mdp.transitions[state][action]
        assert outcome.next_state == terminal and outcome.reward == -1 and outcome.terminated
    for state in range(16):
        for action in range(4):
            outcome, = mdp.transitions[state][action]
            assert outcome.probability == 1
            if state in mdp.terminal_states:
                assert outcome.next_state == state and outcome.reward == 0 and outcome.terminated
            else:
                assert outcome.reward == -1


def test_uniform_random_policy_matches_sutton_example_4_1():
    mdp = sutton_gridworld()
    result = policy_evaluation(mdp, uniform_policy(mdp), gamma=1, theta=1e-10)
    assert result.converged and result.delta < 1e-10
    np.testing.assert_allclose(result.values, RANDOM_VALUES, rtol=0, atol=2e-9)
    assert result.values[0] == result.values[15] == 0
    q = action_values(mdp, result.values, gamma=1)
    np.testing.assert_allclose((result.policy * q).sum(axis=-1), result.values, rtol=0, atol=1e-10)


@pytest.mark.parametrize("algorithm", [policy_iteration, value_iteration])
def test_both_control_algorithms_match_optimal_values_and_select_only_optimal_actions(algorithm):
    mdp = sutton_gridworld()
    result = algorithm(mdp, gamma=1, theta=1e-10)
    assert result.converged
    np.testing.assert_array_equal(result.values, OPTIMAL_VALUES)
    np.testing.assert_allclose(result.policy.sum(axis=-1), 1, rtol=0, atol=1e-15)
    q = action_values(mdp, OPTIMAL_VALUES, gamma=1)
    for state in range(16):
        selected = np.flatnonzero(result.policy[state] > 0)
        assert len(selected) > 0
        np.testing.assert_array_equal(q[state, selected], np.full(len(selected), q[state].max()))
    evaluated = policy_evaluation(mdp, result.policy, gamma=1, theta=1e-10)
    np.testing.assert_array_equal(evaluated.values, OPTIMAL_VALUES)


def test_policy_and_value_iteration_agree_without_fixing_one_correct_policy_array():
    results = solve_gridworld()
    np.testing.assert_array_equal(results["policy_iteration"].values, results["value_iteration"].values)
    assert results["policy_iteration"].evaluation_iterations >= results["policy_iteration"].iterations
    assert results["value_iteration"].evaluation_iterations == 0


def test_gridworld_improper_policy_reports_failure_and_preserves_terminal_zeros():
    mdp = sutton_gridworld()
    up_policy = np.zeros((16, 4))
    up_policy[:, 0] = 1
    result = policy_evaluation(mdp, up_policy, gamma=1, max_iterations=20)
    assert not result.converged and result.iterations == 20
    assert result.values[0] == result.values[15] == 0
    assert result.values[1] == -20  # Up hits a boundary forever.


def test_gridworld_outcomes_can_supply_existing_experience_data_for_future_mc_td():
    mdp = sutton_gridworld()
    outcome, = mdp.transitions[1][2]  # Left enters terminal state 0.
    replay = ReplayBuffer(1, seed=7)
    replay.add(Transition(1, 2, outcome.reward, outcome.next_state, outcome.terminated, False))
    batch = replay.sample(1)
    assert batch.observation.item() == 1 and batch.action.item() == 2
    assert batch.next_observation.item() == 0 and batch.reward.item() == -1
    assert batch.reward.is_floating_point() and batch.terminated.item() and not batch.truncated.item()


def test_gridworld_example_module_reports_tables_actions_errors_and_iteration_counts():
    result = subprocess.run([sys.executable, "-m", "examples.gridworld_dp"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                            check=True, timeout=30)
    assert "Sutton 4x4 Gridworld: gamma=1, synchronous sweeps" in result.stdout
    for name in ("evaluation", "policy_iteration", "value_iteration"):
        assert f"{name}: converged=True iterations=" in result.stdout
    assert "policy_iteration optimal actions" in result.stdout and "value_iteration optimal actions" in result.stdout
    assert "evaluation_sweeps=" in result.stdout and "max_reference_error=" in result.stdout
    assert "policy_iteration_vs_value_iteration_max_error=0.000e+00" in result.stdout


def test_example_iteration_limit_reports_failure_instead_of_hiding_it():
    results = solve_gridworld(max_iterations=1, max_policy_iterations=1)
    assert not any(result.converged for result in results.values())
    assert results["evaluation"].iterations == results["value_iteration"].iterations == 1
    assert results["policy_iteration"].iterations == results["policy_iteration"].evaluation_iterations == 1


def test_example_module_does_not_call_unconverged_policies_optimal():
    result = subprocess.run([sys.executable, "-m", "examples.gridworld_dp", "--max-iterations", "1"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                            check=True, timeout=30)
    assert "converged=False" in result.stdout and "policy actions (not converged)" in result.stdout
    assert "optimal actions" not in result.stdout
