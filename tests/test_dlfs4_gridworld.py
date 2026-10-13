"""Physical transitions checked against the book's common/gridworld.py kernel."""

import numpy as np
import pytest

from modular_rl.dp import action_values, policy_evaluation, policy_iteration, uniform_policy, value_iteration
from modular_rl.environment import DLFS4_GRID_SHAPE, DLFS4_STATES, DLFS4_START_STATE, dlfs4_gridworld


def test_book_coordinates_actions_rewards_and_boundaries_match_original():
    mdp = dlfs4_gridworld()
    assert DLFS4_GRID_SHAPE == (3, 4) and mdp.n_states == 11 and mdp.n_actions == 4
    assert DLFS4_STATES[DLFS4_START_STATE] == (2, 0)
    assert (1, 1) not in DLFS4_STATES and mdp.terminal_states == (3,)
    # Rows: (0,0),(0,1),(0,2),(0,3),(1,0),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3).
    # Columns: up, down, left, right. Terminal rows are storage placeholders.
    expected_next = np.array([[0, 4, 0, 1], [1, 1, 0, 2], [2, 5, 1, 3], [3, 3, 3, 3],
                              [0, 7, 4, 4], [2, 9, 5, 6], [3, 10, 5, 6], [4, 7, 7, 8],
                              [8, 8, 7, 9], [5, 9, 8, 10], [6, 10, 9, 10]])
    for state, actions in enumerate(mdp.transitions):
        for action, outcomes in enumerate(actions):
            assert len(outcomes) == 1
            outcome = outcomes[0]
            next_state = expected_next[state, action]
            assert outcome.probability == 1 and outcome.next_state == next_state
            expected_reward = 0 if state == 3 else 1 if next_state == 3 else -1 if next_state == 6 else 0
            assert outcome.reward == expected_reward
            assert outcome.terminated == (state == 3 or next_state == 3)
    assert mdp.transitions[6][3][0].reward == -1  # Staying in penalty cell also costs -1.
    assert not mdp.transitions[5][3][0].terminated  # Penalty entry does not end an episode.


def test_policy_evaluation_matches_independent_linear_system_for_book_kernel():
    mdp = dlfs4_gridworld()
    policy = uniform_policy(mdp)
    probabilities = np.zeros((11, 11))
    rewards = np.zeros(11)
    for state, actions in enumerate(mdp.transitions):
        for outcomes in actions:
            outcome = outcomes[0]
            rewards[state] += .25 * outcome.reward
            if not outcome.terminated:
                probabilities[state, outcome.next_state] += .25
    expected = np.linalg.solve(np.eye(11) - .9 * probabilities, rewards)
    result = policy_evaluation(mdp, policy, gamma=.9, theta=1e-12)
    assert result.converged
    np.testing.assert_allclose(result.values, expected, atol=1e-10, rtol=0)


def test_policy_and_value_iteration_have_same_optimum_and_five_step_goal_path_value():
    mdp = dlfs4_gridworld()
    pi = policy_iteration(mdp, gamma=.9, theta=1e-12)
    vi = value_iteration(mdp, gamma=.9, theta=1e-12)
    assert pi.converged and vi.converged
    np.testing.assert_allclose(pi.values, vi.values, atol=1e-10, rtol=0)
    assert vi.values[DLFS4_START_STATE] == pytest.approx(.9 ** 4)
    assert vi.values[3] == 0
    q = action_values(mdp, vi.values, gamma=.9)
    np.testing.assert_allclose((pi.policy * q).sum(-1), vi.values, atol=1e-10)
