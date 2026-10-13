"""Known GridWorld kernel from Deep Learning from Scratch 4, chapters 4–6.

    Actions: up, down, left, right. Coordinates are (row, column), top to bottom.
    State IDs index DLFS4_STATES; the wall (1,1) has no state ID. Goal (0,3)
    terminates on entry with +1; entry into (1,3) yields -1 without termination.
    Other entries yield 0. Borders and the wall keep the agent in its state,
    with the reward of that resulting cell, including -1 when staying in (1,3).

    Source: WegraLee/deep-learning-from-scratch-4/common/gridworld.py,
    commit b82cd6432b4e63ce6a4ab2b925fc74a1227fb06a. This defines dynamics only;
    plotting and sampling are separate. Sutton's 4x4 Gridworld is unchanged.
"""

from gymnasium.wrappers import TimeLimit

from modular_rl.data import TabularMDP, TransitionOutcome
from modular_rl.environment.tabular_env import TabularEnv

DLFS4_GRID_SHAPE = (3, 4)
DLFS4_STATES = tuple((row, col) for row in range(3) for col in range(4) if (row, col) != (1, 1))
DLFS4_START_STATE = DLFS4_STATES.index((2, 0))


def dlfs4_gridworld() -> TabularMDP:
    """Return the book's deterministic 11-state, four-action MDP.

    Compact IDs exclude the inaccessible wall; use DLFS4_STATES[id] for book
    coordinates. Terminal outgoing rows are zero-reward placeholders rather
    than additional steps. A negative-reward cell is not a terminal state.
    """
    state_ids = {coordinate: state for state, coordinate in enumerate(DLFS4_STATES)}
    goal = state_ids[(0, 3)]
    transitions = []
    for state, (row, col) in enumerate(DLFS4_STATES):
        actions = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if state == goal:
                outcome = TransitionOutcome(1., state, 0., True)
            else:
                coordinate = (row + dy, col + dx)
                next_state = state_ids.get(coordinate, state)
                next_coordinate = DLFS4_STATES[next_state]
                reward = 1. if next_coordinate == (0, 3) else -1. if next_coordinate == (1, 3) else 0.
                outcome = TransitionOutcome(1., next_state, reward, next_state == goal)
            actions.append((outcome,))
        transitions.append(tuple(actions))
    return TabularMDP(len(DLFS4_STATES), 4, tuple(transitions), (goal,))


def make_dlfs4_gridworld_env(*, render_mode=None, max_episode_steps=2000):
    """Execute the unchanged book kernel; reset starts at coordinate (2,0).

    TimeLimit bounds episodes by default. Pass None for the raw TabularEnv.
    reset(options={'start_state': id}) can override the configured start.
    """
    env = TabularEnv(dlfs4_gridworld(), start_state=DLFS4_START_STATE, render_mode=render_mode)
    return env if max_episode_steps is None else TimeLimit(env, max_episode_steps=max_episode_steps)
