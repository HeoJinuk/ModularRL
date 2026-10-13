"""Sutton & Barto Example 4.1 with corner terminals indexed as 0 and 15."""

from gymnasium.wrappers import TimeLimit

from modular_rl.data.mdp import TabularMDP, TransitionOutcome
from modular_rl.environment.tabular_env import TabularEnv

GRIDWORLD_ACTIONS = ("up", "down", "left", "right")


def sutton_gridworld() -> TabularMDP:
    """Return the 4x4 kernel; each nonterminal move costs -1, including exit.

    Leaving the grid keeps the state unchanged. Terminal outgoing rows are
    zero-reward placeholders, not additional environment steps. This shared
    definition can later supply both full DP backups and sampled MC/TD paths.
    """
    transitions = []
    terminal_states = (0, 15)
    moves = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for state in range(16):
        actions = []
        row, column = divmod(state, 4)
        for dr, dc in moves:
            if state in terminal_states:
                outcome = TransitionOutcome(1.0, state, 0.0, True)
            else:
                next_row, next_column = row + dr, column + dc
                next_state = 4 * next_row + next_column if 0 <= next_row < 4 and 0 <= next_column < 4 else state
                outcome = TransitionOutcome(1.0, next_state, -1.0, next_state in terminal_states)
            actions.append((outcome,))
        transitions.append(tuple(actions))
    return TabularMDP(16, 4, tuple(transitions), terminal_states)


def make_sutton_gridworld_env(*, render_mode=None, max_episode_steps=10_000):
    """Execute the unchanged 4x4 kernel; reset samples a nonterminal start.

    TimeLimit bounds episodes by default. Pass None for the raw TabularEnv.
    Access the shared mathematical model with env.unwrapped.mdp.
    """
    env = TabularEnv(sutton_gridworld(), render_mode=render_mode)
    return env if max_episode_steps is None else TimeLimit(env, max_episode_steps=max_episode_steps)
