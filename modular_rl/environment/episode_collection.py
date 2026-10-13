"""Collect tabular episodes through Gymnasium reset/step, without learning."""

from collections.abc import Callable

import gymnasium as gym

from modular_rl.data import Episode, Transition


def collect_episode(env: gym.Env[int, int], select_action: Callable[[int], int], *,
                    seed: int | None = None, options: dict | None = None) -> Episode:
    """Collect one discrete-state/action episode with an external action callback.

    Use TimeLimit to bound collection; no extra step cap or RNG is owned here.
    Pass seed for the first reset, then omit it to continue the environment's
    stream. The callback owns policy selection and its own RNG. Observations
    and actions are scalar state/action IDs, as in TabularEnv or FrozenLake.

    Stop at either boundary and retain the actual final observation from step;
    the environment must not replace it with an auto-reset observation. Episode
    snapshots the ordered records. log_prob is absent, info is not stored, and
    truncated-only episodes remain explicit fragments for MC to reject.
    """
    observation, _ = env.reset(seed=seed, options=options)
    transitions = []
    while True:
        action = select_action(observation)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        transitions.append(Transition(observation, action, reward, next_observation, terminated, truncated))
        if terminated or truncated:
            return Episode(tuple(transitions))
        observation = next_observation
