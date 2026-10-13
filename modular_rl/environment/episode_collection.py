"""Collect tabular episodes through Gymnasium reset/step, without learning."""

from collections.abc import Callable

import gymnasium as gym
import torch

from modular_rl.data import ActionOutput, Episode, Transition


def collect_episode(env: gym.Env[int, int], select_action: Callable[[int], int | ActionOutput], *,
                    seed: int | None = None, options: dict | None = None) -> Episode:
    """Collect one discrete-state/action episode with an external action callback.

    Use TimeLimit to bound collection; no extra step cap or RNG is owned here.
    Pass seed for the first reset, then omit it to continue the environment's
    stream. The callback owns policy selection and its own RNG. Observations
    and actions are scalar state/action IDs, as in TabularEnv or FrozenLake.
    An integer callback keeps log_prob=None. An ActionOutput callback records
    its sampling-time log_prob in an owned, detached CPU snapshot before step.
    Scalar integer tensor actions are converted at the environment boundary.

    Stop at either boundary and retain the actual final observation from step;
    the environment must not replace it with an auto-reset observation. Episode
    snapshots the ordered records. info is not stored, and
    truncated-only episodes remain explicit fragments for MC to reject.
    """
    observation, _ = env.reset(seed=seed, options=options)
    transitions = []
    while True:
        action, log_prob = _action_record(select_action(observation))
        next_observation, reward, terminated, truncated, _ = env.step(action)
        transitions.append(Transition(observation, action, reward, next_observation, terminated, truncated, log_prob))
        if terminated or truncated:
            return Episode(tuple(transitions))
        observation = next_observation


def _action_record(selected):
    """Extract a discrete action and snapshot its optional behavior log probability."""
    if not isinstance(selected, ActionOutput):
        return selected, None
    action, log_prob = selected.action, selected.log_prob
    if action.ndim != 0 or action.is_floating_point() or action.is_complex() or action.dtype == torch.bool:
        raise TypeError("Episode action must be a scalar integer Tensor")
    if log_prob is not None:
        if log_prob.ndim != 0 or not log_prob.is_floating_point():
            raise TypeError("Episode log_prob must be a scalar floating Tensor")
        log_prob = log_prob.detach().cpu().clone()
    return action.item(), log_prob
