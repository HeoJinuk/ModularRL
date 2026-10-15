"""Ordered, owned vector steps; no environment, RNG or return computation."""

from numbers import Integral

import torch

from modular_rl.component.buffer._transition import _copy_value
from modular_rl.data.rollout import RolloutBatch


class RolloutBuffer:
    """Collect exactly horizon vector steps from n_envs parallel environments.

    add takes one (N, ...) step; get stacks a complete (T, N, ...) rollout.
    Inputs are detached and copied to CPU without dtype conversion. A full
    buffer rejects overwrites. get returns independent tensors and does not
    clear storage: call clear after learning, then collect a new rollout.
    This buffer deliberately does not inherit replay's random sample interface.
    """

    def __init__(self, horizon: int, n_envs: int):
        for name, value in (("horizon", horizon), ("n_envs", n_envs)):
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.horizon, self.n_envs = int(horizon), int(n_envs)
        self._steps: list[dict] = []

    def add(self, observation, action, reward, next_observation, terminated, truncated, *, value, log_prob):
        """Snapshot a single vector step, preserving independent episode flags."""
        if len(self) == self.horizon:
            raise ValueError("RolloutBuffer is full; learn and clear before collecting again")
        if value is None or log_prob is None:
            raise ValueError("RolloutBuffer requires collection-time value and log_prob snapshots")
        step = {name: _copy_value(item) for name, item in dict(
            observation=observation, action=action, reward=reward,
            next_observation=next_observation, terminated=terminated, truncated=truncated,
            value=value, log_prob=log_prob,
        ).items()}
        self._validate(step)
        self._steps.append(step)

    def get(self) -> RolloutBatch:
        """Return an independent complete rollout; partial horizons are rejected."""
        if len(self) != self.horizon:
            raise ValueError("get requires a complete rollout of horizon vector steps")
        return RolloutBatch(**{name: torch.stack([step[name] for step in self._steps])
                               for name in self._steps[0]})

    def clear(self):
        self._steps.clear()

    def __len__(self):
        """Number of collected vector steps, not individual environment steps."""
        return len(self._steps)

    def _validate(self, step):
        observation, following = step["observation"], step["next_observation"]
        if observation.ndim < 1 or observation.shape[0] != self.n_envs:
            raise ValueError("observation must have shape (N, *obs_shape)")
        if following.shape != observation.shape or following.dtype != observation.dtype:
            raise ValueError("observation and next_observation must share shape and dtype")
        if observation.is_complex() or observation.dtype == torch.bool:
            raise TypeError("observation must have real numeric dtype")
        for name in ("action", "reward", "terminated", "truncated", "value", "log_prob"):
            if step[name].shape != (self.n_envs,):
                raise ValueError(f"{name} must have shape (N,)")
        action = step["action"]
        if action.is_floating_point() or action.is_complex() or action.dtype == torch.bool:
            raise TypeError("action must contain discrete integer indices")
        for name in ("reward", "value", "log_prob"):
            if not step[name].is_floating_point():
                raise TypeError(f"{name} must have floating dtype")
        for name in ("terminated", "truncated"):
            if step[name].dtype != torch.bool:
                raise TypeError(f"{name} must have bool dtype")
        if self._steps:
            previous = self._steps[-1]
            for name, item in step.items():
                if item.shape != previous[name].shape or item.dtype != previous[name].dtype:
                    raise ValueError(f"{name} must keep a consistent shape and dtype")
            continuing = ~(previous["terminated"] | previous["truncated"])
            if not torch.equal(observation[continuing], previous["next_observation"][continuing]):
                raise ValueError("Nonboundary observations must follow the previous next_observation")
