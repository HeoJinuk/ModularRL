"""Uniform replay with owned CPU snapshots and an independent NumPy RNG."""

from dataclasses import fields
from numbers import Integral

import numpy as np
import torch
from torch import Tensor

from modular_rl.component.buffer.base import BaseBuffer
from modular_rl.data import Transition, TransitionBatch
from modular_rl.data.experience import ExperienceValue


def _copy_value(value: ExperienceValue) -> Tensor:
    """Keep dtype while removing aliases to caller-owned data and graphs."""
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    return torch.as_tensor(value, device="cpu")


class ReplayBuffer(BaseBuffer):
    """A capacity-limited ring of single transitions, sampled without replacement.

    Each field has a consistent shape and dtype throughout one buffer.
    Observations retain their integer or floating dtype. log_prob is either
    present throughout the buffer or absent throughout; missing values are
    never replaced with an invented probability. clear() resets this schema.

    add() snapshots every field to CPU and detaches tensor graphs. sample()
    returns new stacked tensors, so modifying a batch cannot change storage.
    Randomness uses a private generator; clear() does not reset its state.
    """

    def __init__(self, capacity: int, *, seed: int | None = None) -> None:
        if not isinstance(capacity, Integral) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = int(capacity)
        self._data: list[Transition] = []
        self._position = 0
        self._rng = np.random.default_rng(seed)

    def add(self, transition: Transition) -> None:
        stored = self._snapshot(transition)
        self._validate(stored)
        if len(self._data) < self.capacity:
            self._data.append(stored)
        else:
            self._data[self._position] = stored
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> TransitionBatch:
        """Uniformly sample B distinct transitions; require 1 <= B <= len(self)."""
        self._validate_sample(batch_size)
        indices = self._rng.choice(len(self._data), size=int(batch_size), replace=False)
        selected = [self._data[index] for index in indices]
        values = {}
        for item in fields(TransitionBatch):
            column = [getattr(transition, item.name) for transition in selected]
            values[item.name] = None if column[0] is None else torch.stack(column)
        return TransitionBatch(**values)

    def clear(self) -> None:
        self._data.clear()
        self._position = 0

    def __len__(self) -> int:
        return len(self._data)

    @staticmethod
    def _snapshot(transition: Transition) -> Transition:
        """Convert a single transition into independent CPU tensor values."""
        if not isinstance(transition, Transition):
            raise TypeError("ReplayBuffer.add expects a Transition")
        return Transition(
            observation=_copy_value(transition.observation),
            action=_copy_value(transition.action),
            reward=_copy_value(transition.reward),
            next_observation=_copy_value(transition.next_observation),
            terminated=_copy_value(transition.terminated),
            truncated=_copy_value(transition.truncated),
            log_prob=None if transition.log_prob is None else _copy_value(transition.log_prob),
        )

    def _validate(self, transition: Transition) -> None:
        """Check scalar semantics and a homogeneous field schema, once at add."""
        observation, next_observation = transition.observation, transition.next_observation
        if observation.is_complex() or next_observation.is_complex():
            raise TypeError("Observations must be real numeric values")
        if observation.shape != next_observation.shape or observation.dtype != next_observation.dtype:
            raise ValueError("observation and next_observation must have the same shape and dtype")

        action = transition.action
        if action.ndim > 1 or action.numel() == 0:
            raise ValueError("action must be an integer scalar or a nonempty float vector (A,)")
        if action.ndim == 0:
            if action.is_floating_point() or action.is_complex() or action.dtype == torch.bool:
                raise TypeError("A scalar discrete action must have integer dtype")
        elif not action.is_floating_point():
            raise TypeError("A vector continuous action must have floating dtype")

        reward = transition.reward
        if reward.ndim != 0:
            raise ValueError("reward must be a scalar")
        if reward.is_complex() or reward.dtype == torch.bool:
            raise TypeError("reward must be a real numeric scalar")
        for name in ("terminated", "truncated"):
            flag = getattr(transition, name)
            if flag.ndim != 0:
                raise ValueError(f"{name} must be a scalar")
            if flag.dtype != torch.bool:
                raise TypeError(f"{name} must have bool dtype")
        if transition.log_prob is not None:
            if transition.log_prob.ndim != 0:
                raise ValueError("log_prob must be a scalar summed over the action event")
            if not transition.log_prob.is_floating_point():
                raise TypeError("log_prob must have floating dtype")

        if self._data:
            reference = self._data[0]
            if (transition.log_prob is None) != (reference.log_prob is None):
                raise ValueError("ReplayBuffer cannot mix present and missing log_prob")
            # Next observation matches observation; flag shape/dtype are already fixed.
            for name in ("observation", "action", "reward", "log_prob"):
                value, previous = getattr(transition, name), getattr(reference, name)
                if value is not None and (value.shape != previous.shape or value.dtype != previous.dtype):
                    raise ValueError(f"{name} must have a consistent shape and dtype within a buffer")

    def _validate_sample(self, batch_size: int) -> None:
        if not isinstance(batch_size, Integral) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if batch_size > len(self):
            raise ValueError("batch_size cannot exceed buffer length when sampling without replacement")
