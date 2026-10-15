"""Uniform replay with owned CPU snapshots and an independent NumPy RNG."""

from dataclasses import fields
from numbers import Integral

import numpy as np
import torch

from modular_rl.component.buffer.base import BaseBuffer
from modular_rl.component.buffer._transition import snapshot, transition_from_args, validate
from modular_rl.data import Transition, TransitionBatch


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

    def add(self, *args, **kwargs) -> None:
        """Store Transition, transition=Transition, or its positional/keyword fields."""
        stored = self._snapshot(transition_from_args(args, kwargs))
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

    _snapshot = staticmethod(snapshot)

    def _validate(self, transition: Transition) -> None:
        validate(transition, self._data[0] if self._data else None)

    def _validate_sample(self, batch_size: int) -> None:
        if not isinstance(batch_size, Integral) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if batch_size > len(self):
            raise ValueError("batch_size cannot exceed buffer length when sampling without replacement")
