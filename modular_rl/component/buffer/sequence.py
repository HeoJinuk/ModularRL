"""Uniform replay of ordered windows from one sequential environment."""

from collections import deque
from dataclasses import fields
from numbers import Integral

import numpy as np
import torch

from modular_rl.component.buffer._transition import snapshot, transition_from_args, validate
from modular_rl.data import SequenceBatch, Transition


class SequenceReplayBuffer:
    """Store at most capacity transitions, retaining chronological order.

    add accepts the same forms and homogeneous field schema as ReplayBuffer.
    Each non-boundary next_observation must equal the following observation.
    This contract supports one sequential environment, not interleaved streams.
    terminated OR truncated closes a window; an unfinished short suffix is
    ineligible until more steps arrive. The collector supplies final observations.

    sample returns distinct uniformly chosen starting positions, zero padding,
    valid masks and actual horizons. It computes no returns or bootstrap values.
    Storage and returned tensors are independent detached CPU snapshots. clear
    resets storage/schema, preserving the private RNG's current state.
    """

    def __init__(self, capacity: int, *, seed: int | None = None) -> None:
        if not isinstance(capacity, Integral) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = int(capacity)
        self._data: deque[Transition] = deque(maxlen=self.capacity)
        self._boundaries: deque[bool] = deque(maxlen=self.capacity)
        self._rng = np.random.default_rng(seed)

    def add(self, *args, **kwargs) -> None:
        """Snapshot a Transition or its fields; reject discontinuity before storing."""
        stored = snapshot(transition_from_args(args, kwargs))
        validate(stored, self._data[0] if self._data else None)
        if self._data and not self._boundaries[-1]:
            if not torch.equal(self._data[-1].next_observation, stored.observation):
                raise ValueError("Sequential transitions must have continuous observations")
        boundary = bool(stored.terminated or stored.truncated)
        self._data.append(stored)
        self._boundaries.append(boundary)

    def num_sequences(self, length: int) -> int:
        """Count eligible starts without consuming RNG; useful during warm-up."""
        return len(self._windows(length)[0])

    def sample(self, batch_size: int, *, length: int) -> SequenceBatch:
        """Sample B windows of at most length steps, padded to (B, length, ...)."""
        if not isinstance(batch_size, Integral) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        starts, horizons = self._windows(length)
        if batch_size > len(starts):
            raise ValueError("batch_size exceeds the number of eligible sequences")
        selected = self._rng.choice(len(starts), size=int(batch_size), replace=False)
        data = list(self._data)
        values = {}
        for field in fields(Transition):
            if field.name == "log_prob" and data[0].log_prob is None:
                values[field.name] = None
                continue
            rows = []
            for index in selected:
                start, horizon = starts[index], horizons[index]
                column = torch.stack([getattr(step, field.name) for step in data[start:start + horizon]])
                row = column.new_zeros((length, *column.shape[1:]))
                row[:horizon] = column
                rows.append(row)
            values[field.name] = torch.stack(rows)
        horizon = torch.tensor([horizons[index] for index in selected], dtype=torch.long)
        valid = torch.arange(length).unsqueeze(0) < horizon.unsqueeze(1)
        return SequenceBatch(**values, valid=valid, horizon=horizon)

    def clear(self) -> None:
        self._data.clear()
        self._boundaries.clear()

    def __len__(self) -> int:
        return len(self._data)

    def _windows(self, length: int) -> tuple[list[int], list[int]]:
        """Scan backward: a real boundary certifies all shorter tails before it."""
        if not isinstance(length, Integral) or isinstance(length, bool) or length <= 0:
            raise ValueError("length must be a positive integer")
        starts, horizons = [], []
        remaining, closed = 0, False
        boundaries = list(self._boundaries)
        for index in range(len(boundaries) - 1, -1, -1):
            if boundaries[index]:
                remaining, closed = 1, True
            else:
                remaining += 1
            if remaining >= length or closed:
                starts.append(index)
                horizons.append(min(length, remaining))
        return starts[::-1], horizons[::-1]
