"""One ordered episode, with owned snapshots of existing Transition records."""

from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import Tensor

from modular_rl.data.experience import Transition


def _copy_value(value) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    return torch.as_tensor(value, device="cpu")


@dataclass(frozen=True, slots=True)
class Episode:
    """A nonempty sequence of contiguous environment transitions, in time order.

    All present fields become owned, detached CPU Tensors with original
    NumPy/Tensor dtypes. There is no leading batch axis. Inputs may be reused
    after construction; frozen records do not make stored Tensor contents
    immutable, so callers must preserve this validated contract.

    No transition may follow terminated or truncated. The last record may
    be naturally terminated, time-limit truncated, or still open. MC requires
    terminated=True, including a simultaneous terminated/truncated ending;
    a truncated-only or open episode has an unknown full return.
    """

    transitions: tuple[Transition, ...]

    def __post_init__(self) -> None:
        snapshots = []
        for transition in self.transitions:
            if not isinstance(transition, Transition):
                raise TypeError("Episode expects Transition records")
            values = {}
            for field in fields(Transition):
                value = getattr(transition, field.name)
                if value is None and field.name != "log_prob":
                    raise ValueError(f"Episode {field.name} is required")
                values[field.name] = None if value is None else _copy_value(value)
            snapshots.append(Transition(**values))
        object.__setattr__(self, "transitions", tuple(snapshots))
        self._validate()

    def __len__(self) -> int:
        return len(self.transitions)

    @property
    def terminated(self) -> bool:
        return bool(self.transitions[-1].terminated.item())

    @property
    def truncated(self) -> bool:
        return bool(self.transitions[-1].truncated.item())

    def _validate(self) -> None:
        if not self.transitions:
            raise ValueError("Episode must contain at least one transition")
        for index, transition in enumerate(self.transitions):
            observation, next_observation = transition.observation, transition.next_observation
            if (observation.shape != next_observation.shape or observation.dtype != next_observation.dtype):
                raise ValueError("observation and next_observation must have the same shape and dtype")
            if observation.is_complex() or next_observation.is_complex():
                raise TypeError("Episode observations must be real numeric values")
            reward = transition.reward
            if reward.ndim != 0 or reward.is_complex() or reward.dtype == torch.bool:
                raise ValueError("Episode reward must be a real numeric scalar")
            if not torch.isfinite(reward):
                raise ValueError("Episode reward must be finite")
            for name in ("terminated", "truncated"):
                flag = getattr(transition, name)
                if flag.ndim != 0 or flag.dtype != torch.bool:
                    raise TypeError("Episode boundary flags must be bool scalars")
            if index + 1 < len(self.transitions):
                if transition.terminated or transition.truncated:
                    raise ValueError("Episode cannot continue after terminated or truncated")
                if not torch.equal(next_observation, self.transitions[index + 1].observation):
                    raise ValueError("Episode transitions must be contiguous in time")
