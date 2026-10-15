"""Ordered replay windows; B sampled rows, T ordered time positions."""

from dataclasses import dataclass, fields

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class SequenceBatch:
    """Owned replay tensors with leading (B, T) dimensions.

    Observations: (B, T, *observation_shape). Actions: (B, T) discrete or
    (B, T, A) continuous. Rewards, boundary flags, valid and optional log_prob:
    (B, T). horizon: long (B,). valid is a nonempty prefix in each row;
    horizon counts that prefix. Zero padding carries no experience meaning.

    Each row belongs to one episode in one sequential environment stream.
    A short row ends at an observed terminated or truncated boundary. Replay
    never invents a boundary for an unfinished short window. This is separate
    from TransitionBatch (unordered samples) and vectorized trajectories.
    """

    observation: Tensor
    action: Tensor
    reward: Tensor
    next_observation: Tensor
    terminated: Tensor
    truncated: Tensor
    valid: Tensor
    horizon: Tensor
    log_prob: Tensor | None = None

    def to(self, device: torch.device | str | None = None, *, dtype: torch.dtype | None = None) -> "SequenceBatch":
        """Move tensors and cast floating fields only, preserving indices/masks."""
        if dtype is not None and not dtype.is_floating_point:
            raise ValueError("Batch dtype must be floating-point to preserve integers and booleans")
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = None if value is None else value.to(
                device=device, dtype=dtype if value.is_floating_point() else value.dtype
            )
        return SequenceBatch(**values)
