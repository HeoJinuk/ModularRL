"""Time-major on-policy experience from parallel environments."""

from dataclasses import dataclass, fields

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """A complete, ordered rollout with leading (T, N) dimensions.

    observation/next_observation: (T, N, *obs_shape); all other fields: (T, N).
    Actions are discrete integer indices; flags are bool. value and log_prob
    are detached collection-time snapshots, not gradient sources. A boundary's
    next_observation is the actual final observation, never the autoreset state.

    RolloutBuffer returns owned CPU tensors. .to() casts floating fields only;
    unchanged tensors can share storage, as with Tensor.to(). A frozen dataclass
    does not make its tensors immutable. No padding or replay sampling is used.
    """

    observation: Tensor
    action: Tensor
    reward: Tensor
    next_observation: Tensor
    terminated: Tensor
    truncated: Tensor
    value: Tensor
    log_prob: Tensor

    def to(self, device: torch.device | str | None = None, *, dtype: torch.dtype | None = None) -> "RolloutBatch":
        if dtype is not None and not dtype.is_floating_point:
            raise ValueError("Batch dtype must be floating-point to preserve integers and booleans")
        return RolloutBatch(**{
            field.name: getattr(self, field.name).to(
                device=device, dtype=dtype if getattr(self, field.name).is_floating_point()
                else getattr(self, field.name).dtype
            ) for field in fields(self)
        })
