"""Environment transitions and tensor batches, independent of RL algorithms."""

from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import Tensor

ExperienceValue = np.ndarray | np.generic | Tensor | int | float | bool


@dataclass(frozen=True, slots=True)
class Transition:
    """One real or simulated environment step, with no leading batch axis.

    Observations are numeric scalars/arrays/tensors of arbitrary shape. A
    discrete action is an integer scalar; a continuous action is a float
    vector (A,). Reward and the two boolean boundary flags are scalars.
    log_prob is an optional scalar behavior-policy log probability/density.

    Values are kept as supplied here. ReplayBuffer.add validates the contract
    and takes detached, owned CPU tensor snapshots. A frozen dataclass does
    not by itself make the input arrays or tensors immutable.
    """

    observation: ExperienceValue
    action: ExperienceValue
    reward: ExperienceValue
    next_observation: ExperienceValue
    terminated: bool | np.bool_ | np.ndarray | Tensor
    truncated: bool | np.bool_ | np.ndarray | Tensor
    log_prob: ExperienceValue | None = None

    def to_batch(self) -> "TransitionBatch":
        """Snapshot this transition as a B=1 batch without changing dtypes.

        Every present field is copied and detached, with one leading batch
        axis. Tensor devices are preserved; NumPy/Python inputs become CPU
        tensors. Learner still chooses the training device/floating dtype.
        This conversion does not validate an algorithm's input contract.
        """
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                values[field.name] = None
                continue
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value.copy())
            else:
                tensor = torch.as_tensor(value).detach().clone()
            values[field.name] = tensor.unsqueeze(0)
        return TransitionBatch(**values)


@dataclass(frozen=True, slots=True)
class TransitionBatch:
    """Replay samples with one leading batch dimension B, all as tensors.

    Observations: (B, *observation_shape), including (B,) for tabular states.
    Actions: (B,) for discrete actions or (B, A) for continuous actions.
    Reward, terminated, truncated and a present log_prob: (B,).
    Boundary flags have bool dtype. Other dtypes are preserved by replay.

    Replay returns detached CPU tensors. A future ordered trajectory uses
    (T, B, ...); this class does not represent or infer time ordering.
    """

    observation: Tensor
    action: Tensor
    reward: Tensor
    next_observation: Tensor
    terminated: Tensor
    truncated: Tensor
    log_prob: Tensor | None = None

    def to(
        self, device: torch.device | str | None = None, *, dtype: torch.dtype | None = None
    ) -> "TransitionBatch":
        """Move the batch; optionally cast floating fields only.

        Integer observations/actions/rewards and boolean flags retain dtype.
        The caller, typically Learner, chooses the training device and dtype.
        Like Tensor.to(), unchanged fields may share storage with this batch.
        """
        if dtype is not None and not dtype.is_floating_point:
            raise ValueError("Batch dtype must be floating-point to preserve integers and booleans")
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = None if value is None else value.to(
                device=device, dtype=dtype if value.is_floating_point() else value.dtype
            )
        return TransitionBatch(**values)
