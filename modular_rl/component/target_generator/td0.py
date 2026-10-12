"""One-step TD targets with externally computed bootstrap values."""

from numbers import Real

import torch
from torch import Tensor

from modular_rl.component.target_generator.base import BaseTargetGenerator


class TD0TargetGenerator(BaseTargetGenerator):
    """Compute reward + gamma * (1 - terminated) * next_value.

    All inputs have identical shape and device. reward and next_value share
    a floating dtype; terminated has bool dtype. Scalar (), replay (B,) and
    additional leading dimensions such as (T, B) are accepted without
    broadcasting. The output preserves the reward shape, dtype and device.

    The caller selects next_value (V, next-action Q, max Q, etc.) and manages
    no_grad/detach. This component preserves autograd and does not call a
    model, sample a buffer, or infer trajectory boundaries from truncation.
    """

    def __init__(self, gamma: float) -> None:
        if not isinstance(gamma, Real) or isinstance(gamma, bool) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be a real number in [0, 1]")
        self.gamma = float(gamma)

    def generate(self, reward: Tensor, next_value: Tensor, terminated: Tensor) -> Tensor:
        """Mask terminal bootstrap values, then apply the one-step TD formula."""
        self._validate(reward, next_value, terminated)
        bootstrap_value = next_value.masked_fill(terminated, 0)
        return reward + self.gamma * bootstrap_value

    @staticmethod
    def _validate(reward: Tensor, next_value: Tensor, terminated: Tensor) -> None:
        """Reject broadcasting and ambiguous numeric/device contracts."""
        if not all(isinstance(value, Tensor) for value in (reward, next_value, terminated)):
            raise TypeError("reward, next_value and terminated must be Tensors")
        if reward.shape != next_value.shape or reward.shape != terminated.shape:
            raise ValueError("reward, next_value and terminated must have the same shape")
        if not reward.is_floating_point() or not next_value.is_floating_point():
            raise TypeError("reward and next_value must have floating dtype")
        if reward.dtype != next_value.dtype:
            raise TypeError("reward and next_value must have the same dtype")
        if terminated.dtype != torch.bool:
            raise TypeError("terminated must have bool dtype")
        if reward.device != next_value.device or reward.device != terminated.device:
            raise ValueError("reward, next_value and terminated must be on the same device")
