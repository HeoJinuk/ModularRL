"""Boundary-aware multi-step returns for time-major vector rollouts."""

from numbers import Real

import torch
from torch import Tensor


class RolloutReturnGenerator:
    """Generate returns (T,N) from rewards and actual next-state values (T,N).

    Continuing steps trace into the next collected return. Truncation uses
    that transition's final-observation value and stops the reward trace.
    Natural termination removes bootstrap, including when both flags are true.
    The last continuing step bootstraps from next_value[-1]. No model, implicit
    dtype conversion or detach is used; the caller manages gradient boundaries.
    This is neither GAE nor replay's endpoint-only NStep return.
    """

    def __init__(self, gamma: float = .99):
        if not isinstance(gamma, Real) or isinstance(gamma, bool) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be a real number in [0, 1]")
        self.gamma = float(gamma)

    def generate(self, reward: Tensor, next_value: Tensor, terminated: Tensor, truncated: Tensor) -> Tensor:
        self._validate(reward, next_value, terminated, truncated)
        future_return = next_value[-1]
        returns = []
        for time in reversed(range(reward.shape[0])):
            bootstrap = torch.where(truncated[time], next_value[time], future_return)
            bootstrap = bootstrap.masked_fill(terminated[time], 0)
            future_return = reward[time] + self.gamma * bootstrap
            returns.append(future_return)
        return torch.stack(returns[::-1])

    @staticmethod
    def _validate(reward, next_value, terminated, truncated):
        inputs = (reward, next_value, terminated, truncated)
        if any(not isinstance(item, Tensor) for item in inputs):
            raise TypeError("Rollout return inputs must be Tensors")
        if reward.ndim != 2 or 0 in reward.shape or any(item.shape != reward.shape for item in inputs):
            raise ValueError("All rollout return inputs must share a nonempty (T,N) shape")
        if not reward.is_floating_point() or next_value.dtype != reward.dtype:
            raise TypeError("reward and next_value must share a floating dtype")
        if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
            raise TypeError("terminated and truncated must have bool dtype")
        if any(item.device != reward.device for item in inputs):
            raise ValueError("Rollout return inputs must be on the same device")
