"""Generalized advantage estimation with separate bootstrap and trace masks."""

from numbers import Real

import torch
from torch import Tensor

from modular_rl.data.advantages import GAEOutput


class GAEGenerator:
    """Estimate advantage/returns from matching time-major (T,N) tensors.

    Termination removes bootstrap; either boundary stops the advantage trace.
    next_value is evaluated externally at each actual final/next observation.
    No model calls, conversions or detach: learners own gradient boundaries.
    At lambda=1, equivalence to RolloutReturn requires next_value[t] ==
    value[t+1] for continuing transitions inside the rollout.
    """

    def __init__(self, gamma=.99, lam=.95):
        for name, number in (("gamma", gamma), ("lam", lam)):
            if not isinstance(number, Real) or isinstance(number, bool) or not 0 <= number <= 1:
                raise ValueError(f"{name} must be a real number in [0, 1]")
        self.gamma, self.lam = float(gamma), float(lam)

    def generate(self, reward: Tensor, value: Tensor, next_value: Tensor,
                 terminated: Tensor, truncated: Tensor) -> GAEOutput:
        self._validate(reward, value, next_value, terminated, truncated)
        delta = reward + self.gamma * next_value.masked_fill(terminated, 0) - value
        future_advantage = torch.zeros_like(value[-1])
        advantages = []
        for time in reversed(range(reward.shape[0])):
            continuation = future_advantage.masked_fill(terminated[time] | truncated[time], 0)
            future_advantage = delta[time] + self.gamma * self.lam * continuation
            advantages.append(future_advantage)
        advantage = torch.stack(advantages[::-1])
        return GAEOutput(advantage, advantage + value)

    @staticmethod
    def _validate(reward, value, next_value, terminated, truncated):
        inputs = (reward, value, next_value, terminated, truncated)
        if any(not isinstance(item, Tensor) for item in inputs):
            raise TypeError("GAE inputs must be Tensors")
        if reward.ndim != 2 or 0 in reward.shape or any(item.shape != reward.shape for item in inputs):
            raise ValueError("GAE inputs must share a nonempty (T,N) shape")
        if not reward.is_floating_point() or any(item.dtype != reward.dtype for item in (value, next_value)):
            raise TypeError("reward, value and next_value must share a floating dtype")
        if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
            raise TypeError("terminated and truncated must have bool dtype")
        if any(item.device != reward.device for item in inputs):
            raise ValueError("GAE inputs must be on the same device")
