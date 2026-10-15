"""Discounted rewards plus an externally evaluated endpoint bootstrap."""

from numbers import Real

import torch
from torch import Tensor


class NStepTargetGenerator:
    """Compute sum(gamma**k * reward[k]) + gamma**m * nonterminal * next_value.

    rewards, terminated and valid have shape (B, T); next_value has shape (B,).
    valid is a nonempty contiguous prefix per row, defining actual horizon m.
    A terminal transition can only be the last valid step. Padding is ignored.
    A short row's real episode boundary is the caller/buffer's responsibility;
    truncated-only endpoints retain bootstrap. Caller chooses next_value and
    controls no_grad/detach; this component preserves gradients and devices.

    This reward-only return does not implement Tree Backup or Q(sigma), which
    also need intermediate action values, probabilities and correction terms.
    """

    def __init__(self, gamma: float) -> None:
        if not isinstance(gamma, Real) or isinstance(gamma, bool) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be a real number in [0, 1]")
        self.gamma = float(gamma)

    def generate(self, rewards: Tensor, next_value: Tensor, terminated: Tensor, valid: Tensor) -> Tensor:
        self._validate(rewards, next_value, terminated, valid)
        horizon = valid.sum(dim=1)
        reward_sum = torch.zeros_like(next_value)
        for time in reversed(range(rewards.shape[1])):
            reward = rewards[:, time].masked_fill(~valid[:, time], 0)
            candidate = reward + self.gamma * reward_sum
            reward_sum = torch.where(valid[:, time], candidate, reward_sum)
        last_terminated = terminated.gather(1, (horizon - 1).unsqueeze(1)).squeeze(1)
        bootstrap = next_value.masked_fill(last_terminated, 0)
        continuation = self.gamma ** (horizon.to(rewards.dtype) - 1) * bootstrap
        return reward_sum + self.gamma * continuation

    @staticmethod
    def _validate(rewards: Tensor, next_value: Tensor, terminated: Tensor, valid: Tensor) -> None:
        if not all(isinstance(value, Tensor) for value in (rewards, next_value, terminated, valid)):
            raise TypeError("rewards, next_value, terminated and valid must be Tensors")
        if rewards.ndim != 2 or 0 in rewards.shape or next_value.shape != rewards.shape[:1]:
            raise ValueError("rewards must be nonempty (B, T) and next_value must be (B,)")
        if terminated.shape != rewards.shape or valid.shape != rewards.shape:
            raise ValueError("terminated and valid must have the same (B, T) shape as rewards")
        if not rewards.is_floating_point() or not next_value.is_floating_point() or rewards.dtype != next_value.dtype:
            raise TypeError("rewards and next_value must have the same floating dtype")
        if terminated.dtype != torch.bool or valid.dtype != torch.bool:
            raise TypeError("terminated and valid must have bool dtype")
        if any(value.device != rewards.device for value in (next_value, terminated, valid)):
            raise ValueError("All inputs must be on the same device")
        if not valid[:, 0].all() or (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError("valid must be a nonempty contiguous prefix in each row")
        if (terminated[:, :-1] & valid[:, 1:]).any():
            raise ValueError("A terminal transition must be the last valid step")
