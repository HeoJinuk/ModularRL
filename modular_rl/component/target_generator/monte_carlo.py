"""Complete-episode discounted returns; no bootstrap input or TD base class."""

from numbers import Real

import torch
from torch import Tensor


class MonteCarloReturnGenerator:
    """Generate G_t = reward[t] + gamma * G_{t+1}, with final continuation zero.

    Input/output are floating Tensors (T,) on the same dtype/device. The
    caller must supply rewards from a complete episode; rewards alone cannot
    establish termination. This arithmetic preserves gradients; Learner owns
    no_grad/detach. BaseTargetGenerator's bootstrap interface is not inherited.
    """

    def __init__(self, gamma: float) -> None:
        if not isinstance(gamma, Real) or isinstance(gamma, bool) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be a finite real number in [0, 1]")
        self.gamma = float(gamma)

    def generate(self, rewards: Tensor) -> Tensor:
        self._validate(rewards)
        returns = torch.empty_like(rewards)
        g = rewards.new_zeros(())
        for time in range(rewards.numel() - 1, -1, -1):
            g = rewards[time] + self.gamma * g
            returns[time] = g
        self._validate_returns(returns)
        return returns

    @staticmethod
    def _validate(rewards: Tensor) -> None:
        if not isinstance(rewards, Tensor):
            raise TypeError("rewards must be a Tensor")
        if rewards.ndim != 1 or rewards.numel() == 0:
            raise ValueError("rewards must have nonempty episode shape (T,)")
        if not rewards.is_floating_point():
            raise TypeError("rewards must have floating dtype")
        if not torch.isfinite(rewards).all():
            raise ValueError("rewards must be finite")

    @staticmethod
    def _validate_returns(returns: Tensor) -> None:
        if not torch.isfinite(returns).all():
            raise FloatingPointError("Monte Carlo returns became non-finite; check the numerical scale")
