"""Clipped PPO losses from supplied predictions and fixed rollout estimates."""

from math import isfinite
from numbers import Real

import torch
from torch import Tensor

from modular_rl.data import PPOOutput


class PPOUpdateRule:
    """Compute matching floating (B,) losses without model/optimizer access.

    Old log probabilities and actor advantages are detached. Critic returns
    retain caller-controlled gradients; standard PPO learners supply no_grad
    targets. No value clipping, advantage normalization or KL early stopping.
    Nonfinite inputs, ratios or losses fail rather than being silently clipped.
    """

    def __init__(self, clip_epsilon=.2, value_coef=.5, entropy_coef=.01):
        for name, value in (("clip_epsilon", clip_epsilon), ("value_coef", value_coef),
                            ("entropy_coef", entropy_coef)):
            if not isinstance(value, Real) or isinstance(value, bool) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative real number")
        if clip_epsilon >= 1:
            raise ValueError("clip_epsilon must be in [0, 1)")
        self.clip_epsilon, self.value_coef, self.entropy_coef = clip_epsilon, value_coef, entropy_coef

    def compute(self, new_log_prob: Tensor, old_log_prob: Tensor, current_value: Tensor,
                returns: Tensor, advantage: Tensor, entropy: Tensor) -> PPOOutput:
        self._validate(new_log_prob, old_log_prob, current_value, returns, advantage, entropy)
        log_ratio = new_log_prob - old_log_prob.detach()
        ratio = log_ratio.exp()
        self._validate_finite(ratio, "PPO ratio")
        advantage = advantage.detach()
        surrogate = ratio * advantage
        clipped = ratio.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
        policy_loss = -torch.minimum(surrogate, clipped).mean()
        value_loss = .5 * (current_value - returns).square().mean()
        mean_entropy = entropy.mean()
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * mean_entropy
        approx_kl = (ratio - 1 - log_ratio).mean().detach()
        clip_fraction = ((ratio - 1).abs() > self.clip_epsilon).to(ratio.dtype).mean().detach()
        for metric in (policy_loss, value_loss, total_loss, approx_kl):
            self._validate_finite(metric, "PPO loss/diagnostic")
        return PPOOutput(policy_loss, value_loss, mean_entropy, total_loss, approx_kl, clip_fraction)

    @classmethod
    def _validate(cls, *inputs):
        if any(not isinstance(item, Tensor) or not item.is_floating_point() for item in inputs):
            raise TypeError("PPO inputs must be floating Tensors")
        reference = inputs[0]
        if reference.ndim != 1 or reference.numel() == 0 or any(t.shape != reference.shape for t in inputs):
            raise ValueError("PPO inputs must share a nonempty (B,) shape")
        if any(t.dtype != reference.dtype for t in inputs):
            raise TypeError("PPO inputs must share a floating dtype")
        if any(t.device != reference.device for t in inputs):
            raise ValueError("PPO inputs must be on the same device")
        for item in inputs:
            cls._validate_finite(item, "PPO input")

    @staticmethod
    def _validate_finite(tensor, name):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite")
