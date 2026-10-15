"""One-step actor-critic losses from supplied log probabilities, values and targets."""

from math import isfinite
from numbers import Real

from torch import Tensor

from modular_rl.data import ActorCriticOutput


class ActorCriticUpdateRule:
    """Calculate losses without model calls, sampling, backward or optimization.

    Inputs are matching nonempty floating Tensors (B,). Only the advantage
    used in the actor loss is detached. Critic/target gradients remain caller
    controlled; the learner supplies a no_grad TD target. entropy is per-state
    categorical entropy, not a sampled negative log probability.
    """

    def __init__(self, value_coef=0.5, entropy_coef=0.01):
        for name, value in (("value_coef", value_coef), ("entropy_coef", entropy_coef)):
            if not isinstance(value, Real) or isinstance(value, bool) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative real number")
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def compute(self, log_prob: Tensor, value: Tensor, target: Tensor, entropy: Tensor) -> ActorCriticOutput:
        self._validate(log_prob, value, target, entropy)
        td_error = target - value
        actor_loss = -(log_prob * td_error.detach()).mean()
        critic_loss = (0.5 * td_error.square()).mean()
        mean_entropy = entropy.mean()
        total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * mean_entropy
        return ActorCriticOutput(actor_loss, critic_loss, mean_entropy, total_loss, td_error)

    @staticmethod
    def _validate(log_prob, value, target, entropy):
        inputs = (log_prob, value, target, entropy)
        if any(not isinstance(tensor, Tensor) or not tensor.is_floating_point() for tensor in inputs):
            raise TypeError("log_prob, value, target and entropy must be floating Tensors")
        if value.ndim != 1 or value.numel() == 0 or any(t.shape != value.shape for t in inputs):
            raise ValueError("Actor-critic inputs must have the same nonempty (B,) shape")
        if any(t.dtype != value.dtype for t in inputs):
            raise TypeError("Actor-critic inputs must have the same floating dtype")
        if any(t.device != value.device for t in inputs):
            raise ValueError("Actor-critic inputs must be on the same device")
