"""Immediate one-step categorical actor-critic with a shared representation."""

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.optim import Optimizer

from modular_rl.data import ActorCriticOutput, TransitionBatch
from modular_rl.learner._evaluation import evaluating
from modular_rl.model import Model


class ActorCriticLearner:
    """Learn one fresh on-policy transition (B=1) immediately after env.step.

    The caller selects an action with Model.act, then calls learn before the
    next action. Historical detached batch.log_prob is NOT a gradient source:
    current forward recomputes the categorical action log probability.
    No replay, environment, target network or action selection is owned here.
    Freshness cannot be inferred from TransitionBatch; it is caller discipline.

    A single optimizer owns policy, value and shared representation parameters.
    Current-state forward uses the caller's mode. Bootstrap temporarily uses
    eval/no_grad, preventing Dropout sampling and BatchNorm buffer updates;
    mixed module modes are restored. Natural termination masks bootstrap;
    time-limit truncation retains it using the actual final observation.

    Returned ActorCriticOutput contains detached pre-update scalar metrics
    and signed td_error (1,), with the model's floating dtype/device. Save
    Model and optimizer state_dicts plus collection RNGs to resume training.
    """

    def __init__(self, model, optimizer, target_generator, update_rule):
        self.model = model
        self.optimizer = optimizer
        self.target_generator = target_generator
        self.update_rule = update_rule
        self._validate_setup()

    def learn(self, batch: TransitionBatch) -> ActorCriticOutput:
        self._validate(batch)
        reference = next(self.model.parameters())
        batch = batch.to(reference.device, dtype=reference.dtype)
        actions = batch.action.long()

        estimate = self.model(batch.observation)
        policy, value = self._outputs(estimate, reference)
        log_prob = policy.log_prob(actions)
        entropy = policy.entropy()
        with torch.no_grad(), evaluating(self.model):
            next_value = self._value(self.model(batch.next_observation), reference)
            target = self.target_generator.generate(batch.reward, next_value, batch.terminated)

        result = self.update_rule.compute(log_prob, value, target, entropy)
        self.optimizer.zero_grad()
        result.total_loss.backward()
        self.optimizer.step()
        return ActorCriticOutput(
            result.actor_loss.detach(), result.critic_loss.detach(), result.entropy.detach(),
            result.total_loss.detach(), result.td_error.detach(),
        )

    def _validate_setup(self):
        if not isinstance(self.model, Model):
            raise TypeError("ActorCriticLearner requires a Model")
        parameters = list(self.model.parameters())
        if not parameters or not any(p.requires_grad for p in parameters):
            raise ValueError("Actor-critic Model must have trainable floating parameters")
        reference = parameters[0]
        if any(not p.is_floating_point() or p.dtype != reference.dtype or p.device != reference.device
               for p in parameters):
            raise ValueError("Model parameters must share floating dtype and device")
        if not isinstance(self.optimizer, Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        model_ids = {id(p) for p in parameters}
        optimized = [p for group in self.optimizer.param_groups for p in group["params"]]
        if not optimized or any(id(p) not in model_ids for p in optimized):
            raise ValueError("Optimizer must contain only Actor-Critic Model parameters")

    @staticmethod
    def _validate(batch):
        if not isinstance(batch, TransitionBatch):
            raise TypeError("learn expects a fresh TransitionBatch with B=1")
        if batch.observation.ndim < 1 or batch.observation.shape[0] != 1:
            raise ValueError("ActorCriticLearner supports immediate on-policy B=1 updates only")
        if batch.next_observation.shape != batch.observation.shape:
            raise ValueError("observation and next_observation must have the same shape")
        if any(getattr(batch, name).shape != (1,) for name in ("action", "reward", "terminated", "truncated")):
            raise ValueError("action, reward and boundary flags must have shape (1,)")
        if batch.action.is_floating_point() or batch.action.is_complex() or batch.action.dtype == torch.bool:
            raise TypeError("action must have integer index dtype")
        if not batch.reward.is_floating_point():
            raise TypeError("reward must have floating dtype")
        if batch.terminated.dtype != torch.bool or batch.truncated.dtype != torch.bool:
            raise TypeError("terminated and truncated must have bool dtype")

    @staticmethod
    def _value(estimate, reference):
        if estimate.values is None or estimate.values.v is None:
            raise ValueError("Actor-critic requires EstimatorOutput.values.v")
        value = estimate.values.v
        if not isinstance(value, Tensor) or value.shape != (1,):
            raise ValueError("Actor-critic requires scalar state values with shape (1,)")
        if value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError("State value must match Model floating dtype and device")
        return value

    @classmethod
    def _outputs(cls, estimate, reference):
        policy = estimate.policy
        if not isinstance(policy, Categorical) or policy.batch_shape != (1,):
            raise ValueError("Actor-critic requires a Categorical policy with batch shape (1,)")
        return policy, cls._value(estimate, reference)
