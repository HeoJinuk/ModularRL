"""Synchronous advantage actor-critic from fresh time-major vector rollouts."""

from dataclasses import fields

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.optim import Optimizer

from modular_rl.data import ActorCriticOutput
from modular_rl.data.rollout import RolloutBatch
from modular_rl.learner._evaluation import evaluating
from modular_rl.model import Model


class A2CLearner:
    """Consume one fresh rollout and take one optimizer step over T*N samples.

    Collection is external and must hold policy parameters fixed. Recomputed
    value/log_prob must match collection snapshots within floating tolerance;
    this detects observable policy drift, not every possible parameter change.
    Use consistent deterministic forwards (e.g. model.eval() for Dropout/BN,
    which still allows gradients). Stored log_prob never supplies gradients.

    The same ActorCriticUpdateRule as one-step AC handles the flattened samples.
    Bootstrap values/returns use eval/no_grad. Returned scalar metrics and
    signed multi-step td_error (T,N) are detached, in model dtype/device.
    Learner owns no environment/buffer and implements neither GAE nor PPO.
    Save Model/optimizer state and collection RNGs together for checkpoints.
    """

    def __init__(self, model, optimizer, target_generator, update_rule):
        self.model, self.optimizer = model, optimizer
        self.target_generator, self.update_rule = target_generator, update_rule
        self._validate_setup()

    def learn(self, rollout: RolloutBatch) -> ActorCriticOutput:
        self._validate(rollout)
        reference = next(self.model.parameters())
        rollout = rollout.to(reference.device, dtype=reference.dtype)
        time, n_envs = rollout.reward.shape
        count = time * n_envs
        observations = rollout.observation.reshape(count, *rollout.observation.shape[2:])
        following = rollout.next_observation.reshape(count, *rollout.next_observation.shape[2:])

        estimate = self.model(observations)
        policy, value = self._outputs(estimate, count, reference)
        log_prob = policy.log_prob(rollout.action.long().reshape(count))
        self._validate_freshness(rollout, value, log_prob)
        with torch.no_grad(), evaluating(self.model):
            next_value = self._value(self.model(following), count, reference).reshape(time, n_envs)
            target = self.target_generator.generate(
                rollout.reward, next_value, rollout.terminated, rollout.truncated,
            )

        result = self.update_rule.compute(log_prob, value, target.reshape(count), policy.entropy())
        self.optimizer.zero_grad()
        result.total_loss.backward()
        self.optimizer.step()
        return ActorCriticOutput(
            result.actor_loss.detach(), result.critic_loss.detach(), result.entropy.detach(),
            result.total_loss.detach(), result.td_error.detach().reshape(time, n_envs),
        )

    def _validate_setup(self):
        if not isinstance(self.model, Model):
            raise TypeError("A2CLearner requires a Model")
        parameters = list(self.model.parameters())
        if not parameters or not any(p.requires_grad for p in parameters):
            raise ValueError("A2C Model must have trainable floating parameters")
        reference = parameters[0]
        if any(not p.is_floating_point() or p.dtype != reference.dtype or p.device != reference.device
               for p in parameters):
            raise ValueError("Model parameters must share floating dtype and device")
        if not isinstance(self.optimizer, Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        model_ids = {id(p) for p in parameters}
        optimized = [p for group in self.optimizer.param_groups for p in group["params"]]
        if not optimized or any(id(p) not in model_ids for p in optimized):
            raise ValueError("Optimizer must contain only A2C Model parameters")

    @staticmethod
    def _validate(rollout):
        if not isinstance(rollout, RolloutBatch):
            raise TypeError("learn expects a fresh RolloutBatch")
        tensors = (getattr(rollout, field.name) for field in fields(rollout))
        if any(not isinstance(item, Tensor) for item in tensors):
            raise TypeError("RolloutBatch fields must be Tensors")
        shape = rollout.reward.shape
        if len(shape) != 2 or 0 in shape:
            raise ValueError("Rollout scalar fields must have nonempty (T,N) shape")
        if rollout.observation.shape[:2] != shape or rollout.next_observation.shape != rollout.observation.shape:
            raise ValueError("Rollout observations must have shape (T,N,*obs_shape)")
        if any(getattr(rollout, name).shape != shape for name in
               ("action", "terminated", "truncated", "value", "log_prob")):
            raise ValueError("Rollout action, flags, value and log_prob must have shape (T,N)")
        if rollout.action.is_floating_point() or rollout.action.is_complex() or rollout.action.dtype == torch.bool:
            raise TypeError("action must contain discrete integer indices")
        if any(not getattr(rollout, name).is_floating_point() for name in ("reward", "value", "log_prob")):
            raise TypeError("reward, value and log_prob must have floating dtype")
        if rollout.terminated.dtype != torch.bool or rollout.truncated.dtype != torch.bool:
            raise TypeError("terminated and truncated must have bool dtype")

    @staticmethod
    def _validate_freshness(rollout, value, log_prob):
        if not torch.allclose(value.detach(), rollout.value.reshape(-1), rtol=1e-5, atol=1e-6) or not torch.allclose(
            log_prob.detach(), rollout.log_prob.reshape(-1), rtol=1e-5, atol=1e-6,
        ):
            raise ValueError("On-policy rollout snapshots differ from current Model; parameters or forward modes changed")

    @staticmethod
    def _value(estimate, count, reference):
        if estimate.values is None or estimate.values.v is None:
            raise ValueError("A2C requires EstimatorOutput.values.v")
        value = estimate.values.v
        if not isinstance(value, Tensor) or value.shape != (count,):
            raise ValueError("A2C requires scalar state values with shape (T*N,)")
        if value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError("State value must match Model floating dtype and device")
        return value

    @classmethod
    def _outputs(cls, estimate, count, reference):
        if not isinstance(estimate.policy, Categorical) or estimate.policy.batch_shape != (count,):
            raise ValueError("A2C requires a Categorical policy with batch shape (T*N,)")
        return estimate.policy, cls._value(estimate, count, reference)
