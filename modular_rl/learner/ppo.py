"""PPO with fixed GAE targets and independently shuffled minibatches."""

from dataclasses import fields
from math import isfinite
from numbers import Integral, Real

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.optim import Optimizer

from modular_rl.data import PPOOutput, RolloutBatch
from modular_rl.learner._evaluation import evaluating
from modular_rl.model import Model


class PPOLearner:
    """Learn from one fresh discrete-policy rollout using multiple data passes.

    Collection holds parameters fixed. A one-time value/log_prob comparison
    detects observable drift, not ownership or every possible parameter change.
    Use consistent deterministic forwards, e.g. eval() for Dropout/BatchNorm;
    eval still supports gradients. Model train/eval modes remain caller owned.

    Targets and old snapshots are detached copies, frozen across all epochs.
    Normalization uses all T*N advantages once, population variance and
    sqrt(var+1e-8). A single/constant advantage becomes zero: actor surrogate
    gradients then vanish, while critic/entropy terms remain active. Disable
    normalize_advantage to retain the original actor signal.

    Save Model, optimizer and shuffle_rng.get_state() together; restore the
    latter with shuffle_rng.set_state(). Environment/action RNGs are separate.
    Optional clipping uses the joint norm of unique, trainable optimizer
    parameters; nonfinite norms raise before that minibatch's optimizer step.
    Optional KL stopping evaluates the full rollout in eval/no_grad after an
    epoch when another epoch remains. Module modes are restored; diagnostics
    consume no shuffle/action RNG. Completed minibatch metrics are weighted by
    actual sample counts. No environment or buffer is owned by the learner.
    """

    def __init__(self, model, optimizer, target_generator, update_rule, *,
                 epochs=4, minibatch_size=64, seed=None, normalize_advantage=True,
                 max_grad_norm=None, target_kl=None):
        self.model, self.optimizer = model, optimizer
        self.target_generator, self.update_rule = target_generator, update_rule
        self._validate_setup(epochs, minibatch_size, normalize_advantage)
        self._validate_options(max_grad_norm, target_kl)
        self.epochs, self.minibatch_size = int(epochs), int(minibatch_size)
        self.normalize_advantage = normalize_advantage
        self.max_grad_norm, self.target_kl = max_grad_norm, target_kl
        self.shuffle_rng = torch.Generator(device="cpu")
        if seed is None:
            self.shuffle_rng.seed()
        else:
            if not isinstance(seed, Integral) or isinstance(seed, bool):
                raise ValueError("seed must be an integer or None")
            self.shuffle_rng.manual_seed(int(seed))

    def learn(self, rollout: RolloutBatch) -> PPOOutput:
        self._validate(rollout)
        reference = next(self.model.parameters())
        rollout = rollout.to(reference.device, dtype=reference.dtype)
        time, n_envs = rollout.reward.shape
        count = time * n_envs
        observations = rollout.observation.reshape(count, *rollout.observation.shape[2:])
        following = rollout.next_observation.reshape(count, *rollout.next_observation.shape[2:])
        actions = rollout.action.long().reshape(count)
        old_value = rollout.value.detach().clone()
        old_log_prob = rollout.log_prob.detach().reshape(count).clone()

        with torch.no_grad():
            policy, value = self._outputs(self.model(observations), count, reference)
            self._validate_freshness(old_value, old_log_prob, value, policy.log_prob(actions))
            with evaluating(self.model):
                next_value = self._outputs(self.model(following), count, reference)[1].reshape(time, n_envs)
            generated = self.target_generator.generate(
                rollout.reward, old_value, next_value, rollout.terminated, rollout.truncated,
            )
            advantage = generated.advantage.reshape(count).detach().clone()
            returns = generated.returns.reshape(count).detach().clone()
            if not torch.isfinite(advantage).all() or not torch.isfinite(returns).all():
                raise ValueError("GAE advantage and returns must be finite")
            if self.normalize_advantage:
                advantage = (advantage - advantage.mean()) / torch.sqrt(advantage.var(unbiased=False) + 1e-8)

        totals = reference.new_zeros(6)
        optimizer_steps = 0
        processed_samples, epochs_completed = 0, 0
        early_stopped = False
        parameters = self._optimized_parameters() if self.max_grad_norm is not None else None
        for _ in range(self.epochs):
            permutation = torch.randperm(count, generator=self.shuffle_rng)
            for indices in permutation.split(self.minibatch_size):
                indices = indices.to(reference.device)
                policy, value = self._outputs(self.model(observations[indices]), len(indices), reference)
                result = self.update_rule.compute(
                    policy.log_prob(actions[indices]), old_log_prob[indices], value,
                    returns[indices], advantage[indices], policy.entropy(),
                )
                self.optimizer.zero_grad()
                result.total_loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                metrics = (result.policy_loss, result.value_loss, result.entropy,
                           result.total_loss, result.approx_kl, result.clip_fraction)
                totals += torch.stack([metric.detach() for metric in metrics]) * len(indices)
                optimizer_steps += 1
                processed_samples += len(indices)
            epochs_completed += 1
            if self.target_kl is not None and epochs_completed < self.epochs:
                kl = self._epoch_kl(observations, actions, old_log_prob, reference)
                if kl > self.target_kl:
                    early_stopped = True
                    break
        return PPOOutput(*(totals / processed_samples).unbind(), optimizer_steps, epochs_completed, early_stopped)

    def _optimized_parameters(self):
        return list({id(p): p for group in self.optimizer.param_groups for p in group["params"]
                     if p.requires_grad}.values())

    @torch.no_grad()
    def _epoch_kl(self, observations, actions, old_log_prob, reference):
        with evaluating(self.model):
            policy, _ = self._outputs(self.model(observations), len(actions), reference)
            log_ratio = policy.log_prob(actions) - old_log_prob
            kl = (log_ratio.exp() - 1 - log_ratio).mean()
        if not torch.isfinite(kl):
            raise ValueError("PPO epoch KL must be finite")
        return kl

    @staticmethod
    def _validate_options(max_grad_norm, target_kl):
        for name, value in (("max_grad_norm", max_grad_norm), ("target_kl", target_kl)):
            if value is not None and (not isinstance(value, Real) or isinstance(value, bool)
                                      or not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be None or a positive finite real number")

    def _validate_setup(self, epochs, minibatch_size, normalize_advantage):
        for name, value in (("epochs", epochs), ("minibatch_size", minibatch_size)):
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(normalize_advantage, bool):
            raise TypeError("normalize_advantage must be bool")
        if not isinstance(self.model, Model):
            raise TypeError("PPO requires a Model")
        parameters = list(self.model.parameters())
        if not parameters or not any(p.requires_grad for p in parameters):
            raise ValueError("PPO Model must have trainable floating parameters")
        reference = parameters[0]
        if any(not p.is_floating_point() or p.dtype != reference.dtype or p.device != reference.device for p in parameters):
            raise ValueError("Model parameters must share floating dtype and device")
        if not isinstance(self.optimizer, Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        model_ids = {id(p) for p in parameters}
        optimized = [p for group in self.optimizer.param_groups for p in group["params"]]
        if not optimized or any(id(p) not in model_ids for p in optimized):
            raise ValueError("Optimizer must contain only PPO Model parameters")

    @staticmethod
    def _validate(rollout):
        if not isinstance(rollout, RolloutBatch):
            raise TypeError("learn expects a fresh RolloutBatch")
        inputs = [getattr(rollout, field.name) for field in fields(rollout)]
        if any(not isinstance(item, Tensor) for item in inputs):
            raise TypeError("RolloutBatch fields must be Tensors")
        shape = rollout.reward.shape
        if len(shape) != 2 or 0 in shape:
            raise ValueError("Rollout scalar fields must have nonempty (T,N) shape")
        if rollout.observation.shape[:2] != shape or rollout.next_observation.shape != rollout.observation.shape:
            raise ValueError("Rollout observations must have shape (T,N,*obs_shape)")
        if any(getattr(rollout, name).shape != shape for name in ("action", "terminated", "truncated", "value", "log_prob")):
            raise ValueError("Rollout action, flags, value and log_prob must have shape (T,N)")
        if rollout.observation.dtype == torch.bool or rollout.observation.is_complex() or rollout.next_observation.dtype != rollout.observation.dtype:
            raise TypeError("Observations must share a real numeric dtype")
        if rollout.action.is_floating_point() or rollout.action.is_complex() or rollout.action.dtype == torch.bool:
            raise TypeError("action must contain discrete integer indices")
        if any(not getattr(rollout, name).is_floating_point() for name in ("reward", "value", "log_prob")):
            raise TypeError("reward, value and log_prob must have floating dtype")
        if rollout.terminated.dtype != torch.bool or rollout.truncated.dtype != torch.bool:
            raise TypeError("terminated and truncated must have bool dtype")
        if any(item.device != rollout.reward.device for item in inputs):
            raise ValueError("Rollout fields must be on the same device before preparation")
        if any(not torch.isfinite(item).all() for item in inputs if item.is_floating_point()):
            raise ValueError("Rollout observations, rewards and snapshots must be finite")

    @staticmethod
    def _validate_freshness(old_value, old_log_prob, value, log_prob):
        if not torch.allclose(value, old_value.reshape(-1), rtol=1e-5, atol=1e-6) or not torch.allclose(
            log_prob, old_log_prob, rtol=1e-5, atol=1e-6,
        ):
            raise ValueError("On-policy rollout snapshots differ from current Model; parameters or forward modes changed")

    @staticmethod
    def _outputs(estimate, count, reference):
        if not isinstance(estimate.policy, Categorical) or estimate.policy.batch_shape != (count,):
            raise ValueError("PPO requires a Categorical policy with batch shape (B,)")
        if estimate.values is None or not isinstance(estimate.values.v, Tensor) or estimate.values.v.shape != (count,):
            raise ValueError("PPO requires scalar EstimatorOutput.values.v with shape (B,)")
        value = estimate.values.v
        if any(item.dtype != reference.dtype or item.device != reference.device for item in (value, estimate.policy.logits)):
            raise ValueError("Policy and value must match Model floating dtype and device")
        return estimate.policy, value
