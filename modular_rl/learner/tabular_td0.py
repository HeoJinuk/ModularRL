"""TD(0) prediction by composing a tabular V Model, target and update rule."""

import torch
from torch import Tensor

from modular_rl.component.estimator import TabularVEstimator
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularVUpdateRule
from modular_rl.data import TransitionBatch
from modular_rl.model import Model


class TabularTD0PredictionLearner:
    """Learn fixed-policy state values from an externally collected batch.

    observation, next_observation, reward and both boundary flags have shape
    (B,). Integer state indices are normalized to long; floating rewards use
    the live V-table's dtype/device. The representation must preserve state
    indices. Actions and log probabilities are not used for V prediction.

    All bootstrap targets are evaluated under no_grad before any table write.
    Repeated states are then updated in row order using those frozen targets.
    For online TD, call learn separately with B=1 for each transition in time
    order, so subsequent calls observe the newly updated table.

    Returned signed TD errors have shape (B,), table dtype/device and no graph.
    terminated masks bootstrap; truncated alone does not. The collector must
    supply the actual final observation. No policy, MDP, sampler, buffer or
    optimizer is owned here, and an optional ActionSelector is never used.
    """

    def __init__(
        self, model: Model, target_generator: TD0TargetGenerator, update_rule: TabularVUpdateRule
    ) -> None:
        if not isinstance(model, Model) or not isinstance(model.estimator, TabularVEstimator):
            raise TypeError("TabularTD0PredictionLearner requires a Model with TabularVEstimator")
        self.model = model
        self.target_generator = target_generator
        self.update_rule = update_rule

    def learn(self, batch: TransitionBatch) -> Tensor:
        self._validate(batch)
        v_table = self.model.estimator.v_table
        batch = batch.to(v_table.device, dtype=v_table.dtype)
        states = batch.observation.long()
        with torch.no_grad():
            next_value = self.model.forward(batch.next_observation.long()).values.v
            target = self.target_generator.generate(batch.reward, next_value, batch.terminated)
        return self.update_rule.update(v_table, states, target)

    @staticmethod
    def _validate(batch: TransitionBatch) -> None:
        """Check the fields used for prediction, without silently rounding states."""
        if not isinstance(batch, TransitionBatch):
            raise TypeError("learn expects a TransitionBatch")
        if not isinstance(batch.observation, Tensor):
            raise TypeError("observation must be a Tensor")
        if batch.observation.ndim != 1 or batch.observation.numel() == 0:
            raise ValueError("observation must have nonempty tabular batch shape (B,)")
        for name in ("next_observation", "reward", "terminated", "truncated"):
            value = getattr(batch, name)
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a Tensor")
            if value.shape != batch.observation.shape:
                raise ValueError("prediction fields must have the same (B,) shape")
        for name in ("observation", "next_observation"):
            value = getattr(batch, name)
            if value.is_floating_point() or value.is_complex() or value.dtype == torch.bool:
                raise TypeError(f"{name} must have integer index dtype")
        if not batch.reward.is_floating_point():
            raise TypeError("reward must have floating dtype")
        if batch.terminated.dtype != torch.bool or batch.truncated.dtype != torch.bool:
            raise TypeError("terminated and truncated must have bool dtype")
