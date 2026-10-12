"""Q-learning orchestration for a Model backed by a tabular Q estimator."""

import torch
from torch import Tensor

from modular_rl.component.estimator import TabularQEstimator
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import TransitionBatch
from modular_rl.model import Model


class TabularQLearner:
    """Learn from an external replay batch; do not own or sample its buffer.

    Observations are table state indices, preserved by the Model's
    representation (normally IdentityRepresentation). Actions are discrete
    indices. Each field has shape (B,); rewards are floating and episode
    flags are bool. Integer indices are normalized to long on the table's
    device, and floating fields use its dtype.

    All bootstrap values and targets are computed before the first table
    write. The update rule retains its sequential duplicate-index contract.
    Returned signed errors have shape (B,), the table's dtype/device and no
    graph. The caller supplies actual final observations on episode endings.
    An ActionSelector is optional and is never used by learn().
    """

    def __init__(
        self, model: Model, target_generator: TD0TargetGenerator, update_rule: TabularQUpdateRule
    ) -> None:
        if not isinstance(model, Model) or not isinstance(model.estimator, TabularQEstimator):
            raise TypeError("TabularQLearner requires a Model with TabularQEstimator")
        self.model = model
        self.target_generator = target_generator
        self.update_rule = update_rule

    def learn(self, batch: TransitionBatch) -> Tensor:
        self._validate(batch)
        q_table = self.model.estimator.q_table
        batch = batch.to(q_table.device, dtype=q_table.dtype)
        states, actions = batch.observation.long(), batch.action.long()
        with torch.no_grad():
            next_q = self.model.forward(batch.next_observation.long()).values.q[0]
            next_value = next_q.max(dim=-1).values
            target = self.target_generator.generate(batch.reward, next_value, batch.terminated)
        return self.update_rule.update(q_table, states, actions, target)

    @staticmethod
    def _validate(batch: TransitionBatch) -> None:
        """Require one-dimensional tabular rows without silently rounding indices."""
        if not isinstance(batch, TransitionBatch):
            raise TypeError("learn expects a TransitionBatch")
        if not isinstance(batch.observation, Tensor):
            raise TypeError("observation must be a Tensor")
        if batch.observation.ndim != 1 or batch.observation.numel() == 0:
            raise ValueError("observation must have nonempty tabular batch shape (B,)")
        for name in ("action", "reward", "next_observation", "terminated", "truncated"):
            value = getattr(batch, name)
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a Tensor")
            if value.shape != batch.observation.shape:
                raise ValueError("tabular batch fields must have the same (B,) shape")
        for name in ("observation", "next_observation", "action"):
            value = getattr(batch, name)
            if value.is_floating_point() or value.is_complex() or value.dtype == torch.bool:
                raise TypeError(f"{name} must have integer index dtype")
        if not batch.reward.is_floating_point():
            raise TypeError("reward must have floating dtype")
        if batch.terminated.dtype != torch.bool or batch.truncated.dtype != torch.bool:
            raise TypeError("terminated and truncated must have bool dtype")
