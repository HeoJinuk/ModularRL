"""SARSA orchestration using an externally selected actual next action."""

import torch
from torch import Tensor

from modular_rl.component.estimator import TabularQEstimator
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import TransitionBatch
from modular_rl.model import Model


class TabularSARSALearner:
    """Update the existing Q table with Q(next_state, actual_next_action).

    Batch fields have tabular shape (B,), integer state/action indices,
    floating rewards and bool flags. next_action is an integer Tensor (B,);
    an integer scalar is also accepted for B=1. Indices are normalized to
    long on the live table device. Floating fields use the table dtype.

    None is allowed only when all rows are terminated, without querying the
    next-state model. Otherwise every row has a valid action ID, including
    terminal placeholders (e.g. 0), whose Q values are masked by the target
    generator. Truncated-only rows bootstrap from their final observation.

    The caller selects next_action before learning and executes that same
    action on the next step, unless the episode ended. No sampling, policy,
    environment or buffer is owned here; an ActionSelector is optional.
    All targets are frozen before the existing rule's sequential duplicate
    updates. Online SARSA uses ordered B=1 calls. Signed errors (B,) have no
    graph and preserve the table dtype/device; inputs are never modified.
    """

    def __init__(self, model: Model, target_generator: TD0TargetGenerator,
                 update_rule: TabularQUpdateRule) -> None:
        if not isinstance(model, Model) or not isinstance(model.estimator, TabularQEstimator):
            raise TypeError("TabularSARSALearner requires a Model with TabularQEstimator")
        self.model, self.target_generator, self.update_rule = model, target_generator, update_rule

    def learn(self, batch: TransitionBatch, *, next_action: Tensor | int | None = None) -> Tensor:
        self._validate(batch)
        q_table = self.model.estimator.q_table
        batch = batch.to(q_table.device, dtype=q_table.dtype)
        states, actions = batch.observation.long(), batch.action.long()
        next_actions = self._prepare_next_action(next_action, batch)
        with torch.no_grad():
            if next_actions is None:
                next_value = torch.zeros_like(batch.reward)
            else:
                next_q = self.model.forward(batch.next_observation.long()).values.q[0]
                next_value = next_q.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
            target = self.target_generator.generate(batch.reward, next_value, batch.terminated)
        return self.update_rule.update(q_table, states, actions, target)

    def _prepare_next_action(self, next_action, batch):
        """Validate IDs without rounding; normalize B=1 scalars and dtype/device."""
        if next_action is None:
            if not batch.terminated.all():
                raise ValueError("next_action is required for every nonterminated row, including truncation")
            return None
        actions = torch.as_tensor(next_action)
        if actions.ndim == 0 and batch.reward.numel() == 1:
            actions = actions.reshape(1)
        if actions.shape != batch.reward.shape:
            raise ValueError("next_action must have shape (B,); scalar is allowed only for B=1")
        if actions.is_floating_point() or actions.is_complex() or actions.dtype == torch.bool:
            raise TypeError("next_action must have integer index dtype")
        if torch.any(actions < 0) or torch.any(actions >= self.model.estimator.q_table.shape[1]):
            raise IndexError("next_action index out of bounds; terminal placeholders must also be valid IDs")
        return actions.to(device=batch.reward.device, dtype=torch.long)

    @staticmethod
    def _validate(batch: TransitionBatch) -> None:
        """Match the existing tabular TD batch contract without broadcasting."""
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
