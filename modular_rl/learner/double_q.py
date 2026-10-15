"""Two independent tabular Q Models, with caller-owned update choice RNG."""

import numpy as np
import torch
from torch import Tensor
from modular_rl.component.estimator import TabularQEstimator
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import TransitionBatch, ActionOutput, EstimatorOutput, ValueOutput
from modular_rl.model import Model


class DoubleQLearner:
    """Two independent tabular Q Models, with caller-owned update choice RNG."""

    def __init__(
        self,
        model1: Model,
        model2: Model,
        target_generator: TD0TargetGenerator,
        update_rule: TabularQUpdateRule,
        *,
        action_selector=None,
    ) -> None:
        _validate_models(model1, model2)
        self.model1, self.model2 = model1, model2
        self.target_generator = target_generator
        self.update_rule = update_rule
        self.action_selector = action_selector if action_selector is not None else model1.action_selector

    def learn(self, batch: TransitionBatch, *, update_first: bool) -> Tensor:
        """Cross-bootstrap, then update exactly one independent table in place.

        All batch fields have shape (B,). Integer indices become long; rewards
        follow the two compatible tables' floating dtype/device. update_first is
        one bool for the WHOLE call, including B>1. Random selection belongs to
        the caller, normally once per online B=1 step, and consumes no RNG here.

        Argmax ties select the smallest action index without sampling. Both Q
        estimates and all targets are frozen before writes. Duplicate pairs then
        follow the existing update rule's input-order semantics. Only termination
        masks bootstrap; truncation uses the actual final observation. Returned
        signed errors are (B,), in table dtype/device, without an autograd graph."""
        model1, model2 = self.model1, self.model2
        target_generator, update_rule = self.target_generator, self.update_rule
        _validate_models(model1, model2)
        _validate_batch(batch, update_first)
        selecting, evaluating = (model1, model2) if update_first else (model2, model1)
        table = selecting.estimator.q_table
        batch = batch.to(table.device, dtype=table.dtype)
        states, actions = batch.observation.long(), batch.action.long()
        with torch.no_grad():
            select_q, = selecting.forward(batch.next_observation.long()).values.q
            evaluate_q, = evaluating.forward(batch.next_observation.long()).values.q
            _validate_next_q(select_q, evaluate_q, batch.reward.numel(), table.shape[1])
            best_action = select_q.argmax(dim=-1)
            next_value = evaluate_q.gather(-1, best_action.unsqueeze(-1)).squeeze(-1)
            target = target_generator.generate(batch.reward, next_value, batch.terminated)
        return update_rule.update(table, states, actions, target)

    @torch.no_grad()
    def estimate(self, observation) -> EstimatorOutput:
        """Return Q1 + Q2 for behavior, without allocating a third table."""
        _validate_models(self.model1, self.model2)
        q1, = self.model1.forward(observation).values.q
        q2, = self.model2.forward(observation).values.q
        if q1.shape != q2.shape:
            raise ValueError("Q estimates must have the same shape without broadcasting")
        return EstimatorOutput(values=ValueOutput(q=(q1 + q2,)))

    def act(self, observation, *, deterministic=False) -> ActionOutput:
        """Use the explicitly shared behavior selector; learn() never selects actions."""
        if self.action_selector is None:
            raise ValueError("Double Q act requires an action_selector")
        return self.action_selector.select(self.estimate(observation), deterministic=deterministic)


def _validate_models(model1, model2):
    for model in (model1, model2):
        if not isinstance(model, Model) or not isinstance(model.estimator, TabularQEstimator):
            raise TypeError("Double Q composition requires two Models with TabularQEstimator")
    q1, q2 = model1.estimator.q_table, model2.estimator.q_table
    if q1.shape != q2.shape or q1.dtype != q2.dtype or q1.device != q2.device:
        raise ValueError("Q tables must have the same shape, dtype and device")
    if q1.untyped_storage().data_ptr() == q2.untyped_storage().data_ptr():
        raise ValueError("Q tables must have independent storage")


def _validate_batch(batch, update_first):
    if not isinstance(update_first, (bool, np.bool_)):
        raise TypeError("update_first must be one bool for the whole batch")
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


def _validate_next_q(select_q, evaluate_q, batch_size, n_actions):
    if select_q.shape != (batch_size, n_actions) or evaluate_q.shape != select_q.shape:
        raise ValueError("Both next Q estimates must have the same (B,A) shape")
