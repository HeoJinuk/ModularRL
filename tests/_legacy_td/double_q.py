"""Frozen numerical reference from dc2ecdc; intentional test-only duplication."""

import numpy as np
import torch
from torch import Tensor
from numbers import Integral
from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import TabularQEstimator
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import TransitionBatch, ActionOutput, EstimatorOutput, ValueOutput
from modular_rl.model import Model


@torch.no_grad()
def combined_estimate(model1: Model, model2: Model, observation) -> EstimatorOutput:
    """Combine live Q estimates into one explicit Q sum, without a third table.

    Integer observations may be single or batched. The result has Q shape
    (..., A). The existing selector receives one tensor, not two unreduced
    critics. Behavior and greedy evaluation both use Q1 + Q2.
    """
    _validate_models(model1, model2)
    q1, = model1.forward(observation).values.q
    q2, = model2.forward(observation).values.q
    if q1.shape != q2.shape:
        raise ValueError("Q estimates must have the same shape without broadcasting")
    return EstimatorOutput(values=ValueOutput(q=(q1 + q2,)))


def act_double_q(model1: Model, model2: Model, observation, selector: EpsilonGreedyActionSelector,
                 *, deterministic=False) -> ActionOutput:
    """Use the unchanged selector's sampling, ties, probabilities and log_prob."""
    return selector.select(combined_estimate(model1, model2, observation), deterministic=deterministic)


def choose_update_first(rng: np.random.Generator) -> bool:
    """Draw one fair update coin from an explicit, separate RNG."""
    return bool(rng.random() < .5)


def learn_double_q(model1: Model, model2: Model, batch: TransitionBatch,
                   target_generator: TD0TargetGenerator, update_rule: TabularQUpdateRule,
                   *, update_first: bool) -> Tensor:
    """Cross-bootstrap, then update exactly one independent table in place.

    All batch fields have shape (B,). Integer indices become long; rewards
    follow the two compatible tables' floating dtype/device. update_first is
    one bool for the WHOLE call, including B>1. Random selection belongs to
    the caller, normally once per online B=1 step, and consumes no RNG here.

    Argmax ties select the smallest action index without sampling. Both Q
    estimates and all targets are frozen before writes. Duplicate pairs then
    follow the existing update rule's input-order semantics. Only termination
    masks bootstrap; truncation uses the actual final observation. Returned
    signed errors are (B,), in table dtype/device, without an autograd graph.
    """
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
