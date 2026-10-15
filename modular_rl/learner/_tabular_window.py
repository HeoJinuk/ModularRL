"""Shared chronological B=1 window and endpoint-action checks."""

import torch
from torch import Tensor
from modular_rl.component.estimator import TabularQEstimator
from modular_rl.data import TransitionBatch
from modular_rl.model import Model


def _validate_window(model, window):
    if not isinstance(model, Model) or not isinstance(model.estimator, TabularQEstimator):
        raise TypeError("n-step SARSA requires a Model with TabularQEstimator")
    if not window:
        raise ValueError("window must contain at least one B=1 transition")
    previous = None
    for batch in window:
        if not isinstance(batch, TransitionBatch):
            raise TypeError("window must contain TransitionBatch records")
        for name in ("observation", "action", "reward", "next_observation", "terminated", "truncated"):
            value = getattr(batch, name)
            if not isinstance(value, Tensor) or value.shape != (1,):
                raise ValueError("window fields must be Tensors with online shape (1,)")
        for name in ("observation", "next_observation", "action"):
            value = getattr(batch, name)
            if value.is_floating_point() or value.is_complex() or value.dtype == torch.bool:
                raise TypeError(f"{name} must have integer index dtype")
        if not batch.reward.is_floating_point():
            raise TypeError("reward must have floating dtype")
        if batch.terminated.dtype != torch.bool or batch.truncated.dtype != torch.bool:
            raise TypeError("boundary flags must have bool dtype")
        if previous is not None:
            if previous.terminated.item() or previous.truncated.item():
                raise ValueError("window cannot continue after an episode boundary")
            if not torch.equal(previous.next_observation, batch.observation):
                raise ValueError("window transitions must be contiguous in time")
        previous = batch


def _prepare_next_action(next_action, last, table):
    if next_action is None:
        if not last.terminated.item():
            raise ValueError("next_action is required for nontermination, including truncation")
        return None
    action = torch.as_tensor(next_action)
    if action.ndim == 0:
        action = action.reshape(1)
    if action.shape != (1,):
        raise ValueError("next_action must be scalar or have shape (1,)")
    if action.is_floating_point() or action.is_complex() or action.dtype == torch.bool:
        raise TypeError("next_action must have integer index dtype")
    if torch.any(action < 0) or torch.any(action >= table.shape[1]):
        raise IndexError("next_action index out of bounds")
    return action.to(device=table.device, dtype=torch.long)
