"""Frozen numerical reference from dc2ecdc; intentional test-only duplication."""

from collections import deque
from modular_rl.data import Transition
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


def learn_n_step_sarsa(model: Model, window, target_generator: TD0TargetGenerator,
                       update_rule: TabularQUpdateRule, *, next_action=None) -> Tensor:
    """Update only the oldest row of an ordered window of B=1 batches.

    Window length is the actual reward count m, not a nominal fixed n.
    Contiguous rows cannot cross an episode boundary. next_action is the
    actual action already selected at the last next observation; scalar or
    (1,) integer IDs are accepted. None is valid only for natural termination.

    The caller owns window storage and removes its oldest row after success.
    Each call reads current Q under no_grad, then composes the existing TD0
    backup in reverse reward order. This yields gamma**m bootstrap, including
    shortened time-limit tails. Natural termination dominates truncation.
    Flush calls re-evaluate Q independently after earlier writes, retaining
    the same final selected action. Signed error (1,) follows table dtype/device
    and has no graph. The function never selects an action or consumes RNG.
    """
    _validate_window(model, window)
    table = model.estimator.q_table
    batches = [batch.to(table.device, dtype=table.dtype) for batch in window]
    last, first = batches[-1], batches[0]
    action = _prepare_next_action(next_action, last, table)
    with torch.no_grad():
        if last.terminated.item():
            target = torch.zeros_like(last.reward)
        else:
            next_q = model.forward(last.next_observation.long()).values.q[0]
            target = next_q.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        for batch in reversed(batches):
            target = target_generator.generate(batch.reward, target, batch.terminated)
    return update_rule.update(table, first.observation.long(), first.action.long(), target)


def _validate_n(n):
    if not isinstance(n, Integral) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")


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


def run_episode(env, model, target_generator, update_rule, *, n=3, seed=None):
    """Select next actions before writes; execute them once; flush at boundary.

    pending holds owned Transition.to_batch snapshots in chronological order.
    len(pending)==n is tau=t-n+1>=0; update one head before the next step.
    Ending flush performs all remaining updates without step or act calls.
    Each collected transition is updated once. No full Episode is collected
    before learning. Environment lifetime and checkpoints remain caller-owned.
    """
    _validate_n(n)
    observation, _ = env.reset(seed=seed)
    selected = model.act(observation)
    pending = deque()
    reward_sum, steps, updates = 0., 0, 0
    while True:
        action = selected.action.item()
        log_prob = None if selected.log_prob is None else selected.log_prob.detach().cpu().clone()
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_selected = None if terminated else model.act(next_observation)
        next_action = None if next_selected is None else next_selected.action
        pending.append(Transition(observation, action, reward, next_observation, terminated, truncated, log_prob).to_batch())
        if len(pending) == n:
            learn_n_step_sarsa(model, pending, target_generator, update_rule, next_action=next_action)
            pending.popleft()
            updates += 1
        reward_sum += reward
        steps += 1
        if terminated or truncated:
            while pending:
                learn_n_step_sarsa(model, pending, target_generator, update_rule, next_action=next_action)
                pending.popleft()
                updates += 1
            break
        observation, selected = next_observation, next_selected
    return {"reward": reward_sum, "steps": steps, "updates": updates, "terminated": terminated, "truncated": truncated}
