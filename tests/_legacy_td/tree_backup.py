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
from . import off_policy_n_step_sarsa
_off_policy = vars(off_policy_n_step_sarsa)
_n_step = _off_policy["_on_policy"]


@torch.no_grad()
def learn_tree_backup(model, window, target_generator, update_rule, *, target_policy):
    """Update the oldest row of a contiguous, ordered B=1 window.

    target_policy is a caller-owned fixed floating probability Tensor (S,A),
    or an existing discrete Selector with probabilities(estimate). Fixed rows
    are explicitly cast to Q dtype; device must match. A single Model estimate
    supplies all needed next-state Q rows and dynamic policy probabilities before
    any writes. The natural terminal endpoint is excluded from that estimate.

    Intermediate actions come from the following window row. The final bootstrap
    is an expectation, so no endpoint action, behavior probability or importance
    ratio is used. pi=0 cuts only the sampled branch; pi=1 carries G directly.
    Each TD0 call performs one reward backup. Return the existing rule's signed
    error (1,), in table dtype/device with no graph. The caller pops after success.
    Separate flush calls re-evaluate current Q after earlier successful updates.
    """
    _n_step["_validate_window"](model, window)
    table = model.estimator.q_table
    _validate_actions(window, table)
    batches = [batch.to(table.device, dtype=table.dtype) for batch in window]
    first, last = batches[0], batches[-1]
    next_states = torch.cat([batch.next_observation for batch in batches]).long()
    needed_states = next_states[:-1] if last.terminated.item() else next_states
    if isinstance(target_policy, Tensor):
        _off_policy["_validate_policy"](target_policy, table)
    elif not callable(getattr(target_policy, "probabilities", None)):
        raise TypeError("target_policy must be a fixed probability Tensor or discrete Selector")

    if needed_states.numel():
        estimate = model.forward(needed_states)
        q = estimate.values.q[0]
        probabilities = _probabilities(target_policy, needed_states, estimate)
        expected_values = (probabilities * q).sum(-1)
    target = torch.zeros_like(last.reward) if last.terminated.item() else expected_values[-1:]
    for time in range(len(batches)-1, -1, -1):
        continuation = target
        if time < len(batches)-1:
            next_action = batches[time+1].action.long()
            probability = probabilities[time].gather(0, next_action)
            sampled_q = q[time].gather(0, next_action)
            expected = expected_values[time:time+1]
            if probability.item() == 0:
                continuation = expected
            elif probability.item() != 1:
                continuation = expected + probability * (target - sampled_q)
        batch = batches[time]
        target = target_generator.generate(batch.reward, continuation, batch.terminated)

    states, actions = first.observation.long(), first.action.long()
    _validate_update(table[states, actions], target, update_rule.alpha)
    return update_rule.update(table, states, actions, target)


def _probabilities(policy, states, estimate):
    q = estimate.values.q[0]
    if isinstance(policy, Tensor):
        return policy[states].to(dtype=q.dtype)
    probabilities = policy.probabilities(estimate)
    if not isinstance(probabilities, Tensor) or probabilities.shape != q.shape:
        raise ValueError("Selector probabilities must match the Q shape (M,A)")
    if probabilities.dtype != q.dtype or probabilities.device != q.device:
        raise ValueError("Selector probabilities must share Q dtype and device")
    if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0) or torch.any(probabilities > 1):
        raise ValueError("Selector probabilities must be finite and in [0,1]")
    tolerance = 1e-6 if q.dtype == torch.float32 else 1e-12
    if not torch.allclose(probabilities.sum(-1), torch.ones_like(q[:, 0]), rtol=0, atol=tolerance):
        raise ValueError("Selector probabilities must sum to one per state")
    return probabilities


def _validate_actions(window, table):
    for batch in window:
        if not 0 <= batch.action.item() < table.shape[1]:
            raise IndexError("window action index out of bounds")


def _validate_update(current, target, alpha):
    error = target - current
    if not all(torch.isfinite(x).all() for x in (target, error, current + alpha * error)):
        raise FloatingPointError("Tree Backup target/update is non-finite; head was not updated")


def run_episode(env, model, target_generator, update_rule, *, target_policy, n=3, seed=None):
    """Step -> snapshot -> eligible head update -> next behavior action.

    No bootstrap-only action is selected at either episode boundary. Flush uses
    current expected Q per call, without step/act calls. Each collected transition
    is updated once; each real action is selected once. Window/state/RNG/checkpoint
    and environment lifetime remain caller-owned, with no Runner or learner class.
    """
    _n_step["_validate_n"](n)
    observation, _ = env.reset(seed=seed)
    selected = model.act(observation)
    pending = deque()
    reward_sum, steps, updates = 0., 0, 0
    while True:
        action = selected.action.item()
        log_prob = None if selected.log_prob is None else selected.log_prob.detach().cpu().clone()
        next_observation, reward, terminated, truncated, _ = env.step(action)
        pending.append(Transition(observation, action, reward, next_observation, terminated, truncated, log_prob).to_batch())
        if len(pending) == n:
            learn_tree_backup(model, pending, target_generator, update_rule, target_policy=target_policy)
            pending.popleft()
            updates += 1
        reward_sum += reward
        steps += 1
        if terminated or truncated:
            while pending:
                learn_tree_backup(model, pending, target_generator, update_rule, target_policy=target_policy)
                pending.popleft()
                updates += 1
            break
        observation = next_observation
        selected = model.act(observation)
    return {"reward": reward_sum, "steps": steps, "updates": updates, "terminated": terminated, "truncated": truncated}
