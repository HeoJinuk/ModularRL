"""Expected endpoint and intermediate policy branches from one Q/policy snapshot."""

import torch
from torch import Tensor
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.model import Model
from modular_rl.learner._tabular_window import _validate_window
from modular_rl.learner.off_policy_n_step_sarsa import _validate_policy


class TreeBackupLearner:
    """Expected endpoint and intermediate policy branches from one Q/policy snapshot."""

    def __init__(
        self,
        model: Model,
        target_generator: TD0TargetGenerator,
        update_rule: TabularQUpdateRule,
        *,
        target_policy,
    ) -> None:
        self.model = model
        self.target_generator = target_generator
        self.update_rule = update_rule
        self.target_policy = target_policy

    @torch.no_grad()
    def learn(self, window) -> Tensor:
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
        Separate flush calls re-evaluate current Q after earlier successful updates."""
        model, target_generator, update_rule = self.model, self.target_generator, self.update_rule
        target_policy = self.target_policy
        _validate_window(model, window)
        table = model.estimator.q_table
        _validate_actions(window, table)
        batches = [batch.to(table.device, dtype=table.dtype) for batch in window]
        first, last = batches[0], batches[-1]
        next_states = torch.cat([batch.next_observation for batch in batches]).long()
        needed_states = next_states[:-1] if last.terminated.item() else next_states
        if isinstance(target_policy, Tensor):
            _validate_policy(target_policy, table)
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
