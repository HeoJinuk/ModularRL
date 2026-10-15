"""Sampled endpoint SARSA with recorded behavior probabilities and fixed target policy."""

import torch
from torch import Tensor
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.model import Model
from modular_rl.learner._tabular_window import _validate_window, _prepare_next_action


class OffPolicyNStepSARSALearner:
    """Sampled endpoint SARSA with recorded behavior probabilities and fixed target policy."""

    def __init__(
        self,
        model: Model,
        target_generator: TD0TargetGenerator,
        update_rule: TabularQUpdateRule,
        *,
        target_policy: Tensor,
    ) -> None:
        self.model = model
        self.target_generator = target_generator
        self.update_rule = update_rule
        self.target_policy = target_policy

    @torch.no_grad()
    def learn(self, window, *, bootstrap_action=None, bootstrap_log_prob=None) -> dict[str, Tensor]:
        """Update the window head; return raw/weighted errors, G, effective Y and rho.

        window contains contiguous B=1 snapshots. target_policy is a fixed floating
        (S,A) probability table owned by the caller. rho excludes the head action,
        includes subsequent executed actions and, when bootstrapping, the separately
        saved final action. Behavior logs are read from original snapshots, never
        recomputed or cast through the Q-table dtype. Logs use their recorded precision;
        summation/exp use float64. The caller removes the head only after success.

        Returned tensors have shape (1,), no graph, and table device/dtype except
        importance_ratio (float64). td_error is the actual effective-target error
        returned by the existing rule; raw_td_error is G-Q before the write. rho=1
        passes G directly for bit-exact on-policy equivalence. Numeric failures write
        nothing for this call (earlier successful online calls remain committed).
        No action is sampled. Flush calls read current Q with the same final action."""
        model, target_generator, update_rule = self.model, self.target_generator, self.update_rule
        target_policy = self.target_policy
        _validate_window(model, window)
        table = model.estimator.q_table
        _validate_policy(target_policy, table)
        _validate_logs(window, bootstrap_log_prob, table)
        batches = [batch.to(table.device, dtype=table.dtype) for batch in window]
        first, last = batches[0], batches[-1]
        action = _prepare_next_action(bootstrap_action, last, table)
        if not last.terminated.item() and bootstrap_log_prob is None:
            raise ValueError("bootstrap_log_prob is required when bootstrapping, including truncation")

        # The same TD0 reverse composition as the original n-step example.
        if last.terminated.item():
            target = torch.zeros_like(last.reward)
        else:
            next_q = model.forward(last.next_observation.long()).values.q[0]
            target = next_q.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        for batch in reversed(batches):
            target = target_generator.generate(batch.reward, target, batch.terminated)

        rho = importance_ratio(window, target_policy, bootstrap_action=action,
                               bootstrap_log_prob=bootstrap_log_prob)
        states, actions = first.observation.long(), first.action.long()
        current = table[states, actions].clone()
        raw_error = target - current
        if rho.item() == 1:
            effective_target = target
        elif rho.item() == 0:
            effective_target = current
        else:
            effective_target = current + rho.to(table.dtype) * raw_error
        _validate_update(current, target, raw_error, effective_target, rho, update_rule.alpha)
        error = update_rule.update(table, states, actions, effective_target)
        return {"td_error": error, "raw_td_error": raw_error, "target": target,
                "effective_target": effective_target, "importance_ratio": rho}


def importance_ratio(window, target_policy, *, bootstrap_action=None, bootstrap_log_prob=None):
    """Product pi/b for actions AFTER the head, including the sampled endpoint.

    Called after input validation by learn; useful for inspecting saved windows.
    Natural termination omits the endpoint, even if both flags are true. An empty
    product is one. A zero target probability yields zero, not numerical underflow.
    Selected-action logs cannot verify coverage for actions never observed.
    """
    last = window[-1]
    factors = [(b.observation.item(), b.action.item(), b.log_prob) for b in list(window)[1:]]
    if not last.terminated.item():
        factors.append((last.next_observation.item(), int(torch.as_tensor(bootstrap_action).item()), bootstrap_log_prob))
    log_weight = torch.zeros(1, dtype=torch.float64, device=target_policy.device)
    for state, action, log_b in reversed(factors):
        log_pi = target_policy[state, action].log().double()
        log_weight += log_pi - log_b.detach().to(device=target_policy.device, dtype=torch.float64).reshape(1)
    rho = log_weight.exp()
    if not torch.isfinite(rho).all() or torch.any(torch.isfinite(log_weight) & (rho == 0)):
        raise FloatingPointError("Importance ratio overflow/underflow; window head was not updated")
    return rho.detach()


def _validate_policy(policy, table):
    if not isinstance(policy, Tensor) or not policy.is_floating_point():
        raise TypeError("target_policy must be a floating Tensor")
    if policy.shape != table.shape or policy.device != table.device:
        raise ValueError("target_policy must have table shape (S,A) and device")
    if not torch.isfinite(policy).all() or torch.any(policy < 0) or torch.any(policy > 1):
        raise ValueError("target_policy probabilities must be finite and in [0,1]")
    tolerance = 1e-6 if policy.dtype == torch.float32 else 1e-12
    if not torch.allclose(policy.double().sum(-1), torch.ones(table.shape[0], device=table.device, dtype=torch.float64),
                          rtol=0, atol=tolerance):
        raise ValueError("target_policy probabilities must sum to one per state")


def _validate_logs(window, bootstrap_log_prob, table):
    for batch in window:
        for value, size in ((batch.observation, table.shape[0]), (batch.next_observation, table.shape[0]),
                            (batch.action, table.shape[1])):
            if not 0 <= value.item() < size:
                raise IndexError("window state/action index out of bounds")
    for log_b in [batch.log_prob for batch in window] + ([] if bootstrap_log_prob is None else [bootstrap_log_prob]):
        if not isinstance(log_b, Tensor) or not log_b.is_floating_point() or log_b.shape not in ((), (1,)):
            raise ValueError("sampling-time behavior log_prob must be a scalar or (1,) floating Tensor")
        if not torch.isfinite(log_b).all() or torch.any(log_b > 0):
            raise ValueError("behavior log_prob must be finite and <=0; sampled actions need positive support")


def _validate_update(current, target, raw_error, effective_target, rho, alpha):
    cast_rho = rho.to(current.dtype)
    error = effective_target - current
    proposed = current + alpha * error
    if not all(torch.isfinite(x).all() for x in (cast_rho, target, raw_error, effective_target, error, proposed)):
        raise FloatingPointError("Ratio/target/update is not representable in table dtype; no write committed")
    if torch.any((rho > 0) & (cast_rho == 0)):
        raise FloatingPointError("Positive importance ratio underflows table dtype; no write committed")
    weighted = cast_rho * raw_error
    if torch.any((cast_rho != 0) & (raw_error != 0) & (weighted == 0)):
        raise FloatingPointError("Weighted TD error underflows table dtype; no write committed")
