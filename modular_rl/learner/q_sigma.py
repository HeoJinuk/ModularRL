"""Equation 7.17 with frozen sampling-time sigma and behavior probabilities."""

import torch
from torch import Tensor
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.model import Model
from modular_rl.learner._tabular_window import _validate_window
from modular_rl.learner.tree_backup import _validate_actions, _probabilities, _validate_update
from modular_rl.learner.off_policy_n_step_sarsa import _validate_logs, _validate_policy


class QSigmaLearner:
    """Equation 7.17 with frozen sampling-time sigma and behavior probabilities."""

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
    def learn(self, window, *, sigma) -> dict[str, Tensor]:
        """Compute Eq. 7.17 from one Q/pi snapshot, then update only the window head.

        window contains contiguous owned B=1 TransitionBatch records. sigma is a
        fixed scalar or m frozen values [sigma_(t+1), ..., sigma_(t+m)], where m is
        len(window). The last value is a validated placeholder: the horizon residual
        G_h:h-Q(S_h,A_h) is zero. No endpoint action/log is required or sampled.
        Callbacks belong to collection, never this function. Intermediate b comes
        from original sampling-time log_prob, before batch floating-dtype conversion.

        target_policy is a fixed (S,A) Tensor or existing discrete Selector. Q and
        dynamic pi share one estimate before any write. Natural termination starts
        at the last reward; truncation uses the final observation's expected Q.
        Each reward backup reuses TD0TargetGenerator. There is no second IS product.
        Numeric failures leave this head unchanged; earlier successful calls remain.
        Returned target/td_error are detached (1,) in table dtype/device; coefficients
        and importance_ratios are float64 (m-1,) in forward intermediate-time order.
        The caller pops window/metadata after success; each flush reads current Q."""
        model, target_generator, update_rule = self.model, self.target_generator, self.update_rule
        target_policy = self.target_policy
        _validate_window(model, window)
        table = model.estimator.q_table
        _validate_actions(window, table)
        sigmas = _validate_sigma(sigma, len(window), table.device)
        _validate_logs(window, None, table)
        if isinstance(target_policy, Tensor):
            _validate_policy(target_policy, table)
        elif not callable(getattr(target_policy, 'probabilities', None)):
            raise TypeError('target_policy must be a probability Tensor or discrete Selector')
        batches = [row.to(table.device, dtype=table.dtype) for row in window]
        first, last = batches[0], batches[-1]
        next_states = torch.cat([row.next_observation for row in batches]).long()
        needed = next_states[:-1] if last.terminated.item() else next_states
        ratios, coefficients = [], []
        if needed.numel():
            estimate = model.forward(needed)
            q = estimate.values.q[0]
            probabilities = _probabilities(target_policy, needed, estimate)
            expected_values = (probabilities * q).sum(-1)
            if not torch.isfinite(q).all() or not torch.isfinite(expected_values).all():
                raise FloatingPointError('Non-finite Q/expectation; head was not updated')
            for time, row in enumerate(list(window)[1:]):
                pi = probabilities[time, row.action.item()]
                rho, coefficient = _coefficient(pi, row.log_prob, sigmas[time])
                ratios.append(rho)
                coefficients.append(coefficient)

        continuation = torch.zeros_like(last.reward) if last.terminated.item() else expected_values[-1:]
        target = target_generator.generate(last.reward, continuation, last.terminated)
        if not torch.isfinite(target).all():
            raise FloatingPointError('Non-finite horizon return; head was not updated')
        for time in range(len(batches)-2, -1, -1):
            pi = probabilities[time, batches[time+1].action.item()]
            sampled_q = q[time, batches[time+1].action.item()].reshape(1)
            expected = expected_values[time:time+1]
            coefficient = coefficients[time].to(table.dtype)
            if coefficient.item() == 0:
                continuation = expected
            elif coefficient.item() == 1 and pi.item() == 1:
                continuation = target  # Exact Tree Backup deterministic branch.
            else:
                residual = target - sampled_q
                correction = coefficient * residual
                _validate_product(coefficient, residual, correction)
                continuation = expected + correction
            row = batches[time]
            target = target_generator.generate(row.reward, continuation, row.terminated)
            if not torch.isfinite(target).all():
                raise FloatingPointError('Non-finite recursive target; head was not updated')

        states, actions = first.observation.long(), first.action.long()
        current = table[states, actions]
        _validate_update(current, target, update_rule.alpha)
        _validate_product(target-current, update_rule.alpha, update_rule.alpha*(target-current))
        error = update_rule.update(table, states, actions, target)
        empty = torch.empty(0, dtype=torch.float64, device=table.device)
        return {'target': target, 'td_error': error,
                'importance_ratios': torch.stack(ratios) if ratios else empty,
                'coefficients': torch.stack(coefficients) if coefficients else empty.clone()}


def _validate_sigma(sigma, size, device):
    if callable(sigma):
        raise TypeError('Decide sigma at action sampling time; pass frozen values to learn')
    values = torch.as_tensor(sigma, device=device)
    if values.dtype == torch.bool or values.is_complex():
        raise TypeError('sigma must be real, not bool or complex')
    values = values.double() if isinstance(sigma, Tensor) else torch.as_tensor(sigma, dtype=torch.float64, device=device)
    if values.ndim == 0:
        values = values.expand(size)
    if values.shape != (size,):
        raise ValueError('sigma must be scalar or m values [sigma_(t+1), ..., sigma_(t+m)]')
    if not torch.isfinite(values).all() or torch.any(values < 0) or torch.any(values > 1):
        raise ValueError('sigma must be finite and in [0,1]')
    return values


def _coefficient(pi, log_b, sigma):
    # pi=0 is a real zero branch, not exp(-inf)-related numeric underflow.
    if pi.item() == 0:
        rho = torch.zeros((), dtype=torch.float64, device=pi.device)
    else:
        log_ratio = pi.double().log() - log_b.detach().to(pi.device).double().reshape(())
        rho = log_ratio.exp()
        if not torch.isfinite(rho) or rho.item() == 0:
            raise FloatingPointError('Positive pi/b overflows or underflows; no write committed')
    coefficient = sigma*rho + (1-sigma)*pi.double()
    cast = coefficient.to(pi.dtype)
    if not torch.isfinite(cast) or (coefficient.item() > 0 and cast.item() == 0):
        raise FloatingPointError('Mixture coefficient is not representable in Q dtype')
    return rho, coefficient


def _validate_product(left, right, product):
    if not torch.isfinite(product).all() or torch.any((left != 0) & (right != 0) & (product == 0)):
        raise FloatingPointError('Recursive correction/update overflow or underflow; no write committed')
