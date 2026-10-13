"""Ordinary/weighted full-trajectory IS for fixed-policy tabular V prediction."""

import torch
from torch import Tensor

from modular_rl.component.estimator import TabularVEstimator
from modular_rl.component.target_generator import MonteCarloReturnGenerator
from modular_rl.data import Episode
from modular_rl.model import Model


class OffPolicyMCPredictionLearner:
    """Estimate a fixed target policy's V from complete behavior episodes.

    target_policy is an owned float64 probability Tensor (S,A), copied at
    construction; keep it fixed during learning. Behavior log_prob must be
    scalar log b(a|s) recorded at sampling time, finite and <= 0. Coverage
    (pi(a|s)>0 implies b(a|s)>0) is the collector's responsibility: recorded
    chosen-action probabilities cannot verify support of unobserved actions.

    rho[t] includes every action from t to termination. First-visit means the
    earliest state visit in forward time, independent of reverse rho/return
    calculation. Ordinary IS averages rho*G over all selected visits, including
    rho=0. Weighted IS divides sum(rho*G) by sum(rho); zero total weight leaves
    V unchanged. No bootstrap, behavior model or environment is accessed.

    learn returns raw returns (T,), detached, on the V-table's dtype/device.
    visit_counts (long), weight_sums and weighted_return_sums (float64) are
    separate learner state (S,). Save/restore all three with Model.state_dict,
    target_policy, gamma, method and visit_mode. Stats follow the live table
    device; float64 accumulation does not change the table dtype. A numerical
    failure commits no table/stat updates. Input episodes are never modified.
    """

    def __init__(self, model: Model, return_generator: MonteCarloReturnGenerator,
                 target_policy, method: str = "weighted", visit_mode: str = "first") -> None:
        if not isinstance(model, Model) or not isinstance(model.estimator, TabularVEstimator):
            raise TypeError("OffPolicyMCPredictionLearner requires a Model with TabularVEstimator")
        if method not in ("ordinary", "weighted"):
            raise ValueError("method must be 'ordinary' or 'weighted'")
        if visit_mode not in ("first", "every"):
            raise ValueError("visit_mode must be 'first' or 'every'")
        table = model.estimator.v_table
        policy = torch.as_tensor(target_policy, device=table.device)
        self._validate_policy(policy, table.numel())
        self.target_policy = policy.detach().to(dtype=torch.float64).clone()
        self.model, self.return_generator = model, return_generator
        self.method, self.visit_mode = method, visit_mode
        self.visit_counts = torch.zeros_like(table, dtype=torch.long)
        self.weight_sums = torch.zeros_like(table, dtype=torch.float64)
        self.weighted_return_sums = torch.zeros_like(table, dtype=torch.float64)

    @torch.no_grad()
    def learn(self, episode: Episode) -> Tensor:
        self._validate(episode)
        table = self.model.estimator.v_table
        states = torch.stack([t.observation for t in episode.transitions]).to(device=table.device, dtype=torch.long)
        actions = torch.stack([t.action for t in episode.transitions]).to(device=table.device, dtype=torch.long)
        rewards = torch.stack([t.reward.to(dtype=table.dtype) for t in episode.transitions]).to(table.device)
        log_b = torch.stack([t.log_prob.to(dtype=torch.float64) for t in episode.transitions]).to(table.device)
        returns = self.return_generator.generate(rewards)
        weights = self._importance_weights(states, actions, log_b)

        # Work on copies so overflow cannot leave a partially learned episode.
        counts = self.visit_counts.to(table.device).clone()
        weight_sums = self.weight_sums.to(table.device).clone()
        return_sums = self.weighted_return_sums.to(table.device).clone()
        seen = set()
        for time, state in enumerate(states):
            state_index = state.item()
            if self.visit_mode == "first" and state_index in seen:
                continue
            seen.add(state_index)
            counts[state] += 1
            weight_sums[state] += weights[time]
            return_sums[state] += weights[time] * returns[time].double()

        denominator = counts if self.method == "ordinary" else weight_sums
        observed = denominator > 0
        values = table.clone()
        values[observed] = (return_sums[observed] / denominator[observed]).to(table.dtype)
        self._validate_statistics(weight_sums, return_sums, values)
        self.visit_counts = self.visit_counts.to(table.device)
        self.weight_sums = self.weight_sums.to(table.device)
        self.weighted_return_sums = self.weighted_return_sums.to(table.device)
        self.visit_counts.copy_(counts)
        self.weight_sums.copy_(weight_sums)
        self.weighted_return_sums.copy_(return_sums)
        table.copy_(values)
        return returns

    def _importance_weights(self, states, actions, log_b):
        """rho[t] = product(pi/b) from t onwards, computed by reverse log sums."""
        policy = self.target_policy.to(states.device)
        log_ratios = policy[states, actions].log() - log_b
        log_weights = log_ratios.flip(0).cumsum(0).flip(0)
        weights = log_weights.exp()  # log(0)=-inf correctly makes the affected suffix weight zero.
        if not torch.isfinite(weights).all():
            raise FloatingPointError("Importance weights overflowed; inspect behavior coverage and episode length")
        if torch.any(torch.isfinite(log_weights) & (weights == 0)):
            raise FloatingPointError("Positive importance weights underflowed; no episode update was committed")
        return weights

    def _validate(self, episode):
        if not isinstance(episode, Episode):
            raise TypeError("learn expects an Episode")
        if not episode.terminated:
            raise ValueError("MC requires a terminated episode; open or truncated-only episodes are incomplete")
        n_states, n_actions = self.target_policy.shape
        for transition in episode.transitions:
            for index, size, name in ((transition.observation, n_states, "state"),
                                      (transition.action, n_actions, "action")):
                if index.ndim != 0 or index.is_floating_point() or index.is_complex() or index.dtype == torch.bool:
                    raise TypeError(f"MC {name} must be a scalar integer index")
                if not 0 <= index.item() < size:
                    raise IndexError(f"MC {name} index out of bounds")
            log_b = transition.log_prob
            if log_b is None:
                raise ValueError("Off-policy MC requires sampling-time behavior log_prob for every action")
            if log_b.ndim != 0 or not log_b.is_floating_point():
                raise TypeError("Behavior log_prob must be a scalar floating Tensor")
            if not torch.isfinite(log_b) or log_b > 0:
                raise ValueError("Behavior log_prob must be finite and <= 0 (positive probability <= 1)")

    @staticmethod
    def _validate_policy(policy, n_states):
        if policy.ndim != 2 or policy.shape[0] != n_states or policy.shape[1] == 0:
            raise ValueError("target_policy must have shape (n_states, n_actions)")
        if policy.is_complex() or policy.dtype == torch.bool:
            raise TypeError("target_policy probabilities must be real numeric values")
        if not torch.isfinite(policy).all() or torch.any(policy < 0) or torch.any(policy > 1):
            raise ValueError("target_policy probabilities must be finite and in [0, 1]")
        tolerance = 1e-6 if policy.is_floating_point() and policy.dtype != torch.float64 else 1e-12
        if not torch.allclose(policy.double().sum(-1), torch.ones(n_states, device=policy.device, dtype=torch.float64),
                              rtol=0, atol=tolerance):
            raise ValueError("target_policy probabilities must sum to 1 for each state")

    @staticmethod
    def _validate_statistics(weight_sums, return_sums, values):
        if not all(torch.isfinite(tensor).all() for tensor in (weight_sums, return_sums, values)):
            raise FloatingPointError("IS statistics or values became non-finite; no episode update was committed")
