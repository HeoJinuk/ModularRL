"""Tabular weighted-importance-sampling MC control with a greedy target."""

import torch
from torch import Tensor

from modular_rl.component.estimator import TabularQEstimator
from modular_rl.component.target_generator import MonteCarloReturnGenerator
from modular_rl.data import Episode
from modular_rl.model import Model


class OffPolicyMCControlLearner:
    """Update the existing Q table from complete external behavior episodes.

    Reverse-time weighted IS starts with W=1. Each Q(s,a) update precedes
    target improvement: argmax of the updated row chooses the smallest action
    ID on ties. If the recorded action differs, stop; otherwise divide W by
    its sampling-time behavior probability before the preceding update.
    Q(s,a) conditions on the current action, so its weight excludes that action.
    No ordinary IS or first/every-visit option is added to this control rule.

    All transitions require scalar integer IDs and finite scalar log b(a|s)
    <=0. The caller ensures coverage, collects with Q/epsilon fixed during the
    episode, and supplies the actual final observation. Truncated-only/open
    episodes are incomplete and rejected. No environment, selector, bootstrap
    or optimizer is called. Episode data is never modified.

    learn returns all raw returns (T,) in forward order, including the prefix
    skipped by early break, detached on the live table dtype/device. Q keeps
    its storage; weight_sums C(s,a) is float64 (S,A) on the table device.
    Work copies make numerical failures atomic. Save Model.state_dict and C,
    gamma and collection RNG state together to resume learning.
    """

    def __init__(self, model: Model, return_generator: MonteCarloReturnGenerator) -> None:
        if not isinstance(model, Model) or not isinstance(model.estimator, TabularQEstimator):
            raise TypeError("OffPolicyMCControlLearner requires a Model with TabularQEstimator")
        self.model, self.return_generator = model, return_generator
        self.weight_sums = torch.zeros_like(model.estimator.q_table, dtype=torch.float64)

    @torch.no_grad()
    def learn(self, episode: Episode) -> Tensor:
        self._validate(episode)
        table = self.model.estimator.q_table
        states = torch.stack([t.observation for t in episode.transitions]).to(device=table.device, dtype=torch.long)
        actions = torch.stack([t.action for t in episode.transitions]).to(device=table.device, dtype=torch.long)
        rewards = torch.stack([t.reward.to(dtype=table.dtype) for t in episode.transitions]).to(table.device)
        log_b = torch.stack([t.log_prob.to(dtype=torch.float64) for t in episode.transitions]).to(table.device)
        returns = self.return_generator.generate(rewards)

        q = table.clone()
        cumulative = self.weight_sums.to(table.device).clone()
        weight = cumulative.new_ones(())
        for time in range(len(episode) - 1, -1, -1):
            state, action = states[time], actions[time]
            cumulative[state, action] += weight
            q[state, action] += (weight / cumulative[state, action]) * (returns[time].double() - q[state, action].double())
            self._validate_update(q[state, action], cumulative[state, action])
            greedy_action = q[state].argmax()
            if action != greedy_action:
                break
            if time > 0:  # No weight is needed after the first transition.
                weight *= (-log_b[time]).exp()
                self._validate_weight(weight)

        self.weight_sums = self.weight_sums.to(table.device)
        self.weight_sums.copy_(cumulative)
        table.copy_(q)
        return returns

    def _validate(self, episode: Episode) -> None:
        if not isinstance(episode, Episode):
            raise TypeError("learn expects an Episode")
        if not episode.terminated:
            raise ValueError("MC requires a terminated episode; open or truncated-only episodes are incomplete")
        n_states, n_actions = self.model.estimator.q_table.shape
        for transition in episode.transitions:
            for index, size, name in ((transition.observation, n_states, "state"),
                                      (transition.action, n_actions, "action")):
                if index.ndim != 0 or index.is_floating_point() or index.is_complex() or index.dtype == torch.bool:
                    raise TypeError(f"MC {name} must be a scalar integer index")
                if not 0 <= index.item() < size:
                    raise IndexError(f"MC {name} index out of bounds")
            log_b = transition.log_prob
            if log_b is None:
                raise ValueError("Off-policy MC Control requires sampling-time behavior log_prob for every action")
            if log_b.ndim != 0 or not log_b.is_floating_point():
                raise TypeError("Behavior log_prob must be a scalar floating Tensor")
            if not torch.isfinite(log_b) or log_b > 0:
                raise ValueError("Behavior log_prob must be finite and <= 0 (positive probability <= 1)")

    @staticmethod
    def _validate_update(value, cumulative_weight):
        if not torch.isfinite(value) or not torch.isfinite(cumulative_weight):
            raise FloatingPointError("Q or cumulative weight became non-finite; no episode update was committed")

    @staticmethod
    def _validate_weight(weight):
        if not torch.isfinite(weight):
            raise FloatingPointError("Importance weight overflowed; no episode update was committed")
