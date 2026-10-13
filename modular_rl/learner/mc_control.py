"""Tabular on-policy MC control with first/every-visit sample averages."""

import torch
from torch import Tensor

from modular_rl.component.estimator import TabularQEstimator
from modular_rl.component.target_generator import MonteCarloReturnGenerator
from modular_rl.data import Episode
from modular_rl.model import Model


class TabularMCControlLearner:
    """Update the existing Q table from a complete externally collected episode.

    First-visit selects the earliest occurrence of each (state, action) pair;
    every-visit includes all occurrences. learn returns all discounted returns
    (T,) in chronological order, detached, using the live table dtype/device.
    Scalar integer state/action IDs are normalized to long. The final next
    observation is not another visit. The input episode is never modified.

    The caller collects with the model's epsilon-greedy policy, without Q
    updates during the episode. After learn, Model.act uses the updated table
    for the next episode. Policy selection, epsilon schedules, environment,
    sampling and optimizer operations stay outside this learner.

    visit_counts is a long Tensor (S, A), separate from Model.state_dict. Save
    and restore both table and counts, plus gamma/visit_mode, for continuation.
    Counts follow the table's device at each learn call. Use a representation
    that preserves state indices for Model inference, e.g. IdentityRepresentation.
    """

    def __init__(
        self, model: Model, return_generator: MonteCarloReturnGenerator, visit_mode: str = "first"
    ) -> None:
        if not isinstance(model, Model) or not isinstance(model.estimator, TabularQEstimator):
            raise TypeError("TabularMCControlLearner requires a Model with TabularQEstimator")
        if visit_mode not in ("first", "every"):
            raise ValueError("visit_mode must be 'first' or 'every'")
        self.model = model
        self.return_generator = return_generator
        self.visit_mode = visit_mode
        self.visit_counts = torch.zeros_like(model.estimator.q_table, dtype=torch.long)

    @torch.no_grad()
    def learn(self, episode: Episode) -> Tensor:
        self._validate(episode)
        table = self.model.estimator.q_table
        self.visit_counts = self.visit_counts.to(table.device)
        states = torch.stack([transition.observation for transition in episode.transitions]).to(
            device=table.device, dtype=torch.long
        )
        actions = torch.stack([transition.action for transition in episode.transitions]).to(
            device=table.device, dtype=torch.long
        )
        rewards = torch.stack([transition.reward.to(dtype=table.dtype)
                               for transition in episode.transitions]).to(table.device)
        returns = self.return_generator.generate(rewards)

        seen = set()
        for time, (state, action) in enumerate(zip(states, actions)):
            pair = (state.item(), action.item())
            if self.visit_mode == "first" and pair in seen:
                continue
            seen.add(pair)
            self.visit_counts[state, action] += 1
            table[state, action] += (returns[time] - table[state, action]) / self.visit_counts[state, action]
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
