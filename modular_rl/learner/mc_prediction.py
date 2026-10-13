"""First/every-visit tabular MC prediction with incremental sample averages."""

import torch
from torch import Tensor

from modular_rl.component.estimator import TabularVEstimator
from modular_rl.component.target_generator import MonteCarloReturnGenerator
from modular_rl.data import Episode
from modular_rl.model import Model


class MCPredictionLearner:
    """Learn a fixed policy's state values from complete externally collected episodes.

    Only observation state indices and rewards are used for prediction. The
    final next_observation is not another visit. No MDP, policy, optimizer or
    sampling operation is passed to this learner. ActionSelector is optional.

    learn returns all discounted targets (T,) in time order, detached, even
    when first-visit selection ignores later visits for updates. State indices
    are integer scalars, normalized to long; rewards use the table dtype/device.

    visit_counts (long, N) is learner state, separate from Model.state_dict.
    Save/restore it together with the Model table and identical gamma/visit_mode
    for continuation. Counts follow the live table's device at each learn call.
    The representation should preserve table indices for Model inference.
    """

    def __init__(
        self, model: Model, return_generator: MonteCarloReturnGenerator, visit_mode: str = "first"
    ) -> None:
        if not isinstance(model, Model) or not isinstance(model.estimator, TabularVEstimator):
            raise TypeError("MCPredictionLearner requires a Model with TabularVEstimator")
        if visit_mode not in ("first", "every"):
            raise ValueError("visit_mode must be 'first' or 'every'")
        self.model = model
        self.return_generator = return_generator
        self.visit_mode = visit_mode
        self.visit_counts = torch.zeros_like(model.estimator.v_table, dtype=torch.long)

    @torch.no_grad()
    def learn(self, episode: Episode) -> Tensor:
        self._validate(episode)
        table = self.model.estimator.v_table
        self.visit_counts = self.visit_counts.to(table.device)
        states = torch.stack([transition.observation for transition in episode.transitions]).to(
            device=table.device, dtype=torch.long
        )
        rewards = torch.stack([transition.reward.to(dtype=table.dtype)
                               for transition in episode.transitions]).to(table.device)
        returns = self.return_generator.generate(rewards)

        seen = set()
        for time, state in enumerate(states):
            state_index = state.item()
            if self.visit_mode == "first" and state_index in seen:
                continue
            seen.add(state_index)
            self.visit_counts[state] += 1
            table[state] += (returns[time] - table[state]) / self.visit_counts[state]
        return returns

    def _validate(self, episode: Episode) -> None:
        if not isinstance(episode, Episode):
            raise TypeError("learn expects an Episode")
        if not episode.terminated:
            raise ValueError("MC requires a terminated episode; open or truncated-only episodes are incomplete")
        n_states = self.model.estimator.v_table.numel()
        for transition in episode.transitions:
            state = transition.observation
            if (state.ndim != 0 or state.is_floating_point() or state.is_complex() or state.dtype == torch.bool):
                raise TypeError("MC observations must be scalar integer state indices")
            if not 0 <= state.item() < n_states:
                raise IndexError("MC state index out of bounds")
