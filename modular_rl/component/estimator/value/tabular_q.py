"""Tabular state-index Q values, stored as a non-gradient buffer."""

import torch
from torch import Tensor

from modular_rl.component.estimator.base import QEstimator
from modular_rl.data import EstimatorOutput, ValueOutput


class TabularQEstimator(QEstimator):
    """Look up Q(s, .): long state indices (...) -> q[0] (..., A)."""

    def __init__(self, n_states: int, n_actions: int) -> None:
        super().__init__()
        if n_states <= 0 or n_actions <= 0:
            raise ValueError("n_states and n_actions must be positive")
        self.register_buffer("q_table", torch.zeros(n_states, n_actions))

    def forward(self, features: Tensor) -> EstimatorOutput:
        self._validate(features)
        return EstimatorOutput(values=ValueOutput(q=(self.q_table[features],)))

    def _validate(self, features: Tensor) -> None:
        """Require integer state indices within the table."""
        if features.dtype != torch.long:
            raise TypeError("TabularQ expects state indices of dtype torch.long")
        if torch.any(features < 0) or torch.any(features >= self.q_table.shape[0]):
            raise IndexError("State index out of bounds")
