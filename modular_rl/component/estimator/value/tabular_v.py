"""Tabular state values, stored as a non-gradient buffer."""

import torch
from torch import Tensor

from modular_rl.component.estimator.base import VEstimator
from modular_rl.data import EstimatorOutput, ValueOutput


class TabularVEstimator(VEstimator):
    """Look up V(s): long state indices (...) -> v (...)."""

    def __init__(self, n_states: int) -> None:
        super().__init__()
        if n_states <= 0:
            raise ValueError("n_states must be positive")
        self.register_buffer("v_table", torch.zeros(n_states))

    def forward(self, features: Tensor) -> EstimatorOutput:
        self._validate(features)
        return EstimatorOutput(values=ValueOutput(v=self.v_table[features]))

    def _validate(self, features: Tensor) -> None:
        """Require integer state indices within the table."""
        if features.dtype != torch.long:
            raise TypeError("TabularV expects state indices of dtype torch.long")
        if torch.any(features < 0) or torch.any(features >= self.v_table.shape[0]):
            raise IndexError("State index out of bounds")
