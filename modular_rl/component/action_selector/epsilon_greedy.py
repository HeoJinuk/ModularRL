"""Epsilon-greedy behavior selection from a discrete Q vector."""

from numbers import Real

import torch
from torch import Tensor

from modular_rl.component.action_selector.greedy import GreedyActionSelector
from modular_rl.data import EstimatorOutput


class EpsilonGreedyActionSelector(GreedyActionSelector):
    """P(a|s) = epsilon/A + (1-epsilon) * I(a is maximal)/number of maxima.

    Sampling uses PyTorch's RNG. With deterministic=True, epsilon is ignored
    and the smallest index of a maximal Q action is returned, with no log_prob.
    """

    def __init__(self, epsilon: float = 0.1) -> None:
        super().__init__()
        self.register_buffer("_epsilon", torch.empty(()))
        self.epsilon = epsilon

    @staticmethod
    def _validate_epsilon(value: float) -> None:
        """Check epsilon's numeric type and range when it is configured."""
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError("epsilon must be a real number")
        if not 0 <= value <= 1:
            raise ValueError("epsilon must be in [0, 1]")

    @property
    def epsilon(self) -> float:
        return float(self._epsilon.item())

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        self._validate_epsilon(value)
        self._epsilon.fill_(float(value))

    def probabilities(self, estimate: EstimatorOutput) -> Tensor:
        """Mix uniform exploration with greedy probabilities of shape (..., A)."""
        greedy = super().probabilities(estimate)
        epsilon = self._epsilon.to(device=greedy.device, dtype=greedy.dtype)
        return (1 - epsilon) * greedy + epsilon / greedy.shape[-1]
