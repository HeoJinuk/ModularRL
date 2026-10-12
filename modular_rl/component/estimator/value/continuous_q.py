"""Action-conditioned Q(s,a) for continuous action vectors."""

from numbers import Integral

import torch
from torch import Tensor, nn

from modular_rl.component.estimator.base import QEstimator
from modular_rl.data import EstimatorOutput, ValueOutput


class ContinuousQEstimator(QEstimator):
    """Evaluate float features (..., F) and actions (..., A) -> q[0] (...).

    Batch dimensions must match; inputs are not broadcast, converted, or moved.
    network takes concatenated (..., F+A) vectors and returns (..., 1).
    The default joint MLP allows nonlinear state-action interactions after
    the shared Representation. Use evaluate(), since Q(s,a) needs an action.
    """

    def __init__(
        self, feature_dim: int, action_dim: int, hidden_dim: int = 128,
        *, network: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if any(not isinstance(dim, Integral) or isinstance(dim, bool) or dim <= 0
               for dim in (feature_dim, action_dim)):
            raise ValueError("feature_dim and action_dim must be positive integers")
        if network is None and (not isinstance(hidden_dim, Integral)
                                or isinstance(hidden_dim, bool) or hidden_dim <= 0):
            raise ValueError("hidden_dim must be a positive integer for the default network")
        if network is not None and not isinstance(network, nn.Module):
            raise TypeError("network must be an nn.Module or None")
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.network = network if network is not None else nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> EstimatorOutput:
        raise NotImplementedError("ContinuousQEstimator requires evaluate(features, actions)")

    def evaluate(self, features: Tensor, actions: Tensor) -> EstimatorOutput:
        self._validate(features, actions)
        state_action = torch.cat((features, actions), dim=-1)
        q = self.network(state_action)
        self._validate_output(q, features)
        return EstimatorOutput(values=ValueOutput(q=(q.squeeze(-1),)))

    def _validate(self, features: Tensor, actions: Tensor) -> None:
        """Require float vectors of the configured size and matching batches."""
        if features.ndim < 1 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"features must have shape (..., {self.feature_dim})")
        if actions.ndim < 1 or actions.shape[-1] != self.action_dim:
            raise ValueError(f"actions must have shape (..., {self.action_dim})")
        if features.shape[:-1] != actions.shape[:-1]:
            raise ValueError("features and actions must have matching batch dimensions")
        if not features.is_floating_point() or not actions.is_floating_point():
            raise TypeError("ContinuousQ requires floating-point features and actions")

    @staticmethod
    def _validate_output(q: Tensor, features: Tensor) -> None:
        """Require one scalar Q per supplied state-action pair."""
        if not isinstance(q, Tensor):
            raise TypeError("ContinuousQ network must return a Q-value Tensor")
        expected = features.shape[:-1] + (1,)
        if q.shape != expected:
            raise ValueError(
                f"ContinuousQ network output must have shape {tuple(expected)}, "
                f"got {tuple(q.shape)}"
            )
