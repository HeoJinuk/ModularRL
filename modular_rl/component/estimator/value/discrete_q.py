"""Neural Q(s, all discrete actions)."""

from numbers import Integral

from torch import Tensor, nn

from modular_rl.component.estimator.base import QEstimator
from modular_rl.data import EstimatorOutput, ValueOutput


class DiscreteQEstimator(QEstimator):
    """Map float features (..., F) to q[0] (..., A), one value per action.

    network returns (..., n_actions); the default is Linear.
    """

    def __init__(
        self, feature_dim: int, n_actions: int, *, network: nn.Module | None = None
    ) -> None:
        super().__init__()
        if any(not isinstance(dim, Integral) or isinstance(dim, bool) or dim <= 0
               for dim in (feature_dim, n_actions)):
            raise ValueError("feature_dim and n_actions must be positive integers")
        if network is not None and not isinstance(network, nn.Module):
            raise TypeError("network must be an nn.Module or None")
        self.n_actions = n_actions
        self.network = network if network is not None else nn.Linear(feature_dim, n_actions)

    def forward(self, features: Tensor) -> EstimatorOutput:
        q = self.network(features)
        self._validate_output(q, features)
        return EstimatorOutput(values=ValueOutput(q=(q,)))

    def _validate_output(self, q: Tensor, features: Tensor) -> None:
        """Require one scalar Q per action and preserve the feature batch."""
        if not isinstance(q, Tensor):
            raise TypeError("DiscreteQ network must return a Q-value Tensor")
        expected = features.shape[:-1] + (self.n_actions,)
        if q.shape != expected:
            raise ValueError(
                f"DiscreteQ network output must have shape {tuple(expected)}, "
                f"got {tuple(q.shape)}"
            )
