"""Neural V(s) head."""

from numbers import Integral

from torch import Tensor, nn

from modular_rl.component.estimator.base import VEstimator
from modular_rl.data import EstimatorOutput, ValueOutput


class StateValueEstimator(VEstimator):
    """Map float features (..., F) to scalar V(s) with shape (...).

    network returns (..., 1); the default is Linear. Only that final axis is
    removed, so every leading batch dimension is preserved.
    """

    def __init__(self, feature_dim: int, *, network: nn.Module | None = None) -> None:
        super().__init__()
        if not isinstance(feature_dim, Integral) or isinstance(feature_dim, bool) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        if network is not None and not isinstance(network, nn.Module):
            raise TypeError("network must be an nn.Module or None")
        self.network = network if network is not None else nn.Linear(feature_dim, 1)

    def forward(self, features: Tensor) -> EstimatorOutput:
        value = self.network(features)
        self._validate_output(value, features)
        return EstimatorOutput(values=ValueOutput(v=value.squeeze(-1)))

    @staticmethod
    def _validate_output(value: Tensor, features: Tensor) -> None:
        """Require one scalar value with an explicit final singleton axis."""
        if not isinstance(value, Tensor):
            raise TypeError("StateValue network must return a value Tensor")
        expected = features.shape[:-1] + (1,)
        if value.shape != expected:
            raise ValueError(
                f"StateValue network output must have shape {tuple(expected)}, "
                f"got {tuple(value.shape)}"
            )
