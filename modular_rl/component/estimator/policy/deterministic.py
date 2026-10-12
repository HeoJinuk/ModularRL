"""An unbounded deterministic action head."""

from numbers import Integral

from torch import Tensor, nn

from modular_rl.component.estimator.base import PolicyEstimator
from modular_rl.data import EstimatorOutput


class DeterministicPolicyEstimator(PolicyEstimator):
    """Map float features (..., F) to a policy action Tensor (..., A).

    network maps features to actions (..., action_dim); the default is Linear.
    No sampling, clipping, or action-bound transform is performed.
    An ActionSelector is responsible for producing ActionOutput.
    """

    def __init__(
        self, feature_dim: int, action_dim: int, *, network: nn.Module | None = None
    ) -> None:
        super().__init__()
        if any(not isinstance(dim, Integral) or isinstance(dim, bool) or dim <= 0
               for dim in (feature_dim, action_dim)):
            raise ValueError("feature_dim and action_dim must be positive integers")
        if network is not None and not isinstance(network, nn.Module):
            raise TypeError("network must be an nn.Module or None")
        self.action_dim = action_dim
        self.network = network if network is not None else nn.Linear(feature_dim, action_dim)

    def forward(self, features: Tensor) -> EstimatorOutput:
        action = self.network(features)
        self._validate_output(action, features)
        return EstimatorOutput(policy=action)

    def _validate_output(self, action: Tensor, features: Tensor) -> None:
        """Require one action per coordinate and preserve the feature batch."""
        if not isinstance(action, Tensor):
            raise TypeError("DeterministicPolicy network must return an action Tensor")
        expected = features.shape[:-1] + (self.action_dim,)
        if action.shape != expected:
            raise ValueError(
                f"DeterministicPolicy network output must have shape {tuple(expected)}, "
                f"got {tuple(action.shape)}"
            )
