"""A direct advantage head for discrete actions."""

from numbers import Integral

from torch import Tensor, nn

from modular_rl.component.estimator.base import AdvantageEstimator
from modular_rl.data import EstimatorOutput, ValueOutput


class DiscreteAdvantageEstimator(AdvantageEstimator):
    """Map float features (..., F) to advantage (..., A).

    network returns (..., n_actions); the default is Linear.
    This head is intended to learn A(s,a) from appropriate targets. It does
    not calculate GAE, center its output, or construct a dueling Q network.
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
        advantage = self.network(features)
        self._validate_output(advantage, features)
        return EstimatorOutput(values=ValueOutput(advantage=advantage))

    def _validate_output(self, advantage: Tensor, features: Tensor) -> None:
        """Require one advantage per action and preserve the feature batch."""
        if not isinstance(advantage, Tensor):
            raise TypeError("DiscreteAdvantage network must return an advantage Tensor")
        expected = features.shape[:-1] + (self.n_actions,)
        if advantage.shape != expected:
            raise ValueError(
                f"DiscreteAdvantage network output must have shape {tuple(expected)}, "
                f"got {tuple(advantage.shape)}"
            )
