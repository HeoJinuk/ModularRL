"""Categorical policy for discrete action spaces."""

from numbers import Integral

from torch import Tensor, nn
from torch.distributions import Categorical

from modular_rl.component.estimator.base import PolicyEstimator
from modular_rl.data import EstimatorOutput


class CategoricalPolicyEstimator(PolicyEstimator):
    """Map float features (..., F) to a Categorical policy with batch (...).

    network maps features to logits (..., n_actions); the default is Linear.
    The distribution describes the policy; an ActionSelector samples it.
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
        logits = self.network(features)
        self._validate_output(logits, features)
        return EstimatorOutput(policy=Categorical(logits=logits))

    def _validate_output(self, logits: Tensor, features: Tensor) -> None:
        """Require one logit per action and preserve the feature batch shape."""
        if not isinstance(logits, Tensor):
            raise TypeError("CategoricalPolicy network must return a Tensor of logits")
        expected = features.shape[:-1] + (self.n_actions,)
        if logits.shape != expected:
            raise ValueError(
                f"CategoricalPolicy network output must have shape {tuple(expected)}, "
                f"got {tuple(logits.shape)}"
            )
