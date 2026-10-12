"""An unbounded diagonal Gaussian policy, without action sampling."""

import math
from numbers import Integral

import torch
from torch import Tensor, nn
from torch.distributions import Independent, Normal

from modular_rl.component.estimator.base import PolicyEstimator
from modular_rl.data import EstimatorOutput


class GaussianPolicyEstimator(PolicyEstimator):
    """Map float features (..., F) to an independent Gaussian action vector.

    network maps features to mean (..., A); the default is Linear.
    A learned log_std is shared across states and is not part of the network.
    Batch shape is (...), event shape is (A,), and log_prob sums over actions.
    This is an unbounded Gaussian, not a tanh-squashed or bounded SAC policy.
    Unsafe scale values fail explicitly rather than silently changing the policy.
    """

    def __init__(
        self, feature_dim: int, action_dim: int, log_std_init: float = 0.0,
        *, network: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if any(not isinstance(dim, Integral) or isinstance(dim, bool) or dim <= 0
               for dim in (feature_dim, action_dim)):
            raise ValueError("feature_dim and action_dim must be positive integers")
        if network is not None and not isinstance(network, nn.Module):
            raise TypeError("network must be an nn.Module or None")
        log_std_init = float(log_std_init)
        if not math.isfinite(log_std_init):
            raise ValueError("log_std_init must be finite")
        self.network = network if network is not None else nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def forward(self, features: Tensor) -> EstimatorOutput:
        mean = self.network(features)
        self._validate_output(mean, features)
        std = self.log_std.exp()
        self._validate_scale(std)
        policy = Independent(Normal(mean, std.expand_as(mean)), 1)
        return EstimatorOutput(policy=policy)

    def _validate_output(self, mean: Tensor, features: Tensor) -> None:
        """Require one mean per action coordinate and preserve batch shape."""
        if not isinstance(mean, Tensor):
            raise TypeError("GaussianPolicy network must return a Tensor of means")
        expected = features.shape[:-1] + self.log_std.shape
        if mean.shape != expected:
            raise ValueError(
                f"GaussianPolicy network output must have shape {tuple(expected)}, "
                f"got {tuple(mean.shape)}"
            )

    @staticmethod
    def _validate_scale(std: Tensor) -> None:
        """Keep variance and Normal's 2*variance denominator representable.

        Check only the state-independent action vector, not the whole batch.
        Normal variances also avoid overflow of the reciprocal during backward.
        """
        variance = std.detach().square()
        limits = torch.finfo(std.dtype)
        if torch.any(~torch.isfinite(variance) | (variance < limits.tiny)
                     | (variance > limits.max / 2)):
            raise ValueError(
                f"GaussianPolicy log_std produces an unsafe variance for {std.dtype}; "
                "adjust log_std or use a higher precision dtype"
            )
