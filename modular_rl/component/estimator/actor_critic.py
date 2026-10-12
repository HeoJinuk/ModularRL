"""Simple actor-critic composition with no training logic."""

from torch import Tensor

from modular_rl.component.estimator.base import BaseEstimator, PolicyEstimator, ValueEstimator
from modular_rl.data import EstimatorOutput


class ActorCriticEstimator(BaseEstimator):
    """Combine a policy estimator and a value estimator with shared features.

    For action-conditioned critics (SAC/TD3), model their conditional
    evaluation separately; their critics cannot generally be evaluated with
    forward(features) alone.
    """

    def __init__(self, policy: PolicyEstimator, value: ValueEstimator) -> None:
        super().__init__()
        self.policy = policy
        self.value = value

    def forward(self, features: Tensor) -> EstimatorOutput:
        policy_output = self.policy(features)
        value_output = self.value(features)
        self._validate(policy_output, value_output)
        return EstimatorOutput(policy=policy_output.policy, values=value_output.values)

    @staticmethod
    def _validate(policy_output: EstimatorOutput, value_output: EstimatorOutput) -> None:
        """Require a policy and at least one value from their respective roles."""
        if policy_output.policy is None:
            raise ValueError("PolicyEstimator did not provide policy")
        values = value_output.values
        if values.v is None and not values.q and values.advantage is None:
            raise ValueError("ValueEstimator did not provide values")
