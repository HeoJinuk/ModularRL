"""Sample a categorical policy or choose its mode."""

from torch import Tensor
from torch.distributions import Categorical

from modular_rl.component.action_selector.base import BaseActionSelector
from modular_rl.data import ActionOutput, EstimatorOutput


class CategoricalActionSelector(BaseActionSelector):
    """Use only policy; choose the first mode during deterministic evaluation."""

    @staticmethod
    def _validate(estimate: EstimatorOutput) -> None:
        """Check that the policy payload is a Categorical distribution."""
        if not isinstance(estimate.policy, Categorical):
            raise TypeError("Expected a torch.distributions.Categorical policy")

    def probabilities(self, estimate: EstimatorOutput) -> Tensor:
        """Return the categorical policy's (..., A) action probabilities."""
        self._validate(estimate)
        return estimate.policy.probs

    def select(
        self, estimate: EstimatorOutput, *, deterministic: bool = False
    ) -> ActionOutput:
        """Choose the first mode or sample the policy with its log_prob."""
        self._validate(estimate)
        policy = estimate.policy
        if deterministic:
            action = policy.probs.argmax(dim=-1)
            return ActionOutput(action=action)
        action = policy.sample()
        return ActionOutput(action=action, log_prob=policy.log_prob(action))
