"""Greedy selection from one scalar discrete Q-value tensor."""

import torch
from torch import Tensor
from torch.distributions import Categorical

from modular_rl.component.action_selector.base import BaseActionSelector
from modular_rl.data import ActionOutput, EstimatorOutput


class GreedyActionSelector(BaseActionSelector):
    """Uniformly sample tied maxima; deterministic selection uses the first.

    Expects exactly one scalar discrete Q tensor of shape (..., A).
    The last axis contains scalar values for A discrete actions; every
    preceding axis is a batch axis. Distributional payloads and multiple
    critics need an explicit reduction before using this selector.

    Validation checks the required payload and shape. The estimator is
    responsible for supplying numerically valid Q values.
    """

    @staticmethod
    def _validate(estimate: EstimatorOutput) -> None:
        """Check Q presence, count, Tensor type, and the action axis."""
        if not estimate.values.q:
            raise ValueError("Q selection requires values.q")
        if len(estimate.values.q) != 1:
            raise ValueError("Q selection requires exactly one Q estimate; combine critics explicitly")

        q = estimate.values.q[0]
        if not isinstance(q, Tensor):
            raise TypeError("Expected a scalar discrete Q-value Tensor")
        if q.ndim < 1 or q.shape[-1] == 0:
            raise ValueError("Q-values must have shape (..., A) with at least one action")

    def probabilities(self, estimate: EstimatorOutput) -> Tensor:
        """Return P(a|s), sharing mass uniformly among maximal Q actions."""
        self._validate(estimate)
        q = estimate.values.q[0]
        # Keep double precision; use float32 for integer/low-precision inputs.
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        ties = (q == q.max(dim=-1, keepdim=True).values).to(dtype=dtype)
        return ties / ties.sum(dim=-1, keepdim=True)

    def select(
        self, estimate: EstimatorOutput, *, deterministic: bool = False
    ) -> ActionOutput:
        """Choose a first maximum or sample the behavior policy with log_prob."""
        if deterministic:
            self._validate(estimate)
            q = estimate.values.q[0]
            # Select using Q itself, even when an epsilon-greedy subclass has epsilon=1.
            return ActionOutput(action=q.argmax(dim=-1))

        probs = self.probabilities(estimate)
        # logits = log(P) keeps even tiny exploration probabilities unclamped.
        distribution = Categorical(logits=probs.log())
        action = distribution.sample()
        return ActionOutput(action=action, log_prob=distribution.log_prob(action))
