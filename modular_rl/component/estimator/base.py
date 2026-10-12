"""Minimal estimator roles; each returns an EstimatorOutput."""

from abc import ABC, abstractmethod

from torch import Tensor, nn

from modular_rl.data import EstimatorOutput


class BaseEstimator(nn.Module, ABC):
    @abstractmethod
    def forward(self, features: Tensor) -> EstimatorOutput:
        ...


# Backwards-compatible name from the initial reference package.
Estimator = BaseEstimator


class PolicyEstimator(BaseEstimator):
    """Estimates a policy, stochastic or deterministic."""


# Explicit Base names share the existing roles; no extra inheritance layer.
BasePolicyEstimator = PolicyEstimator


class ValueEstimator(BaseEstimator):
    """Base class for estimates with value-function semantics."""


BaseValueEstimator = ValueEstimator


class VEstimator(ValueEstimator):
    """Estimates V(s)."""


class QEstimator(ValueEstimator):
    """Estimates Q(s, .) where possible.

    Action-conditioned Q(s,a) implementations may additionally provide
    evaluate(features, actions). This is not required of every Q estimator.
    """


class AdvantageEstimator(ValueEstimator):
    """Role for directly estimated A(s,a), not GAE or a dueling head.

    GAE belongs to TargetGenerator. DiscreteAdvantageEstimator is a concrete
    head for this role; a dueling network's centered head has different meaning.
    """
