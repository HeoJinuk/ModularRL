"""Concise estimators; each name aliases its original implementation."""

from modular_rl.component.estimator import TabularVEstimator as TabularV
from modular_rl.component.estimator import TabularQEstimator as TabularQ
from modular_rl.component.estimator import DiscreteQEstimator as DiscreteQ
from modular_rl.component.estimator import ContinuousQEstimator as ContinuousQ
from modular_rl.component.estimator import ActorCriticEstimator as ActorCritic
from modular_rl.component.estimator import StateValueEstimator as StateValue
from modular_rl.component.estimator import DiscreteAdvantageEstimator as DiscreteAdvantage
from modular_rl.component.estimator import CategoricalPolicyEstimator as CategoricalPolicy
from modular_rl.component.estimator import GaussianPolicyEstimator as GaussianPolicy
from modular_rl.component.estimator import DeterministicPolicyEstimator as DeterministicPolicy

__all__ = ['TabularV', 'TabularQ', 'DiscreteQ', 'ContinuousQ', 'ActorCritic', 'StateValue', 'DiscreteAdvantage', 'CategoricalPolicy', 'GaussianPolicy', 'DeterministicPolicy']
