"""Concise selectors; each name aliases its original implementation."""

from modular_rl.component.action_selector import GreedyActionSelector as Greedy
from modular_rl.component.action_selector import EpsilonGreedyActionSelector as EpsilonGreedy
from modular_rl.component.action_selector import CategoricalActionSelector as Categorical

__all__ = ['Greedy', 'EpsilonGreedy', 'Categorical']
