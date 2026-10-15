"""Concise targets; each name aliases its original implementation."""

from modular_rl.component.target_generator import TD0TargetGenerator as TD0
from modular_rl.component.target_generator import NStepTargetGenerator as NStep
from modular_rl.component.target_generator import MonteCarloReturnGenerator as MCReturn

__all__ = ['TD0', 'NStep', 'MCReturn']
