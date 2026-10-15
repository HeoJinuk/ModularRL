"""Concise updates; each name aliases its original implementation."""

from modular_rl.component.update_rule import TabularVUpdateRule as TabularV
from modular_rl.component.update_rule import TabularQUpdateRule as TabularQ
from modular_rl.component.update_rule import TDLossUpdateRule as TDLoss
from modular_rl.component.update_rule import ActorCriticUpdateRule as ActorCritic

__all__ = ['TabularV', 'TabularQ', 'TDLoss', 'ActorCritic']
