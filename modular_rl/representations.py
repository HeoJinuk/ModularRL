"""Concise representations; each name aliases its original implementation."""

from modular_rl.component.representation import IdentityRepresentation as Identity
from modular_rl.component.representation import MLPRepresentation as MLP

__all__ = ['Identity', 'MLP']
