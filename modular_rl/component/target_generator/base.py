"""Minimal tensor-to-target interface, independent of models and buffers."""

from abc import ABC, abstractmethod

from torch import Tensor


class BaseTargetGenerator(ABC):
    @abstractmethod
    def generate(self, reward: Tensor, next_value: Tensor, terminated: Tensor) -> Tensor:
        """Calculate targets from externally supplied, same-shaped tensors."""
        ...
