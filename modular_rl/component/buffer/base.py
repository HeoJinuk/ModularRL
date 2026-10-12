"""Minimal replay-buffer interface, without learning or target generation."""

from abc import ABC, abstractmethod

from modular_rl.data import Transition, TransitionBatch


class BaseBuffer(ABC):
    @abstractmethod
    def add(self, transition: Transition) -> None:
        """Store one transition."""
        ...

    @abstractmethod
    def sample(self, batch_size: int) -> TransitionBatch:
        """Return a replay batch with leading shape (B, ...)."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all stored transitions."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of stored transitions."""
        ...
