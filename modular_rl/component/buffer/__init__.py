"""Replay storage components, independent of Model and learning algorithms."""

from modular_rl.component.buffer.base import BaseBuffer
from modular_rl.component.buffer.replay import ReplayBuffer

__all__ = ["BaseBuffer", "ReplayBuffer"]
