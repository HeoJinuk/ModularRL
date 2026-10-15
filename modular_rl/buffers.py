"""Concise buffers; each name aliases its original implementation."""

from modular_rl.component.buffer import ReplayBuffer as Replay
from modular_rl.component.buffer import SequenceReplayBuffer as SequenceReplay

__all__ = ['Replay', 'SequenceReplay']
