"""Concise buffers; each name aliases its original implementation."""

from modular_rl.component.buffer import ReplayBuffer as Replay
from modular_rl.component.buffer import SequenceReplayBuffer as SequenceReplay
from modular_rl.component.buffer import RolloutBuffer as Rollout

__all__ = ['Replay', 'SequenceReplay', 'Rollout']
