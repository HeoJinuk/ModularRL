"""Observed deterministic tabular dynamics, independent of inference Models."""

import numpy as np
import torch

from modular_rl.data import Transition


class TabularWorldModel:
    """Latest deterministic outcome for each observed (state, action) pair.

    Stored rewards are owned, detached CPU scalar tensors with original dtype.
    State/action IDs and termination are Python scalars. External truncation and
    behavior log probabilities are not dynamics and are not predicted. Sampling
    is uniform over observed pairs, using ONLY the caller's NumPy Generator.
    This is a sample model, not a transition history or a stochastic estimator.
    """

    def __init__(self):
        self._outcomes = {}

    def observe(self, transition: Transition):
        """Replace one pair's prediction, without changing Q or consuming RNG."""
        batch = transition.to_batch()
        self._validate(batch)
        state, action = batch.observation.item(), batch.action.item()
        self._outcomes[state, action] = Transition(
            state, action, batch.reward[0].cpu().clone(),
            batch.next_observation.item(), batch.terminated.item(), False,
        )

    def sample(self, rng: np.random.Generator) -> Transition:
        """Generate one owned prediction; an empty model raises ValueError."""
        if not self._outcomes:
            raise ValueError("Cannot sample an empty World Model")
        keys = tuple(self._outcomes)
        record = self._outcomes[keys[int(rng.integers(len(keys)))]]
        return self._copy(record)

    def reset(self):
        self._outcomes.clear()

    def __len__(self):
        return len(self._outcomes)

    def state_dict(self):
        """Owned records in insertion order; sampling order is checkpoint state."""
        return {"outcomes": [self._copy(record) for record in self._outcomes.values()]}

    def load_state_dict(self, state):
        """Validate a replacement before committing it; retain insertion order."""
        restored = TabularWorldModel()
        for record in state["outcomes"]:
            restored.observe(record)
        self._outcomes = restored._outcomes

    @staticmethod
    def _copy(record):
        return Transition(record.observation, record.action, record.reward.clone(),
                          record.next_observation, record.terminated, False)

    @staticmethod
    def _validate(batch):
        for name in ("observation", "action", "reward", "next_observation", "terminated", "truncated"):
            if getattr(batch, name).shape != (1,):
                raise ValueError("World Model requires scalar tabular transition fields")
        for name in ("observation", "action", "next_observation"):
            value = getattr(batch, name)
            if value.is_floating_point() or value.is_complex() or value.dtype == torch.bool:
                raise TypeError("World Model state/action IDs must have integer dtype")
            if value.item() < 0:
                raise ValueError("World Model state/action IDs must be nonnegative")
        if not batch.reward.is_floating_point():
            raise TypeError("World Model reward must have floating dtype")
        if not torch.isfinite(batch.reward).all():
            raise ValueError("World Model reward must be finite")
        if batch.terminated.dtype != torch.bool or batch.truncated.dtype != torch.bool:
            raise TypeError("World Model boundary flags must have bool dtype")
