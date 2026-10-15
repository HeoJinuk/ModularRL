"""Dyna planning over generated one-step experiences, without environment access."""

from dataclasses import fields
from numbers import Integral

import numpy as np
import torch

from modular_rl.data import TransitionBatch


def synthetic_batch(world_model, *, batch_size, rng) -> TransitionBatch:
    """Stack B independent world predictions, sampling WITH replacement.

    Observations keep their shape and dtype. Predictions have no behavior
    policy log probability; they are one-step inputs to Q-learning or DQN.
    """
    _validate_integer(batch_size, "batch_size", minimum=1)
    rows = [world_model.sample(rng).to_batch() for _ in range(batch_size)]
    return TransitionBatch(**{
        field.name: None if field.name == "log_prob" else torch.cat([getattr(row, field.name) for row in rows])
        for field in fields(TransitionBatch)
    })


class DynaPlanner:
    """Generate batches and call the SAME learner used for actual experience.

    The world supplies len(world) and sample(rng) -> Transition; the learner
    supplies learn(TransitionBatch). No new base class is required. Default
    batch_size=1 gives one tabular backup per planning step. DQN uses B
    predictions per gradient step. Budget and result['updates'] count learn
    calls, not virtual transitions or environment steps.

    Pass seed for an independent NumPy Generator, or rng to retain a caller's
    stream. Save/restore rng.bit_generator.state separately from world and
    learner checkpoints. Planning never selects actions, observes the world,
    samples replay or synchronizes targets.
    """

    def __init__(self, world_model, learner, *, batch_size=1, seed=None, rng=None):
        _validate_integer(batch_size, "batch_size", minimum=1)
        if rng is not None and seed is not None:
            raise ValueError("Supply either seed or rng, not both")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a NumPy Generator")
        self.world_model = world_model
        self.learner = learner
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed) if rng is None else rng

    def plan(self, steps=1, *, record_batches=False):
        """Return requested_updates, updates, results and optional QA batches.

        results contains each unchanged learner return (signed errors for
        tabular Q, detached TDLossOutput for DQN). An empty world or zero budget
        performs no updates and consumes no planning RNG. Batches are retained
        only when explicitly requested.
        """
        _validate_integer(steps, "planning steps", minimum=0)
        results, batches = [], []
        if len(self.world_model):
            for _ in range(steps):
                batch = synthetic_batch(self.world_model, batch_size=self.batch_size, rng=self.rng)
                results.append(self.learner.learn(batch))
                if record_batches:
                    batches.append(batch)
        return {"requested_updates": steps, "updates": len(results), "results": results, "batches": batches}


def _validate_integer(value, name, *, minimum):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
