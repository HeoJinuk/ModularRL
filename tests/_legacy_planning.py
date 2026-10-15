"""Frozen Dyna reference from 32634ce; no delegation to new world/planner code."""

from dataclasses import fields
from numbers import Integral
import numpy as np
import torch
from modular_rl.data import Transition, TransitionBatch


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


def tabular_plan(world_model, learner, *, steps, rng):
    """Run at most `steps` independent one-step Q updates, without an environment.

    A nonempty model yields exactly `steps` updates; an empty model yields none.
    Return signed errors (actual_updates,) in the learner table's dtype/device.
    Each backup reads live Q after the preceding update. No observe/act call,
    behavior RNG, real window or scheduling clock is involved.
    """
    _validate_budget(steps, "planning steps", minimum=0)
    errors = []
    if len(world_model):
        for _ in range(steps):
            synthetic = world_model.sample(rng)
            errors.append(learner.learn(synthetic.to_batch())[0])
    table = learner.model.estimator.q_table
    return torch.stack(errors) if errors else table.new_empty(0)

def synthetic_batch(world_model, *, batch_size, rng):
    """B independent uniform observed-pair predictions WITH replacement.

    One observed pair is sufficient even for B=32. State/action/flags have
    shape (B,), observations remain integer IDs, log_prob is None. The world
    supplies owned predictions; stacking produces an independent CPU batch.
    """
    _positive_integer(batch_size, "batch_size")
    rows = [world_model.sample(rng).to_batch() for _ in range(batch_size)]
    return TransitionBatch(**{
        field.name: None if field.name == "log_prob" else torch.cat([getattr(row, field.name) for row in rows])
        for field in fields(TransitionBatch)
    })


def neural_plan(world_model, learner, *, steps, batch_size, rng, record_batches=False):
    """Run independent 1-step DQN minibatch updates through the SAME learner.

    No environment, observe, replay sampling, epsilon change or target sync.
    With the example's environment-step sync schedule, idle planning never
    advances periodic syncs. Empty models execute zero updates and consume no
    RNG. The result distinguishes requested and actual optimizer steps and
    contains detached TDLossOutput metrics. Batch recording is optional QA data.
    """
    _positive_integer(steps, "planning steps", minimum=0)
    _positive_integer(batch_size, "batch_size")
    results, batches = [], []
    if len(world_model):
        for _ in range(steps):
            batch = synthetic_batch(world_model, batch_size=batch_size, rng=rng)
            results.append(learner.learn(batch))
            if record_batches:
                batches.append(batch)
    return {"requested_updates": steps, "updates": len(results), "results": results, "batches": batches}

def _validate_budget(value, name, *, minimum):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _positive_integer(value, name, *, minimum=1):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
