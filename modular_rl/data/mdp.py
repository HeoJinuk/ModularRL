"""Known finite environment dynamics, separate from the inference Model."""

from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True, slots=True)
class TransitionOutcome:
    """One atom of P(next_state, reward, terminated | state, action).

    Outcomes with the same next state may have different rewards or boundary
    flags. Keep them separate instead of replacing them by an expected reward.
    """

    probability: float
    next_state: int
    reward: float
    terminated: bool = False


@dataclass(frozen=True, slots=True)
class TabularMDP:
    """A complete, known kernel with transitions[state][action][outcome].

    State/action indices are zero-based; all states share n_actions actions.
    Each row is nonempty and its probabilities sum to 1 (absolute tolerance
    1e-12). Input lists are snapshotted into nested tuples and validated once.

    terminal_states is optional. These states have value zero and their
    outgoing rows are placeholders, normally zero-reward terminated self-loops.
    Bootstrap stops for a terminated outcome or entry into a declared terminal
    state. A terminated outcome need not lead to a globally terminal state.
    No Gymnasium format, samples, truncation or learned parameters are stored.
    """

    n_states: int
    n_actions: int
    transitions: tuple[tuple[tuple[TransitionOutcome, ...], ...], ...]
    terminal_states: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "transitions", tuple(
            tuple(tuple(outcomes) for outcomes in actions) for actions in self.transitions
        ))
        object.__setattr__(self, "terminal_states", tuple(self.terminal_states))
        self._validate()

    def _validate(self) -> None:
        if any(not isinstance(size, Integral) or isinstance(size, bool) or size < 1
               for size in (self.n_states, self.n_actions)):
            raise ValueError("n_states and n_actions must be positive integers")
        if len(self.transitions) != self.n_states:
            raise ValueError("transitions must have one row per state")
        for state in self.terminal_states:
            self._validate_state(state)
        for actions in self.transitions:
            if len(actions) != self.n_actions:
                raise ValueError("transitions must have one outcome row per action")
            for outcomes in actions:
                if not outcomes:
                    raise ValueError("each state-action row must have at least one outcome")
                for outcome in outcomes:
                    if not isinstance(outcome, TransitionOutcome):
                        raise TypeError("transition rows must contain TransitionOutcome objects")
                    self._validate_state(outcome.next_state)
                    if (not isinstance(outcome.probability, Real) or isinstance(outcome.probability, bool)
                            or not isfinite(outcome.probability) or not 0 <= outcome.probability <= 1):
                        raise ValueError("outcome probability must be finite and in [0, 1]")
                    if (not isinstance(outcome.reward, Real) or isinstance(outcome.reward, bool)
                            or not isfinite(outcome.reward)):
                        raise ValueError("outcome reward must be a finite real number")
                    if not isinstance(outcome.terminated, (bool, np.bool_)):
                        raise TypeError("outcome terminated must be bool")
                if not isclose(sum(outcome.probability for outcome in outcomes), 1.0,
                               rel_tol=0, abs_tol=1e-12):
                    raise ValueError("outcome probabilities must sum to 1 for each state-action row")

    def _validate_state(self, state: int) -> None:
        if not isinstance(state, Integral) or isinstance(state, bool) or not 0 <= state < self.n_states:
            raise ValueError("state indices must be integers in [0, n_states)")
