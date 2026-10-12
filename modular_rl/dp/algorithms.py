"""Synchronous finite-MDP dynamic programming with explicit full backups."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import numpy as np

from modular_rl.data.mdp import TabularMDP


@dataclass(frozen=True, slots=True)
class DPResult:
    """Owned float64 values (S,) and policy probabilities (S, A), on CPU.

    iterations counts value sweeps for evaluation/value iteration, and outer
    evaluation-improvement cycles for policy iteration. evaluation_iterations
    counts all policy-evaluation sweeps (zero for value iteration). delta is
    the last value-sweep max change, not an absolute solution-error bound.

    converged=False preserves the last computed values. Policy iteration
    returns the last evaluated policy, so its values and policy stay paired,
    and stops without improvement if policy evaluation does not converge.
    Frozen fields do not make the returned NumPy arrays immutable.
    """

    values: np.ndarray
    policy: np.ndarray
    iterations: int
    converged: bool
    delta: float
    evaluation_iterations: int = 0


def uniform_policy(mdp: TabularMDP) -> np.ndarray:
    """Return pi(a|s)=1/A with shape (S, A), including terminal placeholders."""
    return np.full((mdp.n_states, mdp.n_actions), 1.0 / mdp.n_actions)


def action_values(mdp: TabularMDP, values: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Full one-step Q backup: sum p * (r + gamma * nonterminal V(s'))."""
    _validate_gamma(gamma)
    values = _initial_values(mdp, values)
    return _action_values(mdp, values, gamma)


def policy_evaluation(
    mdp: TabularMDP, policy: np.ndarray, gamma: float = 1.0, theta: float = 1e-8,
    max_iterations: int = 10_000, *, initial_values: np.ndarray | None = None,
) -> DPResult:
    """Iterate V <- sum_a pi(a|s) Q_V(s,a), using the previous full V each sweep.

    Stop when max(abs(V_new-V_old)) < theta or the sweep limit is reached.
    gamma=1 requires a suitable episodic policy for convergence; there is no
    automatic proper-policy detection. Invalid inputs raise; reaching the
    iteration limit returns converged=False rather than claiming a solution.
    """
    _validate_settings(gamma, theta, max_iterations)
    policy = _validate_policy(mdp, policy)
    values = _initial_values(mdp, initial_values)
    return _evaluate_policy(mdp, policy, values, gamma, theta, max_iterations)


def policy_improvement(mdp: TabularMDP, values: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Return a greedy policy, distributing probability uniformly over exact Q ties."""
    return _greedy_policy(action_values(mdp, values, gamma))


def policy_iteration(
    mdp: TabularMDP, gamma: float = 1.0, theta: float = 1e-8, max_iterations: int = 100,
    *, max_evaluation_iterations: int = 10_000, initial_policy: np.ndarray | None = None,
) -> DPResult:
    """Evaluate then improve until the canonical greedy probability matrix is stable.

    Start with a uniform policy unless supplied, and warm-start each evaluation
    from the previous values. Ties are uniform and no RNG is used. At either
    iteration limit, return the last evaluated policy and converged=False.
    """
    _validate_settings(gamma, theta, max_iterations)
    _validate_iterations(max_evaluation_iterations)
    policy = uniform_policy(mdp) if initial_policy is None else _validate_policy(mdp, initial_policy)
    values = _initial_values(mdp, None)
    evaluation_iterations = 0
    for iteration in range(1, max_iterations + 1):
        evaluation = _evaluate_policy(mdp, policy, values, gamma, theta, max_evaluation_iterations)
        values = evaluation.values
        evaluation_iterations += evaluation.iterations
        if not evaluation.converged:
            return DPResult(values, policy, iteration, False, evaluation.delta, evaluation_iterations)
        improved_policy = _greedy_policy(_action_values(mdp, values, gamma))
        stable = np.array_equal(improved_policy, policy)
        if stable or iteration == max_iterations:
            return DPResult(values, policy, iteration, stable, evaluation.delta, evaluation_iterations)
        policy = improved_policy


def value_iteration(
    mdp: TabularMDP, gamma: float = 1.0, theta: float = 1e-8, max_iterations: int = 10_000,
    *, initial_values: np.ndarray | None = None,
) -> DPResult:
    """Iterate V <- max_a Q_V(s,a), then extract a uniformly tied greedy policy.

    Uses synchronous sweeps and the same delta/limit rule as policy evaluation.
    A nonconverged result includes a greedy policy for its final approximation,
    without claiming that policy is optimal.
    """
    _validate_settings(gamma, theta, max_iterations)
    values = _initial_values(mdp, initial_values)
    for iteration in range(1, max_iterations + 1):
        updated_values = _action_values(mdp, values, gamma).max(axis=-1)
        delta = float(np.max(np.abs(updated_values - values)))
        values = updated_values
        if delta < theta or not isfinite(delta):
            break
    policy = _greedy_policy(_action_values(mdp, values, gamma))
    return DPResult(values, policy, iteration, delta < theta, delta)


def _action_values(mdp: TabularMDP, values: np.ndarray, gamma: float) -> np.ndarray:
    """Validated internal kernel; sum over joint next-state/reward/termination atoms."""
    q = np.zeros((mdp.n_states, mdp.n_actions))
    for state, actions in enumerate(mdp.transitions):
        if state in mdp.terminal_states:
            continue
        for action, outcomes in enumerate(actions):
            for outcome in outcomes:
                if outcome.probability == 0:
                    continue
                ended = outcome.terminated or outcome.next_state in mdp.terminal_states
                bootstrap = 0.0 if ended or gamma == 0 else values[outcome.next_state]
                q[state, action] += outcome.probability * (outcome.reward + gamma * bootstrap)
    if not np.isfinite(q).all():
        raise FloatingPointError("Bellman backup produced non-finite values; check the numerical scale")
    return q


def _evaluate_policy(mdp, policy, values, gamma, theta, max_iterations) -> DPResult:
    """Shared evaluation loop; callers have already validated and copied inputs."""
    for iteration in range(1, max_iterations + 1):
        updated_values = (policy * _action_values(mdp, values, gamma)).sum(axis=-1)
        delta = float(np.max(np.abs(updated_values - values)))
        values = updated_values
        if delta < theta or not isfinite(delta):
            break
    return DPResult(values, policy, iteration, delta < theta, delta, iteration)


def _greedy_policy(q: np.ndarray) -> np.ndarray:
    ties = q == q.max(axis=-1, keepdims=True)
    return ties / ties.sum(axis=-1, keepdims=True)


def _initial_values(mdp: TabularMDP, values: np.ndarray | None) -> np.ndarray:
    if np.iscomplexobj(values):
        raise TypeError("values must be real numbers")
    values = np.zeros(mdp.n_states) if values is None else np.array(values, dtype=np.float64, copy=True)
    if values.shape != (mdp.n_states,) or not np.isfinite(values).all():
        raise ValueError("values must be finite with shape (n_states,)")
    values[list(mdp.terminal_states)] = 0.0
    return values


def _validate_policy(mdp: TabularMDP, policy: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(policy):
        raise TypeError("policy probabilities must be real numbers")
    policy = np.array(policy, dtype=np.float64, copy=True)
    if policy.shape != (mdp.n_states, mdp.n_actions):
        raise ValueError("policy must have shape (n_states, n_actions)")
    if not np.isfinite(policy).all() or np.any(policy < 0):
        raise ValueError("policy probabilities must be finite and nonnegative")
    if not np.allclose(policy.sum(axis=-1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("policy probabilities must sum to 1 for every state")
    return policy


def _validate_gamma(gamma: float) -> None:
    if not isinstance(gamma, Real) or isinstance(gamma, bool) or not isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be a finite real number in [0, 1]")


def _validate_iterations(max_iterations: int) -> None:
    if not isinstance(max_iterations, Integral) or isinstance(max_iterations, bool) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")


def _validate_settings(gamma: float, theta: float, max_iterations: int) -> None:
    _validate_gamma(gamma)
    if not isinstance(theta, Real) or isinstance(theta, bool) or not isfinite(theta) or theta <= 0:
        raise ValueError("theta must be a positive finite real number")
    _validate_iterations(max_iterations)
