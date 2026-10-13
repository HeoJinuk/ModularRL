"""External policy/outcome sampling from a known MDP, without learning."""

from numbers import Integral

import numpy as np

from modular_rl.data import Episode, TabularMDP, Transition


class TabularEpisodeSampler:
    """Collect ordered transitions with an owned policy and independent NumPy RNG.

    policy is a fixed (S,A) float64 probability matrix, copied at setup. rng is
    an owned Generator; its bit_generator.state can be saved externally for
    exact continuation. A supplied terminal start raises before any RNG draws.

    Like DP, termination means outcome.terminated or entry into a declared
    terminal state. No terminal placeholder actions are sampled. Reaching
    max_steps without natural termination returns a truncated-only Episode;
    the caller must report interruption and MC must not learn that fragment.
    """

    def __init__(self, mdp: TabularMDP, policy: np.ndarray, *, seed: int | None = None) -> None:
        self.mdp = mdp
        if np.iscomplexobj(policy):
            raise TypeError("policy probabilities must be real numbers")
        self.policy = np.array(policy, dtype=np.float64, copy=True)
        self._validate_policy()
        self.rng = np.random.default_rng(seed)

    def sample(self, start_state: int, *, max_steps: int = 10_000) -> Episode:
        self._validate(start_state, max_steps)
        state = int(start_state)
        transitions = []
        for step in range(max_steps):
            action = int(self.rng.choice(self.mdp.n_actions, p=self.policy[state]))
            outcomes = self.mdp.transitions[state][action]
            index = int(self.rng.choice(len(outcomes), p=[outcome.probability for outcome in outcomes]))
            outcome = outcomes[index]
            terminated = bool(outcome.terminated or outcome.next_state in self.mdp.terminal_states)
            truncated = step + 1 == max_steps and not terminated
            transitions.append(Transition(state, action, np.float64(outcome.reward), int(outcome.next_state),
                                          terminated, truncated))
            state = int(outcome.next_state)
            if terminated or truncated:
                break
        return Episode(tuple(transitions))

    def _validate_policy(self) -> None:
        if self.policy.shape != (self.mdp.n_states, self.mdp.n_actions):
            raise ValueError("policy must have shape (n_states, n_actions)")
        if not np.isfinite(self.policy).all() or np.any(self.policy < 0):
            raise ValueError("policy probabilities must be finite and nonnegative")
        if not np.allclose(self.policy.sum(axis=-1), 1.0, rtol=0, atol=1e-12):
            raise ValueError("policy probabilities must sum to 1 for every state")

    def _validate(self, start_state: int, max_steps: int) -> None:
        if (not isinstance(start_state, Integral) or isinstance(start_state, bool)
                or not 0 <= start_state < self.mdp.n_states):
            raise ValueError("start_state must be an integer in [0, n_states)")
        if start_state in self.mdp.terminal_states:
            raise ValueError("start_state must be nonterminal; terminal states have no episode transitions")
        if not isinstance(max_steps, Integral) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
