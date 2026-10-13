"""Gymnasium execution of a known finite MDP, independent of policy and learning."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import ResetNeeded

from modular_rl.data import TabularMDP


class TabularEnv(gym.Env[int, int]):
    """Sample transitions from mdp; expose integer state/action IDs.

    reset uses a configured start_state, or samples uniformly from nonterminal
    states when it is None. options['start_state'] overrides that choice for
    one reset. super().reset(seed=seed) owns the environment RNG; reset without
    a seed continues its stream. Action/policy RNGs are caller-owned.

    Natural termination is outcome.terminated or entry into mdp.terminal_states.
    step before reset or after natural termination raises ResetNeeded. The base
    environment never truncates or auto-resets: apply Gymnasium TimeLimit for a
    step cap and reset after either boundary flag. TimeLimit is an external
    wrapper and its truncation is not visible inside this base environment.

    ansi render returns a state-ID string; human prints on reset/step/render.
    Grid coordinates and value/policy plots belong to separate display code.
    close is idempotent; reset can start a new episode after close.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(self, mdp: TabularMDP, *, start_state: int | None = None,
                 render_mode: str | None = None) -> None:
        if not isinstance(mdp, TabularMDP):
            raise TypeError("TabularEnv requires a TabularMDP")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError("render_mode must be None, 'ansi', or 'human'")
        self.mdp = mdp
        self.observation_space = spaces.Discrete(mdp.n_states)
        self.action_space = spaces.Discrete(mdp.n_actions)
        self._nonterminal = np.array([s for s in range(mdp.n_states) if s not in mdp.terminal_states])
        if self._nonterminal.size == 0:
            raise ValueError("TabularEnv requires at least one nonterminal start state")
        if start_state is not None:
            self._validate_start(start_state)
        self.start_state = None if start_state is None else int(start_state)
        self.render_mode = render_mode
        self._state = None
        self._terminated = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and "start_state" in options:
            self._validate_start(options["start_state"])
            state = int(options["start_state"])
        elif self.start_state is not None:
            state = self.start_state
        else:
            state = int(self.np_random.choice(self._nonterminal))
        self._state, self._terminated = state, False
        if self.render_mode == "human":
            self.render()
        return state, {}

    def step(self, action):
        if self._state is None or self._terminated:
            raise ResetNeeded("Call reset before step and after episode termination")
        self._validate_action(action)
        outcomes = self.mdp.transitions[self._state][int(action)]
        index = int(self.np_random.choice(len(outcomes), p=[outcome.probability for outcome in outcomes]))
        outcome = outcomes[index]
        self._state = int(outcome.next_state)
        self._terminated = bool(outcome.terminated or self._state in self.mdp.terminal_states)
        if self.render_mode == "human":
            self.render()
        return self._state, float(outcome.reward), self._terminated, False, {}

    def render(self):
        if self.render_mode is None:
            return None
        if self._state is None:
            raise ResetNeeded("Call reset before rendering an episode")
        text = f"state={self._state} terminated={self._terminated}\n"
        if self.render_mode == "ansi":
            return text
        print(text, end="")
        return None

    def close(self):
        self._state, self._terminated = None, False

    def _validate_start(self, state):
        if isinstance(state, (bool, np.bool_)) or not self.observation_space.contains(state):
            raise ValueError("start_state must be an integer in the observation space")
        if int(state) in self.mdp.terminal_states:
            raise ValueError("start_state must be nonterminal")

    def _validate_action(self, action):
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError("action must be an integer in the action space")
