"""TD targets, episode returns and boundary-aware rollout returns."""

from modular_rl.component.target_generator.base import BaseTargetGenerator
from modular_rl.component.target_generator.td0 import TD0TargetGenerator
from modular_rl.component.target_generator.monte_carlo import MonteCarloReturnGenerator

from modular_rl.component.target_generator.n_step import NStepTargetGenerator
from modular_rl.component.target_generator.rollout import RolloutReturnGenerator

__all__ = ["BaseTargetGenerator", "TD0TargetGenerator", "MonteCarloReturnGenerator", "NStepTargetGenerator", "RolloutReturnGenerator"]
