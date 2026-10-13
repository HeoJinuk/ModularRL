"""Learners compose inference, targets and updates with explicit contracts."""

from modular_rl.learner.dqn import DQNLearner
from modular_rl.learner.tabular_q import TabularQLearner
from modular_rl.learner.mc_prediction import MCPredictionLearner

__all__ = ["TabularQLearner", "DQNLearner", "MCPredictionLearner"]
