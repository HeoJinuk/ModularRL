"""Learners compose inference, targets and updates with explicit contracts."""

from modular_rl.learner.dqn import DQNLearner
from modular_rl.learner.tabular_q import TabularQLearner
from modular_rl.learner.mc_prediction import MCPredictionLearner
from modular_rl.learner.mc_control import TabularMCControlLearner
from modular_rl.learner.off_policy_mc_prediction import OffPolicyMCPredictionLearner
from modular_rl.learner.off_policy_mc_control import OffPolicyMCControlLearner
from modular_rl.learner.tabular_td0 import TabularTD0PredictionLearner

__all__ = ["TabularQLearner", "DQNLearner", "MCPredictionLearner", "TabularTD0PredictionLearner",
           "TabularMCControlLearner", "OffPolicyMCPredictionLearner", "OffPolicyMCControlLearner"]
