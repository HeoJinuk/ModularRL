"""Concise learners; each name aliases its original implementation."""

from modular_rl.learner import TabularQLearner as QLearning
from modular_rl.learner import TabularSARSALearner as SARSA
from modular_rl.learner import DQNLearner as DQN
from modular_rl.learner import MCPredictionLearner as MCPrediction
from modular_rl.learner import TabularMCControlLearner as MCControl
from modular_rl.learner import OffPolicyMCPredictionLearner as OffPolicyMCPrediction
from modular_rl.learner import OffPolicyMCControlLearner as OffPolicyMCControl
from modular_rl.learner import TabularTD0PredictionLearner as TD0Prediction

__all__ = ['QLearning', 'SARSA', 'DQN', 'MCPrediction', 'MCControl', 'OffPolicyMCPrediction', 'OffPolicyMCControl', 'TD0Prediction']
