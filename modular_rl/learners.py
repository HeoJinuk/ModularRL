"""Concise learners; each name aliases its original implementation."""

from modular_rl.learner import TabularQLearner as QLearning
from modular_rl.learner import TabularSARSALearner as SARSA
from modular_rl.learner import DQNLearner as DQN
from modular_rl.learner import DoubleDQNLearner as DoubleDQN
from modular_rl.learner import ActorCriticLearner as ActorCritic
from modular_rl.learner import MCPredictionLearner as MCPrediction
from modular_rl.learner import TabularMCControlLearner as MCControl
from modular_rl.learner import OffPolicyMCPredictionLearner as OffPolicyMCPrediction
from modular_rl.learner import OffPolicyMCControlLearner as OffPolicyMCControl
from modular_rl.learner import TabularTD0PredictionLearner as TD0Prediction

from modular_rl.learner import ExpectedSARSALearner as ExpectedSARSA
from modular_rl.learner import DoubleQLearner as DoubleQ
from modular_rl.learner import NStepSARSALearner as NStepSARSA
from modular_rl.learner import OffPolicyNStepSARSALearner as OffPolicyNStepSARSA
from modular_rl.learner import TreeBackupLearner as TreeBackup
from modular_rl.learner import QSigmaLearner as QSigma

__all__ = ['QLearning', 'SARSA', 'DQN', 'DoubleDQN', 'ActorCritic', 'MCPrediction', 'MCControl', 'OffPolicyMCPrediction', 'OffPolicyMCControl', 'TD0Prediction', 'ExpectedSARSA', 'DoubleQ', 'NStepSARSA', 'OffPolicyNStepSARSA', 'TreeBackup', 'QSigma']
