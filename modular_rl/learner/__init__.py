"""Learners compose inference, targets and updates with explicit contracts."""

from modular_rl.learner.dqn import DQNLearner
from modular_rl.learner.tabular_q import TabularQLearner
from modular_rl.learner.tabular_sarsa import TabularSARSALearner
from modular_rl.learner.mc_prediction import MCPredictionLearner
from modular_rl.learner.mc_control import TabularMCControlLearner
from modular_rl.learner.off_policy_mc_prediction import OffPolicyMCPredictionLearner
from modular_rl.learner.off_policy_mc_control import OffPolicyMCControlLearner
from modular_rl.learner.tabular_td0 import TabularTD0PredictionLearner
from modular_rl.learner.expected_sarsa import ExpectedSARSALearner
from modular_rl.learner.double_q import DoubleQLearner
from modular_rl.learner.n_step_sarsa import NStepSARSALearner
from modular_rl.learner.off_policy_n_step_sarsa import OffPolicyNStepSARSALearner
from modular_rl.learner.tree_backup import TreeBackupLearner
from modular_rl.learner.q_sigma import QSigmaLearner

__all__ = [
    'DQNLearner',
    'TabularQLearner',
    'TabularSARSALearner',
    'MCPredictionLearner',
    'TabularMCControlLearner',
    'OffPolicyMCPredictionLearner',
    'OffPolicyMCControlLearner',
    'TabularTD0PredictionLearner',
    'ExpectedSARSALearner',
    'DoubleQLearner',
    'NStepSARSALearner',
    'OffPolicyNStepSARSALearner',
    'TreeBackupLearner',
    'QSigmaLearner',
]
