"""DP-exact IS arithmetic and reproducible Gymnasium interaction smoke tests."""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.gridworld_off_policy_mc import train_and_compare
from modular_rl.component.estimator import TabularVEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import MonteCarloReturnGenerator
from modular_rl.data import Episode, TabularMDP, Transition, TransitionOutcome as Outcome
from modular_rl.dp import policy_evaluation
from modular_rl.learner import OffPolicyMCPredictionLearner
from modular_rl.model import Model


@pytest.mark.parametrize("method", ["ordinary", "weighted"])
def test_exact_behavior_frequency_dataset_recovers_target_dp_expectation(method):
    mdp = TabularMDP(1, 2, (((Outcome(1., 0, 4., True),), (Outcome(1., 0, 0., True),)),))
    target = np.array([[.5, .5]])
    reference = policy_evaluation(mdp, target, gamma=1)
    learner = OffPolicyMCPredictionLearner(Model(IdentityRepresentation(), TabularVEstimator(1)).double(),
                                         MonteCarloReturnGenerator(1), target, method=method)
    for action in (0, 1, 1, 1):  # Exactly b=(.25,.75), pi=(.5,.5).
        probability, reward = (.25, 4.) if action == 0 else (.75, 0.)
        learner.learn(Episode((Transition(0, action, reward, 0, True, False,
                                          torch.tensor(math.log(probability), dtype=torch.float64)),)))
    torch.testing.assert_close(learner.model.estimator.v_table, torch.tensor(reference.values))
    assert reference.values[0] == 2 and learner.visit_counts[0] == 4


def test_short_gridworld_experiment_uses_fixed_target_dp_and_shared_behavior_episodes():
    learners, result = train_and_compare(episodes=20, log_every=0)
    assert result["episodes"] == 20 and result["steps"] > 20 and result["truncated_episodes"] == 0
    assert not np.array_equal(result["behavior_policy"], result["target_policy"])
    assert learners["ordinary"].model is not learners["weighted"].model
    for name, metrics in result["methods"].items():
        assert metrics["values"].shape == metrics["visit_counts"].shape == (11,)
        assert metrics["visit_counts"].max() <= 20 and metrics["visit_counts"][3] == 0
        assert metrics["values"][3] == 0
        mask = np.arange(11) != 3
        assert metrics["rmse"] == np.sqrt(np.mean((metrics["values"][mask] - result["reference"][mask]) ** 2))
        assert np.isfinite(metrics["values"]).all() and np.isfinite(metrics["weight_sums"]).all()
        np.testing.assert_array_equal(metrics["values"], learners[name].model.estimator.v_table.numpy())
    for stat in ("visit_counts", "weight_sums", "weighted_return_sums"):
        np.testing.assert_array_equal(result["methods"]["ordinary"][stat], result["methods"]["weighted"][stat])


def test_same_environment_and_behavior_seed_reproduce_all_statistics_and_rmse():
    _, first = train_and_compare(episodes=20, seed=11, log_every=10)
    _, second = train_and_compare(episodes=20, seed=11, log_every=10)
    assert first["steps"] == second["steps"] and first["history"] == second["history"]
    for method in ("ordinary", "weighted"):
        for stat in ("values", "visit_counts", "weight_sums", "weighted_return_sums"):
            np.testing.assert_array_equal(first["methods"][method][stat], second["methods"][method][stat])


def test_uniform_target_matches_behavior_giving_unit_weights_and_equal_methods():
    _, result = train_and_compare(episodes=20, target_probs=(.25,) * 4, log_every=0)
    np.testing.assert_array_equal(result["behavior_policy"], result["target_policy"])
    for method in ("ordinary", "weighted"):
        metrics = result["methods"][method]
        np.testing.assert_array_equal(metrics["weight_sums"], metrics["visit_counts"])
    np.testing.assert_array_equal(result["methods"]["ordinary"]["values"], result["methods"]["weighted"]["values"])


def test_real_truncation_aborts_before_any_update(monkeypatch):
    monkeypatch.setattr(OffPolicyMCPredictionLearner, "learn", lambda *args: pytest.fail("no truncated MC update"))
    with pytest.raises(RuntimeError, match="truncated.*no off-policy learner"):
        train_and_compare(episodes=1, max_steps=1)


def test_example_cli_reports_policies_seed_rmse_and_unit_weights():
    result = subprocess.run([sys.executable, "-m", "examples.gridworld_off_policy_mc", "--episodes", "5",
                             "--seeds", "7", "--target-uniform"], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, check=True, timeout=30)
    for phrase in ("Behavior b=", "target pi=", "Target DP V:", "ordinary V (RMSE=", "weighted V (RMSE=",
                   "unit-weight max |sum rho - count|: 0.0", "RMSE across seeds=[7]"):
        assert phrase in result.stdout
