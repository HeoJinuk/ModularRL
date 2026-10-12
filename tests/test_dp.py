"""Full Bellman backups, synchronous iteration and explicit convergence limits."""

import numpy as np
import pytest
import torch

from modular_rl.component.estimator import TabularVEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.data import TabularMDP, TransitionOutcome as Outcome
from modular_rl.dp import (
    DPResult, action_values, policy_evaluation, policy_improvement,
    policy_iteration, uniform_policy, value_iteration,
)
from modular_rl.model import Model


def stochastic_mdp():
    return TabularMDP(2, 2, (
        ((Outcome(.25, 1, 2, True), Outcome(.75, 1, 4, False)),
         (Outcome(.5, 0, -2, False), Outcome(.5, 1, 8, True))),
        ((Outcome(1, 1, 1, True),), (Outcome(1, 1, 3, True),)),
    ))


def chain_mdp():
    return TabularMDP(2, 1, (((Outcome(1, 0, 2, True),),), ((Outcome(1, 0, 0, False),),)))


@pytest.mark.parametrize("gamma,expected", [
    (0, [[3.5, 3], [1, 3]]), (.5, [[11, 5.5], [1, 3]]), (1, [[18.5, 8], [1, 3]]),
])
def test_action_values_preserve_joint_rewards_termination_and_probability(gamma, expected):
    q = action_values(stochastic_mdp(), np.array([10, 20]), gamma)
    np.testing.assert_allclose(q, expected, rtol=0, atol=1e-14)
    assert q.shape == (2, 2) and q.dtype == np.float64


@pytest.mark.parametrize("gamma,expected", [(0, [3.1, 1.5]), (.5, [6.6, 1.5]), (1, [10.1, 1.5])])
def test_bellman_expectation_backup_with_a_nonuniform_policy(gamma, expected):
    policy = np.array([[.2, .8], [.75, .25]])
    result = policy_evaluation(stochastic_mdp(), policy, gamma, max_iterations=1,
                               initial_values=np.array([10., 20.]))
    np.testing.assert_allclose(result.values, expected, rtol=0, atol=1e-14)
    assert result.iterations == result.evaluation_iterations == 1 and not result.converged


@pytest.mark.parametrize("gamma,expected", [(0, [3.5, 3]), (.5, [11, 3]), (1, [18.5, 3])])
def test_bellman_optimality_backup(gamma, expected):
    result = value_iteration(stochastic_mdp(), gamma, max_iterations=1,
                             initial_values=np.array([10., 20.]))
    np.testing.assert_allclose(result.values, expected)
    assert result.iterations == 1 and not result.converged and result.evaluation_iterations == 0


def test_stochastic_policy_evaluation_matches_an_independent_analytic_solution():
    policy = np.array([[.2, .8], [.75, .25]])
    result = policy_evaluation(stochastic_mdp(), policy, gamma=.5, theta=1e-12)
    # V1=1.5; V0=.2*(.25*2+.75*(4+.5*V1))+.8*(.5*(-2+.5*V0)+.5*8).
    np.testing.assert_allclose(result.values, [4.015625, 1.5], rtol=0, atol=1e-12)
    assert result.converged and result.delta < 1e-12
    np.testing.assert_array_equal(result.policy, policy)


def test_discounted_stochastic_evaluation_matches_linear_algebra():
    rng = np.random.default_rng(13)
    probabilities = rng.dirichlet(np.ones(3), size=(3, 2))
    rewards = rng.normal(size=(3, 2, 3))
    policy = rng.dirichlet(np.ones(2), size=3)
    transitions = tuple(tuple(tuple(Outcome(probabilities[s, a, t], t, rewards[s, a, t])
                                    for t in range(3)) for a in range(2)) for s in range(3))
    mdp = TabularMDP(3, 2, transitions)
    policy_transition = np.einsum("sa,sat->st", policy, probabilities)
    policy_reward = np.einsum("sa,sat,sat->s", policy, probabilities, rewards)
    expected = np.linalg.solve(np.eye(3) - .8 * policy_transition, policy_reward)
    result = policy_evaluation(mdp, policy, gamma=.8, theta=1e-12)
    np.testing.assert_allclose(result.values, expected, rtol=0, atol=5e-12)
    assert result.converged


@pytest.mark.parametrize("algorithm", [policy_evaluation, value_iteration])
def test_updates_are_synchronous_and_do_not_read_earlier_writes_in_the_sweep(algorithm):
    mdp = chain_mdp()
    arguments = {"policy": uniform_policy(mdp)} if algorithm is policy_evaluation else {}
    first = algorithm(mdp, gamma=.5, max_iterations=1, **arguments)
    second = algorithm(mdp, gamma=.5, max_iterations=2, **arguments)
    np.testing.assert_array_equal(first.values, [2, 0])  # In-place state-order update would yield [2, 1].
    np.testing.assert_array_equal(second.values, [2, 1])
    assert not first.converged and not second.converged


def test_declared_terminal_states_have_zero_values_and_block_entry_bootstrap():
    mdp = TabularMDP(2, 1, (((Outcome(1, 1, 5),),), ((Outcome(1, 1, 99),),)), (1,))
    initial = np.array([3., 1e100])
    np.testing.assert_array_equal(action_values(mdp, initial, 1), [[5], [0]])
    result = policy_evaluation(mdp, uniform_policy(mdp), gamma=1, initial_values=initial)
    np.testing.assert_array_equal(result.values, [5, 0])
    np.testing.assert_array_equal(initial, [3, 1e100])
    assert result.converged


def test_terminated_outcomes_do_not_make_the_source_or_next_state_globally_terminal():
    mdp = stochastic_mdp()
    assert mdp.terminal_states == ()
    result = policy_evaluation(mdp, np.array([[1., 0], [0, 1.]]), gamma=1)
    np.testing.assert_array_equal(result.values, [5.75, 3])


def test_policy_improvement_selects_expected_action_values_not_the_largest_individual_reward():
    mdp = TabularMDP(1, 2, (((Outcome(.1, 0, 100, True), Outcome(.9, 0, -100, True)),
                            (Outcome(1, 0, 2, True),)),))
    policy = policy_improvement(mdp, np.zeros(1), gamma=1)
    np.testing.assert_array_equal(policy, [[0, 1]])


def test_greedy_ties_are_uniform_and_policy_iteration_is_stable_without_sampling():
    mdp = TabularMDP(1, 3, (((Outcome(1, 0, 2, True),), (Outcome(1, 0, 0, True),),
                            (Outcome(1, 0, 2, True),)),))
    rng = np.random.get_state()
    policy = policy_improvement(mdp, np.array([999.]))
    np.testing.assert_array_equal(policy, [[.5, 0, .5]])
    result = policy_iteration(mdp, initial_policy=np.array([[.9, 0, .1]]))
    assert result.converged and result.iterations == 2
    np.testing.assert_array_equal(result.policy, policy)
    np.testing.assert_array_equal(result.values, [2])
    repeated = policy_iteration(mdp, initial_policy=result.policy)
    assert repeated.converged and repeated.iterations == 1
    np.testing.assert_array_equal(np.random.get_state()[1], rng[1])
    assert np.random.get_state()[2:] == rng[2:]


@pytest.mark.parametrize("gamma,expected_values,expected_policy", [
    (0, [3.5, 3], [[1, 0], [0, 1]]),
    (.5, [4.625, 3], [[1, 0], [0, 1]]),
    (1, [6, 3], [[0, 1], [0, 1]]),
])
def test_policy_iteration_and_value_iteration_solve_the_same_stochastic_mdp(gamma, expected_values, expected_policy):
    mdp = stochastic_mdp()
    for algorithm in (policy_iteration, value_iteration):
        result = algorithm(mdp, gamma=gamma, theta=1e-12)
        np.testing.assert_allclose(result.values, expected_values, rtol=0, atol=2e-12)
        np.testing.assert_array_equal(result.policy, expected_policy)
        assert result.converged
        evaluated = policy_evaluation(mdp, result.policy, gamma=gamma, theta=1e-12)
        np.testing.assert_allclose(evaluated.values, result.values, rtol=0, atol=2e-12)


@pytest.mark.parametrize("algorithm", [policy_evaluation, value_iteration])
def test_improper_gamma_one_policy_reports_nonconvergence_at_the_sweep_limit(algorithm):
    mdp = TabularMDP(1, 1, (((Outcome(1, 0, 1),),),))
    arguments = {"policy": uniform_policy(mdp)} if algorithm is policy_evaluation else {}
    result = algorithm(mdp, gamma=1, max_iterations=7, **arguments)
    assert not result.converged and result.iterations == 7 and result.delta == 1
    np.testing.assert_array_equal(result.values, [7])


def test_policy_iteration_stops_when_evaluation_fails_without_claiming_policy_stability():
    mdp = TabularMDP(1, 2, (((Outcome(1, 0, -1),), (Outcome(1, 0, 0, True),)),))
    initial = np.array([[1., 0]])  # Improper policy; a terminating alternative exists.
    result = policy_iteration(mdp, gamma=1, max_evaluation_iterations=5, initial_policy=initial)
    assert not result.converged and result.iterations == 1 and result.evaluation_iterations == 5
    np.testing.assert_array_equal(result.values, [-5])
    np.testing.assert_array_equal(result.policy, initial)


def test_outer_iteration_limit_keeps_values_paired_with_the_last_evaluated_policy():
    mdp = TabularMDP(1, 2, (((Outcome(1, 0, 0, True),), (Outcome(1, 0, 2, True),)),))
    result = policy_iteration(mdp, max_iterations=1)
    assert not result.converged and result.iterations == 1
    np.testing.assert_array_equal(result.policy, [[.5, .5]])
    np.testing.assert_array_equal(result.values, [1])
    np.testing.assert_array_equal(policy_evaluation(mdp, result.policy).values, result.values)


def test_initial_fixed_point_converges_in_one_sweep_and_result_arrays_are_owned():
    mdp = chain_mdp()
    policy = uniform_policy(mdp).astype(np.float32)
    values = np.array([2., 1.], dtype=np.float32)
    result = policy_evaluation(mdp, policy, gamma=.5, initial_values=values)
    assert isinstance(result, DPResult) and result.converged and result.iterations == 1
    assert result.values.dtype == result.policy.dtype == np.float64
    result.values[:] = 99
    result.policy[:] = 0
    np.testing.assert_array_equal(values, [2, 1])
    np.testing.assert_array_equal(policy, [[1], [1]])


def test_gamma_zero_and_zero_probability_outcomes_do_not_evaluate_unused_bootstraps():
    mdp = TabularMDP(1, 1, (((Outcome(1, 0, 1, True), Outcome(0, 0, 1e308)),),))
    np.testing.assert_array_equal(action_values(mdp, np.array([1e308]), gamma=1), [[1]])
    continuing = TabularMDP(1, 1, (((Outcome(1, 0, 1),),),))
    np.testing.assert_array_equal(action_values(continuing, np.array([1e308]), gamma=0), [[1]])


def test_numerical_overflow_fails_instead_of_returning_a_nan_greedy_policy():
    mdp = TabularMDP(1, 1, (((Outcome(1, 0, 1e308),),),))
    with np.errstate(over="ignore"), pytest.raises(FloatingPointError, match="non-finite"):
        value_iteration(mdp, gamma=1)


def test_dp_values_can_be_explicitly_loaded_into_the_existing_tabular_estimator():
    result = policy_evaluation(chain_mdp(), np.ones((2, 1)), gamma=.5)
    model = Model(IdentityRepresentation(), TabularVEstimator(2)).double()
    model.estimator.v_table.copy_(torch.from_numpy(result.values))
    torch.testing.assert_close(model.forward(np.array([0, 1])).values.v, torch.tensor([2., 1.], dtype=torch.float64))


@pytest.mark.parametrize("gamma", [-.1, 1.1, np.nan, np.inf, True, "1"])
def test_invalid_gamma_is_rejected(gamma):
    with pytest.raises(ValueError, match="gamma"):
        policy_evaluation(chain_mdp(), np.ones((2, 1)), gamma=gamma)
    with pytest.raises(ValueError, match="gamma"):
        action_values(chain_mdp(), np.zeros(2), gamma=gamma)


@pytest.mark.parametrize("theta", [0, -.1, np.nan, np.inf, True, "small"])
def test_invalid_tolerance_is_rejected(theta):
    with pytest.raises(ValueError, match="theta"):
        value_iteration(chain_mdp(), theta=theta)


@pytest.mark.parametrize("iterations", [0, -1, True, 2.5])
@pytest.mark.parametrize("algorithm", [policy_evaluation, policy_iteration, value_iteration])
def test_iteration_limits_must_be_positive_integers(algorithm, iterations):
    arguments = {"policy": np.ones((2, 1))} if algorithm is policy_evaluation else {}
    with pytest.raises(ValueError, match="max_iterations"):
        algorithm(chain_mdp(), max_iterations=iterations, **arguments)


def test_invalid_policy_evaluation_limit_in_policy_iteration_is_rejected():
    with pytest.raises(ValueError, match="max_iterations"):
        policy_iteration(chain_mdp(), max_evaluation_iterations=0)


@pytest.mark.parametrize("policy,message", [
    (np.array([1, 1]), "shape"), (np.ones((2, 2)), "shape"),
    (np.array([[0.], [1.]]), "sum to 1"), (np.array([[-1.], [1.]]), "nonnegative"),
    (np.array([[np.nan], [1.]]), "finite"), (np.array([[np.inf], [1.]]), "finite"),
])
def test_invalid_policy_matrices_fail(policy, message):
    with pytest.raises(ValueError, match=message):
        policy_evaluation(chain_mdp(), policy)
    with pytest.raises(ValueError, match=message):
        policy_iteration(chain_mdp(), initial_policy=policy)


@pytest.mark.parametrize("values", [np.zeros((2, 1)), np.zeros(3), np.array([np.nan, 0]), np.array([np.inf, 0])])
def test_invalid_value_arrays_fail(values):
    with pytest.raises(ValueError, match="values.*finite.*shape"):
        action_values(chain_mdp(), values)


def test_complex_probabilities_and_values_are_not_silently_cast_to_real():
    with pytest.raises(TypeError, match="policy.*real"):
        policy_evaluation(chain_mdp(), np.ones((2, 1), dtype=complex))
    with pytest.raises(TypeError, match="values.*real"):
        value_iteration(chain_mdp(), initial_values=np.ones(2, dtype=complex))
