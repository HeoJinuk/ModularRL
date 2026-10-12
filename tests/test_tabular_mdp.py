"""Complete known kernels retain joint outcomes and validate probability rows."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from modular_rl.data import TabularMDP, TransitionOutcome


def test_joint_outcomes_keep_different_rewards_and_termination_for_the_same_next_state():
    row = (TransitionOutcome(.25, 0, 1.0, True), TransitionOutcome(.75, 0, 3.0, False))
    mdp = TabularMDP(1, 1, ((row,),))
    assert mdp.n_states == mdp.n_actions == 1
    assert mdp.transitions[0][0] == row
    assert len(mdp.transitions[0][0]) == 2 and mdp.terminal_states == ()


def test_kernel_snapshots_input_lists_and_outcomes_are_frozen():
    outcome = TransitionOutcome(1.0, 0, 2.0)
    rows, terminals = [[[outcome]]], [0]
    mdp = TabularMDP(1, 1, rows, terminals)
    rows[0][0].clear()
    rows.clear()
    terminals.clear()
    assert mdp.transitions == (((outcome,),),) and mdp.terminal_states == (0,)
    with pytest.raises(FrozenInstanceError):
        outcome.reward = 99
    with pytest.raises(FrozenInstanceError):
        mdp.n_states = 99


def test_numpy_numeric_scalars_and_joint_probability_zero_are_supported():
    row = (TransitionOutcome(np.float64(1), np.int32(0), np.float32(-1), np.bool_(False)),
           TransitionOutcome(0.0, 0, 100.0, True))
    mdp = TabularMDP(np.int64(1), np.int32(1), ((row,),))
    assert len(mdp.transitions[0][0]) == 2


@pytest.mark.parametrize("name,value", [
    ("n_states", 0), ("n_states", -1), ("n_states", True), ("n_states", 1.0),
    ("n_actions", 0), ("n_actions", False), ("n_actions", 1.5),
])
def test_invalid_dimensions_fail(name, value):
    fields = {"n_states": 1, "n_actions": 1, "transitions": (((TransitionOutcome(1, 0, 0),),),)}
    fields[name] = value
    with pytest.raises(ValueError, match="positive integers"):
        TabularMDP(**fields)


@pytest.mark.parametrize("probability", [-.1, 1.1, np.nan, np.inf, True, "1", 1j])
def test_invalid_outcome_probability_fails(probability):
    with pytest.raises(ValueError, match="probability.*finite"):
        TabularMDP(1, 1, (((TransitionOutcome(probability, 0, 0),),),))


@pytest.mark.parametrize("reward", [np.nan, np.inf, -np.inf, True, "1", 1j])
def test_rewards_must_be_finite_real_numbers(reward):
    with pytest.raises(ValueError, match="reward.*finite real"):
        TabularMDP(1, 1, (((TransitionOutcome(1, 0, reward),),),))


@pytest.mark.parametrize("state", [-1, 1, True, .5])
def test_state_indices_are_validated_for_both_outcomes_and_terminal_states(state):
    with pytest.raises(ValueError, match="state indices"):
        TabularMDP(1, 1, (((TransitionOutcome(1, state, 0),),),))
    with pytest.raises(ValueError, match="state indices"):
        TabularMDP(1, 1, (((TransitionOutcome(1, 0, 0),),),), (state,))


@pytest.mark.parametrize("terminated", [0, 1, "False", None])
def test_termination_flag_must_be_bool(terminated):
    with pytest.raises(TypeError, match="terminated must be bool"):
        TabularMDP(1, 1, (((TransitionOutcome(1, 0, 0, terminated),),),))


@pytest.mark.parametrize("probabilities", [(.9,), (.4, .4), (.6, .6)])
def test_every_state_action_probability_row_must_sum_to_one(probabilities):
    with pytest.raises(ValueError, match="sum to 1"):
        TabularMDP(1, 1, ((tuple(TransitionOutcome(p, 0, 0) for p in probabilities),),))


def test_kernel_shape_empty_rows_and_wrong_outcome_objects_fail():
    row = (TransitionOutcome(1, 0, 0),)
    with pytest.raises(ValueError, match="row per state"):
        TabularMDP(2, 1, ((row,),))
    with pytest.raises(ValueError, match="row per action"):
        TabularMDP(1, 2, ((row,),))
    with pytest.raises(ValueError, match="at least one outcome"):
        TabularMDP(1, 1, (((),),))
    with pytest.raises(TypeError, match="TransitionOutcome"):
        TabularMDP(1, 1, ((({"probability": 1},),),))


def test_probability_validation_checks_later_rows_too():
    valid = (TransitionOutcome(1, 0, 0),)
    invalid = (TransitionOutcome(.8, 0, 0),)
    with pytest.raises(ValueError, match="sum to 1"):
        TabularMDP(2, 2, ((valid, valid), (valid, invalid)))
