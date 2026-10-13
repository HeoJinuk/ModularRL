"""Policy and joint-outcome sampling, truncation and private RNG ownership."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from modular_rl.component.estimator import TabularVEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import MonteCarloReturnGenerator
from modular_rl.data import TabularMDP, TransitionOutcome as Outcome
from modular_rl.dp import policy_evaluation, uniform_policy
from modular_rl.environment import TabularEpisodeSampler, sutton_gridworld
from modular_rl.learner import MCPredictionLearner
from modular_rl.model import Model


def episode_trace(episode):
    return tuple((t.observation.item(), t.action.item(), t.reward.item(), t.next_observation.item(),
                  t.terminated.item(), t.truncated.item()) for t in episode.transitions)


def test_policy_and_stochastic_outcome_frequencies_match_the_joint_kernel():
    mdp = TabularMDP(3, 2, (
        ((Outcome(.25, 1, 1, True), Outcome(.75, 2, 5, True)),
         (Outcome(.6, 1, -3, True), Outcome(.4, 2, 7, True))),
        ((Outcome(1, 1, 0, True),), (Outcome(1, 1, 0, True),)),
        ((Outcome(1, 2, 0, True),), (Outcome(1, 2, 0, True),)),
    ), (1, 2))
    sampler = TabularEpisodeSampler(mdp, np.array([[.3, .7], [.5, .5], [.5, .5]]), seed=19)
    counts = np.zeros((2, 2), dtype=int)
    rewards = {(0, 1): 1, (0, 2): 5, (1, 1): -3, (1, 2): 7}
    for _ in range(3000):
        episode = sampler.sample(0, max_steps=1)
        assert len(episode) == 1 and episode.terminated and not episode.truncated
        transition, = episode.transitions
        action, next_state = transition.action.item(), transition.next_observation.item()
        counts[action, next_state - 1] += 1
        assert transition.reward.item() == rewards[action, next_state]
    assert counts[0].sum() / counts.sum() == pytest.approx(.3, abs=.04)
    assert counts[0, 0] / counts[0].sum() == pytest.approx(.25, abs=.05)
    assert counts[1, 0] / counts[1].sum() == pytest.approx(.6, abs=.04)


def test_seed_reproduces_whole_episodes_without_using_global_numpy_or_torch_rng():
    mdp = sutton_gridworld()
    policy = uniform_policy(mdp)
    numpy_before, torch_before = np.random.get_state(), torch.get_rng_state().clone()
    first = TabularEpisodeSampler(mdp, policy, seed=7)
    second = TabularEpisodeSampler(mdp, policy, seed=7)
    for start in (1, 6, 12, 14):
        assert episode_trace(first.sample(start)) == episode_trace(second.sample(start))
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
    assert np.random.get_state()[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_policy_is_copied_and_sampler_rng_can_be_restored_for_exact_continuation():
    mdp = sutton_gridworld()
    policy = uniform_policy(mdp)
    sampler = TabularEpisodeSampler(mdp, policy, seed=7)
    policy.fill(0)
    sampler.sample(6)
    saved = deepcopy(sampler.rng.bit_generator.state)
    expected = episode_trace(sampler.sample(12))
    restored = TabularEpisodeSampler(mdp, uniform_policy(mdp), seed=999)
    restored.rng.bit_generator.state = saved
    assert episode_trace(restored.sample(12)) == expected


def test_terminal_starts_make_no_transition_and_consume_no_rng():
    mdp = sutton_gridworld()
    sampler = TabularEpisodeSampler(mdp, uniform_policy(mdp), seed=7)
    before = deepcopy(sampler.rng.bit_generator.state)
    for terminal in mdp.terminal_states:
        with pytest.raises(ValueError, match="nonterminal"):
            sampler.sample(terminal)
    assert sampler.rng.bit_generator.state == before


def test_episode_maximum_length_is_a_truncation_and_mc_rejects_it():
    mdp = TabularMDP(1, 1, (((Outcome(1, 0, 1),),),))
    sampler = TabularEpisodeSampler(mdp, uniform_policy(mdp), seed=7)
    episode = sampler.sample(0, max_steps=5)
    assert len(episode) == 5 and episode.truncated and not episode.terminated
    assert not any(t.truncated.item() for t in episode.transitions[:-1])
    model = Model(IdentityRepresentation(), TabularVEstimator(1))
    learner = MCPredictionLearner(model, MonteCarloReturnGenerator(1))
    with pytest.raises(ValueError, match="truncated-only"):
        learner.learn(episode)
    assert learner.visit_counts[0].item() == model.estimator.v_table[0].item() == 0


@pytest.mark.parametrize("mode", ["first", "every"])
@pytest.mark.parametrize("gamma", [0., .5, 1.])
def test_complete_deterministic_episode_mc_matches_dp_exactly(mode, gamma):
    mdp = TabularMDP(3, 1, (((Outcome(1, 1, 2),),), ((Outcome(1, 2, 3, True),),),
                           ((Outcome(1, 2, 0, True),),)), (2,))
    policy = uniform_policy(mdp)
    reference = policy_evaluation(mdp, policy, gamma=gamma)
    episode = TabularEpisodeSampler(mdp, policy, seed=7).sample(0, max_steps=2)
    assert episode.terminated and not episode.truncated and len(episode) == 2
    learner = MCPredictionLearner(Model(IdentityRepresentation(), TabularVEstimator(3)).double(),
                                  MonteCarloReturnGenerator(gamma), mode)
    learner.learn(episode)
    np.testing.assert_array_equal(learner.model.estimator.v_table.numpy(), reference.values)
    torch.testing.assert_close(learner.visit_counts, torch.tensor([1, 1, 0]))


def test_declared_terminal_entry_stops_sampling_just_as_in_dp():
    mdp = TabularMDP(2, 1, (((Outcome(1, 1, 5, False),),), ((Outcome(1, 1, 99),),)), (1,))
    episode = TabularEpisodeSampler(mdp, uniform_policy(mdp), seed=7).sample(0)
    assert episode.terminated and len(episode) == 1 and episode.transitions[0].reward.item() == 5


def test_outcome_termination_is_preserved_even_without_globally_terminal_states():
    mdp = TabularMDP(1, 1, (((Outcome(1, 0, 5, True),),),))
    episode = TabularEpisodeSampler(mdp, uniform_policy(mdp), seed=7).sample(0)
    assert len(episode) == 1 and episode.terminated and episode.transitions[0].next_observation.item() == 0


@pytest.mark.parametrize("start", [-1, 16, True, 1.5])
def test_bad_start_states_fail(start):
    mdp = sutton_gridworld()
    with pytest.raises(ValueError, match="start_state"):
        TabularEpisodeSampler(mdp, uniform_policy(mdp)).sample(start)


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5])
def test_bad_maximum_lengths_fail(maximum):
    mdp = sutton_gridworld()
    with pytest.raises(ValueError, match="max_steps"):
        TabularEpisodeSampler(mdp, uniform_policy(mdp)).sample(1, max_steps=maximum)


@pytest.mark.parametrize("policy,message", [
    (np.ones(16), "shape"), (np.zeros((16, 4)), "sum to 1"),
    (np.full((16, 4), -.25), "nonnegative"), (np.full((16, 4), np.nan), "finite"),
])
def test_bad_policy_shape_and_probabilities_fail(policy, message):
    with pytest.raises(ValueError, match=message):
        TabularEpisodeSampler(sutton_gridworld(), policy)


def test_complex_policy_is_not_silently_cast_to_real():
    with pytest.raises(TypeError, match="real numbers"):
        TabularEpisodeSampler(sutton_gridworld(), np.ones((16, 4), dtype=complex))
