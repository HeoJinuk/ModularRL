"""Gymnasium episode collection connects unchanged MC, TD and Q-learning APIs."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from modular_rl.component.estimator import TabularVEstimator, TabularQEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import MonteCarloReturnGenerator, TD0TargetGenerator
from modular_rl.component.update_rule import TabularVUpdateRule, TabularQUpdateRule
from modular_rl.data import Episode, TabularMDP, TransitionOutcome as Outcome
from modular_rl.dp import policy_evaluation, uniform_policy
from modular_rl.environment import TabularEnv, TabularEpisodeSampler, collect_episode, make_dlfs4_gridworld_env
from modular_rl.learner import MCPredictionLearner, TabularTD0PredictionLearner, TabularQLearner
from modular_rl.model import Model


def chain():
    return TabularMDP(3, 1, (((Outcome(1., 1, 2.),),), ((Outcome(1., 2, 3., True),),),
                            ((Outcome(1., 2, 0., True),),)), (2,))


def test_collection_preserves_order_transitions_final_observation_and_boundaries():
    env = gym.wrappers.TimeLimit(TabularEnv(chain(), start_state=0), max_episode_steps=3)
    actions = []

    def select_action(state):
        actions.append(state)
        return 0

    episode = collect_episode(env, select_action, seed=7)
    assert isinstance(episode, Episode) and len(episode) == 2 and episode.terminated and not episode.truncated
    assert actions == [0, 1]
    assert [t.observation.item() for t in episode.transitions] == [0, 1]
    assert [t.next_observation.item() for t in episode.transitions] == [1, 2]
    assert [t.reward.item() for t in episode.transitions] == [2., 3.]
    assert all(t.log_prob is None for t in episode.transitions)
    for transition in episode.transitions:
        assert transition.observation.dtype == torch.long
        assert transition.terminated.dtype == transition.truncated.dtype == torch.bool
        assert not transition.reward.requires_grad
    env.close()


def test_policy_rng_is_external_and_action_distribution_is_preserved():
    mdp = TabularMDP(1, 4, (tuple((Outcome(1., 0, float(action), True),) for action in range(4)),))
    env = TabularEnv(mdp, start_state=0)
    rng = np.random.default_rng(17)
    probabilities = np.array([.1, .2, .3, .4])
    actions = [collect_episode(env, lambda state: int(rng.choice(4, p=probabilities)),
                               seed=7 if i == 0 else None).transitions[0].action.item() for i in range(2000)]
    np.testing.assert_allclose(np.bincount(actions, minlength=4) / 2000, probabilities, atol=.035, rtol=0)
    env.close()


def test_environment_and_policy_seeds_reproduce_complete_episodes():
    def collect():
        env = make_dlfs4_gridworld_env()
        rng = np.random.default_rng(8)
        episodes = [collect_episode(env, lambda state: int(rng.integers(4)),
                                    seed=7 if i == 0 else None) for i in range(10)]
        env.close()
        return [[(t.observation.item(), t.action.item(), t.reward.item(), t.next_observation.item(),
                  t.terminated.item(), t.truncated.item()) for t in episode.transitions] for episode in episodes]

    assert collect() == collect()


def test_time_limit_fragment_is_explicit_and_mc_rejects_it_while_td_bootstraps():
    mdp = TabularMDP(1, 1, (((Outcome(1., 0, 1.),),),))
    env = gym.wrappers.TimeLimit(TabularEnv(mdp, start_state=0), max_episode_steps=1)
    episode = collect_episode(env, lambda state: 0, seed=7)
    assert len(episode) == 1 and episode.truncated and not episode.terminated
    transition = episode.transitions[0]
    assert transition.next_observation.item() == 0
    mc = MCPredictionLearner(Model(IdentityRepresentation(), TabularVEstimator(1)), MonteCarloReturnGenerator(.9))
    with pytest.raises(ValueError, match="terminated episode"):
        mc.learn(episode)
    td_model = Model(IdentityRepresentation(), TabularVEstimator(1)).double()
    td_model.estimator.v_table.fill_(4)
    td = TabularTD0PredictionLearner(td_model, TD0TargetGenerator(.9), TabularVUpdateRule(.5))
    torch.testing.assert_close(td.learn(transition.to_batch()), torch.tensor([.6], dtype=torch.float64))
    torch.testing.assert_close(td_model.estimator.v_table, torch.tensor([4.3], dtype=torch.float64))
    env.close()


def test_mc_td_and_q_learning_reuse_the_same_gym_experience_and_dp_reference():
    mdp = chain()
    reference = policy_evaluation(mdp, uniform_policy(mdp), gamma=.5)
    env = gym.wrappers.TimeLimit(TabularEnv(mdp, start_state=0), max_episode_steps=3)
    episode = collect_episode(env, lambda state: 0, seed=7)
    mc_model = Model(IdentityRepresentation(), TabularVEstimator(3)).double()
    mc = MCPredictionLearner(mc_model, MonteCarloReturnGenerator(.5))
    torch.testing.assert_close(mc.learn(episode), torch.tensor([3.5, 3.], dtype=torch.float64))
    np.testing.assert_array_equal(mc_model.estimator.v_table.numpy(), reference.values)
    td_model = Model(IdentityRepresentation(), TabularVEstimator(3)).double()
    q_model = Model(IdentityRepresentation(), TabularQEstimator(3, 1)).double()
    td = TabularTD0PredictionLearner(td_model, TD0TargetGenerator(.5), TabularVUpdateRule(1))
    q = TabularQLearner(q_model, TD0TargetGenerator(.5), TabularQUpdateRule(1))
    for _ in range(2):
        for transition in episode.transitions:
            for learner in (td, q):
                error = learner.learn(transition.to_batch())
                assert not error.requires_grad
    np.testing.assert_array_equal(td_model.estimator.v_table.numpy(), reference.values)
    np.testing.assert_array_equal(q_model.estimator.q_table[:, 0].numpy(), reference.values)
    env.close()


def test_legacy_sampler_keeps_its_api_and_its_original_terminal_at_cap_contract():
    mdp = chain()
    sampler = TabularEpisodeSampler(mdp, uniform_policy(mdp), seed=7)
    legacy = sampler.sample(0, max_steps=2)
    assert len(legacy) == 2 and legacy.terminated and not legacy.truncated
    env = gym.wrappers.TimeLimit(TabularEnv(mdp, start_state=0), max_episode_steps=2)
    current = collect_episode(env, lambda state: 0, seed=7)
    assert current.terminated and current.truncated  # Official TimeLimit semantics, preserved independently.
    env.close()
