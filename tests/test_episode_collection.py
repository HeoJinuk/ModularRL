"""Gymnasium episode collection connects unchanged MC, TD and Q-learning APIs."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import TabularVEstimator, TabularQEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import MonteCarloReturnGenerator, TD0TargetGenerator
from modular_rl.component.update_rule import TabularVUpdateRule, TabularQUpdateRule
from modular_rl.data import ActionOutput, Episode, TabularMDP, Transition, TransitionOutcome as Outcome
from modular_rl.dp import policy_evaluation, uniform_policy
from modular_rl.environment import TabularEnv, TabularEpisodeSampler, collect_episode, make_dlfs4_gridworld_env
from modular_rl.learner import MCPredictionLearner, OffPolicyMCPredictionLearner, TabularTD0PredictionLearner, TabularQLearner
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


def one_step_env(n_actions=4):
    mdp = TabularMDP(1, n_actions, (tuple((Outcome(1., 0, 4., True),) for _ in range(n_actions)),))
    return TabularEnv(mdp, start_state=0)


@pytest.mark.parametrize("epsilon", [0., .4, 1.])
def test_model_action_output_records_actual_sampling_probability_including_greedy_ties(epsilon):
    torch.manual_seed(17)
    env = one_step_env()
    model = Model(IdentityRepresentation(), TabularQEstimator(1, 4), EpsilonGreedyActionSelector(epsilon)).double()
    model.estimator.q_table[0] = torch.tensor([2., 2., 0., -1.])
    theoretical = np.array([(1 - epsilon) / 2 + epsilon / 4] * 2 + [epsilon / 4] * 2)
    counts = np.zeros(4, dtype=int)
    try:
        for number in range(500):
            sample = collect_episode(env, model.act, seed=7 if number == 0 else None)
            transition = sample.transitions[0]
            action = transition.action.item()
            counts[action] += 1
            assert transition.log_prob.exp().item() == pytest.approx(theoretical[action])
            assert transition.log_prob.ndim == 0 and transition.log_prob.dtype == torch.float64
            assert not transition.log_prob.requires_grad and transition.log_prob.grad_fn is None
        np.testing.assert_allclose(counts / 500, theoretical, atol=.065, rtol=0)
    finally:
        env.close()


def test_log_prob_is_snapshotted_before_step_even_if_callback_tensor_is_reused(monkeypatch):
    env = one_step_env(2)
    log_prob = torch.tensor(np.log(.5), dtype=torch.float64, requires_grad=True)
    original_step = env.step

    def step(action):
        with torch.no_grad():
            log_prob.fill_(np.log(.25))
        return original_step(action)

    monkeypatch.setattr(env, "step", step)
    try:
        sample = collect_episode(env, lambda state: ActionOutput(torch.tensor(0), log_prob), seed=7)
        saved = sample.transitions[0].log_prob
        assert saved.exp().item() == pytest.approx(.5) and log_prob.exp().item() == pytest.approx(.25)
        assert saved.data_ptr() != log_prob.data_ptr() and not saved.requires_grad
    finally:
        env.close()


def test_policy_changes_after_collection_do_not_change_behavior_records_or_importance_ratios():
    torch.manual_seed(17)
    env = one_step_env()
    behavior = Model(IdentityRepresentation(), TabularQEstimator(1, 4), EpsilonGreedyActionSelector(.4)).double()
    behavior.estimator.q_table[0] = torch.tensor([2., 2., 0., -1.])
    try:
        sample = collect_episode(env, behavior.act, seed=7)
        action = sample.transitions[0].action.item()
        saved_log_b = sample.transitions[0].log_prob.clone()
        behavior.action_selector.epsilon = 0
        behavior.estimator.q_table.fill_(0)
        behavior.estimator.q_table[0, (action + 1) % 4] = 9
        assert behavior.action_selector.probabilities(behavior.forward(0))[action] == 0
        learner = OffPolicyMCPredictionLearner(Model(IdentityRepresentation(), TabularVEstimator(1)).double(),
                                             MonteCarloReturnGenerator(1), np.full((1, 4), .25), method="ordinary")
        learner.learn(sample)
        assert learner.weight_sums[0].item() == pytest.approx(.25 / saved_log_b.exp().item())
        assert learner.model.estimator.v_table[0].item() == pytest.approx(4 * .25 / saved_log_b.exp().item())
        torch.testing.assert_close(sample.transitions[0].log_prob, saved_log_b)
    finally:
        env.close()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_step_by_step_episode_matches_helper_records_and_off_policy_learning(dtype):
    env = make_dlfs4_gridworld_env()
    behavior = Model(IdentityRepresentation(), TabularQEstimator(env.observation_space.n, env.action_space.n),
                     EpsilonGreedyActionSelector(1)).to(dtype=dtype)
    try:
        torch.manual_seed(7)
        observation, _ = env.reset(seed=7)
        transitions = []
        while True:
            output = behavior.act(observation)
            action = output.action.item()
            log_prob = output.log_prob.detach().cpu().clone()
            next_observation, reward, terminated, truncated, _ = env.step(action)
            transitions.append(Transition(observation, action, reward, next_observation,
                                          terminated, truncated, log_prob))
            if terminated or truncated:
                break
            observation = next_observation
        manual = Episode(tuple(transitions))
        torch.manual_seed(7)
        collected = collect_episode(env, behavior.act, seed=7)
        assert manual.terminated and collected.terminated and len(manual) == len(collected)
        for direct, helper in zip(manual.transitions, collected.transitions):
            for name in ("observation", "action", "reward", "next_observation", "terminated", "truncated", "log_prob"):
                torch.testing.assert_close(getattr(direct, name), getattr(helper, name), rtol=0, atol=0)
            assert direct.log_prob.dtype == dtype and direct.log_prob.grad_fn is None
            assert direct.log_prob.exp().item() == pytest.approx(.25)
        # Neither learning path consults the behavior policy after collection.
        behavior.action_selector.epsilon = 0
        behavior.estimator.q_table[:, 0] = 9
        target_policy = np.tile([.3, .2, .2, .3], (env.observation_space.n, 1))
        for method in ("ordinary", "weighted"):
            learners = [OffPolicyMCPredictionLearner(
                Model(IdentityRepresentation(), TabularVEstimator(env.observation_space.n)).to(dtype=dtype),
                MonteCarloReturnGenerator(.9), target_policy, method) for _ in range(2)]
            torch.testing.assert_close(learners[0].learn(manual), learners[1].learn(collected), rtol=0, atol=0)
            torch.testing.assert_close(learners[0].model.estimator.v_table, learners[1].model.estimator.v_table,
                                       rtol=0, atol=0)
            for name in ("visit_counts", "weight_sums", "weighted_return_sums"):
                torch.testing.assert_close(getattr(learners[0], name), getattr(learners[1], name), rtol=0, atol=0)
    finally:
        env.close()


def test_deterministic_action_output_with_no_log_prob_is_collected_but_cannot_be_used_for_is():
    env = one_step_env(2)
    model = Model(IdentityRepresentation(), TabularQEstimator(1, 2), EpsilonGreedyActionSelector(.1))
    try:
        sample = collect_episode(env, lambda state: model.act(state, deterministic=True), seed=7)
        assert sample.transitions[0].log_prob is None
        learner = OffPolicyMCPredictionLearner(Model(IdentityRepresentation(), TabularVEstimator(1)),
                                             MonteCarloReturnGenerator(1), np.full((1, 2), .5))
        with pytest.raises(ValueError, match="sampling-time"):
            learner.learn(sample)
    finally:
        env.close()


@pytest.mark.parametrize("action,log_prob", [
    (torch.tensor([0]), None), (torch.tensor(0.), None), (torch.tensor(True), None),
    (torch.tensor(0), torch.tensor([-.5])), (torch.tensor(0), torch.tensor(0)),
])
def test_bad_action_output_shape_or_dtype_fails_before_environment_step(action, log_prob, monkeypatch):
    env = one_step_env(2)
    monkeypatch.setattr(env, "step", lambda action: pytest.fail("invalid output must not step"))
    try:
        with pytest.raises(TypeError, match="scalar"):
            collect_episode(env, lambda state: ActionOutput(action, log_prob), seed=7)
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_action_output_can_be_recorded_by_cpu_gymnasium_environment():
    env = one_step_env(2)
    model = Model(IdentityRepresentation(), TabularQEstimator(1, 2), EpsilonGreedyActionSelector(1)).double().cuda()
    try:
        sample = collect_episode(env, model.act, seed=7)
        assert sample.transitions[0].action.device.type == sample.transitions[0].log_prob.device.type == "cpu"
        assert sample.transitions[0].log_prob.exp().item() == pytest.approx(.5)
    finally:
        env.close()
