"""Gymnasium contracts, shared dynamics, RNG ownership and episode boundaries."""

import copy

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import EnvSpec
from gymnasium.error import ResetNeeded
from gymnasium.utils.env_checker import check_env

from modular_rl.data import TabularMDP, TransitionOutcome as Outcome
from modular_rl.environment import (
    TabularEnv, sutton_gridworld, dlfs4_gridworld, DLFS4_START_STATE,
    make_sutton_gridworld_env, make_dlfs4_gridworld_env,
)


@pytest.mark.parametrize("factory", [make_sutton_gridworld_env, make_dlfs4_gridworld_env])
def test_reset_step_types_spaces_and_shared_model(factory):
    env = factory()
    try:
        assert isinstance(env, gym.wrappers.TimeLimit)
        assert isinstance(env.unwrapped, TabularEnv)
        mdp = env.unwrapped.mdp
        assert env.observation_space == gym.spaces.Discrete(mdp.n_states)
        assert env.action_space == gym.spaces.Discrete(mdp.n_actions)
        observation, info = env.reset(seed=7)
        assert type(observation) is int and env.observation_space.contains(observation)
        assert info == {}
        observation, reward, terminated, truncated, info = env.step(np.int64(0))
        assert type(observation) is int and type(reward) is float
        assert type(terminated) is type(truncated) is bool and info == {}
    finally:
        env.close()


@pytest.mark.parametrize("mdp", [sutton_gridworld(), dlfs4_gridworld()])
def test_execution_matches_every_deterministic_nonterminal_model_row(mdp):
    env = TabularEnv(mdp)
    for state in range(mdp.n_states):
        if state in mdp.terminal_states:
            continue
        for action in range(mdp.n_actions):
            env.reset(seed=7, options={"start_state": state})
            observation, reward, terminated, truncated, info = env.step(action)
            outcome = mdp.transitions[state][action][0]
            assert observation == outcome.next_state and reward == outcome.reward
            assert terminated == (outcome.terminated or observation in mdp.terminal_states)
            assert not truncated and info == {}
    env.close()


def test_book_default_start_is_fixed_and_options_override_is_for_one_reset():
    env = make_dlfs4_gridworld_env(max_episode_steps=None)
    assert env.reset(seed=7)[0] == DLFS4_START_STATE
    assert env.reset(options={"start_state": np.int32(2)})[0] == 2
    assert env.reset()[0] == DLFS4_START_STATE
    assert env.reset(options={"start_state": np.array(1)})[0] == 1
    env.close()


def test_sutton_defaults_sample_uniform_nonterminal_starts_without_global_rng():
    env = make_sutton_gridworld_env(max_episode_steps=None)
    global_rng = copy.deepcopy(np.random.get_state())
    starts = [env.reset(seed=7 if index == 0 else None)[0] for index in range(3000)]
    counts = np.bincount(starts, minlength=16)
    assert counts[[0, 15]].sum() == 0
    np.testing.assert_allclose(counts[1:15] / 3000, 1 / 14, atol=.025, rtol=0)
    after = np.random.get_state()
    assert global_rng[0] == after[0] and global_rng[2:] == after[2:]
    np.testing.assert_array_equal(global_rng[1], after[1])
    env.close()


@pytest.mark.parametrize("factory", [make_sutton_gridworld_env, make_dlfs4_gridworld_env])
def test_seed_recreates_rollout_and_unseeded_resets_continue_owned_stream(factory):
    first, second = factory(), factory()

    def rollout(env):
        result = []
        for number in range(20):
            result.append(env.reset(seed=123 if number == 0 else None))
            result.append(env.step(number % 4))
        return result

    assert rollout(first) == rollout(second)
    assert first.unwrapped.np_random.bit_generator.state == second.unwrapped.np_random.bit_generator.state
    first.reset(seed=99)
    before = copy.deepcopy(first.unwrapped.np_random.bit_generator.state)
    first.reset()
    if factory is make_sutton_gridworld_env:
        assert first.unwrapped.np_random.bit_generator.state != before
    else:
        assert first.unwrapped.np_random.bit_generator.state == before  # A fixed start consumes no draw.
    first.close(); second.close()


@pytest.mark.parametrize("action", [-1, 4, True, np.bool_(False), 1., None, "up", np.array([0])])
def test_invalid_actions_fail_without_consuming_rng(action):
    env = make_dlfs4_gridworld_env(max_episode_steps=None, render_mode="ansi")
    env.reset(seed=7)
    rng, state = copy.deepcopy(env.np_random.bit_generator.state), env.render()
    with pytest.raises(ValueError, match="action space"):
        env.step(action)
    assert env.np_random.bit_generator.state == rng and env.render() == state
    env.close()


@pytest.mark.parametrize("start", [-1, 11, True, np.bool_(False), 1., None, np.array([1]), 3])
def test_invalid_or_terminal_start_options_fail_clearly(start):
    env = make_dlfs4_gridworld_env(max_episode_steps=None)
    with pytest.raises(ValueError, match="start_state"):
        env.reset(seed=7, options={"start_state": start})
    env.close()


def test_natural_termination_requires_reset_and_preserves_actual_final_state():
    env = make_dlfs4_gridworld_env(max_episode_steps=None)
    with pytest.raises(ResetNeeded):
        env.step(0)
    env.reset(seed=7, options={"start_state": 2})
    assert env.step(3) == (3, 1., True, False, {})
    with pytest.raises(ResetNeeded):
        env.step(0)
    assert env.reset()[0] == DLFS4_START_STATE
    env.close(); env.close()
    with pytest.raises(ResetNeeded):
        env.step(0)
    assert env.reset(seed=7)[0] == DLFS4_START_STATE
    env.close()


def test_time_limit_truncates_without_replacing_observation_or_terminating_mdp():
    env = make_sutton_gridworld_env(max_episode_steps=1)
    env.reset(seed=7, options={"start_state": 1})
    assert env.step(0) == (1, -1., False, True, {})
    assert not env.unwrapped._terminated  # The standard wrapper owns the cutoff.
    env.reset(options={"start_state": 1})
    assert env.step(2) == (0, -1., True, True, {})  # Natural ending exactly at the cap.
    env.close()


def test_stochastic_joint_outcomes_follow_probabilities_rewards_and_termination():
    mdp = TabularMDP(1, 1, (((Outcome(.25, 0, 2., True), Outcome(.75, 0, -3., False)),),))
    env = TabularEnv(mdp, start_state=0)
    endings = []
    for number in range(3000):
        env.reset(seed=17 if number == 0 else None)
        observation, reward, terminated, truncated, _ = env.step(0)
        assert observation == 0 and not truncated
        assert reward == (2 if terminated else -3)
        endings.append(terminated)
    assert abs(np.mean(endings) - .25) < .025
    env.close()


def test_declared_terminal_entry_also_stops_when_outcome_flag_is_false():
    mdp = TabularMDP(2, 1, (((Outcome(1., 1, 5., False),),), ((Outcome(1., 1, 0.),),)), (1,))
    env = TabularEnv(mdp, start_state=0)
    env.reset(seed=7)
    assert env.step(0) == (1, 5., True, False, {})
    with pytest.raises(ResetNeeded):
        env.step(0)
    env.close()


@pytest.mark.parametrize("factory", [make_sutton_gridworld_env, make_dlfs4_gridworld_env])
@pytest.mark.parametrize("render_mode", [None, "ansi", "human"])
def test_official_checker_on_unwrapped_environment_including_all_render_modes(factory, render_mode, capsys):
    wrapped = factory(render_mode=render_mode)
    env = wrapped.unwrapped
    # An official test-only spec permits check_env to recreate all declared modes;
    # no global registry or product registration is introduced.
    env.spec = EnvSpec(id="ModularRL-Test-v0", entry_point=TabularEnv,
                       kwargs={"mdp": env.mdp, "start_state": env.start_state})
    check_env(env)
    wrapped.close()


def test_ansi_and_human_render_contract_and_close_are_simple(capsys):
    ansi = make_dlfs4_gridworld_env(render_mode="ansi", max_episode_steps=None)
    with pytest.raises(ResetNeeded):
        ansi.render()
    ansi.reset(seed=7)
    first = ansi.render()
    assert isinstance(first, str) and f"state={DLFS4_START_STATE}" in first
    ansi.step(0)
    assert ansi.render() != first and first.endswith("terminated=False\n")
    ansi.close(); ansi.close()
    human = make_dlfs4_gridworld_env(render_mode="human", max_episode_steps=None)
    human.reset(seed=7)
    human.step(0)
    assert human.render() is None and "state=" in capsys.readouterr().out
    human.close(); human.close()


def test_invalid_environment_configuration_fails_at_setup():
    with pytest.raises(TypeError, match="TabularMDP"):
        TabularEnv(None)
    with pytest.raises(ValueError, match="render_mode"):
        TabularEnv(sutton_gridworld(), render_mode="rgb_array")
    with pytest.raises(ValueError, match="start_state"):
        TabularEnv(sutton_gridworld(), start_state=0)
    all_terminal = TabularMDP(1, 1, (((Outcome(1., 0, 0., True),),),), (0,))
    with pytest.raises(ValueError, match="nonterminal"):
        TabularEnv(all_terminal)
