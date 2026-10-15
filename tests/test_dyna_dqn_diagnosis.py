"""Diagnostic instrumentation must not alter the experiment it measures."""

from copy import deepcopy
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch

from experiments.dyna_dqn_diagnosis import Config, run, build_learner, action_values, evaluate, chain_propagation, fingerprint
from modular_rl import world_models as W, planners as P, buffers as B
from modular_rl.data import Transition


def small(**kwargs):
    return Config(steps=80, warmup=8, batch_size=4, target_sync_steps=7,
                  checkpoint_interval=13, evaluation_episodes=2, **kwargs)


def assert_training_equal(first, second):
    assert first['hashes'] == second['hashes']
    assert first['q'] == second['q'] and first['q_batched'] == second['q_batched']
    assert first['episodes'] == second['episodes']
    assert first['state_action_visits'] == second['state_action_visits']


@pytest.mark.parametrize('planning',[0,1,5])
def test_logging_preserves_parameters_optimizer_trajectory_and_all_rngs(planning):
    recorded,_=run(small(planning_steps=planning),diagnostics=True)
    silent,_=run(small(planning_steps=planning),diagnostics=False)
    assert_training_equal(recorded,silent)
    assert recorded['checkpoints'] and not silent['checkpoints']


def test_world_collection_only_matches_basic_dqn_exactly():
    basic,_=run(small(collect_world=False,planning_steps=0))
    observed,_=run(small(collect_world=True,planning_steps=0))
    assert_training_equal(basic,observed)
    assert basic['world']['pairs']==0 and observed['world']['pairs']>0
    assert observed['world']['incorrect_pairs']==0


def test_short_loop_matches_independent_notebook_order_and_diagnostics():
    config=small(planning_steps=1)
    actual,state=run(config)
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        learner=build_learner(config)
        model=learner.online_model
        replay=B.Replay(config.replay_capacity,seed=config.seed)
        world=W.Tabular();rng=np.random.default_rng(config.seed+3)
        planner=P.Dyna(world,learner,batch_size=config.batch_size,rng=rng)
        env=gym.make('FrozenLake-v1',is_slippery=False,max_episode_steps=1000)
        observation,_=env.reset(seed=config.seed)
        trace=[]
        try:
            for steps in range(config.steps):
                model.action_selector.epsilon=max(.05,1-steps/config.epsilon_decay_steps)
                selected=model.act(observation)
                following,reward,terminated,truncated,_=env.step(selected.action.item())
                replay.add(observation,selected.action,float(reward),following,terminated,truncated)
                if len(replay)>=config.warmup:learner.learn(replay.sample(config.batch_size))
                world.observe(Transition(observation,selected.action.item(),float(reward),following,terminated,truncated))
                planner.plan(1)
                if (steps+1)%config.target_sync_steps==0:learner.sync_target()
                trace.append([observation,selected.action.item(),float(reward),following,int(terminated),int(truncated)])
                if steps+1<config.steps:
                    observation=env.reset()[0] if terminated or truncated else following
            expected=dict(online=model.state_dict(),target=learner.target_model.state_dict(),
                          optimizer=learner.optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                          replay_rng=replay._rng.bit_generator.state,planning_rng=rng.bit_generator.state,
                          environment_rng=env.np_random.bit_generator.state,trace=np.array(trace))
            assert actual['hashes']=={key:fingerprint(value) for key,value in expected.items()}
        finally:env.close()
    finally:torch.set_num_threads(old_threads)


@pytest.mark.parametrize('planning',[1,5])
def test_matched_optimizer_quota_is_exact_at_same_environment_budget(planning):
    dyna,_=run(small(planning_steps=planning),diagnostics=False)
    replay,_=run(small(planning_steps=0,collect_world=False,matched_planning=planning),diagnostics=False)
    assert dyna['steps']==replay['steps']==80
    assert dyna['optimizer_steps']==replay['optimizer_steps']==73+80*planning
    assert replay['planning_updates']==0
    assert dyna['target_syncs']==replay['target_syncs']==80//7


def test_independent_values_have_identical_initial_policy_and_behavior_rng():
    config=small()
    mlp=build_learner(config);rng=torch.get_rng_state().clone();q=action_values(mlp.online_model)
    independent=build_learner(replace(config,architecture='independent'))
    assert np.array_equal(q,action_values(independent.online_model))
    assert torch.equal(rng,torch.get_rng_state())


def test_reward_propagates_through_known_chain_in_tabular_and_neural_models():
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:result=chain_propagation()
    finally:torch.set_num_threads(old)
    final=result['trajectory'][-1]
    np.testing.assert_allclose(final['tabular'],[.9,1],atol=1e-5)
    np.testing.assert_allclose(final['neural'],[.9,1],atol=1e-5)
    assert result['trajectory'][0]['neural'][0]==0  # frozen target has not yet copied learned S1


def test_cached_scalar_policy_is_identical_to_actual_model_act_greedy():
    model=build_learner(small()).online_model
    policy=action_values(model).argmax(-1)
    assert all(policy[s]==model.act(s,deterministic=True).action.item() for s in range(16))


def test_visit_counts_partial_boundary_and_sample_statistics():
    result,_=run(small(planning_steps=1))
    assert np.sum(result['state_action_visits'])==80
    assert np.sum(result['state_visits']) in (80+result['completed_episodes'],81+result['completed_episodes'])
    assert result['planning']['samples']==80*4
    assert result['real']['samples']==73*4
    assert result['world']['incorrect_pairs']==0
    assert result['checkpoints'][-1]['step']==80
    # Checkpoints precede that step's periodic sync, unlike final counters.
    for row in result['checkpoints']:assert row['target_syncs']==(row['step']-1)//7


@pytest.mark.parametrize('terminated,truncated,expected',[(True,False,1.),(True,True,1.),(False,True,1.+.99*5)])
def test_synthetic_flags_values_and_bellman_targets(terminated,truncated,expected):
    config=small();learner=build_learner(config);world=W.Tabular()
    with torch.no_grad():
        for p in learner.target_model.parameters():p.zero_()
        learner.target_model.estimator.network.bias.fill_(5.)
    world.observe(Transition(0,2,1.,1,terminated,truncated))
    captured=[]
    generate=learner.target_generator.generate
    def record(*args):
        target=generate(*args);captured.append(target.clone());return target
    learner.target_generator.generate=record
    planner=P.Dyna(world,learner,batch_size=4,seed=7)
    results=planner.plan(1,record_batches=True)
    batch=results['batches'][0]
    assert batch.observation.tolist()==[0]*4 and batch.action.tolist()==[2]*4
    assert batch.reward.tolist()==[1.]*4 and batch.next_observation.tolist()==[1]*4
    assert not batch.truncated.any() and batch.terminated.all().item()==terminated
    torch.testing.assert_close(captured[0],torch.full((4,),expected))
    assert all(p.grad is None for p in learner.target_model.parameters())
