"""Real spawned workers: SAME_STEP endpoints, independent boundaries and cleanup."""

from functools import partial

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, AutoresetMode

from modular_rl import Model, representations as R, estimators as E, selectors as S
from modular_rl import buffers as B, targets as T, learners as L, updates as U
from modular_rl.environment import final_observations, validate_vector_env


class BoundaryEnv(gym.Env):
    observation_space=gym.spaces.Box(-1,100000,(3,),np.float32)
    action_space=gym.spaces.Discrete(2)

    def __init__(self,length=2,truncate=False,fail=False):
        self.length,self.truncate,self.fail=length,truncate,fail

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.episode=int(self.np_random.integers(100,10000));self.time=0
        return np.array([self.episode,0,-1],dtype=np.float32),{}

    def step(self,action):
        if self.fail:raise RuntimeError('intentional worker failure')
        self.time+=1;done=self.time==self.length
        return (np.array([self.episode,self.time,action],dtype=np.float32),float(action+1),
                done and not self.truncate,done and self.truncate,{})


def make_vector(kind=AsyncVectorEnv,**kwargs):
    factories=[partial(BoundaryEnv,length=2),partial(BoundaryEnv,length=3,truncate=True),partial(BoundaryEnv,length=4)]
    if kind is AsyncVectorEnv:kwargs['context']='spawn'
    return kind(factories,autoreset_mode=AutoresetMode.SAME_STEP,**kwargs)


def gather(kind):
    env=make_vector(kind)
    processes=getattr(env,'processes',[])
    buffer=B.Rollout(6,3);trace=[]
    try:
        validate_vector_env(env)
        observation,_=env.reset(seed=[7,11,19])
        starts=observation.copy()
        for time in range(6):
            action=(np.arange(3)+time)%2
            returned,reward,terminated,truncated,info=env.step(action)
            actual=final_observations(returned,terminated,truncated,info)
            np.testing.assert_array_equal(actual[:,2],action)
            boundary=terminated|truncated
            if boundary.any():
                assert (returned[boundary,1]==0).all()
                assert (actual[boundary,1]>0).all()
                np.testing.assert_array_equal(info['_final_obs'],boundary)
            buffer.add(observation,action,reward,actual,terminated,truncated,
                       value=torch.zeros(3),log_prob=torch.zeros(3))
            trace.append(returned.copy());observation=returned
        return buffer.get(),starts,np.array(trace)
    finally:
        env.close(timeout=5) if kind is AsyncVectorEnv else env.close()
        assert all(not process.is_alive() for process in processes)


def test_async_same_step_matches_sync_and_seed_reproducibility():
    first,starts,trace=gather(AsyncVectorEnv)
    second,starts_again,trace_again=gather(AsyncVectorEnv)
    synchronous,_,sync_trace=gather(SyncVectorEnv)
    assert len(set(starts[:,0]))==3
    np.testing.assert_array_equal(starts,starts_again)
    np.testing.assert_array_equal(trace,trace_again)
    np.testing.assert_array_equal(trace,sync_trace)
    for name in first.__dataclass_fields__:
        assert torch.equal(getattr(first,name),getattr(second,name))
        assert torch.equal(getattr(first,name),getattr(synchronous,name))
    assert first.terminated[:,0].tolist()==[False,True,False,True,False,True]
    assert first.truncated[:,1].tolist()==[False,False,True,False,False,True]
    # Bootstrap from final step count 3, never reset count 0; stop later rewards.
    values=first.next_observation[:,:,1].double()
    returns=T.RolloutReturn(.9).generate(first.reward,values,first.terminated,first.truncated)
    assert returns[2,1].item()==pytest.approx(first.reward[2,1].item()+.9*3)
    assert returns[1,0].item()==first.reward[1,0].item()


def test_worker_exception_terminates_all_processes():
    env=AsyncVectorEnv([partial(BoundaryEnv,fail=True),BoundaryEnv],context='spawn',
                       autoreset_mode=AutoresetMode.SAME_STEP)
    processes=env.processes.copy()
    try:
        env.reset(seed=7)
        with pytest.warns(UserWarning,match='ERROR'):
            with pytest.raises(RuntimeError,match='intentional worker failure'):env.step(np.array([0,1]))
    finally:
        env.close(terminate=True)
    assert all(not process.is_alive() for process in processes)


def test_async_collection_to_a2c_uses_final_values_before_one_update(monkeypatch):
    torch.manual_seed(7)
    model=Model(R.Identity(),E.ActorCritic(E.CategoricalPolicy(3,2),E.StateValue(3)),S.Categorical())
    with torch.no_grad():
        for p in model.parameters():p.zero_()
        model.estimator.value.network.weight[0,1]=1  # V(obs) = elapsed episode steps
    learner=L.A2C(model,torch.optim.SGD(model.parameters(),lr=1e-6),T.RolloutReturn(.9),U.ActorCritic())
    buffer=B.Rollout(4,3); env=make_vector(); processes=env.processes.copy()
    targets=[]; generate=learner.target_generator.generate
    def capture(*args):
        target=generate(*args);targets.append(target.clone());return target
    monkeypatch.setattr(learner.target_generator,'generate',capture)
    try:
        validate_vector_env(env);observation,_=env.reset(seed=[7,11,19])
        for _ in range(4):
            with torch.no_grad():
                estimate=model(observation);selected=model.action_selector.select(estimate)
            returned,reward,terminated,truncated,info=env.step(selected.action.numpy())
            actual=final_observations(returned,terminated,truncated,info)
            buffer.add(observation,selected.action,reward,actual,terminated,truncated,
                       value=estimate.values.v,log_prob=selected.log_prob)
            observation=returned
        rollout=buffer.get(); before=[p.clone() for p in model.parameters()]
        result=learner.learn(rollout)
        assert result.td_error.shape==(4,3) and not result.total_loss.requires_grad
        assert targets[0][1,0].item()==rollout.reward[1,0].item()
        assert targets[0][2,1].item()==pytest.approx(rollout.reward[2,1].item()+.9*3)
        assert any(not torch.equal(p,q) for p,q in zip(model.parameters(),before))
    finally:
        env.close(timeout=5)
    assert all(not process.is_alive() for process in processes)


def test_reject_next_step_and_unsupported_spaces():
    env=SyncVectorEnv([BoundaryEnv],autoreset_mode=AutoresetMode.NEXT_STEP)
    try:
        with pytest.raises(ValueError,match='NEXT_STEP'):validate_vector_env(env)
    finally:env.close()
    env=SyncVectorEnv([lambda:gym.make('FrozenLake-v1')],autoreset_mode=AutoresetMode.SAME_STEP)
    try:
        with pytest.raises(TypeError,match='Box'):validate_vector_env(env)
    finally:env.close()


@pytest.mark.parametrize('info',[{}, {'final_obs':[np.ones(3,dtype=np.float32)],'_final_obs':np.array([False])},
 {'final_obs':[np.ones(2,dtype=np.float32)],'_final_obs':np.array([True])}])
def test_missing_or_invalid_final_observation_never_falls_back_to_reset(info):
    with pytest.raises(ValueError,match='final_obs'):
        final_observations(np.zeros((1,3),dtype=np.float32),np.array([False]),np.array([True]),info)
