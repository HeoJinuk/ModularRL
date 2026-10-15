"""A2C math, one-step equivalence, on-policy freshness and checkpointing."""

from copy import deepcopy
from dataclasses import fields, replace

import pytest
import torch
from torch import nn

from modular_rl import Model, representations as R, estimators as E, selectors as S
from modular_rl import buffers as B, learners as L, targets as T, updates as U
from modular_rl.data import RolloutBatch, Transition
from tests.test_actor_critic import make_learner, row, DEVICES


def make_a2c(dtype=torch.float64, device='cpu', **kwargs):
    one = make_learner(dtype, device, **kwargs)
    return L.A2C(one.model, one.optimizer, T.RolloutReturn(one.target_generator.gamma), one.update_rule)


def rollout_for(model, time=3, n_envs=2):
    observation = torch.arange(time*n_envs, dtype=torch.float32).reshape(time,n_envs,1)/10+1
    following = torch.cat((observation[1:], observation[-1:]+1))
    action = torch.arange(time*n_envs).reshape(time,n_envs)%2
    reward = torch.arange(1,time*n_envs+1,dtype=torch.float32).reshape(time,n_envs)
    terminated = torch.zeros(time,n_envs,dtype=torch.bool); truncated = terminated.clone()
    if time>1:
        terminated[0,0]=True; truncated[-1,-1]=True
    buffer = B.Rollout(time,n_envs)
    for index in range(time):
        with torch.no_grad():
            estimate = model(observation[index])
            log_prob = estimate.policy.log_prob(action[index].to(estimate.values.v.device))
        buffer.add(observation[index],action[index],reward[index],following[index],terminated[index],truncated[index],
                   value=estimate.values.v,log_prob=log_prob)
    return buffer.get()


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('terminated,truncated',[(False,False),(True,False),(False,True),(True,True)])
@pytest.mark.parametrize('stateful_network',[False,True])
def test_t1_n1_matches_actor_critic_target_loss_gradient_and_optimizer(dtype,device,terminated,truncated,stateful_network):
    original = make_learner(dtype,device)
    if stateful_network:
        original.model.representation = nn.Sequential(nn.Linear(1,1),nn.BatchNorm1d(1),nn.Dropout(.4)).to(device,dtype)
        original.optimizer = torch.optim.SGD(original.model.parameters(),lr=.05,momentum=.9)
        original.model.eval()  # consistent collection/current/bootstrap modes
    original.learn(row())  # populate optimizer momentum before cloning
    model = deepcopy(original.model)
    optimizer = torch.optim.SGD(model.parameters(),lr=.05,momentum=.9)
    optimizer.load_state_dict(deepcopy(original.optimizer.state_dict()))
    vector = L.A2C(model,optimizer,T.RolloutReturn(.9),deepcopy(original.update_rule))
    data = row(terminated,truncated)
    with torch.no_grad():
        estimate=model(data.observation)
        stored=estimate.policy.log_prob(data.action.to(device))
    buffer = B.Rollout(1,1)
    buffer.add(data.observation,data.action,data.reward,data.next_observation,data.terminated,data.truncated,
               value=estimate.values.v,log_prob=stored)
    targets = []
    for learner in (original,vector):
        generate = learner.target_generator.generate
        def capture(*args,_generate=generate):
            target=_generate(*args);targets.append(target.detach().reshape(-1));return target
        learner.target_generator.generate=capture
    a,b=original.learn(data),vector.learn(buffer.get())
    torch.testing.assert_close(targets[0],targets[1],rtol=0,atol=0)
    for field in fields(a):
        torch.testing.assert_close(getattr(a,field.name).reshape(-1),getattr(b,field.name).reshape(-1),rtol=0,atol=0)
    for p,q in zip(original.model.parameters(),model.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    for a,b in zip(original.optimizer.state.values(),optimizer.state.values()):
        torch.testing.assert_close(a['momentum_buffer'],b['momentum_buffer'],rtol=0,atol=0)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('device',DEVICES)
def test_multistep_loss_gradients_reduction_and_one_optimizer_step(dtype,device,monkeypatch):
    learner=make_a2c(dtype,device); data=rollout_for(learner.model)
    reference=deepcopy(learner.model)
    moved=data.to(device,dtype=dtype)
    estimate=reference(moved.observation.reshape(6,1))
    with torch.no_grad():
        following=reference(moved.next_observation.reshape(6,1)).values.v.reshape(3,2)
        # Independent forward reward sums, stopping at each episode boundary.
        target=torch.zeros_like(moved.reward)
        for env in range(2):
            for start in range(3):
                discount=1.
                for time in range(start,3):
                    target[start,env]+=discount*moved.reward[time,env];discount*=.9
                    if moved.terminated[time,env]:break
                    if moved.truncated[time,env] or time==2:
                        target[start,env]+=discount*following[time,env];break
    delta=target.reshape(-1)-estimate.values.v
    actor=-(estimate.policy.log_prob(moved.action.reshape(-1))*delta.detach()).mean()
    critic=.5*delta.square().mean();entropy=estimate.policy.entropy().mean()
    loss=actor+.5*critic-.01*entropy;loss.backward()
    calls=[]
    for name in ('zero_grad','step'):
        function=getattr(learner.optimizer,name)
        def record(*args,_name=name,_function=function,**kwargs):
            calls.append(_name);return _function(*args,**kwargs)
        monkeypatch.setattr(learner.optimizer,name,record)
    result=learner.learn(data)
    assert calls==['zero_grad','step']
    for name,expected in [('actor_loss',actor),('critic_loss',critic),('entropy',entropy),('total_loss',loss),('td_error',delta.reshape(3,2))]:
        actual=getattr(result,name);torch.testing.assert_close(actual,expected)
        assert not actual.requires_grad and actual.grad_fn is None
        assert actual.dtype==dtype and actual.device.type==device
    for p,q in zip(learner.model.parameters(),reference.parameters()):
        torch.testing.assert_close(p.grad,q.grad)
        torch.testing.assert_close(p,q-.05*q.grad)


def test_stored_snapshots_are_not_gradient_sources_and_modes_are_restored():
    learner=make_a2c(); learner.model.train(); learner.model.estimator.value.eval()
    modes=[m.training for m in learner.model.modules()]
    data=rollout_for(learner.model)
    value=data.value.requires_grad_();log_prob=data.log_prob.requires_grad_()
    result=learner.learn(data)
    assert value.grad is log_prob.grad is None
    assert [m.training for m in learner.model.modules()]==modes
    assert result.td_error.shape==(3,2)


def test_parameter_change_after_collection_is_rejected_before_optimization():
    learner=make_a2c();data=rollout_for(learner.model)
    with torch.no_grad():learner.model.estimator.policy.network.bias[0]+=1
    parameters=[p.clone() for p in learner.model.parameters()]
    with pytest.raises(ValueError,match='On-policy'):learner.learn(data)
    for p,q in zip(learner.model.parameters(),parameters):assert torch.equal(p,q)
    assert not learner.optimizer.state


def test_state_dict_and_optimizer_resume_exactly():
    original=make_a2c();original.learn(rollout_for(original.model))
    resumed=make_a2c()
    resumed.model.load_state_dict(deepcopy(original.model.state_dict()))
    resumed.optimizer.load_state_dict(deepcopy(original.optimizer.state_dict()))
    data=rollout_for(original.model)
    a,b=original.learn(data),resumed.learn(data)
    assert torch.equal(a.total_loss,b.total_loss)
    for p,q in zip(original.model.parameters(),resumed.model.parameters()):assert torch.equal(p,q)


@pytest.mark.parametrize('field,convert',[
 ('action',lambda t:t.float()),('reward',lambda t:t.long()),('terminated',lambda t:t.long()),
 ('truncated',lambda t:t.unsqueeze(-1)),('observation',lambda t:t[:1]),
 ('log_prob',lambda t:t+1),('value',lambda t:t+1),
])
def test_invalid_rollout_and_snapshot_contracts(field,convert):
    learner=make_a2c();data=rollout_for(learner.model)
    with pytest.raises((TypeError,ValueError)):learner.learn(replace(data,**{field:convert(getattr(data,field))}))


def test_wrong_batch_and_estimator_fail_clearly():
    with pytest.raises(TypeError,match='RolloutBatch'):make_a2c().learn(row())
    for estimator in [E.StateValue(1),E.DiscreteQ(1,2)]:
        model=Model(R.Identity(),estimator)
        learner=L.A2C(model,torch.optim.Adam(model.parameters()),T.RolloutReturn(),U.ActorCritic())
        data=rollout_for(make_a2c().model)
        with pytest.raises(ValueError,match='Categorical'):learner.learn(data)
