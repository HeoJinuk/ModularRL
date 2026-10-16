"""Frozen targets, repeated data passes, freshness and reproducible PPO state."""

from copy import deepcopy
from dataclasses import fields, replace

import pytest
import torch

from modular_rl import learners as L, targets as T, updates as U
from tests.test_actor_critic import make_learner, row, DEVICES
from tests.test_a2c import rollout_for


def make_ppo(dtype=torch.float64,device='cpu',**kwargs):
    one=make_learner(dtype,device)
    return L.PPO(one.model,one.optimizer,T.GAE(.9,.95),U.PPO(),seed=7,**kwargs)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('terminated,truncated',[(False,False),(True,False),(False,True),(True,True)])
def test_t1_n1_first_gradient_parameters_and_momentum_match_actor_critic(dtype,device,terminated,truncated):
    ac=make_learner(dtype,device);ac.learn(row())
    model=deepcopy(ac.model)
    optimizer=torch.optim.SGD(model.parameters(),lr=.05,momentum=.9)
    optimizer.load_state_dict(deepcopy(ac.optimizer.state_dict()))
    ppo=L.PPO(model,optimizer,T.GAE(.9,0),U.PPO(),epochs=1,minibatch_size=1,seed=7,normalize_advantage=False)
    batch=row(terminated,truncated).to(device,dtype=dtype)
    with torch.no_grad(): estimate=model(batch.observation)
    from modular_rl.data import RolloutBatch
    rollout=RolloutBatch(*(getattr(batch,name).unsqueeze(0) for name in
        ('observation','action','reward','next_observation','terminated','truncated')),
        estimate.values.v.unsqueeze(0),estimate.policy.log_prob(batch.action.to(device)).detach().unsqueeze(0))
    before=deepcopy(model)
    a,b=ac.learn(batch),ppo.learn(rollout)
    torch.testing.assert_close(a.critic_loss,b.value_loss)
    torch.testing.assert_close(a.entropy,b.entropy)
    # Surrogate loss values differ (-ratio*A versus -log_prob*A); first gradients agree.
    assert b.approx_kl.item()==0 and b.clip_fraction.item()==0 and b.optimizer_steps==1
    for p,q in zip(ac.model.parameters(),model.parameters()):
        torch.testing.assert_close(p.grad,q.grad)
        torch.testing.assert_close(p,q)
    for a,b in zip(ac.optimizer.state.values(),optimizer.state.values()):
        torch.testing.assert_close(a['momentum_buffer'],b['momentum_buffer'])
    assert any(not torch.equal(p,q) for p,q in zip(before.parameters(),model.parameters()))


def test_all_samples_once_per_epoch_frozen_snapshots_and_weighted_metrics(monkeypatch):
    learner=make_ppo(epochs=4,minibatch_size=6,normalize_advantage=False)
    learner.optimizer.param_groups[0]['lr']=.001
    data=rollout_for(learner.model,5,4)
    original=deepcopy(data);records=[];targets=[];generated=[];step_calls=[]
    generate=learner.target_generator.generate
    def capture_generate(*args):
        result=generate(*args);generated.append(result)
        assert not result.returns.requires_grad and not result.advantage.requires_grad
        return result
    monkeypatch.setattr(learner.target_generator,'generate',capture_generate)
    # Unique collection log probabilities identify rows without changing the algorithm.
    expected_old=data.log_prob.reshape(-1).tolist()
    assert len(set(expected_old))==20
    compute=learner.update_rule.compute
    def capture_compute(new,old,value,returns,adv,entropy):
        indices=[expected_old.index(x) for x in old.tolist()]
        result=compute(new,old,value,returns,adv,entropy)
        records.append((indices,result,new.detach().clone(),old.clone()))
        targets.append((returns.clone(),adv.clone()))
        return result
    monkeypatch.setattr(learner.update_rule,'compute',capture_compute)
    step=learner.optimizer.step
    def capture_step(*a,**k):step_calls.append(1);return step(*a,**k)
    monkeypatch.setattr(learner.optimizer,'step',capture_step)
    rng=torch.get_rng_state().clone()
    actual=learner.learn(data)
    assert torch.equal(rng,torch.get_rng_state())
    assert len(generated)==1 and len(step_calls)==actual.optimizer_steps==16
    for epoch in range(4):
        groups=records[epoch*4:epoch*4+4]
        assert [len(r[0]) for r in groups]==[6,6,6,2]
        assert sorted(i for r in groups for i in r[0])==list(range(20))
    for field in fields(data):assert torch.equal(getattr(data,field.name),getattr(original,field.name))
    for (indices,result,new,old),(ret,adv) in zip(records,targets):
        torch.testing.assert_close(ret,generated[0].returns.reshape(-1)[indices],rtol=0,atol=0)
        torch.testing.assert_close(adv,generated[0].advantage.reshape(-1)[indices],rtol=0,atol=0)
    assert any(not torch.allclose(new,old) for _,_,new,old in records[4:])
    for field in fields(actual):
        if field.name in ('optimizer_steps','epochs_completed','early_stopped'):continue
        expected=sum(len(ids)*getattr(r,field.name).detach() for ids,r,_,_ in records)/80
        torch.testing.assert_close(getattr(actual,field.name),expected)
        assert not getattr(actual,field.name).requires_grad


@pytest.mark.parametrize('constant', [False,True])
def test_normalization_once_over_whole_rollout_and_constant_behavior(monkeypatch,constant):
    learner=make_ppo(epochs=2,minibatch_size=4)
    data=rollout_for(learner.model,3,2);seen=[]
    generate=learner.target_generator.generate
    def capture(*args):
        result=generate(*args)
        if constant:result=replace(result,advantage=torch.full_like(result.advantage,7.))
        expected=result.advantage.reshape(-1)
        seen.append((expected-expected.mean())/torch.sqrt(expected.var(unbiased=False)+1e-8))
        return result
    monkeypatch.setattr(learner.target_generator,'generate',capture)
    observed=[];compute=learner.update_rule.compute
    def compute_capture(*args):observed.append(args[4].clone());return compute(*args)
    monkeypatch.setattr(learner.update_rule,'compute',compute_capture)
    learner.learn(data)
    for epoch in range(2):
        torch.testing.assert_close(torch.cat(observed[epoch*2:epoch*2+2]).sort().values,seen[0].sort().values)
    if constant:assert torch.equal(seen[0],torch.zeros(6,dtype=seen[0].dtype))


def test_checkpoint_and_shuffle_rng_reproduce_updates_without_action_rng_consumption():
    a=make_ppo(epochs=2,minibatch_size=4);a.learn(rollout_for(a.model))
    b=make_ppo(epochs=2,minibatch_size=4)
    b.model.load_state_dict(deepcopy(a.model.state_dict()))
    b.optimizer.load_state_dict(deepcopy(a.optimizer.state_dict()))
    b.shuffle_rng.set_state(a.shuffle_rng.get_state())
    data=rollout_for(a.model);rng=torch.get_rng_state().clone()
    x,y=a.learn(data),b.learn(data)
    assert torch.equal(rng,torch.get_rng_state())
    for field in fields(x):
        if isinstance(getattr(x,field.name),torch.Tensor):assert torch.equal(getattr(x,field.name),getattr(y,field.name))
        else:assert getattr(x,field.name)==getattr(y,field.name)
    for p,q in zip(a.model.parameters(),b.model.parameters()):assert torch.equal(p,q)
    assert torch.equal(a.shuffle_rng.get_state(),b.shuffle_rng.get_state())
    c=make_ppo();c.shuffle_rng.manual_seed(11)
    d=make_ppo()
    assert not torch.equal(torch.randperm(20,generator=c.shuffle_rng),torch.randperm(20,generator=d.shuffle_rng))


def test_freshness_only_before_first_step_and_repeated_rollout_rejected():
    learner=make_ppo(epochs=3,minibatch_size=2);data=rollout_for(learner.model)
    data.value.requires_grad_();data.log_prob.requires_grad_()
    assert learner.learn(data).optimizer_steps==9
    assert data.value.grad is data.log_prob.grad is None
    with pytest.raises(ValueError,match='On-policy'):learner.learn(data)
    data=rollout_for(learner.model)
    with torch.no_grad():learner.model.estimator.policy.network.bias[0]+=1
    parameters=[p.clone() for p in learner.model.parameters()]
    state=deepcopy(learner.optimizer.state_dict());rng=learner.shuffle_rng.get_state().clone()
    with pytest.raises(ValueError,match='On-policy'):learner.learn(data)
    for p,q in zip(learner.model.parameters(),parameters):assert torch.equal(p,q)
    assert torch.equal(rng,learner.shuffle_rng.get_state())
    for key,value in state['state'].items():
        for name,tensor in value.items():assert torch.equal(tensor,learner.optimizer.state_dict()['state'][key][name])


def test_a2c_gae_path_uses_explicit_frozen_actor_advantage(monkeypatch):
    ac=make_learner();learner=L.A2C(ac.model,ac.optimizer,T.GAE(.9,.5),ac.update_rule)
    data=rollout_for(learner.model);captured=[]
    compute=learner.update_rule.compute
    def capture(*args,**kwargs):captured.append((args,kwargs));return compute(*args,**kwargs)
    monkeypatch.setattr(learner.update_rule,'compute',capture)
    result=learner.learn(data)
    args,kwargs=captured[0]
    assert 'advantage' in kwargs and not kwargs['advantage'].requires_grad
    torch.testing.assert_close(result.actor_loss,-(args[0]*kwargs['advantage']).mean())
    assert not result.total_loss.requires_grad


@pytest.mark.parametrize('field,convert',[
 ('action',lambda x:x.float()),('reward',lambda x:x.long()),('terminated',lambda x:x.float()),
 ('log_prob',lambda x:x+1),('value',lambda x:x*float('nan')),('truncated',lambda x:x.unsqueeze(-1)),
 ('observation',lambda x:x[:1]),('next_observation',lambda x:x.to('meta'))])
def test_bad_rollout_fails_before_optimizer(field,convert):
    learner=make_ppo();data=rollout_for(learner.model)
    with pytest.raises((TypeError,ValueError)):learner.learn(replace(data,**{field:convert(getattr(data,field))}))
    assert not learner.optimizer.state


@pytest.mark.parametrize('kwargs',[{'epochs':0},{'epochs':True},{'minibatch_size':-1},{'seed':False},{'normalize_advantage':1}])
def test_bad_configuration(kwargs):
    one=make_learner()
    with pytest.raises((TypeError,ValueError)):L.PPO(one.model,one.optimizer,T.GAE(),U.PPO(),**kwargs)


def test_async_rollout_gae_ppo_final_observations_and_worker_cleanup(monkeypatch):
    from modular_rl import Model, representations as R, estimators as E, selectors as S, buffers as B
    from modular_rl.environment import final_observations
    from tests.test_vector_rollout import make_vector
    import numpy as np

    torch.manual_seed(7)
    model=Model(R.Identity(),E.ActorCritic(E.CategoricalPolicy(3,2),E.StateValue(3)),S.Categorical())
    with torch.no_grad():
        for parameter in model.parameters():parameter.zero_()
        model.estimator.value.network.weight[0,1]=1
    learner=L.PPO(model,torch.optim.SGD(model.parameters(),lr=1e-10),T.GAE(.9,.95),U.PPO(),
                  epochs=2,minibatch_size=5,seed=7)
    buffer=B.Rollout(4,3);env=make_vector();processes=env.processes.copy()
    captured=[];generate=learner.target_generator.generate
    def capture(*args):
        result=generate(*args);captured.append((args,result));return result
    monkeypatch.setattr(learner.target_generator,'generate',capture)
    try:
        observation,_=env.reset(seed=[7,11,19])
        for _ in range(4):
            with torch.no_grad():
                estimate=model(observation);selected=model.action_selector.select(estimate)
            returned,reward,terminated,truncated,info=env.step(selected.action.numpy())
            actual=final_observations(returned,terminated,truncated,info)
            buffer.add(observation,selected.action,reward,actual,terminated,truncated,
                       value=estimate.values.v,log_prob=selected.log_prob)
            observation=returned
        data=buffer.get();before=[p.clone() for p in model.parameters()]
        result=learner.learn(data)
        assert result.optimizer_steps==6 and len(captured)==1
        args,generated=captured[0]
        assert all(not item.requires_grad for item in args)
        assert generated.returns[1,0].item()==pytest.approx(data.reward[1,0].item())
        assert generated.returns[2,1].item()==pytest.approx(data.reward[2,1].item()+.9*3)
        assert any(not torch.equal(p,q) for p,q in zip(model.parameters(),before))
    finally:env.close(timeout=5)
    assert all(not process.is_alive() for process in processes)
