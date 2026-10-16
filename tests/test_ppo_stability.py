"""PPO options and an immutable pre-option update reference for compatibility."""
from copy import deepcopy
from dataclasses import fields
from types import MethodType

import numpy as np
import pytest
import torch
from torch import nn

from modular_rl import learners as L, targets as T, updates as U
from modular_rl.data import PPOOutput, RolloutBatch
from modular_rl.learner._evaluation import evaluating
from tests.test_ppo import make_ppo
from tests.test_a2c import rollout_for
from tests.test_actor_critic import DEVICES

# Frozen 89aa872 update order; intentionally independent of the new options.
def baseline_learn(self, rollout: RolloutBatch) -> PPOOutput:
    self._validate(rollout)
    reference = next(self.model.parameters())
    rollout = rollout.to(reference.device, dtype=reference.dtype)
    time, n_envs = rollout.reward.shape
    count = time * n_envs
    observations = rollout.observation.reshape(count, *rollout.observation.shape[2:])
    following = rollout.next_observation.reshape(count, *rollout.next_observation.shape[2:])
    actions = rollout.action.long().reshape(count)
    old_value = rollout.value.detach().clone()
    old_log_prob = rollout.log_prob.detach().reshape(count).clone()

    with torch.no_grad():
        policy, value = self._outputs(self.model(observations), count, reference)
        self._validate_freshness(old_value, old_log_prob, value, policy.log_prob(actions))
        with evaluating(self.model):
            next_value = self._outputs(self.model(following), count, reference)[1].reshape(time, n_envs)
        generated = self.target_generator.generate(
            rollout.reward, old_value, next_value, rollout.terminated, rollout.truncated,
        )
        advantage = generated.advantage.reshape(count).detach().clone()
        returns = generated.returns.reshape(count).detach().clone()
        if not torch.isfinite(advantage).all() or not torch.isfinite(returns).all():
            raise ValueError("GAE advantage and returns must be finite")
        if self.normalize_advantage:
            advantage = (advantage - advantage.mean()) / torch.sqrt(advantage.var(unbiased=False) + 1e-8)

    totals = reference.new_zeros(6)
    optimizer_steps = 0
    for _ in range(self.epochs):
        permutation = torch.randperm(count, generator=self.shuffle_rng)
        for indices in permutation.split(self.minibatch_size):
            indices = indices.to(reference.device)
            policy, value = self._outputs(self.model(observations[indices]), len(indices), reference)
            result = self.update_rule.compute(
                policy.log_prob(actions[indices]), old_log_prob[indices], value,
                returns[indices], advantage[indices], policy.entropy(),
            )
            self.optimizer.zero_grad()
            result.total_loss.backward()
            self.optimizer.step()
            metrics = (result.policy_loss, result.value_loss, result.entropy,
                       result.total_loss, result.approx_kl, result.clip_fraction)
            totals += torch.stack([metric.detach() for metric in metrics]) * len(indices)
            optimizer_steps += 1
    return PPOOutput(*(totals / (count * self.epochs)).unbind(), optimizer_steps)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('device',DEVICES)
def test_default_matches_pre_option_losses_gradients_optimizer_and_rng(dtype,device):
    a=make_ppo(dtype,device,epochs=3,minibatch_size=4)
    a.optimizer.param_groups[0]['lr']=.001
    b=make_ppo(dtype,device,epochs=3,minibatch_size=4,max_grad_norm=None,target_kl=None)
    b.model.load_state_dict(deepcopy(a.model.state_dict()))
    b.optimizer.load_state_dict(deepcopy(a.optimizer.state_dict()))
    a.learn=MethodType(baseline_learn,a)
    for _ in range(2):
        data=rollout_for(a.model);rng=torch.get_rng_state().clone()
        old,new=a.learn(data),b.learn(data)
        for name in ('policy_loss','value_loss','entropy','total_loss','approx_kl','clip_fraction'):
            torch.testing.assert_close(getattr(old,name),getattr(new,name),rtol=0,atol=0)
        assert old.optimizer_steps==new.optimizer_steps==6
        assert new.epochs_completed==3 and not new.early_stopped
        assert torch.equal(rng,torch.get_rng_state())
        assert torch.equal(a.shuffle_rng.get_state(),b.shuffle_rng.get_state())
        for p,q in zip(a.model.parameters(),b.model.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
            torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
        for x,y in zip(a.optimizer.state.values(),b.optimizer.state.values()):
            torch.testing.assert_close(x['momentum_buffer'],y['momentum_buffer'],rtol=0,atol=0)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('device',DEVICES)
def test_joint_clipping_before_step_and_unclipped_gradient(dtype,device,monkeypatch):
    learner=make_ppo(dtype,device,epochs=1,minibatch_size=6,max_grad_norm=.05)
    data=rollout_for(learner.model);records=[];calls=[]
    clip=torch.nn.utils.clip_grad_norm_;step=learner.optimizer.step
    def capture(parameters,max_norm,**kwargs):
        parameters=list(parameters)
        assert len(parameters)==len({id(p) for p in parameters})
        norm=clip(parameters,max_norm,**kwargs)
        after=torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in parameters if p.grad is not None]))
        records.append((norm.item(),after.item()));calls.append('clip')
        return norm
    def take_step(*args,**kwargs):calls.append('step');return step(*args,**kwargs)
    monkeypatch.setattr(torch.nn.utils,'clip_grad_norm_',capture)
    monkeypatch.setattr(learner.optimizer,'step',take_step)
    learner.learn(data)
    assert calls==['clip','step'] and records[0][0]>.05 and records[0][1]<=.050001


def test_clipping_uses_only_unique_trainable_optimizer_parameters(monkeypatch):
    learner=make_ppo(epochs=1,minibatch_size=6,max_grad_norm=.01)
    parameters=list(learner.model.parameters());parameters[-1].requires_grad_(False)
    learner.optimizer=torch.optim.SGD([parameters[0],parameters[1],parameters[-1]],lr=.001)
    learner.optimizer.param_groups[0]['params'].append(parameters[0])
    expected={id(parameters[0]),id(parameters[1])};seen=[]
    clip=torch.nn.utils.clip_grad_norm_
    def capture(params,*args,**kwargs):
        params=list(params);assert {id(p) for p in params}==expected and len(params)==2
        unmanaged=parameters[2].grad.clone()
        result=clip(params,*args,**kwargs)
        assert torch.equal(unmanaged,parameters[2].grad)
        seen.append(result);return result
    monkeypatch.setattr(torch.nn.utils,'clip_grad_norm_',capture)
    monkeypatch.setattr(learner.optimizer,'step',lambda:None)
    learner.learn(rollout_for(learner.model));assert len(seen)==1


def test_nonfinite_gradient_raises_before_optimizer_step(monkeypatch):
    learner=make_ppo(epochs=1,max_grad_norm=.5);steps=[]
    next(learner.model.parameters()).register_hook(lambda grad:grad*float('inf'))
    monkeypatch.setattr(learner.optimizer,'step',lambda:steps.append(1))
    with pytest.raises(RuntimeError,match='non-finite'):learner.learn(rollout_for(learner.model))
    assert not steps


def test_epoch_kl_stopping_counts_samples_preserves_targets_and_shuffle(monkeypatch):
    learner=make_ppo(epochs=4,minibatch_size=4,target_kl=1e-12)
    data=rollout_for(learner.model);saved=deepcopy(data);records=[];generated=[]
    compute=learner.update_rule.compute;generate=learner.target_generator.generate
    def capture(*args):
        result=compute(*args);records.append((len(args[0]),result));return result
    def target(*args):
        result=generate(*args);generated.append(result);return result
    monkeypatch.setattr(learner.update_rule,'compute',capture)
    monkeypatch.setattr(learner.target_generator,'generate',target)
    state=learner.shuffle_rng.get_state();reference_rng=torch.Generator();reference_rng.set_state(state)
    torch.randperm(6,generator=reference_rng)
    rng=torch.get_rng_state().clone();result=learner.learn(data)
    assert result.early_stopped and result.epochs_completed==1 and result.optimizer_steps==2
    assert len(records)==2 and len(generated)==1
    assert torch.equal(learner.shuffle_rng.get_state(),reference_rng.get_state())
    assert torch.equal(rng,torch.get_rng_state())
    for name in ('policy_loss','value_loss','entropy','total_loss','approx_kl','clip_fraction'):
        expected=sum(count*getattr(r,name).detach() for count,r in records)/6
        torch.testing.assert_close(getattr(result,name),expected,rtol=0,atol=0)
    for field in fields(data):assert torch.equal(getattr(data,field.name),getattr(saved,field.name))


def test_kl_below_threshold_has_identical_updates_and_rng_to_disabled():
    a=make_ppo(epochs=3,minibatch_size=4)
    b=make_ppo(epochs=3,minibatch_size=4,target_kl=1e9)
    b.model.load_state_dict(deepcopy(a.model.state_dict()))
    data=rollout_for(a.model);x,y=a.learn(data),b.learn(data)
    assert y.epochs_completed==3 and y.optimizer_steps==6 and not y.early_stopped
    for name in ('policy_loss','value_loss','entropy','total_loss','approx_kl','clip_fraction'):
        assert torch.equal(getattr(x,name),getattr(y,name))
    for p,q in zip(a.model.parameters(),b.model.parameters()):assert torch.equal(p,q)
    assert torch.equal(a.shuffle_rng.get_state(),b.shuffle_rng.get_state())


def test_kl_diagnostic_restores_mixed_modes_buffers_rng_and_matches_rule():
    learner=make_ppo()
    learner.model.representation=nn.Sequential(nn.Linear(1,1),nn.BatchNorm1d(1),nn.Dropout(.5)).double()
    learner.model.train();learner.model.estimator.value.eval()
    modes=[m.training for m in learner.model.modules()]
    buffers={k:v.clone() for k,v in learner.model.named_buffers()}
    obs=torch.arange(6,dtype=torch.float64).reshape(6,1);actions=torch.arange(6)%2
    with torch.no_grad(),evaluating(learner.model):old=learner.model(obs).policy.log_prob(actions)-.1
    rng=torch.get_rng_state().clone();shuffle=learner.shuffle_rng.get_state()
    actual=learner._epoch_kl(obs,actions,old,next(learner.model.parameters()))
    assert [m.training for m in learner.model.modules()]==modes
    for k,v in learner.model.named_buffers():assert torch.equal(v,buffers[k])
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(shuffle,learner.shuffle_rng.get_state())
    with torch.no_grad(),evaluating(learner.model):
        estimate=learner.model(obs);zeros=torch.zeros(6,dtype=torch.float64)
        expected=U.PPO().compute(estimate.policy.log_prob(actions),old,estimate.values.v,zeros,zeros,estimate.policy.entropy()).approx_kl
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_enabled_options_checkpoint_restores_model_optimizer_and_shuffle():
    config=dict(epochs=4,minibatch_size=4,max_grad_norm=.05,target_kl=.001)
    a=make_ppo(**config);a.learn(rollout_for(a.model));b=make_ppo(**config)
    b.model.load_state_dict(deepcopy(a.model.state_dict()))
    b.optimizer.load_state_dict(deepcopy(a.optimizer.state_dict()))
    b.shuffle_rng.set_state(a.shuffle_rng.get_state())
    data=rollout_for(a.model);x,y=a.learn(data),b.learn(data)
    for field in fields(x):
        xv,yv=getattr(x,field.name),getattr(y,field.name)
        assert torch.equal(xv,yv) if isinstance(xv,torch.Tensor) else xv==yv
    for p,q in zip(a.model.parameters(),b.model.parameters()):assert torch.equal(p,q)
    assert torch.equal(a.shuffle_rng.get_state(),b.shuffle_rng.get_state())


@pytest.mark.parametrize('keyword',['max_grad_norm','target_kl'])
@pytest.mark.parametrize('value',[0,-1,True,float('nan'),float('inf')])
def test_invalid_options(keyword,value):
    with pytest.raises(ValueError):make_ppo(**{keyword:value})


def test_numpy_option_scalars_and_legacy_output_constructor():
    make_ppo(max_grad_norm=np.float64(.5),target_kl=np.float32(.02))
    result=PPOOutput(*(torch.zeros(()) for _ in range(6)),3)
    assert result.optimizer_steps==3 and result.epochs_completed==0 and not result.early_stopped
