"""Hand-calculated actor/critic gradients and immediate on-policy learning."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from modular_rl import Model, representations as R, estimators as E, selectors as S
from modular_rl import learners as L, targets as T, updates as U
from modular_rl.data import Transition, ActorCriticOutput


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))]


def make_learner(dtype=torch.float64, device='cpu', *, gamma=.9, entropy_coef=.01):
    model = Model(nn.Linear(1,1,bias=False),E.ActorCritic(E.CategoricalPolicy(1,2),E.StateValue(1)),S.Categorical()).to(dtype=dtype,device=device)
    with torch.no_grad():
        model.representation.weight.fill_(1.)
        model.estimator.policy.network.weight.copy_(torch.tensor([[.1],[-.2]]))
        model.estimator.policy.network.bias.copy_(torch.tensor([.2,-.1]))
        model.estimator.value.network.weight.fill_(2.)
        model.estimator.value.network.bias.zero_()
    return L.ActorCritic(model,torch.optim.SGD(model.parameters(),lr=.05,momentum=.9),T.TD0(gamma),
                         U.ActorCritic(value_coef=.5,entropy_coef=entropy_coef))


def row(terminated=False,truncated=False,log_prob=None):
    return Transition(np.array([1.],dtype=np.float32),0,1.,np.array([2.],dtype=np.float32),terminated,truncated,log_prob).to_batch()


def test_loss_formula_advantage_detach_and_target_gradient_control():
    log_prob = torch.tensor([-.7],dtype=torch.float64,requires_grad=True)
    value = torch.tensor([2.],dtype=torch.float64,requires_grad=True)
    target = torch.tensor([5.],dtype=torch.float64,requires_grad=True)
    entropy = torch.tensor([.4],dtype=torch.float64,requires_grad=True)
    result = U.ActorCritic().compute(log_prob,value,target,entropy)
    assert result.actor_loss.item() == pytest.approx(2.1)
    assert result.critic_loss.item() == pytest.approx(4.5)
    assert result.total_loss.item() == pytest.approx(4.346)
    result.actor_loss.backward(retain_graph=True)
    assert log_prob.grad.item() == -3
    assert value.grad is target.grad is None  # actor advantage is stop-gradient
    log_prob.grad = None
    result.total_loss.backward()
    assert log_prob.grad.item() == -3
    assert value.grad.item() == -1.5
    assert target.grad.item() == 1.5  # UpdateRule does not silently detach target
    assert entropy.grad.item() == -.01
    with pytest.raises(FrozenInstanceError):
        result.total_loss = torch.tensor(0.)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('terminated,truncated',[(False,False),(False,True),(True,False),(True,True)])
def test_exact_target_losses_shared_gradients_and_detached_output(dtype,device,terminated,truncated):
    agent = make_learner(dtype,device)
    reference = deepcopy(agent.model)
    batch = row(terminated,truncated,torch.tensor(123.,requires_grad=True))
    # Independent formula: V(1)=2, V(2)=4, y=1+.9*4=4.6 unless terminal.
    output = reference(torch.tensor([[1.]],dtype=dtype,device=device))
    value = output.values.v
    target = value.new_tensor([1 if terminated else 4.6])
    delta = target-value
    actor = -output.policy.log_prob(torch.tensor([0],device=device))*delta.detach()
    critic = .5*delta.square()
    entropy = output.policy.entropy()
    total = (actor+.5*critic-.01*entropy).mean()
    actor_grad = torch.autograd.grad(actor.mean(),reference.representation.weight,retain_graph=True)[0]
    critic_grad = torch.autograd.grad((.5*critic).mean(),reference.representation.weight,retain_graph=True)[0]
    entropy_grad = torch.autograd.grad((-.01*entropy).mean(),reference.representation.weight,retain_graph=True)[0]
    total.backward()
    assert actor_grad.abs().sum()>0 and critic_grad.abs().sum()>0
    torch.testing.assert_close(reference.representation.weight.grad,actor_grad+critic_grad+entropy_grad)
    actual = agent.learn(batch)
    assert isinstance(actual,ActorCriticOutput)
    for name,expected in [('actor_loss',actor.mean()),('critic_loss',critic.mean()),('entropy',entropy.mean()),('total_loss',total),('td_error',delta)]:
        metric = getattr(actual,name)
        torch.testing.assert_close(metric,expected)
        assert metric.shape == ((1,) if name=='td_error' else ())
        assert metric.dtype==dtype and metric.device.type==device
        assert not metric.requires_grad and metric.grad_fn is None
    for parameter,expected in zip(agent.model.parameters(),reference.parameters()):
        torch.testing.assert_close(parameter.grad,expected.grad)
        torch.testing.assert_close(parameter,expected-.05*expected.grad)
    assert batch.log_prob.item() == 123 and batch.log_prob.grad_fn is None


def test_bootstrap_no_grad_eval_and_optimizer_order(monkeypatch):
    agent=make_learner();calls=[]
    agent.model.train();agent.model.estimator.value.eval()
    modes=[m.training for m in agent.model.modules()]
    agent.model.register_forward_hook(lambda module,args,out:calls.append(('forward',module.training,torch.is_grad_enabled())))
    compute=agent.update_rule.compute
    def capture(*args):
        assert not args[2].requires_grad
        result=compute(*args)
        result.total_loss.register_hook(lambda gradient:calls.append('backward'))
        calls.append('compute');return result
    monkeypatch.setattr(agent.update_rule,'compute',capture)
    for name in ('zero_grad','step'):
        original=getattr(agent.optimizer,name)
        def record(*args,_name=name,_original=original,**kwargs):
            calls.append(_name);return _original(*args,**kwargs)
        monkeypatch.setattr(agent.optimizer,name,record)
    agent.learn(row())
    assert calls==[('forward',True,True),('forward',False,False),'compute','zero_grad','backward','step']
    assert [m.training for m in agent.model.modules()]==modes


def test_detached_behavior_log_prob_is_ignored_and_entropy_bonus_increases_entropy():
    a,b=make_learner(),make_learner()
    first,second=a.learn(row(log_prob=torch.tensor(float('nan')))),b.learn(row())
    assert torch.equal(first.total_loss,second.total_loss)
    for p,q in zip(a.model.parameters(),b.model.parameters()):assert torch.equal(p,q)
    logits=torch.tensor([[2.,-2.]],requires_grad=True)
    distribution=torch.distributions.Categorical(logits=logits)
    log_prob=distribution.log_prob(torch.tensor([0]))
    value=torch.zeros(1,requires_grad=True)
    loss=U.ActorCritic(value_coef=0,entropy_coef=1).compute(log_prob,value,torch.zeros(1),distribution.entropy())
    loss.total_loss.backward()
    after=torch.distributions.Categorical(logits=logits-.1*logits.grad)
    assert after.entropy().item()>distribution.entropy().item()


def test_checkpoint_resumes_model_and_optimizer_updates_exactly():
    original=make_learner();original.learn(row())
    restored=make_learner()
    restored.model.load_state_dict(deepcopy(original.model.state_dict()))
    restored.optimizer.load_state_dict(deepcopy(original.optimizer.state_dict()))
    a,b=original.learn(row(truncated=True)),restored.learn(row(truncated=True))
    assert torch.equal(a.total_loss,b.total_loss)
    for p,q in zip(original.model.parameters(),restored.model.parameters()):assert torch.equal(p,q)


def test_cartpole_online_order_and_seed_reproducibility():
    def run():
        torch.manual_seed(7)
        env=gym.make('CartPole-v1',max_episode_steps=8)
        model=Model(R.MLP(4,(8,)),E.ActorCritic(E.CategoricalPolicy(8,2),E.StateValue(8)),S.Categorical())
        agent=L.ActorCritic(model,torch.optim.Adam(model.parameters(),lr=.001),T.TD0(.99),U.ActorCritic())
        trace=[]
        try:
            observation,_=env.reset(seed=7)
            for _ in range(16):
                selected=model.act(observation)
                assert not selected.log_prob.requires_grad
                following,reward,terminated,truncated,_=env.step(selected.action.item())
                output=agent.learn(Transition(observation,selected.action,float(reward),following,terminated,truncated,selected.log_prob).to_batch())
                trace.append((selected.action.item(),output.total_loss.item()))
                observation=following
                if terminated or truncated:observation,_=env.reset()
            return trace,model.state_dict(),torch.get_rng_state()
        finally:env.close()
    a,b=run(),run()
    assert a[0]==b[0]
    for key,value in a[1].items():assert torch.equal(value,b[1][key])
    assert torch.equal(a[2],b[2])


@pytest.mark.parametrize('bad',[
    lambda b: replace(b,observation=b.observation.repeat(2,1)),
    lambda b: replace(b,action=b.action.float()),
    lambda b: replace(b,reward=b.reward.long()),
    lambda b: replace(b,terminated=b.terminated.long()),
    lambda b: replace(b,action=b.action[:,None]),
])
def test_invalid_online_batch_contract(bad):
    with pytest.raises((ValueError,TypeError)):make_learner().learn(bad(row()))


@pytest.mark.parametrize('keyword,value',[('value_coef',True),('value_coef',-1),('entropy_coef',float('nan')),('entropy_coef',float('inf'))])
def test_invalid_coefficients(keyword,value):
    with pytest.raises(ValueError):U.ActorCritic(**{keyword:value})


@pytest.mark.parametrize('index,replacement',[(0,torch.zeros(1,1)),(1,torch.zeros(1,dtype=torch.long)),(2,torch.zeros(1,dtype=torch.float64)),(3,torch.zeros(0))])
def test_update_rule_rejects_shape_and_dtype_broadcasting(index,replacement):
    inputs=[torch.zeros(1) for _ in range(4)];inputs[index]=replacement
    with pytest.raises((ValueError,TypeError)):U.ActorCritic().compute(*inputs)


@pytest.mark.parametrize('estimator',[lambda:E.DiscreteQ(1,2),lambda:E.StateValue(1)])
def test_wrong_estimator_combination_fails_clearly(estimator):
    model=Model(R.Identity(),estimator())
    agent=L.ActorCritic(model,torch.optim.SGD(model.parameters(),lr=.01),T.TD0(.9),U.ActorCritic())
    with pytest.raises(ValueError,match='Categorical|state values|values.v'):agent.learn(row())
