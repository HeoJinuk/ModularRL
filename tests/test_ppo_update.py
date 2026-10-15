"""Clipped surrogate signs, gradients and diagnostics from independent numbers."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from modular_rl import updates as U
from tests.test_actor_critic import DEVICES


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('device',DEVICES)
def test_clipping_positive_negative_advantage_and_gradients(dtype,device):
    ratios=torch.tensor([1.,1.1,1.4,.6,1.4,.6],dtype=dtype,device=device)
    new=ratios.log().requires_grad_();old=torch.zeros_like(new,requires_grad=True)
    adv=torch.tensor([2.,2.,2.,2.,-2.,-2.],dtype=dtype,device=device,requires_grad=True)
    value=torch.ones_like(new,requires_grad=True);returns=torch.full_like(new,3.,requires_grad=True)
    entropy=torch.full_like(new,.4,requires_grad=True)
    result=U.PPO().compute(new,old,value,returns,adv,entropy)
    # min([2,2.2,2.8,1.2,-2.8,-1.2], [2,2.2,2.4,1.6,-2.4,-1.6])
    assert result.policy_loss.item()==pytest.approx(-3.4/6,abs=1e-6)
    assert result.value_loss.item()==2
    assert result.total_loss.item()==pytest.approx(-3.4/6+1-.004,abs=1e-6)
    torch.testing.assert_close(result.approx_kl,(ratios-1-ratios.log()).mean())
    assert result.clip_fraction.item()==pytest.approx(4/6)
    result.total_loss.backward()
    torch.testing.assert_close(new.grad,torch.tensor([-2,-2.2,0,-1.2,2.8,0],dtype=dtype,device=device)/6)
    assert old.grad is adv.grad is None
    torch.testing.assert_close(value.grad,torch.full_like(value,-1/6))
    torch.testing.assert_close(returns.grad,torch.full_like(value,1/6))
    torch.testing.assert_close(entropy.grad,torch.full_like(value,-.01/6))
    assert not result.approx_kl.requires_grad and not result.clip_fraction.requires_grad
    with pytest.raises(FrozenInstanceError):result.optimizer_steps=9


def test_ratio_one_actor_gradient_matches_explicit_actor_critic_advantage():
    log=torch.tensor([-.7,-.3],dtype=torch.float64,requires_grad=True)
    value=torch.tensor([2.,1.],dtype=torch.float64,requires_grad=True)
    target=torch.tensor([5.,-2.],dtype=torch.float64)
    entropy=torch.tensor([.3,.4],dtype=torch.float64,requires_grad=True)
    advantage=target-value.detach()
    ppo=U.PPO().compute(log,log.detach(),value,target,advantage,entropy)
    ac=U.ActorCritic().compute(log,value,target,entropy)
    assert ppo.approx_kl.item()==0 and ppo.clip_fraction.item()==0
    for x,y in zip(torch.autograd.grad(ppo.total_loss,(log,value,entropy),retain_graph=True),
                   torch.autograd.grad(ac.total_loss,(log,value,entropy))):
        torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_explicit_actor_advantage_is_separate_from_critic_error():
    log=torch.tensor([-.7],requires_grad=True);value=torch.tensor([2.],requires_grad=True)
    adv=torch.tensor([7.],requires_grad=True)
    result=U.ActorCritic().compute(log,value,torch.tensor([5.]),torch.tensor([.4]),advantage=adv)
    assert result.actor_loss.item()==pytest.approx(4.9)
    assert result.td_error.item()==3 and result.critic_loss.item()==4.5
    result.actor_loss.backward();assert log.grad.item()==-7
    assert value.grad is adv.grad is None


@pytest.mark.parametrize('index,replacement',[(0,torch.zeros(2,1)),(1,torch.zeros(2,dtype=torch.long)),
 (2,torch.zeros(2,dtype=torch.float64)),(3,torch.tensor([float('inf'),1.])),
 (4,torch.tensor([float('nan'),1.])),(5,torch.zeros(2,device='meta')),
 (0,torch.tensor([1000.,0.])),(3,torch.full((2,),1e30))])
def test_invalid_shapes_dtypes_devices_nonfinite_and_overflow(index,replacement):
    inputs=[torch.zeros(2) for _ in range(6)];inputs[index]=replacement
    with pytest.raises((ValueError,TypeError)):U.PPO().compute(*inputs)


@pytest.mark.parametrize('keyword,value',[('clip_epsilon',1),('clip_epsilon',-.1),('clip_epsilon',True),
 ('value_coef',float('nan')),('entropy_coef',float('inf'))])
def test_invalid_configuration(keyword,value):
    with pytest.raises(ValueError):U.PPO(**{keyword:value})
