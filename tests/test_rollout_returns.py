"""Independent numeric returns across asynchronous episode boundaries."""

import numpy as np
import pytest
import torch

from modular_rl import targets as T
from modular_rl.component.target_generator import RolloutReturnGenerator

DEVICES = ['cpu', pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))]


@pytest.mark.parametrize('gamma',[0., .9, 1.])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('device',DEVICES)
def test_three_step_return_and_td0_equivalence(gamma,dtype,device):
    reward = torch.tensor([[1.],[2.],[3.]],dtype=dtype,device=device)
    following = torch.tensor([[10.],[20.],[30.]],dtype=dtype,device=device)
    terminated = torch.zeros(3,1,dtype=torch.bool,device=device)
    rule = T.RolloutReturn(gamma)
    actual = rule.generate(reward,following,terminated,terminated)
    expected = reward.new_tensor([[1+gamma*2+gamma**2*3+gamma**3*30],[2+gamma*3+gamma**2*30],[3+gamma*30]])
    torch.testing.assert_close(actual,expected)
    assert T.RolloutReturn is RolloutReturnGenerator
    for terminal in (False,True):
        mask = terminated[:1].clone().fill_(terminal)
        torch.testing.assert_close(rule.generate(reward[:1],following[:1],mask,~mask),
                                   T.TD0(gamma).generate(reward[:1],following[:1],mask))


@pytest.mark.parametrize('gamma',[0., .9, 1.])
def test_five_by_four_independent_streams_and_mixed_boundaries(gamma):
    reward = torch.arange(1,21,dtype=torch.float64).reshape(5,4)
    following = torch.arange(101,121,dtype=torch.float64).reshape(5,4)
    terminated = torch.zeros(5,4,dtype=torch.bool); truncated = terminated.clone()
    terminated[1,0]=True; truncated[2,1]=True; truncated[-1,2]=True
    terminated[3,3]=True; truncated[3,3]=True
    actual = T.RolloutReturn(gamma).generate(reward,following,terminated,truncated)
    expected = np.zeros((5,4))
    for env in range(4):
        for start in range(5):
            discount=1.; total=0.
            for time in range(start,5):
                total += discount*reward[time,env].item()
                discount *= gamma
                if terminated[time,env]: break
                if truncated[time,env] or time==4:
                    total += discount*following[time,env].item(); break
            expected[start,env]=total
    torch.testing.assert_close(actual,torch.tensor(expected))
    changed = reward.clone(); changed[3:,1] += 10000
    other = T.RolloutReturn(gamma).generate(changed,following,terminated,truncated)
    torch.testing.assert_close(actual[:3,1],other[:3,1])


def test_terminal_nan_mask_and_generator_preserves_caller_gradient_policy():
    reward = torch.tensor([[1.],[2.]],requires_grad=True)
    following = torch.tensor([[5.],[float('nan')]],requires_grad=True)
    terminal = torch.tensor([[False],[True]])
    truncated = torch.tensor([[True],[True]])
    out = T.RolloutReturn(.9).generate(reward,following,terminal,truncated)
    torch.testing.assert_close(out,torch.tensor([[5.5],[2.]]))
    out.sum().backward()
    torch.testing.assert_close(reward.grad,torch.ones_like(reward))
    torch.testing.assert_close(following.grad,torch.tensor([[.9],[0.]]))


@pytest.mark.parametrize('index,bad',[(0,torch.zeros(3)),(1,torch.zeros(3,1,1)),
 (1,torch.zeros(3,1,dtype=torch.float64)),(2,torch.zeros(3,1)),(3,torch.zeros(3,1,dtype=torch.long)),(0,torch.zeros(0,1))])
def test_invalid_shape_dtype_and_masks(index,bad):
    args=[torch.ones(3,1),torch.ones(3,1),torch.zeros(3,1,dtype=torch.bool),torch.zeros(3,1,dtype=torch.bool)]
    args[index]=bad
    with pytest.raises((TypeError,ValueError)): T.RolloutReturn().generate(*args)


@pytest.mark.parametrize('gamma',[-.1,1.1,True,float('nan'),float('inf')])
def test_invalid_gamma(gamma):
    with pytest.raises(ValueError): T.RolloutReturn(gamma)


def test_device_mismatch_is_rejected_before_computation():
    with pytest.raises(ValueError,match='device'):
        T.RolloutReturn().generate(torch.ones(1,1),torch.ones(1,1,device='meta'),
                                  torch.zeros(1,1,dtype=torch.bool),torch.zeros(1,1,dtype=torch.bool))
