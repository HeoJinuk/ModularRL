"""GAE arithmetic, boundary masks and caller-owned gradient policy."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from modular_rl import targets as T
from tests.test_actor_critic import DEVICES


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('gamma', [0., .9, 1.])
@pytest.mark.parametrize('lam', [0., .5, .95, 1.])
def test_forward_sum_oracle_and_lambda_endpoints(dtype, device, gamma, lam):
    reward = torch.arange(20, dtype=dtype, device=device).reshape(5,4)/10
    value = torch.arange(20, dtype=dtype, device=device).reshape(5,4)/7
    following = torch.cat((value[1:], value[-1:]+1))
    term = torch.zeros(5,4,dtype=torch.bool,device=device); trunc = term.clone()
    term[1,0] = True; term[-1,2] = True
    trunc[2,1] = True; trunc[-1,3] = True; trunc[1,0] = True
    result = T.GAE(gamma,lam).generate(reward,value,following,term,trunc)
    delta = reward+gamma*following.masked_fill(term,0)-value
    expected = torch.zeros_like(value)
    for env in range(4):
        for start in range(5):
            for end in range(start,5):
                expected[start,env] += (gamma*lam)**(end-start)*delta[end,env]
                if term[end,env] or trunc[end,env]: break
    torch.testing.assert_close(result.advantage,expected)
    torch.testing.assert_close(result.returns,expected+value)
    if lam==0: torch.testing.assert_close(result.advantage,delta)
    if lam==1:
        torch.testing.assert_close(result.returns,T.RolloutReturn(gamma).generate(reward,following,term,trunc))
    assert result.returns.shape == (5,4)
    assert result.returns.dtype==dtype and result.returns.device.type==device


def test_hand_calculated_lambda_and_no_trace_across_reset():
    reward=torch.tensor([[1.],[2.],[3.]],dtype=torch.float64)
    value=torch.tensor([[2.],[4.],[6.]],dtype=torch.float64)
    next_value=torch.tensor([[4.],[6.],[8.]],dtype=torch.float64)
    term=torch.zeros(3,1,dtype=torch.bool);trunc=term.clone()
    actual=T.GAE(.9,.95).generate(reward,value,next_value,term,trunc)
    # deltas [2.6,3.4,4.2], recursive multiplier .855.
    torch.testing.assert_close(actual.advantage,torch.tensor([[8.577305],[6.991],[4.2]],dtype=torch.float64))
    trunc[1]=True
    before=T.GAE(.9,.95).generate(reward,value,next_value,term,trunc)
    reward[-1]=1e6
    after=T.GAE(.9,.95).generate(reward,value,next_value,term,trunc)
    assert torch.equal(before.advantage[:2],after.advantage[:2])
    assert before.advantage[1].item()==pytest.approx(3.4)  # final V bootstraps
    term[1]=True
    both=T.GAE(.9,.95).generate(reward,value,next_value,term,trunc)
    assert both.advantage[1].item()==-2  # no bootstrap, term wins


@pytest.mark.parametrize('terminated,truncated',[(False,False),(False,True),(True,False),(True,True)])
def test_t1_n1_gradient_preservation_and_terminal_nan(terminated,truncated):
    reward=torch.tensor([[1.]],requires_grad=True)
    value=torch.tensor([[2.]],requires_grad=True)
    following=torch.tensor([[float('nan') if terminated else 4.]],requires_grad=True)
    flags=[torch.tensor([[flag]]) for flag in (terminated,truncated)]
    result=T.GAE(.9,.95).generate(reward,value,following,*flags)
    assert result.advantage.item()==pytest.approx(-1 if terminated else 2.6)
    result.advantage.sum().backward()
    assert reward.grad.item()==1 and value.grad.item()==-1
    assert following.grad.item()==pytest.approx(0 if terminated else .9)
    with pytest.raises(FrozenInstanceError): result.returns=value


@pytest.mark.parametrize('kwargs',[{'gamma':True},{'gamma':-1},{'gamma':float('nan')},
                                   {'lam':1.1},{'lam':float('inf')},{'lam':False}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):T.GAE(**kwargs)


@pytest.mark.parametrize('index,replacement',[(0,torch.ones(3)),(1,torch.ones(3,1,dtype=torch.float64)),
 (2,torch.ones(3,1,dtype=torch.long)),(3,torch.ones(3,1)),(4,torch.zeros(3,1,dtype=torch.bool,device='meta'))])
def test_invalid_contract(index,replacement):
    inputs=[torch.ones(3,1) for _ in range(3)]+[torch.zeros(3,1,dtype=torch.bool) for _ in range(2)]
    inputs[index]=replacement
    with pytest.raises((TypeError,ValueError)):T.GAE().generate(*inputs)
