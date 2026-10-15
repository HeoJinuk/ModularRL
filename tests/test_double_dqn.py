"""Double DQN changes only bootstrap selection; DQN optimization remains shared."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

from modular_rl import Model, estimators as E, learners as L, targets as T, updates as U
from modular_rl.data import Transition, TransitionBatch, SequenceBatch


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))]


def learner(kind=L.DoubleDQN, dtype=torch.float64, device='cpu'):
    model = Model(nn.Embedding.from_pretrained(torch.eye(3), freeze=True),
                  E.DiscreteQ(3, 2, network=nn.Linear(3,2,bias=False))).to(dtype=dtype,device=device)
    target = deepcopy(model)
    with torch.no_grad():
        model.estimator.network.weight.copy_(torch.tensor([[2,1,0],[0,4,3]]))
        target.estimator.network.weight.copy_(torch.tensor([[0,10,2],[0,2,1]]))
    return kind(model,target,torch.optim.SGD(model.parameters(),lr=.01,momentum=.9),T.TD0(.9),U.TDLoss())


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('terminated,truncated',[(False,False),(False,True),(True,False),(True,True)])
def test_online_selects_four_target_evaluates_two_not_ten(dtype,device,terminated,truncated):
    double, dqn = learner(dtype=dtype,device=device), learner(L.DQN,dtype,device)
    batch = Transition(0,0,1.,1,terminated,truncated).to_batch()
    target_before = deepcopy(double.target_model.state_dict())
    actual, ordinary = double.learn(batch), dqn.learn(batch)
    expected = 1 if terminated else 1+.9*2
    torch.testing.assert_close(actual.td_error,torch.tensor([expected-2],dtype=dtype,device=device))
    torch.testing.assert_close(ordinary.td_error,torch.tensor([(1 if terminated else 10)-2],dtype=dtype,device=device))
    assert actual.loss.shape == () and actual.td_error.shape == (1,)
    assert actual.loss.dtype == dtype and actual.loss.device.type == device
    assert not actual.loss.requires_grad and not actual.td_error.requires_grad
    assert all(p.grad is None for p in double.target_model.parameters())
    assert any(p.grad is not None for p in double.online_model.parameters())
    for key,value in target_before.items():
        assert torch.equal(double.target_model.state_dict()[key],value)


def test_batched_action_gather_and_terminal_row_order():
    agent = learner()
    batch = TransitionBatch(torch.tensor([0,2]),torch.tensor([0,1]),torch.tensor([1.,2.]),
                            torch.tensor([1,1]),torch.tensor([False,True]),torch.tensor([True,False]))
    result = agent.learn(batch)
    torch.testing.assert_close(result.td_error,torch.tensor([.8,-1.],dtype=torch.float64))


@pytest.mark.parametrize('terminated',[False,True])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_n_step_reuses_valid_horizon_and_reward_generator(terminated,dtype):
    agent = learner(dtype=dtype)
    batch = SequenceBatch(
        observation=torch.tensor([[0,1,2],[2,0,0]]), action=torch.tensor([[0,1,0],[1,0,0]]),
        reward=torch.tensor([[1.,2.,3.],[4.,999.,999.]]), next_observation=torch.tensor([[1,2,1],[1,0,0]]),
        terminated=torch.tensor([[False,False,terminated],[False,False,False]]),
        truncated=torch.tensor([[False,False,not terminated],[True,False,False]]),
        valid=torch.tensor([[True,True,True],[True,False,False]]),horizon=torch.tensor([3,1]))
    result = agent.learn(batch)
    target = 1+.9*2+.9**2*3+(0 if terminated else .9**3*2)
    torch.testing.assert_close(result.td_error,torch.tensor([target-2,4+.9*2-3],dtype=dtype))


def test_positional_n_step_generator_and_identical_networks_equal_dqn():
    double, dqn = learner(), learner(L.DQN)
    double.sync_target(); dqn.sync_target()
    generator = T.NStep(.9)
    double = L.DoubleDQN(double.online_model,double.target_model,double.optimizer,generator,U.TDLoss())
    assert double.n_step_target_generator is generator
    batch = Transition(0,0,1.,1,False,False).to_batch()
    a,b = double.learn(batch),dqn.learn(batch)
    assert torch.equal(a.td_error,b.td_error) and torch.equal(a.loss,b.loss)
    for p,q in zip(double.online_model.parameters(),dqn.online_model.parameters()):
        assert torch.equal(p,q)


def test_bootstrap_eval_preserves_dropout_rng_batchnorm_buffers_and_mixed_modes():
    torch.manual_seed(7)
    model = Model(nn.Sequential(nn.Linear(2,4),nn.BatchNorm1d(4),nn.Dropout(.5)), E.DiscreteQ(4,2))
    target = deepcopy(model)
    agent = L.DoubleDQN(model,target,torch.optim.SGD(model.parameters(),lr=.01),T.TD0(.9),U.TDLoss())
    model.train(); model.representation[0].eval()  # mixed caller modes
    modes = [m.training for m in model.modules()]
    observations = torch.tensor([[1.,2.],[3.,4.]])
    rng = torch.get_rng_state()
    snapshot = deepcopy(model.state_dict())
    model(observations)
    expected_rng = torch.get_rng_state()
    model.load_state_dict(snapshot);torch.set_rng_state(rng)
    calls = []
    model.register_forward_hook(lambda module,args,result: calls.append((module.training,torch.is_grad_enabled())))
    batch = TransitionBatch(observations,torch.tensor([0,1]),torch.tensor([1.,2.]),observations+1,
                            torch.tensor([False,False]),torch.tensor([False,True]))
    agent.learn(batch)
    assert calls == [(True,True),(False,False)]
    assert [m.training for m in model.modules()] == modes
    assert model.representation[1].num_batches_tracked.item() == 1
    assert target.representation[1].num_batches_tracked.item() == 0
    assert torch.equal(expected_rng,torch.get_rng_state())


def test_checkpoint_optimizer_resume_and_hard_sync():
    first = learner()
    batch = Transition(0,0,1.,1,False,False).to_batch()
    first.learn(batch)
    restored = learner()
    restored.online_model.load_state_dict(deepcopy(first.online_model.state_dict()))
    restored.target_model.load_state_dict(deepcopy(first.target_model.state_dict()))
    restored.optimizer.load_state_dict(deepcopy(first.optimizer.state_dict()))
    a,b = first.learn(batch),restored.learn(batch)
    assert torch.equal(a.loss,b.loss)
    for p,q in zip(first.online_model.parameters(),restored.online_model.parameters()):
        assert torch.equal(p,q)
    restored.sync_target()
    for p,q in zip(restored.online_model.parameters(),restored.target_model.parameters()):
        assert torch.equal(p,q) and p.data_ptr() != q.data_ptr()
    assert not restored.target_model.training


def test_invalid_multi_critic_does_not_silently_choose_first(monkeypatch):
    from modular_rl.data import EstimatorOutput,ValueOutput
    agent = learner()
    monkeypatch.setattr(agent.target_model,'forward',lambda obs: EstimatorOutput(values=ValueOutput(q=(torch.ones(1,2),torch.ones(1,2)))))
    with pytest.raises(ValueError,match='exactly one'):
        agent.learn(Transition(0,0,1.,1,False,False).to_batch())
