"""N-step return mathematics and DQN compatibility with the saved legacy path."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from examples.gridworld_dyna_dqn import learn_real_n_step, make_n_step_experience
from modular_rl.component.buffer import SequenceReplayBuffer
from modular_rl.component.target_generator import NStepTargetGenerator, TD0TargetGenerator
from modular_rl.data import Transition
from tests.test_dqn_optimization import DEVICES, make_learner, make_batch, legacy_learn
from tests.test_dyna_dqn_composition import assert_tree_equal, one_cpu_thread


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('gamma', [0., .9, 1.])
def test_returns_masks_short_horizons_and_gradients(device, dtype, gamma):
    rewards = torch.tensor([[1., 2., 3.], [1., 2., float('nan')], [4., float('inf'), 999.]], device=device, dtype=dtype, requires_grad=True)
    valid = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]], device=device, dtype=torch.bool)
    terminated = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], device=device, dtype=torch.bool)
    next_value = torch.tensor([4., float('nan'), 10.], device=device, dtype=dtype, requires_grad=True)
    target = NStepTargetGenerator(gamma).generate(rewards, next_value, terminated, valid)
    expected = torch.tensor([1+gamma*2+gamma**2*3+gamma**3*4, 1+gamma*2, 4+gamma*10], device=device, dtype=dtype)
    torch.testing.assert_close(target, expected)
    if gamma == .9: assert target[0].item() == pytest.approx(8.146)
    target.sum().backward()
    torch.testing.assert_close(rewards.grad, torch.tensor([[1, gamma, gamma**2], [1, gamma, 0], [1, 0, 0]], device=device, dtype=dtype))
    torch.testing.assert_close(next_value.grad, torch.tensor([gamma**3, 0, gamma], device=device, dtype=dtype))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('gamma', [0., .9, 1.])
def test_n1_is_bit_exact_td0(dtype, gamma):
    torch.manual_seed(1)
    r, v, d = torch.randn(100, dtype=dtype), torch.randn(100, dtype=dtype), torch.arange(100) % 2 == 0
    new = NStepTargetGenerator(gamma).generate(r[:, None], v, d[:, None], torch.ones((100, 1), dtype=torch.bool))
    assert torch.equal(new, TD0TargetGenerator(gamma).generate(r, v, d))


@pytest.mark.parametrize('gamma', [-1, 1.1, True, float('nan'), float('inf')])
def test_bad_gamma(gamma):
    with pytest.raises(ValueError): NStepTargetGenerator(gamma)


@pytest.mark.parametrize('problem', ['rewards_shape', 'next_shape', 'mask_shape', 'dtype', 'bool', 'device', 'empty', 'holes', 'no_valid', 'interior_terminal'])
def test_invalid_return_contract(problem):
    r, v = torch.ones(2, 3), torch.ones(2)
    d, valid = torch.zeros(2, 3, dtype=torch.bool), torch.ones(2, 3, dtype=torch.bool)
    if problem == 'rewards_shape': r = r[:, 0]
    elif problem == 'next_shape': v = v[:, None]
    elif problem == 'mask_shape': d = d[:, 0]
    elif problem == 'dtype': v = v.double()
    elif problem == 'bool': d = d.long()
    elif problem == 'device': v = v.to('meta')
    elif problem == 'empty': r, v, d, valid = r[:0], v[:0], d[:0], valid[:0]
    elif problem == 'holes': valid[0, 1] = False
    elif problem == 'no_valid': valid[0] = False
    elif problem == 'interior_terminal': d[0, 0] = True
    with pytest.raises((ValueError, TypeError)): NStepTargetGenerator(.9).generate(r, v, d, valid)


def windows(dtype):
    result = []
    for m in (1, 3, 5):
        window = []
        for i in range(m):
            x = torch.tensor([i/10, (i+1)/10], dtype=dtype)
            window.append(Transition(x, i % 3, torch.tensor((i-1)/3, dtype=dtype), torch.tensor([(i+1)/10, (i+2)/10], dtype=dtype),
                                     m == 3 and i == m-1, m == 1 and i == m-1))
        result.append(window)
    return result


def sequence_batch(dtype):
    replay = SequenceReplayBuffer(30, seed=3)
    for window in windows(dtype):
        # Last long window is open, hence only its complete n=5 row is eligible.
        for t in window: replay.add(t)
    batch = replay.sample(5, length=5)
    # Exactly tails 1,3,2,1 and one length-5 start, in replay's random order.
    return batch


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('optimizer', ['adam', 'sgd'])
@pytest.mark.parametrize('batch_size', [1, 5])
def test_sequence_dqn_matches_legacy_targets_online_target_and_optimizer(device, dtype, optimizer, batch_size):
    torch.manual_seed(7)
    new = make_learner(dtype, device, optimizer)
    old = deepcopy(new)
    from examples.gridworld_dyna_dqn import create_learner
    new = create_learner(n_states=16, n_actions=3)
    new.online_model.to(device=device, dtype=dtype)
    new.target_model.to(device=device, dtype=dtype)
    factory = torch.optim.Adam if optimizer == 'adam' else torch.optim.SGD
    new.optimizer = factory(new.online_model.parameters(), lr=.01)
    old = deepcopy(new)
    replay = SequenceReplayBuffer(30, seed=3)
    for m in (1, 3, 5):
        for i in range(m):
            replay.add(i, i % 3, torch.tensor((i-1)/3, dtype=dtype), i+1,
                       m == 3 and i == m-1, m == 1 and i == m-1)
    batch = replay.sample(batch_size, length=5)
    records = []
    for row, m in enumerate(batch.horizon.tolist()):
        window = [Transition(*(getattr(batch, name)[row, t] for name in
                              ('observation', 'action', 'reward', 'next_observation', 'terminated', 'truncated'))).to_batch()
                  for t in range(m)]
        records.append(make_n_step_experience(window, old.target_generator, dtype=dtype))
    target_before = deepcopy(new.target_model.state_dict())
    calls = []
    optimize = new.optimize
    new.optimize = lambda p, t: (calls.append(t.detach().clone()) or optimize(p, t))
    actual, expected = new.learn(batch), learn_real_n_step(old, records)
    assert len(calls) == 1
    assert_tree_equal(actual, expected)
    assert_tree_equal(new.online_model.state_dict(), old.online_model.state_dict())
    assert_tree_equal(new.optimizer.state_dict(), old.optimizer.state_dict())
    assert_tree_equal(new.target_model.state_dict(), target_before)
    assert all(p.grad is None for p in new.target_model.parameters())
    # Planning retains the same optimizer/target but uses a separate one-step horizon.
    from modular_rl.data import TransitionBatch
    planning = TransitionBatch(torch.arange(4), torch.arange(4) % 3, torch.ones(4, dtype=dtype),
                               torch.arange(4)+1, torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool))
    actual, expected = new.learn(planning), legacy_learn(old, planning)
    assert_tree_equal(actual, expected)
    assert_tree_equal(new.optimizer.state_dict(), old.optimizer.state_dict())
    new.sync_target(); old.sync_target()
    assert_tree_equal(new.target_model.state_dict(), old.target_model.state_dict())


def test_sequence_endpoint_uses_current_target_and_terminal_mask():
    learner = make_learner()
    batch = sequence_batch(torch.float32)
    with torch.no_grad():
        for p in learner.target_model.parameters(): p.zero_()
        learner.target_model.estimator.network[-1].bias.fill_(4.)
    captured = []
    optimize = learner.optimize
    learner.optimize = lambda p, t: (captured.append(t.clone()) or optimize(p, t))
    learner.learn(batch)
    expected = NStepTargetGenerator(.9).generate(batch.reward, torch.full((5,), 4.), batch.terminated, batch.valid)
    assert torch.equal(captured[0], expected)


@pytest.mark.parametrize('problem', ['horizon', 'mask', 'action', 'short_open', 'interior_boundary'])
def test_bad_sequence_is_rejected_before_optimizer_mutation(problem):
    learner = make_learner()
    learner.learn(make_batch(4))
    before = deepcopy(learner.optimizer.state_dict())
    batch = sequence_batch(torch.float32)
    if problem == 'horizon': batch = replace(batch, horizon=batch.horizon + 1)
    elif problem == 'mask': batch.valid[0, 0] = False
    elif problem == 'action': batch = replace(batch, action=batch.action.float())
    elif problem == 'short_open': batch = replace(batch, terminated=torch.zeros_like(batch.terminated), truncated=torch.zeros_like(batch.truncated))
    elif problem == 'interior_boundary': batch.truncated[-1, 0] = True; batch.valid[-1] = True; batch.horizon[-1] = 5
    with pytest.raises((ValueError, TypeError)): learner.learn(batch)
    assert_tree_equal(before, learner.optimizer.state_dict())
