"""Replay API compatibility and ordered-window ownership/boundaries."""
from dataclasses import asdict, fields

import numpy as np
import pytest
import torch

from modular_rl.component.buffer import ReplayBuffer, SequenceReplayBuffer
from modular_rl.data import SequenceBatch, Transition


def step(i, *, end=False, truncate=False):
    return Transition(i, i % 2, float(i + 1), i + 1, end, truncate)


def assert_batch_equal(a, b):
    for field in fields(a):
        x, y = getattr(a, field.name), getattr(b, field.name)
        if x is None:
            assert y is None
        else:
            assert torch.equal(x, y)


@pytest.mark.parametrize('mode', ['object', 'transition_keyword', 'positional', 'keyword', 'mixed'])
def test_add_forms_preserve_storage_and_sampling_rng(mode):
    old, new = ReplayBuffer(4, seed=9), ReplayBuffer(4, seed=9)
    for i in range(8):
        t = step(i)
        old.add(t)
        if mode == 'object': new.add(t)
        elif mode == 'transition_keyword': new.add(transition=t)
        elif mode == 'positional': new.add(*asdict(t).values())
        elif mode == 'keyword': new.add(**asdict(t))
        else: new.add(t.observation, t.action, reward=t.reward, next_observation=t.next_observation,
                      terminated=t.terminated, truncated=t.truncated)
    for _ in range(3):
        assert_batch_equal(old.sample(3), new.sample(3))


@pytest.mark.parametrize('buffer_type', [ReplayBuffer, SequenceReplayBuffer])
@pytest.mark.parametrize('args,kwargs', [
    ((), {}), ((1, 0), {}), ((step(0), 1), {}), ((step(0),), {'reward': 2.}),
    ((), {'transition': step(0), 'reward': 2.}), ((0, 0, 1., 1, False, False), {'reward': 2.}),
    ((), {'transition': 42}),
])
def test_bad_binding_does_not_change_storage_or_rng(buffer_type, args, kwargs):
    replay = buffer_type(2, seed=1)
    replay.add(step(0))
    state = str(replay._rng.bit_generator.state)
    with pytest.raises(TypeError): replay.add(*args, **kwargs)
    assert len(replay) == 1
    assert state == str(replay._rng.bit_generator.state)


@pytest.mark.parametrize('terminal,truncated', [(True, False), (False, True), (True, True)])
def test_short_closed_episode_has_all_tails_and_zero_padding(terminal, truncated):
    replay = SequenceReplayBuffer(20, seed=7)
    replay.add(step(0)); replay.add(step(1)); replay.add(step(2, end=terminal, truncate=truncated))
    assert replay.num_sequences(5) == 3
    batch = replay.sample(3, length=5)
    assert isinstance(batch, SequenceBatch)
    assert sorted(batch.horizon.tolist()) == [1, 2, 3]
    for row in range(3):
        start, n = batch.observation[row, 0].item(), batch.horizon[row].item()
        assert batch.observation[row, :n].tolist() == list(range(start, 3))
        assert batch.reward[row, :n].tolist() == list(range(start + 1, 4))
        assert batch.valid[row].tolist() == [True] * n + [False] * (5 - n)
        assert batch.reward[row, n:].count_nonzero() == 0
        assert batch.terminated[row, n-1].item() == terminal
        assert batch.truncated[row, n-1].item() == truncated


def test_open_future_is_ineligible_and_probability_queries_do_not_use_rng():
    replay = SequenceReplayBuffer(10, seed=7)
    replay.add(step(0)); replay.add(step(1))
    state = str(replay._rng.bit_generator.state)
    assert replay.num_sequences(3) == 0
    with pytest.raises(ValueError, match='eligible'): replay.sample(1, length=3)
    assert state == str(replay._rng.bit_generator.state)
    replay.add(step(2))
    assert replay.num_sequences(3) == 1
    assert replay.sample(1, length=3).horizon.item() == 3
    replay.add(step(3, truncate=True))
    assert replay.num_sequences(3) == 4


def test_ring_overwrite_excludes_removed_steps_and_does_not_cross_reset():
    replay = SequenceReplayBuffer(5, seed=11)
    for i in range(4): replay.add(step(i, end=i == 3))
    for i in range(10, 14): replay.add(step(i))
    assert len(replay) == 5
    assert replay.num_sequences(3) == 3  # terminal tail 3, full 10..12 and 11..13
    batch = replay.sample(3, length=3)
    assert set(batch.observation[:, 0].tolist()) == {3, 10, 11}
    for row, n in enumerate(batch.horizon.tolist()):
        start = batch.observation[row, 0].item()
        assert batch.observation[row, :n].tolist() == list(range(start, start+n))


@pytest.mark.parametrize('continuous', [False, True])
def test_snapshot_dtype_graph_sample_independence_and_seed(continuous):
    a, b = SequenceReplayBuffer(8, seed=7), SequenceReplayBuffer(8, seed=7)
    observation = np.array([0., 1.], dtype=np.float64) if continuous else np.array(0, dtype=np.int32)
    next_observation = observation + 1
    action = torch.tensor([.1, .2], requires_grad=True) if continuous else np.int32(1)
    log_prob = torch.tensor(-.5, requires_grad=True)
    for replay in (a, b):
        replay.add(observation, action, np.float64(2), next_observation, False, True, log_prob)
    observation[...] = 99
    if continuous:
        with torch.no_grad(): action.fill_(99)
    log_prob.detach().fill_(99)
    x, y = a.sample(1, length=3), b.sample(1, length=3)
    assert_batch_equal(x, y)
    assert x.observation.dtype == (torch.float64 if continuous else torch.int32)
    assert x.action.shape == ((1, 3, 2) if continuous else (1, 3))
    assert x.observation[0, 0].flatten()[0] == 0
    assert x.log_prob[0, 0] == -.5
    assert all(v.grad_fn is None and v.device.type == 'cpu' for v in (x.action, x.log_prob))
    x.reward.fill_(99)
    assert a.sample(1, length=3).reward[0, 0] == 2
    moved = y.to('cpu', dtype=torch.float32)
    assert moved.reward.dtype == torch.float32
    assert moved.horizon.dtype == torch.long and moved.valid.dtype == torch.bool
    if not continuous: assert moved.observation.dtype == torch.int32


def test_multiple_seeded_samples_clear_and_continuity_rejection():
    a, b = SequenceReplayBuffer(20, seed=4), SequenceReplayBuffer(20, seed=4)
    for replay in (a, b):
        for i in range(10): replay.add(step(i, end=i == 9))
    for _ in range(4): assert_batch_equal(a.sample(4, length=3), b.sample(4, length=3))
    a.clear(); b.clear()
    assert len(a) == a.num_sequences(3) == 0
    a.add(step(0)); b.add(step(0))
    with pytest.raises(ValueError, match='continuous'): a.add(step(3))
    assert len(a) == 1
    a.add(step(1, end=True)); b.add(step(1, end=True))
    assert_batch_equal(a.sample(2, length=3), b.sample(2, length=3))


@pytest.mark.parametrize('bad', [0, -1, True, 1.5])
def test_invalid_sizes(bad):
    with pytest.raises(ValueError): SequenceReplayBuffer(bad)
    replay = SequenceReplayBuffer(2)
    with pytest.raises(ValueError): replay.num_sequences(bad)
    with pytest.raises(ValueError): replay.sample(bad, length=2)
