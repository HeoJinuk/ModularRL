"""Time-major ownership, boundaries and device/dtype contracts."""

from dataclasses import fields, FrozenInstanceError

import numpy as np
import pytest
import torch

from modular_rl import buffers as B
from modular_rl.component.buffer import RolloutBuffer
from modular_rl.data import RolloutBatch


def step(n=2, dtype=np.float32):
    return dict(observation=np.zeros((n, 3), dtype=dtype), action=np.arange(n, dtype=np.int32),
                reward=np.ones(n, dtype=dtype), next_observation=np.ones((n, 3), dtype=dtype),
                terminated=np.zeros(n, dtype=bool), truncated=np.zeros(n, dtype=bool),
                value=torch.ones(n, requires_grad=True), log_prob=torch.zeros(n, requires_grad=True))


def test_snapshots_shapes_order_and_independent_get():
    buffer = B.Rollout(2, 2)
    first = step(); first['terminated'][0] = True; first['truncated'][1] = True
    buffer.add(**first)
    for value in first.values():
        if isinstance(value, np.ndarray): value.fill(99)
        else:
            with torch.no_grad(): value.fill_(99)
    second = step(); second['observation'].fill(7); second['next_observation'].fill(8)
    buffer.add(**second)
    rollout = buffer.get()
    assert isinstance(rollout, RolloutBatch) and B.Rollout is RolloutBuffer
    assert rollout.observation.shape == (2, 2, 3)
    assert rollout.reward.shape == rollout.value.shape == rollout.log_prob.shape == (2, 2)
    assert rollout.action.dtype == torch.int32
    assert rollout.terminated.tolist() == [[True, False], [False, False]]
    assert rollout.truncated.tolist() == [[False, True], [False, False]]
    assert rollout.observation[:, 0, 0].tolist() == [0, 7]
    for field in fields(rollout):
        tensor = getattr(rollout, field.name)
        assert tensor.device.type == 'cpu' and tensor.grad_fn is None and not tensor.requires_grad
    rollout.reward.zero_(); rollout.observation.zero_()
    assert buffer.get().reward.sum() == 4
    assert buffer.get().observation[1, 0, 0] == 7
    with pytest.raises(FrozenInstanceError): rollout.reward = torch.zeros(1)


def test_full_partial_clear_and_reuse():
    buffer = B.Rollout(1, 2)
    with pytest.raises(ValueError, match='complete'): buffer.get()
    buffer.add(**step())
    with pytest.raises(ValueError, match='full'): buffer.add(**step())
    assert len(buffer) == 1
    buffer.clear(); assert len(buffer) == 0
    changed = step(dtype=np.float64); buffer.add(**changed)
    assert buffer.get().observation.dtype == torch.float64


def test_continuity_only_for_nonboundary_streams():
    buffer = B.Rollout(2, 2)
    first = step(); first['truncated'][0] = True; buffer.add(**first)
    second = step(); second['observation'][0] = 99
    with pytest.raises(ValueError, match='Nonboundary'): buffer.add(**second)
    assert len(buffer) == 1
    second['observation'][1] = 1; buffer.add(**second)
    assert buffer.get().observation[1, 0, 0] == 99


@pytest.mark.parametrize('field,bad',[
    ('observation', np.zeros((1, 3))), ('next_observation', np.zeros((2, 3), dtype=np.float64)),
    ('action', np.zeros(2)), ('reward', np.ones((2, 1))), ('terminated', np.zeros(2, dtype=int)),
    ('truncated', np.zeros((1, 2), dtype=bool)), ('value', torch.ones(2, dtype=torch.long)),
    ('log_prob', None),
])
def test_invalid_vector_step(field, bad):
    data = step(); data[field] = bad
    with pytest.raises((ValueError, TypeError, RuntimeError)): B.Rollout(1, 2).add(**data)


@pytest.mark.parametrize('kwargs',[dict(horizon=0,n_envs=2),dict(horizon=True,n_envs=2),dict(horizon=1,n_envs=-1)])
def test_invalid_dimensions(kwargs):
    with pytest.raises(ValueError): B.Rollout(**kwargs)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_to_and_source_tensor_snapshot(dtype, device):
    data = step(); data['observation'] = torch.ones(2, 3, dtype=torch.int16, device=device)
    data['next_observation'] = data['observation'] + 1
    data['reward'] = torch.ones(2, dtype=dtype, device=device, requires_grad=True)
    buffer = B.Rollout(1, 2); buffer.add(**data)
    rollout = buffer.get().to(device, dtype=dtype)
    assert rollout.observation.dtype == torch.int16
    assert rollout.action.dtype == torch.int32 and rollout.terminated.dtype == torch.bool
    assert rollout.reward.dtype == rollout.value.dtype == rollout.log_prob.dtype == dtype
    assert rollout.reward.device.type == device
    assert all(not getattr(rollout, f.name).requires_grad for f in fields(rollout))
    with pytest.raises(ValueError): rollout.to(dtype=torch.long)
