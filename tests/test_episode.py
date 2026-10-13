"""Ordered, owned Transition snapshots and distinct natural/time-limit endings."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from modular_rl.data import Episode, Transition


def test_episode_preserves_time_order_fields_and_distinct_boundary_flags():
    steps = [Transition(1, 2, 1.0, 2, False, False, -.5),
             Transition(2, 0, 2.0, 3, False, True, None)]
    episode = Episode(steps)
    steps.clear()
    assert len(episode) == 2 and not episode.terminated and episode.truncated
    assert [t.observation.item() for t in episode.transitions] == [1, 2]
    assert [t.reward.item() for t in episode.transitions] == [1, 2]
    assert episode.transitions[0].action.item() == 2
    assert episode.transitions[0].log_prob.item() == -.5 and episode.transitions[1].log_prob is None
    assert episode.transitions[1].next_observation.item() == 3
    assert not episode.transitions[0].terminated and not episode.transitions[0].truncated


def test_numpy_array_reuse_cannot_change_stored_episode_values_or_dtypes():
    observation = np.array([1., 2.], dtype=np.float32)
    following = np.array([3., 4.], dtype=np.float32)
    action = np.array([.25], dtype=np.float64)
    episode = Episode((Transition(observation, action, np.float64(5), following, True, False),))
    observation[:] = following[:] = 99
    action[:] = 99
    stored, = episode.transitions
    torch.testing.assert_close(stored.observation, torch.tensor([1., 2.]))
    torch.testing.assert_close(stored.next_observation, torch.tensor([3., 4.]))
    torch.testing.assert_close(stored.action, torch.tensor([.25], dtype=torch.float64))
    assert stored.reward.dtype == torch.float64 and stored.observation.dtype == torch.float32
    assert stored.terminated.dtype == stored.truncated.dtype == torch.bool


def test_tensor_snapshots_detach_all_graphs_and_do_not_alias_the_source():
    observation = torch.tensor([1., 2.], requires_grad=True)
    reward, log_prob = torch.tensor(3., requires_grad=True), torch.tensor(-.5, requires_grad=True)
    episode = Episode((Transition(observation, torch.tensor(0), reward, observation * 2, True, False, log_prob),))
    stored, = episode.transitions
    for value in (stored.observation, stored.next_observation, stored.reward, stored.log_prob):
        assert not value.requires_grad and value.grad_fn is None and value.device.type == "cpu"
    with torch.no_grad():
        reward.fill_(99)
        observation.fill_(99)
    assert stored.reward.item() == 3 and stored.observation.tolist() == [1, 2]


@pytest.mark.parametrize("terminated,truncated", [(False, False), (True, False), (False, True), (True, True)])
def test_open_terminated_truncated_and_simultaneous_endings_are_preserved(terminated, truncated):
    episode = Episode((Transition(1, 0, 1., 0, terminated, truncated),))
    assert episode.terminated == terminated and episode.truncated == truncated


@pytest.mark.parametrize("boundary", ["terminated", "truncated"])
def test_multiple_episodes_cannot_be_concatenated_after_a_boundary(boundary):
    first = replace(Transition(1, 0, 1., 2, False, False), **{boundary: True})
    with pytest.raises(ValueError, match="cannot continue"):
        Episode((first, Transition(2, 0, 2., 0, True, False)))


def test_discontinuous_or_out_of_order_transitions_are_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        Episode((Transition(1, 0, 1., 2, False, False), Transition(3, 0, 2., 0, True, False)))
    with pytest.raises(ValueError, match="at least one"):
        Episode(())
    with pytest.raises(TypeError, match="Transition records"):
        Episode(({},))


@pytest.mark.parametrize("field,value,error,message", [
    ("reward", np.array([1.]), ValueError, "scalar"),
    ("reward", True, ValueError, "real numeric"),
    ("reward", float("nan"), ValueError, "finite"),
    ("reward", 1j, ValueError, "real numeric"),
    ("reward", None, ValueError, "reward is required"),
    ("terminated", 1, TypeError, "bool scalars"),
    ("truncated", np.array([False]), TypeError, "bool scalars"),
    ("next_observation", np.array([0]), ValueError, "same shape"),
    ("next_observation", 0., ValueError, "same shape and dtype"),
])
def test_important_episode_field_contracts_are_validated(field, value, error, message):
    transition = replace(Transition(1, 0, 1., 0, True, False), **{field: value})
    with pytest.raises(error, match=message):
        Episode((transition,))
