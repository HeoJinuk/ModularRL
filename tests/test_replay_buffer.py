"""Experience meanings, snapshot ownership, replay schemas and private RNG."""

from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest
import torch

from modular_rl.component.buffer import BaseBuffer, ReplayBuffer
from modular_rl.data import Transition, TransitionBatch


def transition(index, *, log_prob=None):
    return Transition(index, index % 2, float(index), index + 1, False, False, log_prob)


@pytest.fixture(autouse=True)
def preserve_global_rng():
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        np.random.set_state(numpy_state)


def test_transition_has_only_the_shared_step_fields_and_optional_behavior_log_prob():
    observation = np.array([1.0, 2.0], dtype=np.float32)
    step = Transition(observation, 1, 0.5, observation + 1, False, True)
    assert [item.name for item in fields(step)] == [
        "observation", "action", "reward", "next_observation", "terminated", "truncated", "log_prob"
    ]
    assert step.observation is observation
    assert step.log_prob is None
    assert not step.terminated and step.truncated
    with pytest.raises(FrozenInstanceError):
        step.action = 2


def test_replay_implements_the_minimal_buffer_contract():
    replay = ReplayBuffer(4)
    assert isinstance(replay, BaseBuffer)
    assert len(replay) == 0
    with pytest.raises(TypeError, match="abstract"):
        BaseBuffer()


@pytest.mark.parametrize("capacity", [1, 3])
def test_capacity_replaces_oldest_transitions_across_multiple_wraps(capacity):
    replay = ReplayBuffer(capacity, seed=5)
    count = capacity * 3 + 2
    for index in range(count):
        replay.add(transition(index))
        assert len(replay) == min(index + 1, capacity)
    sampled = replay.sample(capacity)
    assert set(sampled.observation.tolist()) == set(range(count - capacity, count))
    assert len(set(sampled.observation.tolist())) == capacity


@pytest.mark.parametrize("shape,dtype", [
    ((), np.int32), ((4,), np.float32), ((2, 3, 4), np.float64), ((2, 3), np.uint8),
])
def test_observation_shape_and_dtype_are_preserved_without_float_coercion(shape, dtype):
    replay = ReplayBuffer(4, seed=1)
    for index in range(4):
        observation = np.full(shape, index, dtype=dtype)
        replay.add(Transition(observation, index % 2, 0.5, observation + np.ones(shape, dtype=dtype), False, False))
    batch = replay.sample(3)
    assert isinstance(batch, TransitionBatch)
    assert batch.observation.shape == batch.next_observation.shape == (3,) + shape
    assert batch.observation.dtype == torch.from_numpy(np.zeros(shape, dtype=dtype)).dtype
    assert batch.action.shape == batch.reward.shape == batch.terminated.shape == batch.truncated.shape == (3,)
    assert batch.action.dtype == torch.long
    assert batch.terminated.dtype == batch.truncated.dtype == torch.bool
    assert batch.log_prob is None


@pytest.mark.parametrize("action", [np.int32(1), torch.tensor(1)])
def test_discrete_action_is_scalar_per_transition_and_vector_per_batch(action):
    replay = ReplayBuffer(2, seed=1)
    replay.add(replace(transition(0), action=action))
    replay.add(replace(transition(1), action=action))
    batch = replay.sample(2)
    assert batch.action.shape == (2,)
    assert batch.action.dtype == torch.as_tensor(action).dtype


@pytest.mark.parametrize("action_dim", [1, 3])
def test_continuous_action_keeps_the_action_axis_and_positive_log_density(action_dim):
    replay = ReplayBuffer(2, seed=1)
    for index in range(2):
        action = np.linspace(-0.2, 0.2, action_dim, dtype=np.float32)
        replay.add(replace(transition(index, log_prob=0.5), action=action))
    batch = replay.sample(2)
    assert batch.action.shape == (2, action_dim)
    assert batch.action.dtype == torch.float32
    torch.testing.assert_close(batch.log_prob, torch.full((2,), 0.5))


def test_add_copies_every_mutable_numpy_field_before_the_environment_reuses_it():
    observation = np.array([1.0, 2.0], dtype=np.float32)
    action = np.array([-0.25, 0.5], dtype=np.float32)
    reward = np.array(3.0, dtype=np.float32)
    next_observation = np.array([4.0, 5.0], dtype=np.float32)
    terminated, truncated = np.array(False), np.array(True)
    log_prob = np.array(-0.75, dtype=np.float32)
    replay = ReplayBuffer(1)
    replay.add(Transition(observation, action, reward, next_observation, terminated, truncated, log_prob))
    for value in (observation, action, reward, next_observation, log_prob):
        value.fill(99)
    terminated.fill(True)
    truncated.fill(False)
    batch = replay.sample(1)
    torch.testing.assert_close(batch.observation, torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(batch.action, torch.tensor([[-0.25, 0.5]]))
    torch.testing.assert_close(batch.reward, torch.tensor([3.0]))
    torch.testing.assert_close(batch.next_observation, torch.tensor([[4.0, 5.0]]))
    torch.testing.assert_close(batch.log_prob, torch.tensor([-0.75]))
    assert not batch.terminated.item() and batch.truncated.item()


def test_numpy_views_with_negative_strides_and_readonly_arrays_are_snapshotted():
    source = np.arange(4, dtype=np.float32)
    reversed_observation = source[::-1]
    reversed_observation.flags.writeable = False
    replay = ReplayBuffer(1)
    replay.add(Transition(reversed_observation, 0, 1.0, reversed_observation, False, False))
    source.fill(-1)
    torch.testing.assert_close(replay.sample(1).observation, torch.tensor([[3.0, 2.0, 1.0, 0.0]]))


def test_add_detaches_and_copies_all_tensor_values_without_retaining_graphs():
    observation = torch.tensor([1.0, 2.0], requires_grad=True)
    action = torch.tensor([0.1, 0.2], requires_grad=True)
    reward = torch.tensor(0.5, requires_grad=True)
    log_prob = (action.square().sum() + reward).log()
    next_observation = observation * 2
    terminated, truncated = torch.tensor(False), torch.tensor(True)
    replay = ReplayBuffer(1)
    replay.add(Transition(observation, action, reward, next_observation, terminated, truncated, log_prob))
    expected_log_prob = log_prob.detach().clone()
    with torch.no_grad():
        for value in (observation, action, reward, next_observation, log_prob):
            value.add_(10)
        terminated.fill_(True)
        truncated.fill_(False)
    batch = replay.sample(1)
    for item in fields(batch):
        value = getattr(batch, item.name)
        assert value.device.type == "cpu"
        assert not value.requires_grad and value.grad_fn is None
    torch.testing.assert_close(batch.observation, torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(batch.next_observation, torch.tensor([[2.0, 4.0]]))
    torch.testing.assert_close(batch.log_prob, expected_log_prob.unsqueeze(0))
    assert not batch.terminated.item() and batch.truncated.item()


def test_modifying_a_sample_does_not_modify_buffer_storage_or_other_samples():
    replay = ReplayBuffer(2, seed=10)
    for index in range(2):
        replay.add(transition(index, log_prob=-0.5))
    batch = replay.sample(2)
    for item in fields(batch):
        getattr(batch, item.name).fill_(0)
    again = replay.sample(2)
    assert set(again.observation.tolist()) == {0, 1}
    assert set(again.next_observation.tolist()) == {1, 2}
    torch.testing.assert_close(again.log_prob, torch.full((2,), -0.5))


def test_terminated_and_truncated_are_independent_and_final_observation_is_preserved():
    combinations = [(False, False), (True, False), (False, True), (True, True)]
    replay = ReplayBuffer(4, seed=7)
    for index, (terminated, truncated) in enumerate(combinations):
        replay.add(replace(transition(index), terminated=terminated, truncated=truncated, next_observation=index + 100))
    batch = replay.sample(4)
    for row, index in enumerate(batch.observation.tolist()):
        assert (batch.terminated[row].item(), batch.truncated[row].item()) == combinations[index]
        assert batch.next_observation[row].item() == index + 100
    # Bootstrap and episode-boundary masks serve different purposes in future targets.
    bootstrap = ~batch.terminated
    boundary = batch.terminated | batch.truncated
    time_limit_row = (batch.observation == 2).nonzero().item()
    assert bootstrap[time_limit_row] and boundary[time_limit_row]


@pytest.mark.parametrize("known_first", [False, True])
def test_missing_behavior_log_prob_is_never_fabricated(known_first):
    replay = ReplayBuffer(1)
    first = transition(0, log_prob=-0.5 if known_first else None)
    replay.add(first)
    with pytest.raises(ValueError, match="cannot mix present and missing log_prob"):
        replay.add(transition(1, log_prob=None if known_first else -0.5))
    assert replay.sample(1).observation.item() == 0
    assert (replay.sample(1).log_prob is None) == (not known_first)


def test_seed_and_identical_operation_sequences_reproduce_sampling():
    first, second, different = [ReplayBuffer(10, seed=seed) for seed in (123, 123, 124)]
    for index in range(16):
        for replay in (first, second, different):
            replay.add(transition(index, log_prob=-0.25))
    different_order_seen = False
    for _ in range(4):
        one, two, other = first.sample(5), second.sample(5), different.sample(5)
        for item in fields(one):
            torch.testing.assert_close(getattr(one, item.name), getattr(two, item.name))
        different_order_seen |= not torch.equal(one.observation, other.observation)
        assert len(set(one.observation.tolist())) == 5
    assert different_order_seen


def test_replay_rng_does_not_consume_global_numpy_or_torch_randomness():
    np.random.seed(19)
    torch.manual_seed(19)
    expected_numpy, expected_torch = np.random.random(5), torch.rand(5)
    np.random.seed(19)
    torch.manual_seed(19)
    replay = ReplayBuffer(8, seed=99)
    for index in range(8):
        replay.add(transition(index))
    replay.sample(4)
    np.testing.assert_array_equal(np.random.random(5), expected_numpy)
    torch.testing.assert_close(torch.rand(5), expected_torch)


def test_clear_resets_storage_ring_and_schema_but_not_rng_state():
    replay = ReplayBuffer(6, seed=7)
    expected_rng = np.random.default_rng(7)
    for index in range(6):
        replay.add(transition(index))
    np.testing.assert_array_equal(replay.sample(3).observation.numpy(), expected_rng.choice(6, 3, replace=False))
    replay.clear()
    assert len(replay) == 0
    with pytest.raises(ValueError, match="cannot exceed buffer length"):
        replay.sample(1)
    for index in range(6):
        replay.add(transition(index))
    np.testing.assert_array_equal(replay.sample(3).observation.numpy(), expected_rng.choice(6, 3, replace=False))
    replay.clear()
    replay.add(Transition(np.ones(2, dtype=np.float64), np.zeros(3, dtype=np.float32), 0.0,
                          np.zeros(2, dtype=np.float64), True, False, -1.0))
    assert replay.sample(1).action.shape == (1, 3)


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_capacity_fails_during_configuration(capacity):
    with pytest.raises(ValueError, match="capacity must be a positive integer"):
        ReplayBuffer(capacity)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, 3])
def test_invalid_sample_sizes_fail_without_replacement(batch_size):
    replay = ReplayBuffer(2)
    replay.add(transition(0))
    replay.add(transition(1))
    with pytest.raises(ValueError, match="batch_size"):
        replay.sample(batch_size)


@pytest.mark.parametrize("changes,error,message", [
    ({"reward": np.ones(1)}, ValueError, "reward must be a scalar"),
    ({"terminated": 1}, TypeError, "terminated must have bool dtype"),
    ({"truncated": np.array([True])}, ValueError, "truncated must be a scalar"),
    ({"action": 0.0}, TypeError, "scalar discrete action"),
    ({"action": np.array([1, 2])}, TypeError, "vector continuous action"),
    ({"action": np.zeros((2, 1), dtype=np.float32)}, ValueError, "action must be"),
    ({"log_prob": torch.zeros(2)}, ValueError, "summed over the action event"),
    ({"next_observation": np.ones(2)}, ValueError, "same shape and dtype"),
])
def test_semantic_contract_errors_leave_existing_storage_unchanged(changes, error, message):
    replay = ReplayBuffer(1)
    replay.add(transition(0))
    with pytest.raises(error, match=message):
        replay.add(replace(transition(1), **changes))
    assert len(replay) == 1
    assert replay.sample(1).observation.item() == 0


@pytest.mark.parametrize("changes", [
    {"observation": np.ones(2), "next_observation": np.ones(2)},
    {"observation": np.int32(1), "next_observation": np.int32(2)},
    {"action": np.zeros(2, dtype=np.float32)},
    {"reward": np.float64(1.0)},
])
def test_buffer_schema_prevents_implicit_shape_or_dtype_promotion(changes):
    replay = ReplayBuffer(2)
    replay.add(transition(0))
    with pytest.raises(ValueError, match="consistent shape and dtype"):
        replay.add(replace(transition(1), **changes))
    assert len(replay) == 1


def test_add_requires_transition_and_numpy_integer_sizes_are_supported():
    replay = ReplayBuffer(np.int64(2), seed=5)
    with pytest.raises(TypeError, match="expects a Transition"):
        replay.add({"observation": 0})
    replay.add(transition(0))
    assert replay.sample(np.int64(1)).observation.shape == (1,)


@pytest.mark.parametrize("continuous", [False, True])
def test_batch_to_casts_floating_fields_and_preserves_indices_and_flags(continuous):
    replay = ReplayBuffer(1)
    if continuous:
        replay.add(Transition(np.ones(2, dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0,
                              np.zeros(2, dtype=np.float32), False, True, -0.5))
    else:
        replay.add(transition(0, log_prob=-0.5))
    batch = replay.sample(1)
    converted = batch.to("cpu", dtype=torch.float64)
    assert converted.reward.dtype == converted.log_prob.dtype == torch.float64
    assert converted.observation.dtype == converted.action.dtype == (torch.float64 if continuous else torch.long)
    assert converted.terminated.dtype == converted.truncated.dtype == torch.bool
    assert batch.reward.dtype == torch.float32
    with pytest.raises(ValueError, match="dtype must be floating-point"):
        batch.to(dtype=torch.long)


def test_batch_to_keeps_absent_log_prob_and_integer_reward_dtype():
    replay = ReplayBuffer(1)
    replay.add(replace(transition(0), reward=1))
    batch = replay.sample(1).to(dtype=torch.float64)
    assert batch.log_prob is None and batch.reward.dtype == torch.long


@pytest.mark.parametrize("backend", ["numpy", "tensor"])
def test_tabular_model_action_outputs_round_trip_through_replay(backend):
    from modular_rl.component.action_selector import EpsilonGreedyActionSelector
    from modular_rl.component.estimator import TabularQEstimator
    from modular_rl.component.representation import IdentityRepresentation
    from modular_rl.model import Model

    model = Model(IdentityRepresentation(), TabularQEstimator(6, 2), EpsilonGreedyActionSelector(0.2))
    model.estimator.q_table.copy_(torch.arange(12.0).reshape(6, 2))
    replay = ReplayBuffer(4, seed=2)
    for index in range(4):
        state = np.int32(index)
        next_state = np.int32(index + 1)
        if backend == "tensor":
            state, next_state = torch.as_tensor(state), torch.as_tensor(next_state)
        action = model.act(state)
        replay.add(Transition(state, action.action, 1.0, next_state, index == 1, index == 3, action.log_prob))
    batch = replay.sample(4).to("cpu", dtype=torch.float64)
    assert batch.observation.dtype == torch.int32
    estimate = model.forward(batch.observation)
    assert estimate.values.q[0].shape == (4, 2)
    next_estimate = model.forward(batch.next_observation)
    torch.testing.assert_close(next_estimate.values.q[0], model.estimator.q_table[batch.next_observation.long()])
    time_limit_row = (batch.observation == 3).nonzero().item()
    assert batch.truncated[time_limit_row] and not batch.terminated[time_limit_row]
    assert batch.next_observation[time_limit_row] == 4
    probabilities = model.action_selector.probabilities(estimate)
    expected = probabilities.gather(-1, batch.action.unsqueeze(-1)).squeeze(-1).log()
    torch.testing.assert_close(batch.log_prob, expected.double())


@pytest.mark.parametrize("backend", ["numpy", "tensor"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_actor_critic_recomputes_training_gradients_from_detached_replay_data(backend, dtype):
    from modular_rl.component.action_selector import CategoricalActionSelector
    from modular_rl.component.estimator import ActorCriticEstimator, CategoricalPolicyEstimator, StateValueEstimator
    from modular_rl.component.representation import MLPRepresentation
    from modular_rl.model import Model

    tensor_dtype = torch.from_numpy(np.empty((), dtype=dtype)).dtype
    model = Model(
        MLPRepresentation(3, hidden_dims=(5,), activation=torch.nn.Tanh),
        ActorCriticEstimator(CategoricalPolicyEstimator(5, 2), StateValueEstimator(5)),
        CategoricalActionSelector(),
    ).to(dtype=tensor_dtype)
    replay = ReplayBuffer(4, seed=4)
    for index in range(4):
        observation = np.full(3, index, dtype=dtype)
        if backend == "tensor":
            observation = torch.from_numpy(observation).requires_grad_()
        selected = model.act(observation)
        replay.add(Transition(observation, selected.action, float(index), observation + 1,
                              False, index == 3, selected.log_prob))
    batch = replay.sample(3).to("cpu", dtype=tensor_dtype)
    assert batch.observation.dtype == batch.log_prob.dtype == tensor_dtype
    assert not batch.observation.requires_grad and batch.observation.grad_fn is None
    assert not batch.log_prob.requires_grad and batch.log_prob.grad_fn is None
    estimate = model.forward(batch.observation)
    log_prob = estimate.policy.log_prob(batch.action)
    loss = estimate.values.v.square().mean() - log_prob.mean()
    loss.backward()
    assert log_prob.requires_grad
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize("backend", ["numpy", "tensor"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_neural_q_action_outputs_round_trip_through_replay(backend, deterministic):
    from modular_rl.component.action_selector import EpsilonGreedyActionSelector
    from modular_rl.component.estimator import DiscreteQEstimator
    from modular_rl.component.representation import MLPRepresentation
    from modular_rl.model import Model

    model = Model(MLPRepresentation(3, hidden_dims=(5,)), DiscreteQEstimator(5, 2),
                  EpsilonGreedyActionSelector(0.2))
    replay = ReplayBuffer(4, seed=8)
    for index in range(4):
        observation = np.full(3, index, dtype=np.float32)
        if backend == "tensor":
            observation = torch.from_numpy(observation).requires_grad_()
        selected = model.act(observation, deterministic=deterministic)
        replay.add(Transition(observation, selected.action, float(index), observation + 1,
                              index == 1, index == 3, selected.log_prob))
    batch = replay.sample(4)
    estimate = model.forward(batch.observation)
    chosen_q = estimate.values.q[0].gather(-1, batch.action.unsqueeze(-1)).squeeze(-1)
    assert chosen_q.shape == (4,)
    if deterministic:
        assert batch.log_prob is None
        assert torch.equal(batch.action, estimate.values.q[0].argmax(-1))
    else:
        probabilities = model.action_selector.probabilities(estimate)
        expected_log_prob = probabilities.gather(-1, batch.action.unsqueeze(-1)).squeeze(-1).log()
        torch.testing.assert_close(batch.log_prob, expected_log_prob)
    chosen_q.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize("backend", ["numpy", "tensor"])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_continuous_replay_batch_feeds_model_evaluate_q(backend, batch_size):
    from modular_rl.component.estimator import ContinuousQEstimator
    from modular_rl.component.representation import MLPRepresentation
    from modular_rl.model import Model

    critic = Model(MLPRepresentation(3, hidden_dims=(5,)), ContinuousQEstimator(5, 2, hidden_dim=8)).double()
    replay = ReplayBuffer(3, seed=6)
    expected = []
    for index in range(3):
        observation = np.full(3, index, dtype=np.float64)
        action = np.array([index / 4, -index / 4], dtype=np.float64)
        if backend == "tensor":
            observation = torch.from_numpy(observation).requires_grad_()
            action = torch.from_numpy(action).requires_grad_()
        expected.append(critic.evaluate_q(observation, action).values.q[0].detach())
        replay.add(Transition(observation, action, float(index), observation + 1, False, index == 2))
    batch = replay.sample(batch_size).to("cpu", dtype=torch.float64)
    assert batch.observation.dtype == batch.action.dtype == torch.float64
    assert not batch.observation.requires_grad and not batch.action.requires_grad
    q = critic.evaluate_q(batch.observation, batch.action).values.q[0]
    assert q.shape == (batch_size,)
    torch.testing.assert_close(q, torch.stack([expected[int(index)] for index in batch.reward]))
    next_q = critic.evaluate_q(batch.next_observation, batch.action).values.q[0]
    assert next_q.shape == (batch_size,) and torch.isfinite(next_q).all()
    q.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in critic.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_inputs_are_stored_detached_on_cpu_and_batch_transfer_is_explicit():
    replay = ReplayBuffer(1)
    observation = torch.ones(2, device="cuda", requires_grad=True)
    action = torch.zeros(3, device="cuda", requires_grad=True)
    replay.add(Transition(observation, action, torch.tensor(1.0, device="cuda"), observation * 2,
                          torch.tensor(False, device="cuda"), torch.tensor(True, device="cuda"), action.sum()))
    batch = replay.sample(1)
    assert all(getattr(batch, item.name).device.type == "cpu" for item in fields(batch))
    converted = batch.to("cuda", dtype=torch.float64)
    for item in fields(converted):
        value = getattr(converted, item.name)
        assert value.device.type == "cuda" and not value.requires_grad
    assert converted.terminated.dtype == converted.truncated.dtype == torch.bool
