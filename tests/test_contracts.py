import numpy as np
import pytest
import torch
from torch import nn, Tensor
from torch.distributions import Categorical

from modular_rl.component.action_selector import (
    EpsilonGreedyActionSelector, CategoricalActionSelector,
)
from modular_rl.component.estimator.actor_critic import ActorCriticEstimator
from modular_rl.component.estimator.base import QEstimator, AdvantageEstimator
from modular_rl.component.estimator import (
    TabularQEstimator, DiscreteQEstimator, CategoricalPolicyEstimator, StateValueEstimator,
)
from modular_rl.component.representation import IdentityRepresentation, MLPRepresentation
from modular_rl.model import Model, ActionOutput, EstimatorOutput, ValueOutput, InputAdapter


def make_tabular():
    return Model(IdentityRepresentation(), TabularQEstimator(16, 4),
                 EpsilonGreedyActionSelector(0.2))


def make_actor_critic():
    return Model(MLPRepresentation(4, hidden_dims=(64,)),
                 ActorCriticEstimator(CategoricalPolicyEstimator(64, 2), StateValueEstimator(64)),
                 CategoricalActionSelector())


def test_tabular_scalar_numpy_and_torch_indices():
    m = make_tabular()
    for observation in (5, np.int64(5), np.array(5), torch.tensor(5)):
        estimate = m.forward(observation)
        assert estimate.policy is None
        assert estimate.values.v is None
        assert len(estimate.values.q) == 1
        assert estimate.values.q[0].shape == (4,)
        selected = m.act(observation)
        assert selected.action.ndim == 0
        assert 0 <= selected.action.item() < 4
        assert selected.log_prob is not None


def test_tabular_batched_indices():
    m = make_tabular()
    assert m.forward(np.array([0, 1, 2], dtype=np.int32)).values.q[0].shape == (3, 4)
    assert m.act(np.array([0, 1, 2])).action.shape == (3,)


def test_tabular_rejects_invalid_index():
    m = make_tabular()
    with pytest.raises(IndexError):
        m.forward(16)
    with pytest.raises(TypeError):
        m.forward('not a state')


def test_mlp_q_shapes_with_numpy_and_tensor():
    m = Model(MLPRepresentation(4, hidden_dims=(64,)), DiscreteQEstimator(64, 2),
              EpsilonGreedyActionSelector())
    assert m.forward(np.zeros(4, dtype=np.float64)).values.q[0].shape == (2,)
    assert m.forward(torch.zeros(8, 4)).values.q[0].shape == (8, 2)
    assert m.forward(torch.zeros(3, 8, 4)).values.q[0].shape == (3, 8, 2)


def test_actor_critic_policy_value_and_actions():
    m = make_actor_critic()
    o = m.forward(np.zeros((10, 4), dtype=np.float32))
    assert isinstance(o.policy, Categorical)
    assert o.values.v.shape == (10,)
    assert o.values.q == ()
    a = m.act(torch.zeros(10, 4))
    assert a.action.shape == (10,)
    assert a.log_prob.shape == (10,)
    assert not a.log_prob.requires_grad


def test_gradient_flows_through_forward_not_act():
    m = make_actor_critic()
    obs = torch.randn(6, 4, requires_grad=True)
    loss = m.forward(obs).values.v.square().mean()
    loss.backward()
    assert obs.grad is not None
    assert any(p.grad is not None for p in m.parameters())
    assert not m.act(obs).log_prob.requires_grad


def test_deterministic_mode_returns_action_without_policy_logprob():
    m = make_actor_critic()
    a = m.act(np.zeros(4), deterministic=True)
    assert a.log_prob is None
    assert a.action.shape == ()


def test_numpy_conversion_casts_floats_and_preserves_integral_indices():
    m = make_actor_critic()
    x = m.encode(np.zeros(4, dtype=np.float64))
    assert x.dtype == torch.float32
    m = make_tabular()
    assert m.encode(np.int32(2)).dtype == torch.long


def test_float64_model_moves_input_floats_to_model_dtype():
    m = make_actor_critic().double()
    est = m.forward(np.zeros(4, dtype=np.float32))
    assert est.values.v.dtype == torch.float64


def test_force_cast_integer_box_observation():
    m = Model(MLPRepresentation(4, hidden_dims=(16,)), StateValueEstimator(16),
              input_adapter=InputAdapter(dtype=torch.float32))
    assert m.forward(np.array([1, 2, 3, 4], dtype=np.int32)).values.v.ndim == 0


def test_q_tuple_requires_tuple_and_open_payload():
    with pytest.raises(TypeError):
        ValueOutput(q=torch.ones(4))
    custom_distribution = {'atoms': torch.tensor([0., 1.]), 'logits': torch.zeros(3, 2)}
    result = EstimatorOutput(values=ValueOutput(q=(custom_distribution,)))
    assert result.values.q[0] is custom_distribution
    with pytest.raises(TypeError):
        EstimatorOutput(values={'q': 1})


def test_non_action_value_estimator_has_no_selector():
    m = Model(MLPRepresentation(4, hidden_dims=(16,)), StateValueEstimator(16))
    assert m.forward(torch.zeros(4)).values.v.shape == ()
    with pytest.raises(RuntimeError):
        m.act(torch.zeros(4))


def test_selector_validates_missing_semantic_field():
    with pytest.raises(ValueError):
        EpsilonGreedyActionSelector().select(EstimatorOutput())
    with pytest.raises(TypeError):
        CategoricalActionSelector().select(EstimatorOutput(values=ValueOutput(q=(torch.zeros(4),))))


def test_equal_greedy_values_share_probability_mass():
    s = EpsilonGreedyActionSelector(epsilon=0.2)
    est = EstimatorOutput(values=ValueOutput(q=(torch.tensor([1., 1., 0., 0.]),)))
    p = s.probabilities(est)
    assert torch.allclose(p, torch.tensor([0.45, 0.45, 0.05, 0.05]))


def test_selector_state_saved_as_part_of_model_and_restored():
    first = make_tabular()
    first.action_selector.epsilon = 0.7
    saved = first.state_dict()
    assert 'action_selector._epsilon' in saved
    second = make_tabular()
    second.load_state_dict(saved)
    assert second.action_selector.epsilon == pytest.approx(0.7)


def test_future_action_conditioned_q_without_altering_estimator_output():
    class ContinuousQ(QEstimator):
        def __init__(self):
            super().__init__()
            self.network = nn.Linear(6, 1)

        def forward(self, features: Tensor) -> EstimatorOutput:
            return EstimatorOutput()  # no Q(s,a) until action is supplied

        def evaluate(self, features: Tensor, actions: Tensor) -> EstimatorOutput:
            q = self.network(torch.cat([features, actions], dim=-1)).squeeze(-1)
            return EstimatorOutput(values=ValueOutput(q=(q,)))

    m = Model(IdentityRepresentation(), ContinuousQ())
    out = m.evaluate_q(torch.ones(5, 4), np.ones((5, 2), dtype=np.float32))
    assert out.values.q[0].shape == (5,)


def test_action_output_requires_tensor():
    with pytest.raises(TypeError):
        ActionOutput(action=1)
    with pytest.raises(TypeError):
        ActionOutput(action=torch.tensor(1), log_prob=0.0)
