"""Regression tests for the user-supplied MLPRepresentation contract."""

import numpy as np
import pytest
import torch
from torch import nn

from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import DiscreteQEstimator
from modular_rl.component.representation import (
    BaseRepresentation,
    IdentityRepresentation,
    MLPRepresentation,
)
from modular_rl.data.outputs import ActionOutput, EstimatorOutput, ValueOutput
from modular_rl.model import Model


def test_mlp_keeps_user_hidden_dims_and_default_activation():
    rep = MLPRepresentation(input_dim=4, hidden_dims=(128, 64))
    assert isinstance(rep, BaseRepresentation)
    assert isinstance(rep, nn.Module)
    assert rep.input_dim == 4
    assert rep.output_dim == 64
    assert [type(layer) for layer in rep.network] == [
        nn.Linear, nn.ReLU, nn.Linear, nn.ReLU
    ]
    assert isinstance(MLPRepresentation(4).network[0], nn.Linear)
    assert MLPRepresentation(4).output_dim == 128


def test_mlp_supports_user_selected_activation():
    rep = MLPRepresentation(3, hidden_dims=(16, 8), activation=nn.Tanh)
    assert type(rep.network[1]) is nn.Tanh
    assert type(rep.network[3]) is nn.Tanh
    assert rep(torch.ones(3)).shape == (8,)


@pytest.mark.parametrize("shape,expected", [((4,), (12,)), ((7, 4), (7, 12)), ((5, 7, 4), (5, 7, 12))])
def test_mlp_preserves_leading_dimensions(shape, expected):
    rep = MLPRepresentation(4, hidden_dims=(12,))
    assert rep(torch.ones(*shape)).shape == expected


@pytest.mark.parametrize("kwargs", [
    {"input_dim": 0},
    {"input_dim": -2},
    {"input_dim": 4, "hidden_dims": ()},
    {"input_dim": 4, "hidden_dims": (64, -1)},
    {"input_dim": 4, "hidden_dims": (0,)},
    {"input_dim": 4, "hidden_dims": [64]},
])
def test_mlp_rejects_invalid_dimensions(kwargs):
    with pytest.raises(ValueError):
        MLPRepresentation(**kwargs)


def test_mlp_rejects_non_tensor_and_wrong_last_dimension():
    rep = MLPRepresentation(4)
    with pytest.raises(TypeError, match="torch.Tensor"):
        rep(np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match="last observation dimension"):
        rep(torch.zeros(3, 5))


def test_model_adapts_numpy_but_representation_does_not():
    rep = MLPRepresentation(4, hidden_dims=(32, 16))
    model = Model(rep, DiscreteQEstimator(rep.output_dim, 2), EpsilonGreedyActionSelector())
    for obs in [np.zeros(4, dtype=np.float64), torch.zeros(4, dtype=torch.float32)]:
        result = model.forward(obs)
        assert result.values.q[0].shape == (2,)
    assert model.forward(torch.zeros(8, 4)).values.q[0].shape == (8, 2)


def test_model_device_and_dtype_tracked_by_adapter():
    rep = MLPRepresentation(4, hidden_dims=(5,))
    model = Model(rep, DiscreteQEstimator(5, 2)).double()
    out = model.forward(np.ones(4, dtype=np.float32))
    assert out.values.q[0].dtype == torch.float64


def test_representation_preserves_gradient():
    rep = MLPRepresentation(4, hidden_dims=(8,), activation=nn.Tanh)
    x = torch.randn(4, requires_grad=True)
    rep(x).sum().backward()
    assert x.grad is not None
    assert any(param.grad is not None for param in rep.parameters())


def test_identity_keeps_scalar_index_and_tensor_identity():
    index = torch.tensor(4, dtype=torch.long)
    assert IdentityRepresentation()(index) is index


def test_new_named_modules_and_data_imports():
    from modular_rl.model.rl_model import Model as DirectModel
    from modular_rl.component.estimator.policy.categorical import CategoricalPolicyEstimator
    from modular_rl.component.estimator.value.tabular_q import TabularQEstimator
    from modular_rl.component.action_selector.policy_sampling import CategoricalActionSelector

    assert DirectModel is Model
    assert CategoricalPolicyEstimator is not None
    assert TabularQEstimator is not None
    assert CategoricalActionSelector is not None
    assert EstimatorOutput(values=ValueOutput(q=(torch.ones(2),))).values.q[0].shape == (2,)
    assert ActionOutput(torch.tensor(1)).log_prob is None
