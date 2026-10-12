"""Minimal tensor-to-tensor representation contract."""

from abc import ABC, abstractmethod

from torch import Tensor, nn


class BaseRepresentation(nn.Module, ABC):
    """Map tensor observations to tensor features.

    Vector observations have shape (..., input_dim). Keep the leading dimensions.
    IdentityRepresentation also accepts scalar state indices for tabular RL.
    Input conversion is the Model/InputAdapter's responsibility.
    """

    @abstractmethod
    def forward(self, observations: Tensor) -> Tensor:
        ...


# Keeps code written against the first reference version importable.
Representation = BaseRepresentation
