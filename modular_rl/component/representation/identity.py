"""No-op feature representation, including tabular state indices."""

from torch import Tensor

from modular_rl.component.representation.base import BaseRepresentation


class IdentityRepresentation(BaseRepresentation):
    def forward(self, observations: Tensor) -> Tensor:
        return observations
