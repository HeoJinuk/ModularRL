"""Action selection contract."""

from abc import ABC, abstractmethod
from torch import nn
from modular_rl.data import ActionOutput, EstimatorOutput


class BaseActionSelector(nn.Module, ABC):
    """Consumes semantic EstimatorOutput; returns action and optional log-prob."""

    @abstractmethod
    def select(
        self, estimate: EstimatorOutput, *, deterministic: bool = False
    ) -> ActionOutput:
        ...


ActionSelector = BaseActionSelector
