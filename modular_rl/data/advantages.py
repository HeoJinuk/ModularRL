"""Return and advantage estimates, independent of models and optimization."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class GAEOutput:
    """Time-major advantage and critic return (T,N), retaining input gradients."""

    advantage: Tensor
    returns: Tensor
