"""Results of neural TD loss calculation, independent of optimizer execution."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class TDLossOutput:
    """Signed target-prediction errors and a scalar mean loss.

    td_error has the prediction shape; loss has shape (). Both retain the
    caller's autograd graph. Learner detaches errors for logs or priorities.
    """

    td_error: Tensor
    loss: Tensor
