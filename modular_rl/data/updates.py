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


@dataclass(frozen=True, slots=True)
class ActorCriticOutput:
    """Actor/critic mean losses, mean entropy, total loss and signed TD error.

    The four metrics are scalar Tensors; td_error has shape (B,). All share
    the input floating dtype/device. The UpdateRule retains graphs except
    the actor's advantage; ActorCriticLearner returns detached metrics.
    """

    actor_loss: Tensor
    critic_loss: Tensor
    entropy: Tensor
    total_loss: Tensor
    td_error: Tensor
