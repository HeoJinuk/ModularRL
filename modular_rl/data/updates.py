"""Neural loss results, independent of optimizer execution."""

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

    The four metrics are scalar Tensors; Rule td_error has shape (B,), and
    A2CLearner restores its time-major (T,N) shape. All share
    the input floating dtype/device. The UpdateRule retains graphs except
    the actor's advantage; ActorCriticLearner/A2CLearner return detached metrics.
    """

    actor_loss: Tensor
    critic_loss: Tensor
    entropy: Tensor
    total_loss: Tensor
    td_error: Tensor
