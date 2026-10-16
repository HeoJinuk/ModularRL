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


@dataclass(frozen=True, slots=True)
class PPOOutput:
    """Six scalar metrics in input dtype/device and executed optimizer steps.

    Rule losses retain gradients; diagnostics are detached and steps is zero.
    Learner metrics are detached, weighted by minibatch sample counts over all
    completed epochs (measured before each step, not a final-policy reevaluation).
    approx_kl = mean(exp(log_ratio)-1-log_ratio); clip_fraction measures
    abs(ratio-1)>clip_epsilon, regardless of advantage sign.
    """

    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    total_loss: Tensor
    approx_kl: Tensor
    clip_fraction: Tensor
    optimizer_steps: int = 0
    epochs_completed: int = 0
    early_stopped: bool = False
