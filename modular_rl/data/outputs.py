"""Shared semantic output contracts. Payload representations stay extensible."""

from dataclasses import dataclass, field

from torch import Tensor


@dataclass(frozen=True, slots=True)
class ValueOutput:
    """Value estimates, named by mathematical meaning, not downstream usage.

    Values intentionally allow arbitrary payload objects: raw scalar tensors,
    categorical return distributions, quantile objects, etc. Consumers must
    validate the specific representation they support.

    q is ALWAYS a tuple. A single DQN critic is (q_tensor,), a SAC twin
    critic is (q1, q2). All elements of one tuple should describe the same
    action domain and compatible estimates.
    """

    v: object | None = None
    q: tuple[object, ...] = ()
    advantage: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.q, tuple):
            raise TypeError("ValueOutput.q must be a tuple (use (q,) for one critic)")


@dataclass(frozen=True, slots=True)
class EstimatorOutput:
    """Policy and/or value predictions; at least one is expected in practice.

    policy can be a torch Distribution or a deterministic action tensor.
    A selector explicitly validates the form it accepts. This is one
    semantic policy field, not separate stochastic/deterministic fields.
    """

    policy: object | None = None
    values: ValueOutput = field(default_factory=ValueOutput)

    def __post_init__(self) -> None:
        if not isinstance(self.values, ValueOutput):
            raise TypeError("EstimatorOutput.values must be ValueOutput")


@dataclass(frozen=True, slots=True)
class ActionOutput:
    """Internal tensor action and optional policy log-probability.

    A Runner converts action tensors to Gymnasium-compatible Python/NumPy
    objects at the environment boundary. A deterministic evaluation action
    may have log_prob=None.
    """

    action: Tensor
    log_prob: Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, Tensor):
            raise TypeError("ActionOutput.action must be a torch.Tensor")
        if self.log_prob is not None and not isinstance(self.log_prob, Tensor):
            raise TypeError("ActionOutput.log_prob must be a torch.Tensor or None")
