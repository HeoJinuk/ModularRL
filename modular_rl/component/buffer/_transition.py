"""Shared binding, ownership and single-transition schema checks for replay."""

import numpy as np
import torch
from torch import Tensor

from modular_rl.data import Transition
from modular_rl.data.experience import ExperienceValue


def transition_from_args(args: tuple, kwargs: dict) -> Transition:
    """Bind old Transition calls or the fields of one Transition, before storage."""
    if "transition" in kwargs:
        if args or len(kwargs) != 1:
            raise TypeError("A Transition cannot be combined with transition fields")
        transition = kwargs["transition"]
    elif args and isinstance(args[0], Transition):
        if len(args) != 1 or kwargs:
            raise TypeError("A Transition cannot be combined with transition fields")
        transition = args[0]
    else:
        if len(args) == 1 and not kwargs:
            raise TypeError("ReplayBuffer.add expects a Transition or its required fields")
        return Transition(*args, **kwargs)
    if not isinstance(transition, Transition):
        raise TypeError("ReplayBuffer.add expects a Transition")
    return transition


def _copy_value(value: ExperienceValue) -> Tensor:
    """Keep dtype while removing aliases to caller-owned data and graphs."""
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    return torch.as_tensor(value, device="cpu")


def snapshot(transition: Transition) -> Transition:
    """Convert a single transition into independent CPU tensor values."""
    if not isinstance(transition, Transition):
        raise TypeError("ReplayBuffer.add expects a Transition")
    return Transition(
        observation=_copy_value(transition.observation),
        action=_copy_value(transition.action),
        reward=_copy_value(transition.reward),
        next_observation=_copy_value(transition.next_observation),
        terminated=_copy_value(transition.terminated),
        truncated=_copy_value(transition.truncated),
        log_prob=None if transition.log_prob is None else _copy_value(transition.log_prob),
    )


def validate(transition: Transition, reference: Transition | None) -> None:
    """Check scalar semantics and a homogeneous field schema, once at add."""
    observation, next_observation = transition.observation, transition.next_observation
    if observation.is_complex() or next_observation.is_complex():
        raise TypeError("Observations must be real numeric values")
    if observation.shape != next_observation.shape or observation.dtype != next_observation.dtype:
        raise ValueError("observation and next_observation must have the same shape and dtype")

    action = transition.action
    if action.ndim > 1 or action.numel() == 0:
        raise ValueError("action must be an integer scalar or a nonempty float vector (A,)")
    if action.ndim == 0:
        if action.is_floating_point() or action.is_complex() or action.dtype == torch.bool:
            raise TypeError("A scalar discrete action must have integer dtype")
    elif not action.is_floating_point():
        raise TypeError("A vector continuous action must have floating dtype")

    reward = transition.reward
    if reward.ndim != 0:
        raise ValueError("reward must be a scalar")
    if reward.is_complex() or reward.dtype == torch.bool:
        raise TypeError("reward must be a real numeric scalar")
    for name in ("terminated", "truncated"):
        flag = getattr(transition, name)
        if flag.ndim != 0:
            raise ValueError(f"{name} must be a scalar")
        if flag.dtype != torch.bool:
            raise TypeError(f"{name} must have bool dtype")
    if transition.log_prob is not None:
        if transition.log_prob.ndim != 0:
            raise ValueError("log_prob must be a scalar summed over the action event")
        if not transition.log_prob.is_floating_point():
            raise TypeError("log_prob must have floating dtype")

    if reference is not None:
        if (transition.log_prob is None) != (reference.log_prob is None):
            raise ValueError("ReplayBuffer cannot mix present and missing log_prob")
        # Next observation matches observation; flag shape/dtype are already fixed.
        for name in ("observation", "action", "reward", "log_prob"):
            value, previous = getattr(transition, name), getattr(reference, name)
            if value is not None and (value.shape != previous.shape or value.dtype != previous.dtype):
                raise ValueError(f"{name} must have a consistent shape and dtype within a buffer")
