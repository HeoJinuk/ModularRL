"""Small Gymnasium SAME_STEP boundary helpers; collection remains external."""

import numpy as np
from gymnasium.spaces import Box, Discrete


def validate_vector_env(env):
    """Require homogeneous Box observations, discrete actions and SAME_STEP.

    Call once before reset/collection. NEXT_STEP reset-only outputs cannot be
    represented as real fixed-(T,N) transitions and are deliberately rejected.
    Gymnasium's vector constructor checks identical spaces across workers.
    """
    mode = getattr(env, "autoreset_mode", None)
    if getattr(mode, "value", mode) != "SameStep":
        raise ValueError("Vector rollout requires AutoresetMode.SAME_STEP; NEXT_STEP is unsupported")
    if not isinstance(env.single_observation_space, Box) or not isinstance(env.single_action_space, Discrete):
        raise TypeError("Vector rollout supports Box observations and Discrete actions")
    if env.single_action_space.start != 0:
        raise ValueError("Categorical vector actions require zero-based Discrete indices")


def final_observations(next_observation, terminated, truncated, info):
    """Copy actual transition endpoints from a validated SAME_STEP env.step.

    Returned env observations belong to the next policy step, including reset
    observations. This separate array belongs in rollout.next_observation.
    Every boundary must have a true _final_obs mask and an actual final_obs.
    Never fall back to the reset observation when endpoint data is missing.
    """
    terminated, truncated = np.asarray(terminated), np.asarray(truncated)
    if terminated.dtype != np.bool_ or truncated.dtype != np.bool_:
        raise TypeError("terminated and truncated must be bool arrays")
    if terminated.ndim != 1 or terminated.shape != truncated.shape:
        raise ValueError("terminated and truncated must share shape (N,)")
    boundary = terminated | truncated
    observation = np.array(next_observation, copy=True)
    if boundary.ndim != 1 or observation.shape[:1] != boundary.shape:
        raise ValueError("Vector observations and boundary flags must share the N dimension")
    if np.any(boundary):
        final = info.get("final_obs")
        mask = np.asarray(info.get("_final_obs", np.zeros_like(boundary)))
        if final is None or mask.dtype != np.bool_ or mask.shape != boundary.shape or not np.all(mask[boundary]):
            raise ValueError("SAME_STEP boundaries require final_obs and a true _final_obs mask")
        for index in np.flatnonzero(boundary):
            value = np.asarray(final[index])
            if value.shape != observation.shape[1:] or value.dtype != observation.dtype:
                raise ValueError("final_obs must match observation shape and dtype")
            observation[index] = value
    return observation
