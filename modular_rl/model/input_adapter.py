"""Convert environment-facing observations to PyTorch tensors."""

import numpy as np
import torch
from torch import Tensor

Observation = np.ndarray | np.generic | Tensor | int | float


class InputAdapter:
    """Keep indices integral and normal floating observations floating.

    dtype=None (default): integer observations -> torch.long, float
    observations -> the Model's floating dtype.
    dtype=torch.float32: force ALL numeric observations to float32; useful
    for a Box space whose sensor readings happen to be stored as integers.
    """

    def __init__(self, dtype: torch.dtype | None = None) -> None:
        self.dtype = dtype

    def __call__(
        self,
        observation: Observation,
        *,
        device: torch.device,
        float_dtype: torch.dtype,
    ) -> Tensor:
        if not isinstance(observation, (np.ndarray, np.generic, Tensor, int, float)):
            raise TypeError("Expected a NumPy value, torch.Tensor, or numeric scalar")

        tensor = torch.as_tensor(observation)
        if tensor.is_complex():
            raise TypeError("Complex observations are not supported")

        dtype = self.dtype
        if dtype is None:
            dtype = float_dtype if tensor.is_floating_point() else torch.long

        # torch.Tensor.to maintains autograd when converting differentiable tensors.
        return tensor.to(device=device, dtype=dtype)
