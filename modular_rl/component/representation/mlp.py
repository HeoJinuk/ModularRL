"""Simple MLP representation with a predictable tensor interface."""

from torch import Tensor, nn

from modular_rl.component.representation.base import BaseRepresentation


class MLPRepresentation(BaseRepresentation):
    """Linear + activation at each hidden dimension, including the last.

    Input shape: (..., input_dim)
    Output shape: (..., hidden_dims[-1])

    This representation receives tensors only. NumPy, device, and dtype
    conversions belong to Model's InputAdapter.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128,),
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if not isinstance(input_dim, int) or isinstance(input_dim, bool) or input_dim <= 0:
            raise ValueError("input_dim must be greater than 0")
        if not isinstance(hidden_dims, tuple) or not hidden_dims:
            raise ValueError("hidden_dims must be a non-empty tuple")
        if any(not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in hidden_dims):
            raise ValueError("hidden dimensions must be greater than 0")
        if not isinstance(activation, type) or not issubclass(activation, nn.Module):
            raise TypeError("activation must be a torch.nn.Module class")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.extend((nn.Linear(in_dim, out_dim), activation()))
            in_dim = out_dim
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1]

    def forward(self, observations: Tensor) -> Tensor:
        if not isinstance(observations, Tensor):
            raise TypeError("MLPRepresentation requires a torch.Tensor; use Model for NumPy")
        if observations.ndim < 1 or observations.shape[-1] != self.input_dim:
            raise ValueError(f"Expected last observation dimension {self.input_dim}")
        return self.network(observations)
