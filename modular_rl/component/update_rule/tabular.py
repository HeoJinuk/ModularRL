"""Sequential tabular TD updates with a fixed target snapshot."""

from numbers import Real

import torch
from torch import Tensor


def _validate_alpha(alpha: float) -> None:
    if not isinstance(alpha, Real) or isinstance(alpha, bool) or not 0 < alpha <= 1:
        raise ValueError("alpha must be a finite real number in (0, 1]")


def _validate_inputs(table: Tensor, states: Tensor, target: Tensor, *, table_ndim: int) -> None:
    """Validate the shared V/Q table, state and target contract before writing."""
    if not all(isinstance(value, Tensor) for value in (table, states, target)):
        raise TypeError("table, states and target must be Tensors")
    if table.ndim != table_ndim:
        shape = "(N,)" if table_ndim == 1 else "(N, A)"
        raise ValueError(f"table must have shape {shape}")
    if not table.is_floating_point():
        raise TypeError("table must have floating dtype")
    if states.dtype != torch.long:
        raise TypeError("states must have torch.long dtype")
    if states.ndim > 1:
        raise ValueError("states must be scalar () or a batch (B,)")
    if states.shape != target.shape:
        raise ValueError("states and target must have the same shape")
    if target.dtype != table.dtype:
        raise TypeError("target must have the same floating dtype as table")
    if table.device != states.device or table.device != target.device:
        raise ValueError("table, states and target must be on the same device")
    if torch.any(states < 0) or torch.any(states >= table.shape[0]):
        raise IndexError("State index out of bounds")


class TabularVUpdateRule:
    """Apply V[s] += alpha * (target - V[s]) in input order.

    v_table is a floating (N,) Tensor. states is long with shape () or (B,);
    target has the same shape and the table's dtype/device. The original
    table is updated in place under no_grad. Returned signed errors have
    target's shape and no graph. Repeated states see earlier row updates.

    Targets are copied once before writing, including when they alias the
    table. Bootstrap values are not recomputed during the sequential loop.
    """

    def __init__(self, alpha: float) -> None:
        _validate_alpha(alpha)
        self.alpha = float(alpha)

    @torch.no_grad()
    def update(self, v_table: Tensor, states: Tensor, target: Tensor) -> Tensor:
        _validate_inputs(v_table, states, target, table_ndim=1)
        targets = target.clone().reshape(-1)
        td_error = torch.empty_like(targets)
        for index, state in enumerate(states.reshape(-1)):
            td_error[index] = targets[index] - v_table[state]
            v_table[state] += self.alpha * td_error[index]
        return td_error.reshape(target.shape)


class TabularQUpdateRule:
    """Apply Q[s,a] += alpha * (target - Q[s,a]) in input order.

    q_table is a floating (N,A) Tensor. states/actions are long, scalar ()
    or batch (B,), matching target's shape. All inputs share the table's
    device; target shares its dtype. The original table is updated in place
    under no_grad and returned signed errors have no graph.

    Targets are copied before any writes. Duplicate pairs are updated
    sequentially, with an error calculated immediately before each update.
    """

    def __init__(self, alpha: float) -> None:
        _validate_alpha(alpha)
        self.alpha = float(alpha)

    @torch.no_grad()
    def update(self, q_table: Tensor, states: Tensor, actions: Tensor, target: Tensor) -> Tensor:
        self._validate(q_table, states, actions, target)
        targets = target.clone().reshape(-1)
        td_error = torch.empty_like(targets)
        for index, (state, action) in enumerate(zip(states.reshape(-1), actions.reshape(-1))):
            td_error[index] = targets[index] - q_table[state, action]
            q_table[state, action] += self.alpha * td_error[index]
        return td_error.reshape(target.shape)

    @staticmethod
    def _validate(q_table: Tensor, states: Tensor, actions: Tensor, target: Tensor) -> None:
        _validate_inputs(q_table, states, target, table_ndim=2)
        if not isinstance(actions, Tensor):
            raise TypeError("actions must be a Tensor")
        if actions.dtype != torch.long:
            raise TypeError("actions must have torch.long dtype")
        if actions.shape != states.shape:
            raise ValueError("states, actions and target must have the same shape")
        if actions.device != q_table.device:
            raise ValueError("actions must be on the same device as table")
        if torch.any(actions < 0) or torch.any(actions >= q_table.shape[1]):
            raise IndexError("Action index out of bounds")
