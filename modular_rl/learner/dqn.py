"""Basic DQN: online action values, target bootstrap and one optimizer step."""

import torch
from torch import Tensor
from torch.optim import Optimizer

from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TDLossUpdateRule
from modular_rl.data import EstimatorOutput, TDLossOutput, TransitionBatch
from modular_rl.model import Model


class DQNLearner:
    """Learn from external replay batches, without sampling or choosing actions.

    Observations preserve their leading batch axis B. Discrete actions,
    floating rewards and bool boundary flags have shape (B,). Each model
    must return exactly one floating scalar-discrete Q Tensor (B, A).

    Supply independent, structurally matching Models on the same device and
    floating dtype, and an optimizer containing only online parameters.
    Target parameters are frozen and evaluated in eval mode. Construction
    does not copy weights: call sync_target() for the initial hard copy and
    whenever the external training schedule requests it.

    learn() runs backward/step and returns detached pre-update signed TD
    errors (B,) and a scalar loss. Time-limit truncation retains bootstrap;
    the caller supplies the actual final observation, not an autoreset one.
    Online train/eval mode remains the caller's choice.
    """

    def __init__(
        self,
        online_model: Model,
        target_model: Model,
        optimizer: Optimizer,
        target_generator: TD0TargetGenerator,
        update_rule: TDLossUpdateRule,
    ) -> None:
        self.online_model = online_model
        self.target_model = target_model
        self.optimizer = optimizer
        self.target_generator = target_generator
        self.update_rule = update_rule
        self._validate_models()
        self._validate_optimizer()
        self.target_model.requires_grad_(False)
        self.target_model.zero_grad(set_to_none=True)
        self.target_model.eval()

    def learn(self, batch: TransitionBatch) -> TDLossOutput:
        self._validate(batch)
        reference = next(self.online_model.parameters())
        batch = batch.to(reference.device, dtype=reference.dtype)
        actions = batch.action.long()
        batch_size = actions.numel()

        q = self._q_values(self.online_model(batch.observation), batch_size, reference)
        self._validate_actions(actions, q.shape[-1])
        prediction = q.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            self.target_model.eval()
            next_q = self._q_values(self.target_model(batch.next_observation), batch_size, reference,
                                   n_actions=q.shape[-1])
            next_value = next_q.max(dim=-1).values
            target = self.target_generator.generate(batch.reward, next_value, batch.terminated)

        return self.optimize(prediction, target)

    def optimize(self, prediction: Tensor, target: Tensor) -> TDLossOutput:
        """Apply one semi-gradient update to externally prepared Q predictions.

        prediction and target are nonempty floating Tensors (B,) on the Online
        Model's dtype/device. The caller builds prediction from that model with
        gradients enabled and supplies a fixed target without a graph. Target
        gradients are rejected, never silently detached; graph provenance is
        the caller's responsibility. The existing UpdateRule checks matching
        shape/dtype/device before any optimizer call.

        No model evaluation, target generation/sync, sampling or action selection
        occurs here. Return detached pre-update signed errors and scalar loss.
        """
        self._validate_optimization(prediction, target)
        result = self.update_rule.compute(prediction, target)
        self.optimizer.zero_grad()
        result.loss.backward()
        self.optimizer.step()
        return TDLossOutput(td_error=result.td_error.detach(), loss=result.loss.detach())

    @torch.no_grad()
    def sync_target(self) -> None:
        """Hard-copy model state and all buffers; preserve target storage and eval mode."""
        self._validate_models()
        state = self.online_model.state_dict()
        self.target_model.load_state_dict(state, strict=True)
        # state_dict omits nonpersistent buffers, which can also affect inference.
        target_buffers = dict(self.target_model.named_buffers())
        for name, buffer in self.online_model.named_buffers():
            if name not in state:
                target_buffers[name].copy_(buffer)
        self.target_model.eval()

    def _validate_models(self) -> None:
        """Check replica compatibility and storage independence at setup/synchronization."""
        if not isinstance(self.online_model, Model) or not isinstance(self.target_model, Model):
            raise TypeError("DQNLearner requires Online and Target Models")
        online_parameters = dict(self.online_model.named_parameters())
        if not online_parameters or not any(p.requires_grad for p in online_parameters.values()):
            raise ValueError("Online Model must have trainable floating parameters")
        reference = next(iter(online_parameters.values()))
        online = {**online_parameters, **dict(self.online_model.named_buffers())}
        target = {**dict(self.target_model.named_parameters()), **dict(self.target_model.named_buffers())}
        online_modules = {name: type(module) for name, module in self.online_model.named_modules()}
        target_modules = {name: type(module) for name, module in self.target_model.named_modules()}
        if online_modules != target_modules or online.keys() != target.keys():
            raise ValueError("Online and Target Models must have matching structure")
        if self.online_model.state_dict().keys() != self.target_model.state_dict().keys():
            raise ValueError("Online and Target Models must have matching state_dict structure")
        for name, value in online.items():
            other = target[name]
            if value.shape != other.shape:
                raise ValueError(f"Online and Target Models must have matching shape: {name}")
            if value.device != reference.device or other.device != reference.device:
                raise ValueError("Online and Target Models must be on the same device")
            if value.dtype != other.dtype or (value.is_floating_point() and value.dtype != reference.dtype):
                raise TypeError("Online and Target Models must have the same floating dtype")
        if any(not p.is_floating_point() for p in online_parameters.values()):
            raise TypeError("Online parameters must have floating dtype")
        online_storage = {value.untyped_storage().data_ptr() for value in online.values() if value.numel()}
        target_storage = {value.untyped_storage().data_ptr() for value in target.values() if value.numel()}
        if self.online_model is self.target_model or online_storage & target_storage:
            raise ValueError("Online and Target Models must have independent parameter and buffer storage")

    def _validate_optimizer(self) -> None:
        if not isinstance(self.optimizer, Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        online_ids = {id(parameter) for parameter in self.online_model.parameters()}
        parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
        if not parameters or any(id(p) not in online_ids for p in parameters):
            raise ValueError("Optimizer must contain only Online Model parameters, never Target parameters")

    def _validate_optimization(self, prediction: Tensor, target: Tensor) -> None:
        """Check DQN-specific batch, gradient and Online Model contracts."""
        if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
            raise TypeError("prediction and target must be Tensors")
        if prediction.ndim != 1:
            raise ValueError("DQN prediction must have shape (B,)")
        if target.requires_grad:
            raise ValueError("DQN target must be fixed; compute it under no_grad or detach it explicitly")
        if not prediction.requires_grad:
            raise ValueError("DQN prediction must retain its gradient path")
        reference = next(self.online_model.parameters())
        if prediction.dtype != reference.dtype or prediction.device != reference.device:
            raise ValueError("DQN prediction must match Online Model dtype and device")

    @staticmethod
    def _validate(batch: TransitionBatch) -> None:
        if not isinstance(batch, TransitionBatch):
            raise TypeError("learn expects a TransitionBatch")
        for name in ("observation", "next_observation", "action", "reward", "terminated", "truncated"):
            if not isinstance(getattr(batch, name), Tensor):
                raise TypeError(f"{name} must be a Tensor")
        if batch.observation.ndim < 1 or batch.observation.shape[0] == 0:
            raise ValueError("observation must have a nonempty leading batch dimension")
        if batch.next_observation.shape != batch.observation.shape:
            raise ValueError("observation and next_observation must have the same shape")
        expected = (batch.observation.shape[0],)
        if any(getattr(batch, name).shape != expected for name in ("action", "reward", "terminated", "truncated")):
            raise ValueError("actions, rewards and boundary flags must have shape (B,)")
        if batch.action.is_floating_point() or batch.action.is_complex() or batch.action.dtype == torch.bool:
            raise TypeError("action must have integer index dtype")
        if not batch.reward.is_floating_point():
            raise TypeError("reward must have floating dtype")
        if batch.terminated.dtype != torch.bool or batch.truncated.dtype != torch.bool:
            raise TypeError("terminated and truncated must have bool dtype")

    @staticmethod
    def _q_values(
        estimate: EstimatorOutput, batch_size: int, reference: Tensor, *, n_actions: int | None = None
    ) -> Tensor:
        """Extract one scalar-discrete critic; never choose among multiple critics."""
        if estimate.values is None or not estimate.values.q:
            raise ValueError("DQN requires Q-values in EstimatorOutput.values.q")
        if len(estimate.values.q) != 1:
            raise ValueError("DQN requires exactly one Q estimate")
        q = estimate.values.q[0]
        if not isinstance(q, Tensor) or not q.is_floating_point():
            raise TypeError("DQN requires a floating scalar-discrete Q Tensor")
        if q.ndim != 2 or q.shape[0] != batch_size or q.shape[-1] == 0:
            raise ValueError("DQN Q-values must have shape (B, A) with a nonempty action axis")
        if n_actions is not None and q.shape[-1] != n_actions:
            raise ValueError("Online and Target Q must have the same (B, A) shape")
        if q.dtype != reference.dtype or q.device != reference.device:
            raise ValueError("DQN Q-values must match Online Model dtype and device")
        return q

    @staticmethod
    def _validate_actions(actions: Tensor, n_actions: int) -> None:
        if torch.any((actions < 0) | (actions >= n_actions)):
            raise ValueError("action indices must be in [0, n_actions)")
