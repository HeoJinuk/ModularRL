"""Expected SARSA using the Model’s own discrete policy selector."""

import torch
from torch import Tensor
from modular_rl.component.estimator import TabularQEstimator
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import TransitionBatch
from modular_rl.model import Model


class ExpectedSARSALearner:
    """Expected SARSA using the Model’s own discrete policy selector."""

    def __init__(
        self,
        model: Model,
        target_generator: TD0TargetGenerator,
        update_rule: TabularQUpdateRule,
    ) -> None:
        self.model = model
        self.target_generator = target_generator
        self.update_rule = update_rule

    def learn(self, batch: TransitionBatch) -> Tensor:
        """Compose a frozen TD target and in-place Q update from an external batch.

        The Model’s own action selector supplies
        discrete probabilities. Batch rows are (B,); states/actions are integer,
        rewards floating and boundary flags bool. Q/probabilities are (B,A) with
        equal dtype/device. The live table determines batch dtype/device.

        One estimate supplies both Q and pi before any writes. No action/RNG is
        sampled. Termination is masked by the existing target generator; time-limit
        truncation bootstraps from the collector's actual final observation.
        Existing sequential duplicate-pair updates return signed errors (B,) with
        no graph. The learner owns no environment, buffer or sampling RNG."""
        model, target_generator, update_rule = self.model, self.target_generator, self.update_rule
        _validate(model, batch)
        policy_selector = model.action_selector
        table = model.estimator.q_table
        batch = batch.to(table.device, dtype=table.dtype)
        states, actions = batch.observation.long(), batch.action.long()
        with torch.no_grad():
            estimate = model.forward(batch.next_observation.long())
            probabilities = policy_selector.probabilities(estimate)
            next_q = estimate.values.q[0]
            _validate_estimate(next_q, probabilities, batch.reward.numel(), table)
            next_value = (probabilities * next_q).sum(dim=-1)
            target = target_generator.generate(batch.reward, next_value, batch.terminated)
        return update_rule.update(table, states, actions, target)


def _validate(model, batch):
    if not isinstance(model, Model) or not isinstance(model.estimator, TabularQEstimator):
        raise TypeError("Expected SARSA composition requires a Model with TabularQEstimator")
    policy_selector = model.action_selector
    if not callable(getattr(policy_selector, "probabilities", None)):
        raise TypeError("policy_selector must support discrete probabilities")
    if not isinstance(batch, TransitionBatch):
        raise TypeError("learn expects a TransitionBatch")
    if not isinstance(batch.observation, Tensor):
        raise TypeError("observation must be a Tensor")
    if batch.observation.ndim != 1 or batch.observation.numel() == 0:
        raise ValueError("observation must have nonempty tabular batch shape (B,)")
    for name in ("action", "reward", "next_observation", "terminated", "truncated"):
        value = getattr(batch, name)
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a Tensor")
        if value.shape != batch.observation.shape:
            raise ValueError("tabular batch fields must have the same (B,) shape")
    for name in ("observation", "next_observation", "action"):
        value = getattr(batch, name)
        if value.is_floating_point() or value.is_complex() or value.dtype == torch.bool:
            raise TypeError(f"{name} must have integer index dtype")
    if not batch.reward.is_floating_point():
        raise TypeError("reward must have floating dtype")
    if batch.terminated.dtype != torch.bool or batch.truncated.dtype != torch.bool:
        raise TypeError("terminated and truncated must have bool dtype")


def _validate_estimate(q, probabilities, batch_size, table):
    if not isinstance(probabilities, Tensor):
        raise TypeError("probabilities must return a Tensor")
    if q.shape != (batch_size, table.shape[1]) or probabilities.shape != q.shape:
        raise ValueError("Q and probabilities must have the same (B,A) shape without broadcasting")
    if probabilities.dtype != q.dtype or probabilities.device != q.device:
        raise ValueError("Q and probabilities must share dtype and device")
