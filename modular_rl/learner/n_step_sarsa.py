"""Ordered online SARSA windows; collection and tail scheduling stay external."""

import torch
from torch import Tensor
from numbers import Integral
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.model import Model
from modular_rl.learner._tabular_window import _validate_window, _prepare_next_action


class NStepSARSALearner:
    """Ordered online SARSA windows; collection and tail scheduling stay external."""

    def __init__(
        self,
        model: Model,
        target_generator: TD0TargetGenerator,
        update_rule: TabularQUpdateRule,
    ) -> None:
        self.model = model
        self.target_generator = target_generator
        self.update_rule = update_rule

    def learn(self, window, *, next_action=None) -> Tensor:
        """Update only the oldest row of an ordered window of B=1 batches.

        Window length is the actual reward count m, not a nominal fixed n.
        Contiguous rows cannot cross an episode boundary. next_action is the
        actual action already selected at the last next observation; scalar or
        (1,) integer IDs are accepted. None is valid only for natural termination.

        The caller owns window storage and removes its oldest row after success.
        Each call reads current Q under no_grad, then composes the existing TD0
        backup in reverse reward order. This yields gamma**m bootstrap, including
        shortened time-limit tails. Natural termination dominates truncation.
        Flush calls re-evaluate Q independently after earlier writes, retaining
        the same final selected action. Signed error (1,) follows table dtype/device
        and has no graph. The function never selects an action or consumes RNG."""
        model, target_generator, update_rule = self.model, self.target_generator, self.update_rule
        _validate_window(model, window)
        table = model.estimator.q_table
        batches = [batch.to(table.device, dtype=table.dtype) for batch in window]
        last, first = batches[-1], batches[0]
        action = _prepare_next_action(next_action, last, table)
        with torch.no_grad():
            if last.terminated.item():
                target = torch.zeros_like(last.reward)
            else:
                next_q = model.forward(last.next_observation.long()).values.q[0]
                target = next_q.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            for batch in reversed(batches):
                target = target_generator.generate(batch.reward, target, batch.terminated)
        return update_rule.update(table, first.observation.long(), first.action.long(), target)


def _validate_n(n):
    if not isinstance(n, Integral) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
