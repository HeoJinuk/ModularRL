"""Double DQN shares DQN replay, n-step targets, optimization and target sync."""

from modular_rl.learner._evaluation import evaluating
from modular_rl.learner.dqn import DQNLearner


class DoubleDQNLearner(DQNLearner):
    """Select the bootstrap action online, then evaluate it with the target.

    Constructor, learn(TransitionBatch | SequenceBatch), optimize() and
    sync_target() retain DQNLearner's contracts. Bootstrap runs under no_grad.
    Online selection temporarily uses eval mode to avoid Dropout sampling and
    BatchNorm updates; all caller modes are restored before optimization.
    Current-state prediction still uses the caller's online train/eval mode.
    Argmax ties select the smallest action index, as in PyTorch argmax.
    """

    def _bootstrap(self, next_observation, batch_size, reference, *, n_actions):
        self.target_model.eval()
        with evaluating(self.online_model):
            online_q = self._q_values(self.online_model(next_observation), batch_size, reference,
                                     n_actions=n_actions)
            next_action = online_q.argmax(dim=-1, keepdim=True)
        target_q = self._q_values(self.target_model(next_observation), batch_size, reference,
                                 n_actions=n_actions)
        return target_q.gather(-1, next_action).squeeze(-1)
