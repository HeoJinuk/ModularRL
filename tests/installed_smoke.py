"""Exercise installed algorithms without importing repository examples or tests.

Copy this script outside the checkout and run it with the installation's
python -I. Third-party runtime dependencies may come from system site packages.
"""

from copy import deepcopy
import importlib
import importlib.util
from pathlib import Path
import sys

import torch
import numpy as np

from modular_rl import Model, representations as R, estimators as E, selectors as S
from modular_rl import buffers as B, targets as T, updates as U, learners as L
from modular_rl import world_models as W, planners as P
from modular_rl.data import Transition


def actor_model():
    model = Model(R.MLP(1, (4,)), E.ActorCritic(E.CategoricalPolicy(4, 2), E.StateValue(4)), S.Categorical())
    return model.eval()


def rollout(model):
    buffer = B.Rollout(2, 1)
    for step in range(2):
        observation = torch.tensor([[float(step)]])
        with torch.no_grad():
            estimate = model(observation)
            action = model.action_selector.select(estimate)
        buffer.add(observation, action.action, torch.ones(1), observation + 1,
                   torch.tensor([False]), torch.tensor([step == 1]),
                   value=estimate.values.v, log_prob=action.log_prob)
    return buffer.get()


def main():
    import modular_rl
    assert Path(modular_rl.__file__).is_relative_to(Path(sys.prefix)), modular_rl.__file__
    assert importlib.util.find_spec('examples') is None
    canonical = dict(representations='component.representation', estimators='component.estimator',
                     selectors='component.action_selector', buffers='component.buffer',
                     targets='component.target_generator', updates='component.update_rule',
                     learners='learner', world_models='component.world_model', planners='component.planner')
    for name, path in canonical.items():
        public = importlib.import_module('modular_rl.' + name)
        original = importlib.import_module('modular_rl.' + path)
        for alias in public.__all__:
            cls = getattr(public, alias)
            assert getattr(original, cls.__name__) is cls

    observation = np.array([1., 2.], dtype=np.float32)
    probe = Model(R.Identity(), E.DiscreteQ(2, 2), S.Greedy())
    assert probe(observation).values.q[0].shape == (2,)
    assert probe.act(observation).action.cpu().numpy().shape == ()

    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = Model(R.Identity(), E.TabularQ(3, 2), S.EpsilonGreedy(.1))
    learner = L.QLearning(model, T.TD0(.9), U.TabularQ(.5))
    transition = Transition(0, 1, 1., 1, True, False)
    assert learner.learn(transition.to_batch()).item() == 1
    assert model.estimator.q_table[0, 1].item() == .5
    world = W.Tabular(); world.observe(transition)
    assert P.Dyna(world, learner, seed=7).plan(1)['updates'] == 1
    assert model.estimator.q_table[0, 1].item() == .75

    neural_batch = Transition(torch.tensor([0.]), 0, 1., torch.tensor([1.]), True, False).to_batch()
    for cls in (L.DQN, L.DoubleDQN):
        online = Model(R.MLP(1, (4,)), E.DiscreteQ(4, 2))
        target = deepcopy(online)
        trainer = cls(online, target, torch.optim.Adam(online.parameters()), T.TD0(.9), U.TDLoss())
        trainer.sync_target()
        result = trainer.learn(neural_batch)
        assert torch.isfinite(result.loss) and not result.loss.requires_grad
        assert any(p.grad is not None for p in online.parameters())

    model = actor_model()
    trainer = L.ActorCritic(model, torch.optim.Adam(model.parameters()), T.TD0(.9), U.ActorCritic())
    assert torch.isfinite(trainer.learn(neural_batch).total_loss)
    for target_generator in (T.RolloutReturn(.9), T.GAE(.9, .95)):
        model = actor_model()
        trainer = L.A2C(model, torch.optim.Adam(model.parameters()), target_generator, U.ActorCritic())
        assert torch.isfinite(trainer.learn(rollout(model)).total_loss)
    model = actor_model()
    trainer = L.PPO(model, torch.optim.Adam(model.parameters()), T.GAE(.9, .95), U.PPO(),
                    epochs=2, minibatch_size=1, max_grad_norm=.5, target_kl=.02, seed=7)
    result = trainer.learn(rollout(model))
    assert result.optimizer_steps > 0 and 1 <= result.epochs_completed <= 2
    assert torch.isfinite(result.total_loss) and not result.total_loss.requires_grad
    print('PASS: installed aliases, Q-learning, DQN, Double DQN, Dyna, ActorCritic, A2C, GAE, PPO')
    print('Loaded from:', modular_rl.__file__)


if __name__ == '__main__':
    main()
