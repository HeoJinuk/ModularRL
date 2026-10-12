"""Run from the package root: python examples/quickstart.py"""

import numpy as np
import torch

from modular_rl.component.action_selector import (
    CategoricalActionSelector,
    EpsilonGreedyActionSelector,
)
from modular_rl.component.estimator.actor_critic import ActorCriticEstimator
from modular_rl.component.estimator import (
    CategoricalPolicyEstimator,
    StateValueEstimator,
    TabularQEstimator,
)
from modular_rl.component.representation import IdentityRepresentation, MLPRepresentation
from modular_rl.model import Model


def main():
    tabular = Model(
        representation=IdentityRepresentation(),
        estimator=TabularQEstimator(16, 4),
        action_selector=EpsilonGreedyActionSelector(epsilon=0.1),
    )
    print("Tabular Q:", tabular.forward(5).values.q[0])
    print("Tabular action:", tabular.act(5).action.item())

    actor_critic = Model(
        representation=MLPRepresentation(4, hidden_dims=(64,)),
        estimator=ActorCriticEstimator(
            policy=CategoricalPolicyEstimator(64, 2),
            value=StateValueEstimator(64),
        ),
        action_selector=CategoricalActionSelector(),
    )
    output = actor_critic.forward(np.zeros(4, dtype=np.float32))
    action = actor_critic.act(torch.zeros(4))
    print("Actor-Critic value:", output.values.v)
    print("Actor-Critic action/log_prob:", action.action, action.log_prob)


if __name__ == "__main__":
    main()
