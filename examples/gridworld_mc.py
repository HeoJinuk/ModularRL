"""Run python -m examples.gridworld_mc; compare fixed-policy MC with exact DP."""

import argparse

import numpy as np
import torch

from modular_rl.component.estimator import TabularVEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import MonteCarloReturnGenerator
from modular_rl.dp import policy_evaluation, uniform_policy
from modular_rl.environment import TabularEpisodeSampler, sutton_gridworld
from modular_rl.learner import MCPredictionLearner
from modular_rl.model import Model


def train_and_compare(episodes=20_000, seed=7, max_steps=10_000, log_every=5000):
    """Train both modes on identical episodes with uniformly chosen nonterminal starts.

    A censored episode aborts the experiment before either learner sees it.
    This avoids treating a time-limit return as a full MC sample or silently
    selecting only short successful episodes. No monotonic-RMSE claim is made.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    mdp = sutton_gridworld()
    policy = uniform_policy(mdp)
    reference = policy_evaluation(mdp, policy, gamma=1, theta=1e-10)
    if not reference.converged:
        raise RuntimeError("DP reference did not converge")
    nonterminal = np.array([state for state in range(mdp.n_states) if state not in mdp.terminal_states])
    start_rng = np.random.default_rng(seed)
    sampler = TabularEpisodeSampler(mdp, policy, seed=seed + 1)
    learners = {}
    for mode in ("first", "every"):
        model = Model(IdentityRepresentation(), TabularVEstimator(mdp.n_states)).double()
        learners[mode] = MCPredictionLearner(model, MonteCarloReturnGenerator(gamma=1), mode)

    def rmse(learner):
        values = learner.model.estimator.v_table.cpu().numpy()
        return float(np.sqrt(np.mean((values[nonterminal] - reference.values[nonterminal]) ** 2)))

    initial_rmse = rmse(learners["first"])
    steps = 0
    start_counts = np.zeros(mdp.n_states, dtype=np.int64)
    for number in range(1, episodes + 1):
        start = int(start_rng.choice(nonterminal))
        episode = sampler.sample(start, max_steps=max_steps)
        if not episode.terminated:
            raise RuntimeError(f"episode={number} was truncated at {len(episode)} steps; "
                               "MC did not learn it. Increase max_steps or inspect the policy.")
        start_counts[start] += 1
        steps += len(episode)
        for learner in learners.values():
            learner.learn(episode)
        if log_every and (number % log_every == 0 or number == episodes):
            print(f"episodes={number}/{episodes} steps={steps} "
                  f"first_rmse={rmse(learners['first']):.6f} every_rmse={rmse(learners['every']):.6f}", flush=True)

    results = {"episodes": episodes, "steps": steps, "initial_rmse": initial_rmse,
               "reference": reference.values.copy(), "start_counts": start_counts}
    for mode, learner in learners.items():
        values = learner.model.forward(np.arange(mdp.n_states)).values.v
        results[mode] = {"values": values.cpu().numpy().copy(),
                         "visit_counts": learner.visit_counts.cpu().numpy().copy(), "rmse": rmse(learner)}
    return learners, results


def main():
    parser = argparse.ArgumentParser(description="First/every-visit MC on Sutton's uniform Gridworld policy")
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=10_000)
    args = parser.parse_args()
    _, results = train_and_compare(args.episodes, args.seed, args.max_steps)
    print(f"Sutton 4x4 MC Prediction: gamma=1 seed={args.seed} "
          f"episodes={results['episodes']} steps={results['steps']} truncated=0")
    print(f"initial_rmse={results['initial_rmse']:.6f} (14 nonterminal states)")
    with np.printoptions(precision=4, suppress=True):
        print("DP uniform-policy V:")
        print(results["reference"].reshape(4, 4))
        print("Start counts:")
        print(results["start_counts"].reshape(4, 4))
        for mode in ("first", "every"):
            metrics = results[mode]
            print(f"{mode}-visit V (RMSE={metrics['rmse']:.6f}):")
            print(metrics["values"].reshape(4, 4))
            print(f"{mode}-visit update counts:")
            print(metrics["visit_counts"].reshape(4, 4))


if __name__ == "__main__":
    main()
