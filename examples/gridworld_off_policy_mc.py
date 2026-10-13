"""Fixed-target V prediction with recorded behavior probabilities in Gymnasium."""

import argparse

import numpy as np
import torch

from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import TabularQEstimator, TabularVEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import MonteCarloReturnGenerator
from modular_rl.dp import policy_evaluation
from modular_rl.environment import collect_episode, make_dlfs4_gridworld_env
from modular_rl.learner import OffPolicyMCPredictionLearner
from modular_rl.model import Model


def train_and_compare(episodes=1000, seed=7, target_probs=(.3, .2, .2, .3), max_steps=2000, log_every=200):
    """Give ordinary/weighted first-visit estimators the same complete episodes.

    Behavior is a fixed uniform Model (epsilon=1); target probabilities are
    fixed and independent. Environment seed and PyTorch action seed are explicit.
    A cutoff aborts before learning rather than selectively discarding long
    episodes. RMSE uses the target DP value over ten nonterminal states.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    torch.manual_seed(seed)
    env = make_dlfs4_gridworld_env(max_episode_steps=max_steps)
    try:
        mdp = env.unwrapped.mdp
        target = np.tile(np.asarray(target_probs, dtype=np.float64), (mdp.n_states, 1))
        if target.shape != (mdp.n_states, mdp.n_actions):
            raise ValueError("target_probs must contain one probability per action")
        behavior = Model(IdentityRepresentation(), TabularQEstimator(mdp.n_states, mdp.n_actions),
                         EpsilonGreedyActionSelector(1)).double()
        learners = {}
        for method in ("ordinary", "weighted"):
            model = Model(IdentityRepresentation(), TabularVEstimator(mdp.n_states)).double()
            learners[method] = OffPolicyMCPredictionLearner(model, MonteCarloReturnGenerator(.9), target, method=method)
        reference = policy_evaluation(mdp, target, gamma=.9, theta=1e-12)
        if not reference.converged:
            raise RuntimeError("Target DP policy evaluation did not converge")
        nonterminal = np.setdiff1d(np.arange(mdp.n_states), mdp.terminal_states)

        def errors():
            return {name: float(np.sqrt(np.mean((learner.model.estimator.v_table.numpy()[nonterminal] -
                                                 reference.values[nonterminal]) ** 2)))
                    for name, learner in learners.items()}

        history = [{"episodes": 0, "steps": 0, "rmse": errors()}]
        steps, truncated_episodes = 0, 0
        for number in range(1, episodes + 1):
            episode = collect_episode(env, behavior.act, seed=seed if number == 1 else None)
            if not episode.terminated:
                raise RuntimeError(f"episode={number} was truncated; no off-policy learner processed it")
            for learner in learners.values():
                learner.learn(episode)
            steps += len(episode)
            truncated_episodes += int(episode.truncated)
            if number == episodes or (log_every and number % log_every == 0):
                metrics = errors()
                history.append({"episodes": number, "steps": steps, "rmse": metrics})
                if log_every:
                    print(f"seed={seed} episodes={number}/{episodes} steps={steps} " +
                          " ".join(f"{method}_rmse={value:.6f}" for method, value in metrics.items()), flush=True)
    finally:
        env.close()

    result = {"episodes": episodes, "steps": steps, "seed": seed, "gamma": .9, "history": history,
              "target_policy": target.copy(), "behavior_policy": np.full_like(target, 1 / mdp.n_actions),
              "reference": reference.values.copy(), "truncated_episodes": truncated_episodes, "methods": {}}
    for name, learner in learners.items():
        result["methods"][name] = {"values": learner.model.estimator.v_table.numpy().copy(),
                                   "rmse": history[-1]["rmse"][name],
                                   **{stat: getattr(learner, stat).numpy().copy() for stat in
                                      ("visit_counts", "weight_sums", "weighted_return_sums")}}
    return learners, result


def main():
    parser = argparse.ArgumentParser(description="Ordinary/weighted MC prediction of a fixed Gridworld target policy")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--target-uniform", action="store_true", help="Use pi=b to verify unit importance weights")
    args = parser.parse_args()
    target_probs = (.25,) * 4 if args.target_uniform else (.3, .2, .2, .3)
    runs = []
    print(f"Behavior b=(.25,.25,.25,.25); target pi={target_probs}; actions=up/down/left/right")
    for seed in args.seeds:
        _, result = train_and_compare(args.episodes, seed, target_probs, args.max_steps, args.log_every)
        runs.append(result)
        print(f"DLFS4 Off-policy MC Prediction: gamma=.9 seed={seed} episodes={result['episodes']} "
              f"steps={result['steps']} truncated={result['truncated_episodes']}")
        with np.printoptions(precision=4, suppress=True):
            print("Target DP V:", result["reference"])
            for name, metrics in result["methods"].items():
                print(f"{name} V (RMSE={metrics['rmse']:.6f}):", metrics["values"])
                print("visit counts:", metrics["visit_counts"])
                print("weight sums:", metrics["weight_sums"])
                if args.target_uniform:
                    print("unit-weight max |sum rho - count|:",
                          np.max(np.abs(metrics["weight_sums"] - metrics["visit_counts"])))
    print(f"RMSE across seeds={args.seeds}: mean +/- std (ddof=0)")
    for name in ("ordinary", "weighted"):
        errors = np.array([run["methods"][name]["rmse"] for run in runs])
        print(f"{name}: {errors.mean():.6f} +/- {errors.std():.6f}")


if __name__ == "__main__":
    main()
