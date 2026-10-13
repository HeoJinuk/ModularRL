"""Compare DP, MC and transition-by-transition TD on the same Gridworld episodes."""

import argparse

import numpy as np

from modular_rl.component.estimator import TabularVEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import MonteCarloReturnGenerator, TD0TargetGenerator
from modular_rl.component.update_rule import TabularVUpdateRule
from modular_rl.dp import policy_evaluation, uniform_policy
from modular_rl.environment import TabularEpisodeSampler, sutton_gridworld
from modular_rl.learner import MCPredictionLearner, TabularTD0PredictionLearner
from modular_rl.model import Model


def train_and_compare(episodes=5000, seed=7, alphas=(0.01, 0.05, 0.1), max_steps=10_000, log_every=1000):
    """Give independent learners identical complete episodes under uniform pi, gamma=1.

    Starts are uniform over the 14 nonterminal states, using the same two RNG
    seeds as gridworld_mc. A cutoff aborts before any learner processes that
    episode. MC receives it whole; TD receives B=1 rows in chronological order.
    The episode is collected before learning, rather than using a live streaming
    Runner. RMSE history records both episode and environment-step coordinates.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not alphas or len(set(alphas)) != len(alphas):
        raise ValueError("alphas must be a nonempty sequence of distinct step sizes")
    mdp = sutton_gridworld()
    policy = uniform_policy(mdp)
    reference = policy_evaluation(mdp, policy, gamma=1, theta=1e-10)
    if not reference.converged:
        raise RuntimeError("DP reference did not converge")
    nonterminal = np.array([s for s in range(mdp.n_states) if s not in mdp.terminal_states])
    start_rng = np.random.default_rng(seed)
    sampler = TabularEpisodeSampler(mdp, policy, seed=seed + 1)
    learners = {}
    for mode in ("first", "every"):
        model = Model(IdentityRepresentation(), TabularVEstimator(mdp.n_states)).double()
        learners[mode] = MCPredictionLearner(model, MonteCarloReturnGenerator(1), mode)
    td_learners = {}
    for alpha in alphas:
        model = Model(IdentityRepresentation(), TabularVEstimator(mdp.n_states)).double()
        name = f"td(alpha={float(alpha)})"
        td_learners[name] = TabularTD0PredictionLearner(model, TD0TargetGenerator(1), TabularVUpdateRule(alpha))
    learners.update(td_learners)

    def errors():
        return {name: float(np.sqrt(np.mean(
            (learner.model.estimator.v_table.cpu().numpy()[nonterminal] - reference.values[nonterminal]) ** 2
        ))) for name, learner in learners.items()}

    history = [{"episodes": 0, "steps": 0, "rmse": errors()}]
    steps = 0
    start_counts = np.zeros(mdp.n_states, dtype=np.int64)
    for number in range(1, episodes + 1):
        start = int(start_rng.choice(nonterminal))
        episode = sampler.sample(start, max_steps=max_steps)
        if not episode.terminated:
            raise RuntimeError(f"episode={number} was truncated at {len(episode)} steps; "
                               "no learner processed it. Increase max_steps or inspect the policy.")
        start_counts[start] += 1
        steps += len(episode)
        for mode in ("first", "every"):
            learners[mode].learn(episode)
        for transition in episode.transitions:
            batch = transition.to_batch()
            for learner in td_learners.values():
                learner.learn(batch)
        if number == episodes or (log_every and number % log_every == 0):
            metrics = errors()
            history.append({"episodes": number, "steps": steps, "rmse": metrics})
            if log_every:
                print(f"seed={seed} episodes={number}/{episodes} steps={steps} " +
                      " ".join(f"{name}={rmse:.6f}" for name, rmse in metrics.items()), flush=True)

    results = {"episodes": episodes, "steps": steps, "seed": seed, "gamma": 1,
               "reference": reference.values.copy(), "start_counts": start_counts, "history": history,
               "methods": {}}
    for name, learner in learners.items():
        results["methods"][name] = {"values": learner.model.estimator.v_table.cpu().numpy().copy(),
                                    "rmse": history[-1]["rmse"][name]}
        if name in ("first", "every"):
            results["methods"][name]["visit_counts"] = learner.visit_counts.cpu().numpy().copy()
    return learners, results


def main():
    parser = argparse.ArgumentParser(description="DP/MC/TD(0) prediction using identical Gridworld episodes")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args()
    runs = []
    for seed in args.seeds:
        _, results = train_and_compare(args.episodes, seed, args.alphas, args.max_steps, args.log_every)
        runs.append(results)
        print(f"Sutton 4x4 DP/MC/TD(0): gamma=1 seed={seed} episodes={results['episodes']} "
              f"steps={results['steps']} truncated=0 (RMSE: 14 nonterminal states)")
        with np.printoptions(precision=4, suppress=True):
            print("DP uniform-policy V:")
            print(results["reference"].reshape(4, 4))
            for name, metrics in results["methods"].items():
                print(f"{name} V (RMSE={metrics['rmse']:.6f}):")
                print(metrics["values"].reshape(4, 4))
    print(f"Final RMSE across seeds={args.seeds}: mean +/- std (ddof=0)")
    for name in runs[0]["methods"]:
        rmse = np.array([run["methods"][name]["rmse"] for run in runs])
        print(f"{name}: {rmse.mean():.6f} +/- {rmse.std():.6f}")


if __name__ == "__main__":
    main()
