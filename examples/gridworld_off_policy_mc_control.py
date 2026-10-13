"""Weighted IS MC control with epsilon-greedy behavior in DLFS4 Gridworld."""

import argparse

import numpy as np
import torch

from examples.gridworld_mc_control import evaluate_greedy
from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import TabularQEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import MonteCarloReturnGenerator
from modular_rl.dp import value_iteration
from modular_rl.environment import DLFS4_START_STATE, DLFS4_STATES, collect_episode, make_dlfs4_gridworld_env
from modular_rl.learner import OffPolicyMCControlLearner
from modular_rl.model import Model


def train_and_evaluate(episodes=1000, seed=7, gamma=.9, epsilon=.1, max_steps=2000, log_every=100):
    """Collect with fixed Q/epsilon per episode, then update the same model.

    Exploration remains positive. Target greediness is deterministic with
    smallest-ID ties; evaluation uses the existing separate-env helper.
    A truncated-only training episode aborts before learning. DP compares the
    optimal values/path without imposing a stochastic test score threshold.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not 0 < epsilon <= 1:
        raise ValueError("epsilon must be in (0, 1] to cover target actions")
    torch.manual_seed(seed)
    model = Model(IdentityRepresentation(), TabularQEstimator(len(DLFS4_STATES), 4),
                  EpsilonGreedyActionSelector(epsilon)).double()
    learner = OffPolicyMCControlLearner(model, MonteCarloReturnGenerator(gamma))
    env = make_dlfs4_gridworld_env(max_episode_steps=max_steps)
    eval_env = make_dlfs4_gridworld_env(max_episode_steps=20)
    rewards, steps, truncated_episodes = [], 0, 0
    try:
        optimal = value_iteration(env.unwrapped.mdp, gamma=gamma, theta=1e-12)
        if not optimal.converged:
            raise RuntimeError("DP reference did not converge")
        before = evaluate_greedy(model, eval_env, seed=seed + 1)
        dp_episode = collect_episode(eval_env, lambda state: int(np.argmax(optimal.policy[state])), seed=seed + 1)
        history = [{"episodes": 0, "steps": 0, "training_mean_reward": None, "greedy": before}]
        for number in range(1, episodes + 1):
            episode = collect_episode(env, model.act, seed=seed if number == 1 else None)
            if not episode.terminated:
                raise RuntimeError(f"episode={number} was truncated; Off-policy MC Control did not learn it")
            learner.learn(episode)
            steps += len(episode)
            truncated_episodes += int(episode.truncated)
            rewards.append(sum(t.reward.item() for t in episode.transitions))
            if number == episodes or (log_every and number % log_every == 0):
                evaluation = evaluate_greedy(model, eval_env, seed=seed + 1)
                recent_reward = float(np.mean(rewards[-100:]))
                history.append({"episodes": number, "steps": steps, "training_mean_reward": recent_reward,
                                "episode_reward": rewards[-1], "greedy": evaluation})
                if log_every:
                    print(f"seed={seed} episode={number}/{episodes} reward={rewards[-1]:.1f} steps={steps} "
                          f"epsilon={model.action_selector.epsilon:.3f} train_mean_reward={recent_reward:.3f} "
                          f"greedy_goal={evaluation['terminated']} greedy_steps={evaluation['steps']}", flush=True)
    finally:
        env.close()
        eval_env.close()

    q_values = model.estimator.q_table.numpy().copy()
    nonterminal = np.setdiff1d(np.arange(len(DLFS4_STATES)), env.unwrapped.mdp.terminal_states)
    return learner, {"episodes": episodes, "steps": steps, "seed": seed, "gamma": gamma, "epsilon": epsilon,
                     "before": before, "after": history[-1]["greedy"], "history": history,
                     "training_rewards": np.array(rewards), "truncated_episodes": truncated_episodes,
                     "q_values": q_values, "weight_sums": learner.weight_sums.numpy().copy(),
                     "greedy_actions": q_values.argmax(axis=1), "dp_values": optimal.values.copy(),
                     "dp_steps": len(dp_episode),
                     "max_q_rmse": float(np.sqrt(np.mean((q_values.max(axis=1)[nonterminal] -
                                                          optimal.values[nonterminal]) ** 2)))}


def main():
    parser = argparse.ArgumentParser(description="Weighted IS off-policy MC control using recorded log b")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--gamma", type=float, default=.9)
    parser.add_argument("--epsilon", type=float, default=.1)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    for seed in args.seeds:
        _, result = train_and_evaluate(args.episodes, seed, args.gamma, args.epsilon, args.max_steps, args.log_every)
        print(f"DLFS4 Off-policy MC Control: seed={seed} gamma={args.gamma} epsilon={args.epsilon} "
              f"episodes={result['episodes']} steps={result['steps']} truncated={result['truncated_episodes']}")
        for name in ("before", "after"):
            evaluation = result[name]
            print(f"{name} greedy: goal={evaluation['terminated']} steps={evaluation['steps']} "
                  f"reward={evaluation['reward']:.1f} truncated={evaluation['truncated']}")
        print("Greedy path:", [DLFS4_STATES[state] for state in result["after"]["path"]])
        print(f"DP optimal: steps={result['dp_steps']} V(start)={result['dp_values'][DLFS4_START_STATE]:.4f} "
              f"max Q RMSE(nonterminal)={result['max_q_rmse']:.6f}")
        with np.printoptions(precision=4, suppress=True):
            print("Q (state rows; actions up/down/left/right):\n", result["q_values"])
            print("C cumulative weights:\n", result["weight_sums"])
            print("Deterministic greedy action IDs:", result["greedy_actions"])


if __name__ == "__main__":
    main()
