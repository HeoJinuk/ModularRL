"""On-policy MC control in the book's 3x4 Gymnasium Gridworld."""

import argparse

import numpy as np
import torch

from modular_rl.component.action_selector import EpsilonGreedyActionSelector, GreedyActionSelector
from modular_rl.component.estimator import TabularQEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import MonteCarloReturnGenerator
from modular_rl.dp import value_iteration
from modular_rl.environment import DLFS4_START_STATE, DLFS4_STATES, collect_episode, make_dlfs4_gridworld_env
from modular_rl.learner import TabularMCControlLearner
from modular_rl.model import Model


def evaluate_greedy(model, env, *, seed):
    """Evaluate without learning/exploration; report a cutoff as a cutoff."""
    episode = collect_episode(env, lambda state: model.act(state, deterministic=True).action.item(), seed=seed)
    return {"steps": len(episode), "reward": sum(t.reward.item() for t in episode.transitions),
            "terminated": episode.terminated, "truncated": episode.truncated,
            "path": [episode.transitions[0].observation.item()] +
                    [t.next_observation.item() for t in episode.transitions]}


def train_and_evaluate(episodes=1000, seed=7, gamma=.9, epsilon=.1, max_steps=2000, log_every=100):
    """Keep Q fixed during collection; update sample averages at episode boundaries.

    Model owns the epsilon-greedy behavior policy. Environment seed and PyTorch
    action seed are explicit; deterministic evaluation uses a separate capped
    environment and does not draw from the behavior RNG. Truncated-only training
    episodes abort before learning, rather than biasing sampling by discarding them.
    Fixed epsilon is exploration, not a guarantee of all-action Q* convergence.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    torch.manual_seed(seed)
    model = Model(IdentityRepresentation(), TabularQEstimator(len(DLFS4_STATES), 4),
                  EpsilonGreedyActionSelector(epsilon)).double()
    learner = TabularMCControlLearner(model, MonteCarloReturnGenerator(gamma))
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
            episode = collect_episode(env, lambda state: model.act(state).action.item(),
                                      seed=seed if number == 1 else None)
            if not episode.terminated:
                raise RuntimeError(f"episode={number} was truncated; MC Control did not learn it. "
                                   "Increase max_steps or inspect the policy.")
            learner.learn(episode)
            steps += len(episode)
            truncated_episodes += int(episode.truncated)
            rewards.append(sum(t.reward.item() for t in episode.transitions))
            if number == episodes or (log_every and number % log_every == 0):
                evaluation = evaluate_greedy(model, eval_env, seed=seed + 1)
                recent_reward = float(np.mean(rewards[-100:]))
                history.append({"episodes": number, "steps": steps, "training_mean_reward": recent_reward,
                                "greedy": evaluation})
                if log_every:
                    print(f"episodes={number}/{episodes} steps={steps} train_mean_reward={recent_reward:.3f} "
                          f"greedy_goal={evaluation['terminated']} greedy_steps={evaluation['steps']} "
                          f"greedy_reward={evaluation['reward']:.1f}", flush=True)
    finally:
        env.close()
        eval_env.close()

    policy = GreedyActionSelector().probabilities(model.forward(np.arange(len(DLFS4_STATES))))
    return learner, {"episodes": episodes, "steps": steps, "seed": seed, "gamma": gamma, "epsilon": epsilon,
                     "before": before, "after": history[-1]["greedy"], "history": history,
                     "training_rewards": np.array(rewards), "truncated_episodes": truncated_episodes,
                     "q_values": model.estimator.q_table.cpu().numpy().copy(),
                     "visit_counts": learner.visit_counts.cpu().numpy().copy(),
                     "greedy_policy": policy.cpu().numpy().copy(), "dp_values": optimal.values.copy(),
                     "dp_policy": optimal.policy.copy(), "dp_steps": len(dp_episode)}


def main():
    parser = argparse.ArgumentParser(description="First-visit on-policy MC control using Model.act")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=.9)
    parser.add_argument("--epsilon", type=float, default=.1)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    _, result = train_and_evaluate(args.episodes, args.seed, args.gamma, args.epsilon, args.max_steps, args.log_every)
    print(f"DLFS4 on-policy MC Control: seed={args.seed} gamma={args.gamma} epsilon={args.epsilon} "
          f"episodes={result['episodes']} steps={result['steps']} truncated={result['truncated_episodes']}")
    for name in ("before", "after"):
        evaluation = result[name]
        print(f"{name} greedy: goal={evaluation['terminated']} steps={evaluation['steps']} "
              f"reward={evaluation['reward']:.1f} truncated={evaluation['truncated']}")
    print("Greedy path:", [DLFS4_STATES[state] for state in result["after"]["path"]])
    print(f"DP optimal: steps={result['dp_steps']} V(start)={result['dp_values'][DLFS4_START_STATE]:.4f}")
    with np.printoptions(precision=4, suppress=True):
        print("Q (state rows; actions up/down/left/right):\n", result["q_values"])
        print("State-action visit counts:\n", result["visit_counts"])
        print("Greedy policy probabilities (uniform ties):\n", result["greedy_policy"])


if __name__ == "__main__":
    main()
