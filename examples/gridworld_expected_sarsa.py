"""Expected SARSA by composing unchanged components, without a new learner."""

import argparse

from modular_rl.learner.expected_sarsa import ExpectedSARSALearner, _validate, _validate_estimate

import numpy as np
import torch
from torch import Tensor

from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import TabularQEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import Transition, TransitionBatch
from modular_rl.dp import value_iteration
from modular_rl.environment import DLFS4_START_STATE, DLFS4_STATES, collect_episode, make_dlfs4_gridworld_env
from modular_rl.learner import TabularQLearner, TabularSARSALearner
from modular_rl.model import Model


def learn_expected_sarsa(model: Model, batch: TransitionBatch, target_generator: TD0TargetGenerator,
                         update_rule: TabularQUpdateRule, *, policy_selector) -> Tensor:
    """Compatibility entry point; learning lives in ExpectedSARSALearner."""
    if policy_selector is not model.action_selector:
        raise ValueError("policy_selector must be the same object used by model.act")
    return ExpectedSARSALearner(model, target_generator, update_rule).learn(batch)


def _evaluate_greedy(model, env, seed):
    episode = collect_episode(env, lambda state: model.act(state, deterministic=True).action.item(), seed=seed)
    return {"steps": len(episode), "reward": sum(t.reward.item() for t in episode.transitions),
            "terminated": episode.terminated, "truncated": episode.truncated,
            "path": [episode.transitions[0].observation.item()] + [t.next_observation.item() for t in episode.transitions]}


def train_and_evaluate(algorithm="expected_sarsa", episodes=1000, seed=7, gamma=.9, alpha=.8,
                       epsilon=.1, max_steps=2000, log_every=100):
    """Independent online runs share budgets/settings, not sampled trajectories.

    Expected SARSA uses the example function and the model's own selector.
    Existing Q/SARSA learners supply the comparators. SARSA carries its actual
    pre-update next action; expected/max-Q control samples its next executed
    action after updating. Exploration and separate greedy evaluation differ.
    """
    if algorithm not in ("expected_sarsa", "q_learning", "sarsa"):
        raise ValueError("algorithm must be expected_sarsa, q_learning or sarsa")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    torch.manual_seed(seed)
    model = Model(IdentityRepresentation(), TabularQEstimator(len(DLFS4_STATES), 4),
                  EpsilonGreedyActionSelector(epsilon)).double()
    target_generator, update_rule = TD0TargetGenerator(gamma), TabularQUpdateRule(alpha)
    learner = None
    if algorithm == "q_learning":
        learner = TabularQLearner(model, target_generator, update_rule)
    elif algorithm == "sarsa":
        learner = TabularSARSALearner(model, target_generator, update_rule)
    env = make_dlfs4_gridworld_env(max_episode_steps=max_steps)
    eval_env = make_dlfs4_gridworld_env(max_episode_steps=20)
    rewards, lengths, steps, terminated_episodes, truncated_episodes = [], [], 0, 0, 0
    try:
        optimal = value_iteration(env.unwrapped.mdp, gamma=gamma, theta=1e-12)
        if not optimal.converged:
            raise RuntimeError("DP reference did not converge")
        before = _evaluate_greedy(model, eval_env, seed + 1)
        dp_episode = collect_episode(eval_env, lambda state: int(np.argmax(optimal.policy[state])), seed=seed + 1)
        history = [{"episodes": 0, "steps": 0, "training_mean_reward": None, "greedy": before}]
        for number in range(1, episodes + 1):
            observation, _ = env.reset(seed=seed if number == 1 else None)
            selected = model.act(observation)
            episode_reward, length = 0., 0
            while True:
                action = selected.action.item()
                log_prob = selected.log_prob.detach().cpu().clone()
                next_observation, reward, terminated, truncated, _ = env.step(action)
                transition = Transition(observation, action, reward, next_observation, terminated, truncated, log_prob)
                if algorithm == "sarsa":
                    next_selected = None if terminated else model.act(next_observation)
                    next_action = None if next_selected is None else next_selected.action.item()
                    learner.learn(transition.to_batch(), next_action=next_action)
                else:
                    if algorithm == "expected_sarsa":
                        learn_expected_sarsa(model, transition.to_batch(), target_generator, update_rule,
                                             policy_selector=model.action_selector)
                    else:
                        learner.learn(transition.to_batch())
                    next_selected = None if terminated or truncated else model.act(next_observation)
                episode_reward += reward
                length += 1
                if terminated or truncated:
                    break
                observation, selected = next_observation, next_selected
            steps += length
            terminated_episodes += int(terminated)
            truncated_episodes += int(truncated)
            rewards.append(episode_reward)
            lengths.append(length)
            if number == episodes or (log_every and number % log_every == 0):
                evaluation = _evaluate_greedy(model, eval_env, seed + 1)
                recent_reward = float(np.mean(rewards[-100:]))
                history.append({"episodes": number, "steps": steps, "training_mean_reward": recent_reward,
                                "episode_reward": episode_reward, "episode_length": length, "greedy": evaluation})
                if log_every:
                    print(f"algorithm={algorithm} seed={seed} episode={number}/{episodes} reward={episode_reward:.1f} "
                          f"length={length} steps={steps} epsilon={epsilon:.3f} alpha={alpha:.3f} "
                          f"train_mean_reward={recent_reward:.3f} greedy_goal={evaluation['terminated']} "
                          f"greedy_steps={evaluation['steps']}", flush=True)
    finally:
        env.close()
        eval_env.close()

    q_values = model.estimator.q_table.numpy().copy()
    nonterminal = np.setdiff1d(np.arange(len(DLFS4_STATES)), env.unwrapped.mdp.terminal_states)
    return model, {"algorithm": algorithm, "episodes": episodes, "steps": steps, "seed": seed,
                   "gamma": gamma, "alpha": alpha, "epsilon": epsilon, "before": before,
                   "after": history[-1]["greedy"], "history": history,
                   "training_rewards": np.array(rewards), "episode_lengths": np.array(lengths),
                   "terminated_episodes": terminated_episodes, "truncated_episodes": truncated_episodes,
                   "q_values": q_values, "dp_values": optimal.values.copy(), "dp_steps": len(dp_episode),
                   "max_q_rmse": float(np.sqrt(np.mean((q_values.max(axis=1)[nonterminal] -
                                                        optimal.values[nonterminal]) ** 2)))}


def main():
    parser = argparse.ArgumentParser(description="Expected SARSA composition vs unchanged Q/SARSA learners")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    parser.add_argument("--gamma", type=float, default=.9)
    parser.add_argument("--alpha", type=float, default=.8)
    parser.add_argument("--epsilon", type=float, default=.1)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    for seed in args.seeds:
        for algorithm in ("expected_sarsa", "q_learning", "sarsa"):
            _, result = train_and_evaluate(algorithm, args.episodes, seed, args.gamma, args.alpha,
                                          args.epsilon, args.max_steps, args.log_every)
            print(f"DLFS4 {algorithm}: seed={seed} episodes={result['episodes']} steps={result['steps']} "
                  f"terminated={result['terminated_episodes']} truncated={result['truncated_episodes']}")
            for name in ("before", "after"):
                evaluation = result[name]
                print(f"{name} greedy: goal={evaluation['terminated']} steps={evaluation['steps']} "
                      f"reward={evaluation['reward']:.1f} truncated={evaluation['truncated']}")
            print("Greedy path:", [DLFS4_STATES[state] for state in result["after"]["path"]])
            print(f"DP optimal: steps={result['dp_steps']} V(start)={result['dp_values'][DLFS4_START_STATE]:.4f} "
                  f"max Q RMSE(nonterminal)={result['max_q_rmse']:.6f}")
            with np.printoptions(precision=4, suppress=True):
                print("Q (state rows; actions up/down/left/right):\n", result["q_values"])


if __name__ == "__main__":
    main()
