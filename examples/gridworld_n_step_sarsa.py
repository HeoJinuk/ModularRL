"""Online n-step SARSA through unchanged one-step target and table components."""

import argparse

from modular_rl.learner.n_step_sarsa import NStepSARSALearner, _validate_n
from modular_rl.learner._tabular_window import _validate_window, _prepare_next_action
from collections import deque
from numbers import Integral

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
from modular_rl.model import Model


def learn_n_step_sarsa(model: Model, window, target_generator: TD0TargetGenerator,
                       update_rule: TabularQUpdateRule, *, next_action=None) -> Tensor:
    """Compatibility entry point; collection stays in this example."""
    return NStepSARSALearner(model, target_generator, update_rule).learn(window, next_action=next_action)


def run_episode(env, model, target_generator, update_rule, *, n=3, seed=None):
    """Select next actions before writes; execute them once; flush at boundary.

    pending holds owned Transition.to_batch snapshots in chronological order.
    len(pending)==n is tau=t-n+1>=0; update one head before the next step.
    Ending flush performs all remaining updates without step or act calls.
    Each collected transition is updated once. No full Episode is collected
    before learning. Environment lifetime and checkpoints remain caller-owned.
    """
    _validate_n(n)
    observation, _ = env.reset(seed=seed)
    selected = model.act(observation)
    pending = deque()
    reward_sum, steps, updates = 0., 0, 0
    while True:
        action = selected.action.item()
        log_prob = None if selected.log_prob is None else selected.log_prob.detach().cpu().clone()
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_selected = None if terminated else model.act(next_observation)
        next_action = None if next_selected is None else next_selected.action
        pending.append(Transition(observation, action, reward, next_observation, terminated, truncated, log_prob).to_batch())
        if len(pending) == n:
            learn_n_step_sarsa(model, pending, target_generator, update_rule, next_action=next_action)
            pending.popleft()
            updates += 1
        reward_sum += reward
        steps += 1
        if terminated or truncated:
            while pending:
                learn_n_step_sarsa(model, pending, target_generator, update_rule, next_action=next_action)
                pending.popleft()
                updates += 1
            break
        observation, selected = next_observation, next_selected
    return {"reward": reward_sum, "steps": steps, "updates": updates, "terminated": terminated, "truncated": truncated}


def _evaluate_greedy(model, env, seed):
    episode = collect_episode(env, lambda state: model.act(state, deterministic=True), seed=seed)
    return {"steps": len(episode), "reward": sum(t.reward.item() for t in episode.transitions),
            "terminated": episode.terminated, "truncated": episode.truncated,
            "path": [episode.transitions[0].observation.item()] + [t.next_observation.item() for t in episode.transitions]}


def train_and_evaluate(n=3, episodes=1000, seed=7, gamma=.9, alpha=.8, epsilon=.1,
                       max_steps=2000, log_every=100):
    """Independent Gymnasium runs; no buffer sampling or Learner class.

    Environment uses reset(seed); behavior uses PyTorch manual_seed(seed).
    Greedy evaluation and target composition do not consume behavior RNG.
    n=1 uses the same action timing and arithmetic as TabularSARSALearner.
    Results include actual update counts, one per collected environment step.
    """
    _validate_n(n)
    if episodes < 1:
        raise ValueError("episodes must be positive")
    torch.manual_seed(seed)
    model = Model(IdentityRepresentation(), TabularQEstimator(len(DLFS4_STATES), 4),
                  EpsilonGreedyActionSelector(epsilon)).double()
    target_generator, update_rule = TD0TargetGenerator(gamma), TabularQUpdateRule(alpha)
    env = make_dlfs4_gridworld_env(max_episode_steps=max_steps)
    eval_env = make_dlfs4_gridworld_env(max_episode_steps=20)
    rewards, lengths, steps, updates, terminated_episodes, truncated_episodes = [], [], 0, 0, 0, 0
    try:
        optimal = value_iteration(env.unwrapped.mdp, gamma=gamma, theta=1e-12)
        if not optimal.converged:
            raise RuntimeError("DP reference did not converge")
        nonterminal = np.setdiff1d(np.arange(len(DLFS4_STATES)), env.unwrapped.mdp.terminal_states)
        before = _evaluate_greedy(model, eval_env, seed + 1)
        dp_episode = collect_episode(eval_env, lambda state: int(np.argmax(optimal.policy[state])), seed=seed + 1)
        history = [{"episodes": 0, "steps": 0, "updates": 0, "training_mean_reward": None, "greedy": before}]
        for number in range(1, episodes + 1):
            episode = run_episode(env, model, target_generator, update_rule, n=n, seed=seed if number == 1 else None)
            rewards.append(episode["reward"])
            lengths.append(episode["steps"])
            steps += episode["steps"]
            updates += episode["updates"]
            terminated_episodes += int(episode["terminated"])
            truncated_episodes += int(episode["truncated"])
            if number == episodes or (log_every and number % log_every == 0):
                evaluation = _evaluate_greedy(model, eval_env, seed + 1)
                recent_reward = float(np.mean(rewards[-100:]))
                history.append({"episodes": number, "steps": steps, "updates": updates, "training_mean_reward": recent_reward,
                                "episode_reward": episode["reward"], "episode_length": episode["steps"], "greedy": evaluation})
                if log_every:
                    print(f"algorithm=sarsa n={n} seed={seed} episode={number}/{episodes} reward={episode['reward']:.1f} "
                          f"length={episode['steps']} steps={steps} updates={updates} epsilon={epsilon:.3f} alpha={alpha:.3f} "
                          f"train_mean_reward={recent_reward:.3f} greedy_goal={evaluation['terminated']} greedy_steps={evaluation['steps']}", flush=True)
    finally:
        env.close()
        eval_env.close()

    q_values = model.estimator.q_table.numpy().copy()
    return model, {"algorithm": "n_step_sarsa", "n": int(n), "episodes": episodes, "steps": steps, "updates": updates,
            "seed": seed, "gamma": gamma, "alpha": alpha, "epsilon": epsilon, "before": before,
            "after": history[-1]["greedy"], "history": history, "training_rewards": np.array(rewards),
            "episode_lengths": np.array(lengths), "terminated_episodes": terminated_episodes,
            "truncated_episodes": truncated_episodes, "q_values": q_values, "dp_values": optimal.values.copy(),
            "dp_steps": len(dp_episode), "dp_path": [dp_episode.transitions[0].observation.item()] +
                                                    [t.next_observation.item() for t in dp_episode.transitions],
            "max_q_rmse": float(np.sqrt(np.mean((q_values.max(axis=1)[nonterminal] - optimal.values[nonterminal]) ** 2)))}


def main():
    from examples.gridworld_sarsa import train_and_evaluate as train_reference

    parser = argparse.ArgumentParser(description="Online n=1/3/5 SARSA composition vs existing SARSA/Q-learning")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    parser.add_argument("--ns", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--gamma", type=float, default=.9)
    parser.add_argument("--alpha", type=float, default=.8)
    parser.add_argument("--epsilon", type=float, default=.1)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    for seed in args.seeds:
        for n in args.ns:
            _, result = train_and_evaluate(n, args.episodes, seed, args.gamma, args.alpha, args.epsilon, args.max_steps, args.log_every)
            print(f"DLFS4 n-step SARSA: n={n} seed={seed} episodes={result['episodes']} steps={result['steps']} updates={result['updates']} "
                  f"terminated={result['terminated_episodes']} truncated={result['truncated_episodes']}")
            for name in ("before", "after"):
                value = result[name]
                print(f"{name} greedy: goal={value['terminated']} steps={value['steps']} reward={value['reward']:.1f} truncated={value['truncated']}")
            print("Greedy path:", [DLFS4_STATES[s] for s in result["after"]["path"]])
            print("DP optimal path:", [DLFS4_STATES[s] for s in result["dp_path"]])
            print(f"DP optimal: steps={result['dp_steps']} V(start)={result['dp_values'][DLFS4_START_STATE]:.4f} max Q RMSE={result['max_q_rmse']:.6f}")
            with np.printoptions(precision=4, suppress=True):
                print("Q (up/down/left/right):\n", result["q_values"])
        for algorithm in ("sarsa", "q_learning"):
            _, result = train_reference(algorithm, args.episodes, seed, args.gamma, args.alpha, args.epsilon, args.max_steps, args.log_every)
            print(f"Reference {algorithm}: seed={seed} steps={result['steps']} greedy_goal={result['after']['terminated']} "
                  f"greedy_steps={result['after']['steps']} max Q RMSE={result['max_q_rmse']:.6f}")


if __name__ == "__main__":
    main()
