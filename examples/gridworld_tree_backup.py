"""Online n-step Tree Backup by composing existing ModularRL components."""

import argparse

from modular_rl.learner.tree_backup import TreeBackupLearner, _probabilities, _validate_actions, _validate_update
from collections import deque
from pathlib import Path
import runpy

import numpy as np
import torch
from torch import Tensor

from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import TabularQEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import Transition
from modular_rl.dp import action_values, policy_evaluation
from modular_rl.environment import DLFS4_STATES, make_dlfs4_gridworld_env
from modular_rl.model import Model

# Existing examples also load from a notebook cwd without installing examples.
_off_policy = runpy.run_path(str(Path(__file__).with_name("gridworld_off_policy_n_step_sarsa.py")))
_n_step = _off_policy["_on_policy"]


def learn_tree_backup(model, window, target_generator, update_rule, *, target_policy):
    """Compatibility entry point; learning lives in TreeBackupLearner."""
    return TreeBackupLearner(model, target_generator, update_rule, target_policy=target_policy).learn(window)


def run_episode(env, model, target_generator, update_rule, *, target_policy, n=3, seed=None):
    """Step -> snapshot -> eligible head update -> next behavior action.

    No bootstrap-only action is selected at either episode boundary. Flush uses
    current expected Q per call, without step/act calls. Each collected transition
    is updated once; each real action is selected once. Window/state/RNG/checkpoint
    and environment lifetime remain caller-owned, with no Runner or learner class.
    """
    _n_step["_validate_n"](n)
    observation, _ = env.reset(seed=seed)
    selected = model.act(observation)
    pending = deque()
    reward_sum, steps, updates = 0., 0, 0
    while True:
        action = selected.action.item()
        log_prob = None if selected.log_prob is None else selected.log_prob.detach().cpu().clone()
        next_observation, reward, terminated, truncated, _ = env.step(action)
        pending.append(Transition(observation, action, reward, next_observation, terminated, truncated, log_prob).to_batch())
        if len(pending) == n:
            learn_tree_backup(model, pending, target_generator, update_rule, target_policy=target_policy)
            pending.popleft()
            updates += 1
        reward_sum += reward
        steps += 1
        if terminated or truncated:
            while pending:
                learn_tree_backup(model, pending, target_generator, update_rule, target_policy=target_policy)
                pending.popleft()
                updates += 1
            break
        observation = next_observation
        selected = model.act(observation)
    return {"reward": reward_sum, "steps": steps, "updates": updates, "terminated": terminated, "truncated": truncated}


def train_and_evaluate(target_mode="different", n=3, episodes=200, seed=7, gamma=.9, alpha=.05,
                       max_steps=2000, log_every=50):
    """Uniform behavior; fixed target evaluation against the same policy's DP Qpi."""
    if target_mode not in ("matched", "different") or episodes < 1:
        raise ValueError("target_mode must be matched/different and episodes positive")
    torch.manual_seed(seed)
    model = Model(IdentityRepresentation(), TabularQEstimator(len(DLFS4_STATES), 4),
                  EpsilonGreedyActionSelector(1.)).double()
    policy = torch.full_like(model.estimator.q_table, .25)
    if target_mode == "different":
        policy[:] = policy.new_tensor([.3, .2, .2, .3])
    policy = policy.detach().clone()
    target, rule = TD0TargetGenerator(gamma), TabularQUpdateRule(alpha)
    env = make_dlfs4_gridworld_env(max_episode_steps=max_steps)
    rewards, lengths, steps, updates, terminals, truncations = [], [], 0, 0, 0, 0
    try:
        reference = policy_evaluation(env.unwrapped.mdp, policy.numpy(), gamma=gamma, theta=1e-12)
        if not reference.converged:
            raise RuntimeError("DP fixed-policy reference did not converge")
        reference_q = action_values(env.unwrapped.mdp, reference.values, gamma)
        nonterminal = np.setdiff1d(np.arange(len(DLFS4_STATES)), env.unwrapped.mdp.terminal_states)

        def rmse():
            return float(np.sqrt(np.mean((model.estimator.q_table.numpy()[nonterminal] - reference_q[nonterminal])**2)))

        history = [{"episodes": 0, "steps": 0, "q_rmse": rmse()}]
        for number in range(1, episodes+1):
            result = run_episode(env, model, target, rule, target_policy=policy, n=n, seed=seed if number == 1 else None)
            rewards.append(result["reward"])
            lengths.append(result["steps"])
            steps += result["steps"]
            updates += result["updates"]
            terminals += int(result["terminated"])
            truncations += int(result["truncated"])
            if number == episodes or number % max(1, log_every) == 0:
                history.append({"episodes": number, "steps": steps, "q_rmse": rmse()})
                if log_every:
                    print(f"tree_backup {target_mode} n={n} seed={seed} episode={number} steps={steps} updates={updates} "
                          f"reward={result['reward']:.1f} length={result['steps']} Qpi_RMSE={rmse():.6f}", flush=True)
    finally:
        env.close()
    return model, {"algorithm": "tree_backup", "target_mode": target_mode, "n": n, "seed": seed,
                   "episodes": episodes, "steps": steps, "updates": updates, "gamma": gamma, "alpha": alpha,
                   "behavior_epsilon": 1., "target_policy": policy.numpy().copy(),
                   "q_values": model.estimator.q_table.numpy().copy(), "dp_values": reference.values.copy(),
                   "dp_q_values": reference_q.copy(), "q_rmse": history[-1]["q_rmse"], "history": history,
                   "training_rewards": np.asarray(rewards), "episode_lengths": np.asarray(lengths),
                   "terminated_episodes": terminals, "truncated_episodes": truncations}


def run_comparison(seed=7, episodes=200, ns=(1, 3, 5), **kwargs):
    """Reuse on/off-policy SARSA comparators; add matched/different Tree Backup.

    Uniform behavior makes natural-termination trajectories identical despite
    different Q updates/action timing. SARSA selects an extra final action at a
    time limit; Tree Backup does not. Such runs have independent RNG trajectories.
    No equality between stochastic multi-step SARSA and Tree Backup is assumed.
    """
    results = []
    for n in ns:
        group = []
        for mode in ("on_policy", "matched", "different"):
            _, result = _off_policy["train_and_evaluate"](mode, n, episodes, seed, **kwargs)
            result["algorithm"] = "on_policy_sarsa" if mode == "on_policy" else "off_policy_sarsa"
            result["target_mode"] = "different" if mode == "different" else "matched"
            group.append(result)
        for mode in ("matched", "different"):
            _, result = train_and_evaluate(mode, n, episodes, seed, **kwargs)
            group.append(result)
        for key in ("q_values", "training_rewards", "episode_lengths"):
            np.testing.assert_array_equal(group[0][key], group[1][key])
        if all(r["truncated_episodes"] == 0 for r in group):
            for result in group[1:]:
                np.testing.assert_array_equal(result["training_rewards"], group[0]["training_rewards"])
                np.testing.assert_array_equal(result["episode_lengths"], group[0]["episode_lengths"])
        results.extend(group)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    parser.add_argument("--ns", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--gamma", type=float, default=.9)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    all_results = []
    for seed in args.seeds:
        for result in run_comparison(seed, args.episodes, args.ns, alpha=args.alpha, gamma=args.gamma,
                                     max_steps=args.max_steps, log_every=args.log_every):
            all_results.append(result)
            print(f"algorithm={result['algorithm']} target={result['target_mode']} n={result['n']} seed={seed} "
                  f"steps={result['steps']} updates={result['updates']} terminated={result['terminated_episodes']} "
                  f"truncated={result['truncated_episodes']} Qpi_RMSE={result['q_rmse']:.6f}")
            with np.printoptions(precision=4, suppress=True):
                print("Target pi:\n", result["target_policy"])
                print("Qpi estimate:\n", result["q_values"])
    for n in args.ns:
        for algorithm, mode in (("on_policy_sarsa", "matched"), ("off_policy_sarsa", "matched"),
                                ("off_policy_sarsa", "different"), ("tree_backup", "matched"), ("tree_backup", "different")):
            errors = [r["q_rmse"] for r in all_results if r["n"] == n and r["algorithm"] == algorithm and r["target_mode"] == mode]
            print(f"n={n} {algorithm} {mode}: RMSE mean={np.mean(errors):.6f} std={np.std(errors):.6f}")


if __name__ == "__main__":
    main()
