"""Fixed-policy sampled n-step SARSA using unchanged ModularRL components."""

import argparse

from modular_rl.learner.off_policy_n_step_sarsa import OffPolicyNStepSARSALearner, importance_ratio, _validate_policy, _validate_logs, _validate_update
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

# examples is not an installed package. File loading also works from notebooks.
# Reuse the original window/index validation and the on-policy comparator as-is.
_on_policy = runpy.run_path(str(Path(__file__).with_name("gridworld_n_step_sarsa.py")))


def learn_off_policy_n_step_sarsa(model, window, target_generator, update_rule, *,
                                target_policy, bootstrap_action=None, bootstrap_log_prob=None):
    """Compatibility entry point; retain original sampling-time endpoint metadata."""
    return OffPolicyNStepSARSALearner(model, target_generator, update_rule, target_policy=target_policy).learn(
        window, bootstrap_action=bootstrap_action, bootstrap_log_prob=bootstrap_log_prob)


def run_episode(env, model, target_generator, update_rule, *, target_policy, n=3, seed=None):
    """Online deque schedule; save the endpoint ActionOutput before any write.

    Termination selects no final action. Truncation selects one bootstrap-only
    action and preserves its log_prob through all flush calls without resampling.
    Each real transition is updated once. The caller owns environment/checkpoint
    state; there is no full Episode, Buffer, Runner or new learner here.
    """
    _on_policy["_validate_n"](n)
    observation, _ = env.reset(seed=seed)
    selected = model.act(observation)
    pending = deque()
    reward_sum, steps, updates, ratios = 0., 0, 0, []
    while True:
        action = selected.action.item()
        log_prob = None if selected.log_prob is None else selected.log_prob.detach().cpu().clone()
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_selected = None if terminated else model.act(next_observation)
        next_action = None if next_selected is None else next_selected.action.detach().clone()
        next_log = None if next_selected is None or next_selected.log_prob is None else next_selected.log_prob.detach().clone()
        pending.append(Transition(observation, action, reward, next_observation, terminated, truncated, log_prob).to_batch())
        if len(pending) == n:
            result = learn_off_policy_n_step_sarsa(model, pending, target_generator, update_rule,
                                                 target_policy=target_policy, bootstrap_action=next_action,
                                                 bootstrap_log_prob=next_log)
            ratios.append(result["importance_ratio"].item())
            pending.popleft()
            updates += 1
        reward_sum += reward
        steps += 1
        if terminated or truncated:
            while pending:
                result = learn_off_policy_n_step_sarsa(model, pending, target_generator, update_rule,
                                                     target_policy=target_policy, bootstrap_action=next_action,
                                                     bootstrap_log_prob=next_log)
                ratios.append(result["importance_ratio"].item())
                pending.popleft()
                updates += 1
            break
        observation, selected = next_observation, next_selected
    return {"reward": reward_sum, "steps": steps, "updates": updates, "terminated": terminated,
            "truncated": truncated, "importance_ratios": np.asarray(ratios)}


def train_and_evaluate(mode="different", n=3, episodes=200, seed=7, gamma=.9, alpha=.05,
                       max_steps=2000, log_every=50):
    """Fixed-policy Q prediction, referenced to DP Q^pi, not optimal control.

    Behavior epsilon=1 gives actual uniform probabilities independent of Q.
    on_policy and matched use uniform pi; different uses pi=[.3,.2,.2,.3].
    Copies own pi/Q results. All runs share budget/start distribution and RNG seed;
    uniform behavior makes trajectories identical even when Q updates differ.
    """
    if mode not in ("on_policy", "matched", "different") or episodes < 1:
        raise ValueError("mode must be on_policy/matched/different and episodes positive")
    torch.manual_seed(seed)
    model = Model(IdentityRepresentation(), TabularQEstimator(len(DLFS4_STATES), 4),
                  EpsilonGreedyActionSelector(1.)).double()
    policy = torch.full_like(model.estimator.q_table, .25)
    if mode == "different":
        policy[:] = policy.new_tensor([.3, .2, .2, .3])
    policy = policy.detach().clone()  # Caller-owned fixed target; never improved using Q.
    target, rule = TD0TargetGenerator(gamma), TabularQUpdateRule(alpha)
    env = make_dlfs4_gridworld_env(max_episode_steps=max_steps)
    rewards, lengths, ratios, steps, updates, terminals, truncations = [], [], [], 0, 0, 0, 0
    try:
        reference = policy_evaluation(env.unwrapped.mdp, policy.numpy(), gamma=gamma, theta=1e-12)
        if not reference.converged:
            raise RuntimeError("DP fixed-policy reference did not converge")
        reference_q = action_values(env.unwrapped.mdp, reference.values, gamma)
        nonterminal = np.setdiff1d(np.arange(len(DLFS4_STATES)), env.unwrapped.mdp.terminal_states)

        def rmse():
            return float(np.sqrt(np.mean((model.estimator.q_table.numpy()[nonterminal] - reference_q[nonterminal])**2)))

        history = [{"episodes": 0, "steps": 0, "q_rmse": rmse()}]
        for number in range(1, episodes + 1):
            kwargs = {"n": n, "seed": seed if number == 1 else None}
            if mode == "on_policy":
                result = _on_policy["run_episode"](env, model, target, rule, **kwargs)
                ratio = np.ones(result["updates"])
            else:
                result = run_episode(env, model, target, rule, target_policy=policy, **kwargs)
                ratio = result["importance_ratios"]
            rewards.append(result["reward"])
            lengths.append(result["steps"])
            ratios.extend(ratio)
            steps += result["steps"]
            updates += result["updates"]
            terminals += int(result["terminated"])
            truncations += int(result["truncated"])
            if number == episodes or number % max(1, log_every) == 0:
                history.append({"episodes": number, "steps": steps, "q_rmse": rmse()})
                if log_every:
                    print(f"mode={mode} n={n} seed={seed} episode={number} steps={steps} updates={updates} "
                          f"reward={result['reward']:.1f} length={result['steps']} Qpi_RMSE={rmse():.6f}", flush=True)
    finally:
        env.close()
    ratios = np.asarray(ratios)
    return model, {"mode": mode, "n": n, "seed": seed, "episodes": episodes, "steps": steps, "updates": updates,
                   "gamma": gamma, "alpha": alpha, "behavior_epsilon": 1., "target_policy": policy.numpy().copy(),
                   "q_values": model.estimator.q_table.numpy().copy(), "dp_values": reference.values.copy(),
                   "dp_q_values": reference_q.copy(), "q_rmse": history[-1]["q_rmse"], "history": history,
                   "training_rewards": np.asarray(rewards), "episode_lengths": np.asarray(lengths),
                   "terminated_episodes": terminals, "truncated_episodes": truncations,
                   "importance_ratios": ratios, "rho_mean": float(ratios.mean()), "rho_max": float(ratios.max()),
                   "rho_zero_fraction": float(np.mean(ratios == 0)), "ess": float(ratios.sum()**2 / np.square(ratios).sum())}


def run_comparison(seed=7, episodes=200, ns=(1, 3, 5), **kwargs):
    """Independent fixed-policy runs; matched mode must exactly equal on-policy."""
    results = []
    for n in ns:
        group = []
        for mode in ("on_policy", "matched", "different"):
            _, result = train_and_evaluate(mode, n, episodes, seed, **kwargs)
            results.append(result)
            group.append(result)
        for key in ("q_values", "training_rewards", "episode_lengths"):
            np.testing.assert_array_equal(group[0][key], group[1][key])
        assert group[0]["steps"] == group[1]["steps"] == group[2]["steps"]
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
    for seed in args.seeds:
        for result in run_comparison(seed, args.episodes, args.ns, alpha=args.alpha, gamma=args.gamma,
                                     max_steps=args.max_steps, log_every=args.log_every):
            print(f"Fixed pi: mode={result['mode']} n={result['n']} seed={seed} episodes={result['episodes']} "
                  f"steps={result['steps']} updates={result['updates']} Qpi_RMSE={result['q_rmse']:.6f} "
                  f"rho_mean={result['rho_mean']:.6f} rho_max={result['rho_max']:.6f} "
                  f"rho_zero={result['rho_zero_fraction']:.6f} ESS={result['ess']:.1f}")
            with np.printoptions(precision=4, suppress=True):
                print("Target pi:\n", result["target_policy"])
                print("Qpi estimate:\n", result["q_values"])


if __name__ == "__main__":
    main()
