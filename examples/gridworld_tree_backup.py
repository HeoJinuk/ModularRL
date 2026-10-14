"""Online n-step Tree Backup by composing existing ModularRL components."""

import argparse
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


@torch.no_grad()
def learn_tree_backup(model, window, target_generator, update_rule, *, target_policy):
    """Update the oldest row of a contiguous, ordered B=1 window.

    target_policy is a caller-owned fixed floating probability Tensor (S,A),
    or an existing discrete Selector with probabilities(estimate). Fixed rows
    are explicitly cast to Q dtype; device must match. A single Model estimate
    supplies all needed next-state Q rows and dynamic policy probabilities before
    any writes. The natural terminal endpoint is excluded from that estimate.

    Intermediate actions come from the following window row. The final bootstrap
    is an expectation, so no endpoint action, behavior probability or importance
    ratio is used. pi=0 cuts only the sampled branch; pi=1 carries G directly.
    Each TD0 call performs one reward backup. Return the existing rule's signed
    error (1,), in table dtype/device with no graph. The caller pops after success.
    Separate flush calls re-evaluate current Q after earlier successful updates.
    """
    _n_step["_validate_window"](model, window)
    table = model.estimator.q_table
    _validate_actions(window, table)
    batches = [batch.to(table.device, dtype=table.dtype) for batch in window]
    first, last = batches[0], batches[-1]
    next_states = torch.cat([batch.next_observation for batch in batches]).long()
    needed_states = next_states[:-1] if last.terminated.item() else next_states
    if isinstance(target_policy, Tensor):
        _off_policy["_validate_policy"](target_policy, table)
    elif not callable(getattr(target_policy, "probabilities", None)):
        raise TypeError("target_policy must be a fixed probability Tensor or discrete Selector")

    if needed_states.numel():
        estimate = model.forward(needed_states)
        q = estimate.values.q[0]
        probabilities = _probabilities(target_policy, needed_states, estimate)
        expected_values = (probabilities * q).sum(-1)
    target = torch.zeros_like(last.reward) if last.terminated.item() else expected_values[-1:]
    for time in range(len(batches)-1, -1, -1):
        continuation = target
        if time < len(batches)-1:
            next_action = batches[time+1].action.long()
            probability = probabilities[time].gather(0, next_action)
            sampled_q = q[time].gather(0, next_action)
            expected = expected_values[time:time+1]
            if probability.item() == 0:
                continuation = expected
            elif probability.item() != 1:
                continuation = expected + probability * (target - sampled_q)
        batch = batches[time]
        target = target_generator.generate(batch.reward, continuation, batch.terminated)

    states, actions = first.observation.long(), first.action.long()
    _validate_update(table[states, actions], target, update_rule.alpha)
    return update_rule.update(table, states, actions, target)


def _probabilities(policy, states, estimate):
    q = estimate.values.q[0]
    if isinstance(policy, Tensor):
        return policy[states].to(dtype=q.dtype)
    probabilities = policy.probabilities(estimate)
    if not isinstance(probabilities, Tensor) or probabilities.shape != q.shape:
        raise ValueError("Selector probabilities must match the Q shape (M,A)")
    if probabilities.dtype != q.dtype or probabilities.device != q.device:
        raise ValueError("Selector probabilities must share Q dtype and device")
    if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0) or torch.any(probabilities > 1):
        raise ValueError("Selector probabilities must be finite and in [0,1]")
    tolerance = 1e-6 if q.dtype == torch.float32 else 1e-12
    if not torch.allclose(probabilities.sum(-1), torch.ones_like(q[:, 0]), rtol=0, atol=tolerance):
        raise ValueError("Selector probabilities must sum to one per state")
    return probabilities


def _validate_actions(window, table):
    for batch in window:
        if not 0 <= batch.action.item() < table.shape[1]:
            raise IndexError("window action index out of bounds")


def _validate_update(current, target, alpha):
    error = target - current
    if not all(torch.isfinite(x).all() for x in (target, error, current + alpha * error)):
        raise FloatingPointError("Tree Backup target/update is non-finite; head was not updated")


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
