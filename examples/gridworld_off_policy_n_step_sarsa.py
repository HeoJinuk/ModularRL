"""Fixed-policy sampled n-step SARSA using unchanged ModularRL components."""

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

# examples is not an installed package. File loading also works from notebooks.
# Reuse the original window/index validation and the on-policy comparator as-is.
_on_policy = runpy.run_path(str(Path(__file__).with_name("gridworld_n_step_sarsa.py")))


@torch.no_grad()
def learn_off_policy_n_step_sarsa(model, window, target_generator, update_rule, *,
                                target_policy, bootstrap_action=None, bootstrap_log_prob=None):
    """Update the window head; return raw/weighted errors, G, effective Y and rho.

    window contains contiguous B=1 snapshots. target_policy is a fixed floating
    (S,A) probability table owned by the caller. rho excludes the head action,
    includes subsequent executed actions and, when bootstrapping, the separately
    saved final action. Behavior logs are read from original snapshots, never
    recomputed or cast through the Q-table dtype. Logs use their recorded precision;
    summation/exp use float64. The caller removes the head only after success.

    Returned tensors have shape (1,), no graph, and table device/dtype except
    importance_ratio (float64). td_error is the actual effective-target error
    returned by the existing rule; raw_td_error is G-Q before the write. rho=1
    passes G directly for bit-exact on-policy equivalence. Numeric failures write
    nothing for this call (earlier successful online calls remain committed).
    No action is sampled. Flush calls read current Q with the same final action.
    """
    _on_policy["_validate_window"](model, window)
    table = model.estimator.q_table
    _validate_policy(target_policy, table)
    _validate_logs(window, bootstrap_log_prob, table)
    batches = [batch.to(table.device, dtype=table.dtype) for batch in window]
    first, last = batches[0], batches[-1]
    action = _on_policy["_prepare_next_action"](bootstrap_action, last, table)
    if not last.terminated.item() and bootstrap_log_prob is None:
        raise ValueError("bootstrap_log_prob is required when bootstrapping, including truncation")

    # The same TD0 reverse composition as the original n-step example.
    if last.terminated.item():
        target = torch.zeros_like(last.reward)
    else:
        next_q = model.forward(last.next_observation.long()).values.q[0]
        target = next_q.gather(-1, action.unsqueeze(-1)).squeeze(-1)
    for batch in reversed(batches):
        target = target_generator.generate(batch.reward, target, batch.terminated)

    rho = importance_ratio(window, target_policy, bootstrap_action=action,
                           bootstrap_log_prob=bootstrap_log_prob)
    states, actions = first.observation.long(), first.action.long()
    current = table[states, actions].clone()
    raw_error = target - current
    if rho.item() == 1:
        effective_target = target
    elif rho.item() == 0:
        effective_target = current
    else:
        effective_target = current + rho.to(table.dtype) * raw_error
    _validate_update(current, target, raw_error, effective_target, rho, update_rule.alpha)
    error = update_rule.update(table, states, actions, effective_target)
    return {"td_error": error, "raw_td_error": raw_error, "target": target,
            "effective_target": effective_target, "importance_ratio": rho}


def importance_ratio(window, target_policy, *, bootstrap_action=None, bootstrap_log_prob=None):
    """Product pi/b for actions AFTER the head, including the sampled endpoint.

    Called after input validation by learn; useful for inspecting saved windows.
    Natural termination omits the endpoint, even if both flags are true. An empty
    product is one. A zero target probability yields zero, not numerical underflow.
    Selected-action logs cannot verify coverage for actions never observed.
    """
    last = window[-1]
    factors = [(b.observation.item(), b.action.item(), b.log_prob) for b in list(window)[1:]]
    if not last.terminated.item():
        factors.append((last.next_observation.item(), int(torch.as_tensor(bootstrap_action).item()), bootstrap_log_prob))
    log_weight = torch.zeros(1, dtype=torch.float64, device=target_policy.device)
    for state, action, log_b in reversed(factors):
        log_pi = target_policy[state, action].log().double()
        log_weight += log_pi - log_b.detach().to(device=target_policy.device, dtype=torch.float64).reshape(1)
    rho = log_weight.exp()
    if not torch.isfinite(rho).all() or torch.any(torch.isfinite(log_weight) & (rho == 0)):
        raise FloatingPointError("Importance ratio overflow/underflow; window head was not updated")
    return rho.detach()


def _validate_policy(policy, table):
    if not isinstance(policy, Tensor) or not policy.is_floating_point():
        raise TypeError("target_policy must be a floating Tensor")
    if policy.shape != table.shape or policy.device != table.device:
        raise ValueError("target_policy must have table shape (S,A) and device")
    if not torch.isfinite(policy).all() or torch.any(policy < 0) or torch.any(policy > 1):
        raise ValueError("target_policy probabilities must be finite and in [0,1]")
    tolerance = 1e-6 if policy.dtype == torch.float32 else 1e-12
    if not torch.allclose(policy.double().sum(-1), torch.ones(table.shape[0], device=table.device, dtype=torch.float64),
                          rtol=0, atol=tolerance):
        raise ValueError("target_policy probabilities must sum to one per state")


def _validate_logs(window, bootstrap_log_prob, table):
    for batch in window:
        for value, size in ((batch.observation, table.shape[0]), (batch.next_observation, table.shape[0]),
                            (batch.action, table.shape[1])):
            if not 0 <= value.item() < size:
                raise IndexError("window state/action index out of bounds")
    for log_b in [batch.log_prob for batch in window] + ([] if bootstrap_log_prob is None else [bootstrap_log_prob]):
        if not isinstance(log_b, Tensor) or not log_b.is_floating_point() or log_b.shape not in ((), (1,)):
            raise ValueError("sampling-time behavior log_prob must be a scalar or (1,) floating Tensor")
        if not torch.isfinite(log_b).all() or torch.any(log_b > 0):
            raise ValueError("behavior log_prob must be finite and <=0; sampled actions need positive support")


def _validate_update(current, target, raw_error, effective_target, rho, alpha):
    cast_rho = rho.to(current.dtype)
    error = effective_target - current
    proposed = current + alpha * error
    if not all(torch.isfinite(x).all() for x in (cast_rho, target, raw_error, effective_target, error, proposed)):
        raise FloatingPointError("Ratio/target/update is not representable in table dtype; no write committed")
    if torch.any((rho > 0) & (cast_rho == 0)):
        raise FloatingPointError("Positive importance ratio underflows table dtype; no write committed")
    weighted = cast_rho * raw_error
    if torch.any((cast_rho != 0) & (raw_error != 0) & (weighted == 0)):
        raise FloatingPointError("Weighted TD error underflows table dtype; no write committed")


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
