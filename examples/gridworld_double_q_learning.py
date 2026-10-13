"""Double Q-learning composed from two unchanged Models and existing components."""

import argparse

import numpy as np
import torch
from torch import Tensor

from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import TabularQEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import ActionOutput, EstimatorOutput, Transition, TransitionBatch, ValueOutput
from modular_rl.dp import value_iteration
from modular_rl.environment import DLFS4_START_STATE, DLFS4_STATES, collect_episode, make_dlfs4_gridworld_env
from modular_rl.model import Model


@torch.no_grad()
def combined_estimate(model1: Model, model2: Model, observation) -> EstimatorOutput:
    """Combine live Q estimates into one explicit Q sum, without a third table.

    Integer observations may be single or batched. The result has Q shape
    (..., A). The existing selector receives one tensor, not two unreduced
    critics. Behavior and greedy evaluation both use Q1 + Q2.
    """
    _validate_models(model1, model2)
    q1, = model1.forward(observation).values.q
    q2, = model2.forward(observation).values.q
    if q1.shape != q2.shape:
        raise ValueError("Q estimates must have the same shape without broadcasting")
    return EstimatorOutput(values=ValueOutput(q=(q1 + q2,)))


def act_double_q(model1: Model, model2: Model, observation, selector: EpsilonGreedyActionSelector,
                 *, deterministic=False) -> ActionOutput:
    """Use the unchanged selector's sampling, ties, probabilities and log_prob."""
    return selector.select(combined_estimate(model1, model2, observation), deterministic=deterministic)


def choose_update_first(rng: np.random.Generator) -> bool:
    """Draw one fair update coin from an explicit, separate RNG."""
    return bool(rng.random() < .5)


def learn_double_q(model1: Model, model2: Model, batch: TransitionBatch,
                   target_generator: TD0TargetGenerator, update_rule: TabularQUpdateRule,
                   *, update_first: bool) -> Tensor:
    """Cross-bootstrap, then update exactly one independent table in place.

    All batch fields have shape (B,). Integer indices become long; rewards
    follow the two compatible tables' floating dtype/device. update_first is
    one bool for the WHOLE call, including B>1. Random selection belongs to
    the caller, normally once per online B=1 step, and consumes no RNG here.

    Argmax ties select the smallest action index without sampling. Both Q
    estimates and all targets are frozen before writes. Duplicate pairs then
    follow the existing update rule's input-order semantics. Only termination
    masks bootstrap; truncation uses the actual final observation. Returned
    signed errors are (B,), in table dtype/device, without an autograd graph.
    """
    _validate_models(model1, model2)
    _validate_batch(batch, update_first)
    selecting, evaluating = (model1, model2) if update_first else (model2, model1)
    table = selecting.estimator.q_table
    batch = batch.to(table.device, dtype=table.dtype)
    states, actions = batch.observation.long(), batch.action.long()
    with torch.no_grad():
        select_q, = selecting.forward(batch.next_observation.long()).values.q
        evaluate_q, = evaluating.forward(batch.next_observation.long()).values.q
        _validate_next_q(select_q, evaluate_q, batch.reward.numel(), table.shape[1])
        best_action = select_q.argmax(dim=-1)
        next_value = evaluate_q.gather(-1, best_action.unsqueeze(-1)).squeeze(-1)
        target = target_generator.generate(batch.reward, next_value, batch.terminated)
    return update_rule.update(table, states, actions, target)


def _validate_models(model1, model2):
    for model in (model1, model2):
        if not isinstance(model, Model) or not isinstance(model.estimator, TabularQEstimator):
            raise TypeError("Double Q composition requires two Models with TabularQEstimator")
    q1, q2 = model1.estimator.q_table, model2.estimator.q_table
    if q1.shape != q2.shape or q1.dtype != q2.dtype or q1.device != q2.device:
        raise ValueError("Q tables must have the same shape, dtype and device")
    if q1.untyped_storage().data_ptr() == q2.untyped_storage().data_ptr():
        raise ValueError("Q tables must have independent storage")


def _validate_batch(batch, update_first):
    if not isinstance(update_first, (bool, np.bool_)):
        raise TypeError("update_first must be one bool for the whole batch")
    if not isinstance(batch, TransitionBatch):
        raise TypeError("learn expects a TransitionBatch")
    if not isinstance(batch.observation, Tensor):
        raise TypeError("observation must be a Tensor")
    if batch.observation.ndim != 1 or batch.observation.numel() == 0:
        raise ValueError("observation must have nonempty tabular batch shape (B,)")
    for name in ("action", "reward", "next_observation", "terminated", "truncated"):
        value = getattr(batch, name)
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a Tensor")
        if value.shape != batch.observation.shape:
            raise ValueError("tabular batch fields must have the same (B,) shape")
    for name in ("observation", "next_observation", "action"):
        value = getattr(batch, name)
        if value.is_floating_point() or value.is_complex() or value.dtype == torch.bool:
            raise TypeError(f"{name} must have integer index dtype")
    if not batch.reward.is_floating_point():
        raise TypeError("reward must have floating dtype")
    if batch.terminated.dtype != torch.bool or batch.truncated.dtype != torch.bool:
        raise TypeError("terminated and truncated must have bool dtype")


def _validate_next_q(select_q, evaluate_q, batch_size, n_actions):
    if select_q.shape != (batch_size, n_actions) or evaluate_q.shape != select_q.shape:
        raise ValueError("Both next Q estimates must have the same (B,A) shape")


def _episode_result(episode):
    return {"steps": len(episode), "reward": sum(t.reward.item() for t in episode.transitions),
            "terminated": episode.terminated, "truncated": episode.truncated,
            "path": [episode.transitions[0].observation.item()] + [t.next_observation.item() for t in episode.transitions]}


def evaluate_greedy(model1, model2, selector, env, *, seed):
    episode = collect_episode(env, lambda state: act_double_q(model1, model2, state, selector, deterministic=True), seed=seed)
    return _episode_result(episode)


def train_and_evaluate(episodes=1000, seed=7, gamma=.9, alpha=.8, epsilon=.1,
                       max_steps=2000, log_every=100):
    """Gymnasium online B=1 updates, with a fresh fair update coin per step.

    Environment reset(seed=seed) owns its NumPy stream. Behavior sampling uses
    PyTorch seeded with seed. Update choice uses default_rng(seed+2), separate
    from both. Greedy evaluation draws no behavior or update-choice randomness.
    Returns (model1, model2, selector) and owned experimental snapshots.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    torch.manual_seed(seed)
    model1, model2 = (Model(IdentityRepresentation(), TabularQEstimator(len(DLFS4_STATES), 4)).double()
                      for _ in range(2))
    selector = EpsilonGreedyActionSelector(epsilon).double()
    target_generator, update_rule = TD0TargetGenerator(gamma), TabularQUpdateRule(alpha)
    update_rng = np.random.default_rng(seed + 2)
    update_counts = np.zeros(2, dtype=np.int64)
    env = make_dlfs4_gridworld_env(max_episode_steps=max_steps)
    eval_env = make_dlfs4_gridworld_env(max_episode_steps=20)
    rewards, lengths, steps, terminated_episodes, truncated_episodes = [], [], 0, 0, 0
    try:
        optimal = value_iteration(env.unwrapped.mdp, gamma=gamma, theta=1e-12)
        if not optimal.converged:
            raise RuntimeError("DP reference did not converge")
        nonterminal = np.setdiff1d(np.arange(len(DLFS4_STATES)), env.unwrapped.mdp.terminal_states)
        before = evaluate_greedy(model1, model2, selector, eval_env, seed=seed + 1)
        dp_episode = collect_episode(eval_env, lambda state: int(np.argmax(optimal.policy[state])), seed=seed + 1)
        history = [{"episodes": 0, "steps": 0, "training_mean_reward": None, "greedy": before}]
        for number in range(1, episodes + 1):
            observation, _ = env.reset(seed=seed if number == 1 else None)
            episode_reward, length = 0., 0
            while True:
                selected = act_double_q(model1, model2, observation, selector)
                log_prob = selected.log_prob.detach().cpu().clone()
                next_observation, reward, terminated, truncated, _ = env.step(selected.action.item())
                transition = Transition(observation, selected.action.item(), reward, next_observation, terminated, truncated, log_prob)
                update_first = choose_update_first(update_rng)
                learn_double_q(model1, model2, transition.to_batch(), target_generator, update_rule, update_first=update_first)
                update_counts[0 if update_first else 1] += 1
                episode_reward += reward
                length += 1
                if terminated or truncated:
                    break
                observation = next_observation
            steps += length
            terminated_episodes += int(terminated)
            truncated_episodes += int(truncated)
            rewards.append(episode_reward)
            lengths.append(length)
            if number == episodes or (log_every and number % log_every == 0):
                evaluation = evaluate_greedy(model1, model2, selector, eval_env, seed=seed + 1)
                recent_reward = float(np.mean(rewards[-100:]))
                history.append({"episodes": number, "steps": steps, "training_mean_reward": recent_reward,
                                "episode_reward": episode_reward, "episode_length": length, "greedy": evaluation})
                if log_every:
                    print(f"algorithm=double_q seed={seed} episode={number}/{episodes} reward={episode_reward:.1f} "
                          f"length={length} steps={steps} epsilon={epsilon:.3f} alpha={alpha:.3f} "
                          f"train_mean_reward={recent_reward:.3f} greedy_goal={evaluation['terminated']} "
                          f"greedy_steps={evaluation['steps']}", flush=True)
    finally:
        env.close()
        eval_env.close()

    q1, q2 = model1.estimator.q_table.numpy().copy(), model2.estimator.q_table.numpy().copy()
    average_q = (q1 + q2) / 2
    return (model1, model2, selector), {"algorithm": "double_q", "episodes": episodes, "steps": steps, "seed": seed,
            "gamma": gamma, "alpha": alpha, "epsilon": epsilon, "before": before, "after": history[-1]["greedy"],
            "history": history, "training_rewards": np.array(rewards), "episode_lengths": np.array(lengths),
            "terminated_episodes": terminated_episodes, "truncated_episodes": truncated_episodes,
            "q1_values": q1, "q2_values": q2, "q_values": average_q, "update_counts": update_counts.copy(),
            "dp_values": optimal.values.copy(), "dp_steps": len(dp_episode), "dp_path": _episode_result(dp_episode)["path"],
            "max_q_rmse": float(np.sqrt(np.mean((average_q.max(axis=1)[nonterminal] - optimal.values[nonterminal]) ** 2)))}


def main():
    # Project-local comparator import is only needed by the module CLI.
    from examples.gridworld_expected_sarsa import train_and_evaluate as train_comparator

    parser = argparse.ArgumentParser(description="Double Q-learning composition vs existing tabular TD control")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    parser.add_argument("--gamma", type=float, default=.9)
    parser.add_argument("--alpha", type=float, default=.8)
    parser.add_argument("--epsilon", type=float, default=.1)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--compare-all", action="store_true", help="Also run existing SARSA/Expected SARSA comparators")
    args = parser.parse_args()
    algorithms = ("double_q", "q_learning", "sarsa", "expected_sarsa") if args.compare_all else ("double_q", "q_learning")
    for seed in args.seeds:
        for algorithm in algorithms:
            if algorithm == "double_q":
                _, result = train_and_evaluate(args.episodes, seed, args.gamma, args.alpha, args.epsilon, args.max_steps, args.log_every)
            else:
                _, result = train_comparator(algorithm, args.episodes, seed, args.gamma, args.alpha, args.epsilon, args.max_steps, args.log_every)
            print(f"DLFS4 {algorithm}: seed={seed} episodes={result['episodes']} steps={result['steps']} "
                  f"terminated={result['terminated_episodes']} truncated={result['truncated_episodes']}")
            for name in ("before", "after"):
                value = result[name]
                print(f"{name} greedy: goal={value['terminated']} steps={value['steps']} reward={value['reward']:.1f} truncated={value['truncated']}")
            print("Greedy path:", [DLFS4_STATES[state] for state in result["after"]["path"]])
            print(f"DP optimal: steps={result['dp_steps']} V(start)={result['dp_values'][DLFS4_START_STATE]:.4f} "
                  f"max Q RMSE(nonterminal)={result['max_q_rmse']:.6f}")
            if algorithm == "double_q":
                print("DP optimal path:", [DLFS4_STATES[state] for state in result["dp_path"]])
                print("Q1/Q2 update counts:", result["update_counts"].tolist())
            with np.printoptions(precision=4, suppress=True):
                for field in (("q1_values", "q2_values", "q_values") if algorithm == "double_q" else ("q_values",)):
                    print(field + " (up/down/left/right):\n", result[field])


if __name__ == "__main__":
    main()
