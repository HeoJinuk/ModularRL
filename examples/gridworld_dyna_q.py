"""Dyna-Q composition: a learned sample model and unchanged Q-learning components."""

import argparse
from collections import deque
from numbers import Integral

import numpy as np
import torch

from examples.gridworld_mc_control import evaluate_greedy
from examples.gridworld_n_step_sarsa import _validate_n, _validate_window
from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import TabularQEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import Transition
from modular_rl.dp import value_iteration
from modular_rl.environment import DLFS4_STATES, collect_episode, make_dlfs4_gridworld_env
from modular_rl.learner import TabularQLearner
from modular_rl.model import Model


# Historical example import remains valid for callers and old checkpoints.
from modular_rl.component.world_model import TabularWorldModel
from modular_rl.component.planner import DynaPlanner


def plan(world_model, learner, *, steps, rng):
    """Run at most `steps` independent one-step Q updates, without an environment.

    A nonempty model yields exactly `steps` updates; an empty model yields none.
    Return signed errors (actual_updates,) in the learner table's dtype/device.
    Each backup reads live Q after the preceding update. No observe/act call,
    behavior RNG, real window or scheduling clock is involved.
    """
    result = DynaPlanner(world_model, learner, rng=rng).plan(steps)
    errors = [row[0] for row in result["results"]]
    table = learner.model.estimator.q_table
    return torch.stack(errors) if errors else table.new_empty(0)


@torch.no_grad()
def learn_real_n_step(learner, window):
    """Update only the oldest real row using actual m rewards and max-Q bootstrap.

    window contains contiguous owned B=1 batches; callers pop after success.
    n=1 delegates directly to TabularQLearner. For m>1, reuse TD0.generate in
    reverse reward order, then the same Q UpdateRule. Only termination masks
    bootstrap. Each call freezes its target before writing; flush calls read
    current Q independently. There is no action sampling or planning data here.
    """
    if len(window) == 1 and window[0].observation.shape == (1,):
        return learner.learn(window[0])
    _validate_window(learner.model, window)
    table = learner.model.estimator.q_table
    batches = [row.to(table.device, dtype=table.dtype) for row in window]
    first, last = batches[0], batches[-1]
    next_q, = learner.model.forward(last.next_observation.long()).values.q
    target = next_q.max(-1).values
    for row in reversed(batches):
        target = learner.target_generator.generate(row.reward, target, row.terminated)
    return learner.update_rule.update(table, first.observation.long(), first.action.long(), target)


def run_steps(env, real_learner, *, steps, world_model=None, planning_learner=None,
              planning_steps=0, rng=None, n=1, seed=None):
    """Interact for EXACTLY `steps` environment steps; own no environment lifetime.

    Per step: act -> step -> eligible real update -> observe -> one-step plan.
    Ending windows flush AFTER that step's planning, before reset/another action.
    A budget cutoff is recorded as external truncation and flushes the window;
    a partial episode is reported separately from completed episodes. All real
    transitions start exactly one real update. For n>1 the first n-1 steps can
    already observe/plan while no real head is eligible. No endpoint action is
    sampled. The caller seeds behavior Torch RNG; environment uses reset(seed).

    Trace rows contain only real transitions, per-step update counts and real
    targets/errors (possibly several at a boundary). Checkpointed mid-episode
    operation uses explicit caller-owned windows and learn_real_n_step/plan;
    this convenience loop does not expose a live environment checkpoint.
    """
    _validate_run(steps, n, planning_steps, world_model, rng, real_learner, planning_learner)
    planning_learner = real_learner if planning_learner is None else planning_learner
    model = real_learner.model
    observation, _ = env.reset(seed=seed)
    pending = deque()
    trace, rewards, lengths = [], [], []
    episode_reward, length = 0., 0
    real_updates = planning_updates = model_updates = flush_updates = 0
    terminated_episodes = truncated_episodes = 0
    for time in range(steps):
        selected = model.act(observation)
        action = selected.action.item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        budget_cutoff = time == steps - 1 and not (terminated or truncated)
        boundary = terminated or truncated or budget_cutoff
        transition = Transition(observation, action, reward, next_state,
                                terminated, truncated or budget_cutoff, selected.log_prob)
        pending.append(transition.to_batch())
        targets, errors = [], []

        def update_head():
            # Reconstruct the target from old Q + signed error for diagnostics.
            head = pending[0]
            old_q = model.estimator.q_table[head.observation.item(), head.action.item()].clone()
            error = learn_real_n_step(real_learner, pending)
            targets.append((old_q + error[0]).item())
            errors.append(error[0].item())
            pending.popleft()

        if len(pending) == n:
            update_head()
            real_updates += 1
        if world_model is not None:
            world_model.observe(transition)
            model_updates += 1
        planned = 0 if world_model is None else plan(
            world_model, planning_learner, steps=planning_steps, rng=rng).numel()
        planning_updates += planned
        flushed = 0
        if boundary:
            while pending:
                update_head()
                real_updates += 1
                flushed += 1
            flush_updates += flushed
        episode_reward += reward
        length += 1
        trace.append({"state": observation, "action": action, "reward": reward, "next_state": next_state,
                      "terminated": terminated, "truncated": truncated, "budget_cutoff": budget_cutoff,
                      "targets": targets, "td_errors": errors, "real_updates": len(errors),
                      "planning_updates": planned, "flush_updates": flushed})
        if boundary:
            if not budget_cutoff:
                rewards.append(episode_reward)
                lengths.append(length)
                terminated_episodes += int(terminated)
                truncated_episodes += int(truncated)
            if time < steps - 1:
                observation, _ = env.reset()
                episode_reward, length = 0., 0
        else:
            observation = next_state
    return {"steps": steps, "real_updates": real_updates, "planning_updates": planning_updates,
            "model_updates": model_updates, "flush_updates": flush_updates, "trace": trace,
            "training_rewards": np.array(rewards), "episode_lengths": np.array(lengths),
            "episodes": len(rewards), "terminated_episodes": terminated_episodes,
            "truncated_episodes": truncated_episodes,
            "partial_episode": {"steps": length, "reward": episode_reward} if budget_cutoff else None}


def _validate_budget(value, name, *, minimum):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _validate_run(steps, n, planning_steps, world_model, rng, real_learner, planning_learner):
    _validate_budget(steps, "environment steps", minimum=1)
    _validate_n(n)
    _validate_budget(planning_steps, "planning steps", minimum=0)
    if planning_steps and (world_model is None or rng is None):
        raise ValueError("planning requires a World Model and an independent RNG")
    if planning_learner is not None and planning_learner.model is not real_learner.model:
        raise ValueError("real and planning learners must share the same Q Model")


def train_and_evaluate(*, steps=2000, seed=7, planning_steps=5, n=1, baseline=False,
                       gamma=.9, alpha=.8, epsilon=.1, max_steps=2000):
    """Compare equal REAL step budgets, unlike earlier episode-budget examples."""
    if baseline and (planning_steps or n != 1):
        raise ValueError("baseline is unchanged one-step Q-learning with planning_steps=0")
    torch.manual_seed(seed)
    model = Model(IdentityRepresentation(), TabularQEstimator(len(DLFS4_STATES), 4),
                  EpsilonGreedyActionSelector(epsilon)).double()
    learner = TabularQLearner(model, TD0TargetGenerator(gamma), TabularQUpdateRule(alpha))
    world_model = None if baseline else TabularWorldModel()
    rng = np.random.default_rng(seed + 1)
    env = make_dlfs4_gridworld_env(max_episode_steps=max_steps)
    eval_env = make_dlfs4_gridworld_env(max_episode_steps=20)
    try:
        optimal = value_iteration(env.unwrapped.mdp, gamma=gamma, theta=1e-12)
        if not optimal.converged:
            raise RuntimeError("DP reference did not converge")
        before = evaluate_greedy(model, eval_env, seed=seed + 2)
        dp_episode = collect_episode(eval_env, lambda state: int(np.argmax(optimal.policy[state])), seed=seed + 2)
        result = run_steps(env, learner, steps=steps, world_model=world_model, planning_steps=planning_steps,
                           rng=rng, n=n, seed=seed)
        after = evaluate_greedy(model, eval_env, seed=seed + 2)
        # Accuracy is queried ONLY on observed pairs; no unseen generalization claim.
        correct = 0
        if world_model is not None:
            for record in world_model.state_dict()["outcomes"]:
                outcome, = env.unwrapped.mdp.transitions[record.observation][record.action]
                correct += int(record.next_observation == outcome.next_state and
                               record.reward.item() == outcome.reward and
                               record.terminated == outcome.terminated)
        terminal = env.unwrapped.mdp.terminal_states
    finally:
        env.close()
        eval_env.close()
    q = model.estimator.q_table.cpu().numpy().copy()
    nonterminal = np.setdiff1d(np.arange(len(DLFS4_STATES)), terminal)
    result.update({"seed": seed, "n": n, "planning_steps": planning_steps, "gamma": gamma,
                   "alpha": alpha, "epsilon": epsilon, "baseline": baseline,
                   "before": before, "after": after, "q_values": q, "dp_values": optimal.values,
                   "dp_steps": len(dp_episode), "world_pairs": 0 if world_model is None else len(world_model),
                   "model_correct_pairs": correct, "world_model": world_model,
                   "max_q_rmse": float(np.sqrt(np.mean((q.max(-1)[nonterminal] -
                                                        optimal.values[nonterminal]) ** 2)))})
    return learner, result


def comparison(*, steps=2000, seeds=(7, 11, 19)):
    """Q-learning, Dyna budgets 0/5/10, and real n=3/5 with five 1-step plans."""
    results = []
    settings = [(True, 0, 1), (False, 0, 1), (False, 5, 1), (False, 10, 1),
                (False, 5, 3), (False, 5, 5)]
    for seed in seeds:
        for baseline, budget, n in settings:
            _, result = train_and_evaluate(steps=steps, seed=seed, planning_steps=budget, n=n, baseline=baseline)
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    args = parser.parse_args()
    for result in comparison(steps=args.steps, seeds=args.seeds):
        print(f"seed={result['seed']} baseline={result['baseline']} n={result['n']} plan={result['planning_steps']} "
              f"env={result['steps']} real={result['real_updates']} model={result['model_updates']} "
              f"planning={result['planning_updates']} flush={result['flush_updates']} "
              f"episodes={result['episodes']} pairs={result['world_pairs']} "
              f"accuracy={result['model_correct_pairs']}/{result['world_pairs']} "
              f"greedy={result['after']['terminated']}/{result['after']['steps']} "
              f"DP={result['dp_steps']} RMSE={result['max_q_rmse']:.6f}")
        print("Episode rewards:", result["training_rewards"].tolist())
        print("Episode lengths:", result["episode_lengths"].tolist(), "partial:", result["partial_episode"])
        with np.printoptions(precision=4, suppress=True):
            print("Q-table:\n", result["q_values"])


if __name__ == "__main__":
    main()
