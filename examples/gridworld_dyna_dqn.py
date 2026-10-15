"""Dyna-DQN using one unchanged learner, optimizer, target and a tabular sample model."""

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from time import perf_counter

import numpy as np
import torch
from torch import Tensor, nn

from modular_rl.component.world_model import TabularWorldModel
from modular_rl.component.planner import DynaPlanner, synthetic_batch
from examples.gridworld_mc_control import evaluate_greedy
from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.buffer import ReplayBuffer
from modular_rl.component.estimator import DiscreteQEstimator
from modular_rl.component.representation import MLPRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TDLossUpdateRule
from modular_rl.data import TDLossOutput, Transition, TransitionBatch
from modular_rl.dp import value_iteration
from modular_rl.environment import DLFS4_STATES, collect_episode, make_dlfs4_gridworld_env
from modular_rl.learner import DQNLearner
from modular_rl.model import Model


def create_learner(*, n_states=11, n_actions=4, hidden_dims=(128, 128),
                   learning_rate=1e-4, gamma=.9, dtype=torch.float32, device="cpu"):
    """Compose a frozen one-hot lookup and the existing MLP/Q head.

    InputAdapter preserves scalar/batched integer IDs. The identity embedding
    maps (...) -> (..., S) in the Model's floating dtype/device; it is frozen.
    No World Model or ReplayBuffer observation is converted to one-hot storage.
    The caller seeds Torch before initialization. One Adam owns trainable online
    parameters; initial hard copy is explicit and is not a periodic sync count.
    """
    representation = nn.Sequential(
        nn.Embedding.from_pretrained(torch.eye(n_states), freeze=True),
        MLPRepresentation(n_states, hidden_dims=hidden_dims),
    )
    model = Model(representation, DiscreteQEstimator(hidden_dims[-1], n_actions),
                  EpsilonGreedyActionSelector(1.)).to(device=device, dtype=dtype)
    target = deepcopy(model)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    learner = DQNLearner(model, target, optimizer, TD0TargetGenerator(gamma),
                         TDLossUpdateRule(nn.SmoothL1Loss(reduction="none")))
    learner.sync_target()
    return learner


def plan(world_model, learner, *, steps, batch_size, rng, record_batches=False):
    """Run independent 1-step DQN minibatch updates through the SAME learner.

    No environment, observe, replay sampling, epsilon change or target sync.
    With the example's environment-step sync schedule, idle planning never
    advances periodic syncs. Empty models execute zero updates and consume no
    RNG. The result distinguishes requested and actual optimizer steps and
    contains detached TDLossOutput metrics. Batch recording is optional QA data.
    """
    return DynaPlanner(world_model, learner, batch_size=batch_size, rng=rng).plan(
        steps, record_batches=record_batches)


@dataclass(frozen=True, slots=True)
class NStepExperience:
    """Example-only replay metadata; NOT a one-step Transition.

    Owned CPU scalar fields, discounted reward sum and actual horizon m. gamma
    records the sum's meaning; a different discount requires rebuilding records.
    No bootstrap Q is stored: the current target network is read when sampled.
    """

    observation: Tensor
    action: Tensor
    reward_sum: Tensor
    next_observation: Tensor
    terminated: Tensor
    truncated: Tensor
    horizon: int
    gamma: float


@torch.no_grad()
def make_n_step_experience(window, target_generator, *, dtype=None):
    """Summarize a contiguous ordered window of actual owned B=1 batches."""
    _validate_window(window)
    first, last = window[0], window[-1]
    reward_sum = torch.zeros((), dtype=dtype or first.reward.dtype)
    for row in reversed(window):
        reward = row.reward[0].to(device="cpu", dtype=reward_sum.dtype)
        reward_sum = target_generator.generate(reward, reward_sum, row.terminated[0].cpu())
    return NStepExperience(
        first.observation[0].detach().cpu().clone(), first.action[0].detach().cpu().clone(), reward_sum,
        last.next_observation[0].detach().cpu().clone(), last.terminated[0].detach().cpu().clone(),
        last.truncated[0].detach().cpu().clone(), len(window), target_generator.gamma,
    )


def learn_real_n_step(learner, experiences):
    """Combine public DQN components with variable-horizon replay ingredients.

    DQNLearner.learn fixes one-step bootstrap and accepts no external target.
    This small example function reuses its SAME online/target/optimizer and
    public TargetGenerator/UpdateRule; it calls no private learner method.
    Prediction retains gradients and target uses no_grad/current target eval.
    DQNLearner.optimize owns the shared loss/optimizer step and detached metrics.
    """
    _validate_experiences(experiences, learner.target_generator.gamma)
    reference = next(learner.online_model.parameters())
    states = torch.stack([row.observation for row in experiences]).to(reference.device).long()
    actions = torch.stack([row.action for row in experiences]).to(reference.device).long()
    next_states = torch.stack([row.next_observation for row in experiences]).to(reference.device).long()
    rewards = torch.stack([row.reward_sum for row in experiences]).to(reference)
    terminated = torch.stack([row.terminated for row in experiences]).to(reference.device)
    horizons = reference.new_tensor([row.horizon for row in experiences])
    q, = learner.online_model(states).values.q
    prediction = q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        learner.target_model.eval()
        next_q, = learner.target_model(next_states).values.q
        next_value = next_q.max(-1).values
        # TD0 contributes one gamma; scale its continuation by gamma**(m-1).
        continuation = learner.target_generator.gamma ** (horizons - 1) * next_value
        target = learner.target_generator.generate(rewards, continuation, terminated)
    return learner.optimize(prediction, target)


def run_steps(env, learner, replay, *, steps, world_model=None, planning_steps=0,
              planning_rng=None, batch_size=32, minimum_buffer_size=64, train_frequency=1,
              gradient_steps=1, target_sync_steps=250, epsilon_decay_steps=2000,
              n=1, n_step_rng=None, seed=7, record=False):
    """Fixed REAL-step budget, one learner, independent data sources/schedules.

    Real update -> world.observe -> one-step planning -> periodic hard sync.
    Sync uses actual ENVIRONMENT steps, as in the existing CartPole example;
    it is not multiplied by planning budget. Initial sync is done at setup.
    At a boundary, remaining real n-step records flush AFTER planning, before
    sync/reset. Flushing only creates replay metadata; it does not silently
    add gradient updates or consume RNG. Training frequency still governs
    learning. A final partial episode is an external bootstrap-capable cutoff.

    Raw ReplayBuffer stores ONLY actual transitions. For n>1 an example-local
    deque stores eligible NStepExperience records because ReplayBuffer has no
    horizon/metadata/index API. This is not a new general replay framework.
    Uniform n-step sampling is without replacement with an explicit RNG.
    Planning is allowed before real warm-up and during real-window delays.
    """
    _validate_run(steps, batch_size, minimum_buffer_size, train_frequency, gradient_steps,
                  target_sync_steps, epsilon_decay_steps, n, replay, world_model, planning_steps, planning_rng)
    if n_step_rng is None:
        n_step_rng = np.random.default_rng(seed + 3)
    pending, n_step_replay = deque(), deque(maxlen=replay.capacity)
    counters = {name: 0 for name in ("environment_steps", "real_transitions", "world_observations",
                                    "real_updates", "planning_updates", "requested_planning_updates",
                                    "total_updates", "target_syncs", "eligible_records", "flush_records")}
    trace, real_losses, planning_losses, rewards, lengths = [], [], [], [], []
    observation, _ = env.reset(seed=seed)
    episode_reward, episode_length = 0., 0
    terminated_episodes = truncated_episodes = 0
    for time in range(1, steps + 1):
        learner.online_model.action_selector.epsilon = 1. - .95 * min(1., (time - 1) / epsilon_decay_steps)
        selected = learner.online_model.act(observation)
        next_state, reward, terminated, truncated, _ = env.step(selected.action.item())
        cutoff = time == steps and not (terminated or truncated)
        transition = Transition(observation, selected.action, float(reward), next_state,
                                terminated, truncated or cutoff, selected.log_prob)
        replay.add(transition)
        counters["environment_steps"] += 1
        counters["real_transitions"] += 1
        if n > 1:
            pending.append(transition.to_batch())
            if len(pending) == n:
                n_step_replay.append(make_n_step_experience(pending, learner.target_generator,
                                                          dtype=next(learner.online_model.parameters()).dtype))
                pending.popleft()
                counters["eligible_records"] += 1
        available = len(replay) if n == 1 else len(n_step_replay)
        step_results, sampled = [], []
        if len(replay) >= minimum_buffer_size and available >= batch_size and time % train_frequency == 0:
            for _ in range(gradient_steps):
                if n == 1:
                    batch = replay.sample(batch_size)
                    result = learner.learn(batch)
                else:
                    indices = n_step_rng.choice(len(n_step_replay), size=batch_size, replace=False)
                    batch = [n_step_replay[int(index)] for index in indices]
                    result = learn_real_n_step(learner, batch)
                step_results.append(result)
                real_losses.append(result.loss.item())
                counters["real_updates"] += 1
                if record:
                    sampled.append(batch)
        if world_model is not None:
            world_model.observe(transition)
            counters["world_observations"] += 1
        planned = {"updates": 0, "results": []}
        counters["requested_planning_updates"] += planning_steps
        if world_model is not None:
            planned = plan(world_model, learner, steps=planning_steps, batch_size=batch_size, rng=planning_rng)
        counters["planning_updates"] += planned["updates"]
        planning_losses.extend(result.loss.item() for result in planned["results"])
        flushed = 0
        if n > 1 and (terminated or truncated or cutoff):
            while pending:
                n_step_replay.append(make_n_step_experience(pending, learner.target_generator,
                                                          dtype=next(learner.online_model.parameters()).dtype))
                pending.popleft()
                flushed += 1
            counters["eligible_records"] += flushed
            counters["flush_records"] += flushed
        synced = time % target_sync_steps == 0
        if synced:
            learner.sync_target()
            counters["target_syncs"] += 1
        counters["total_updates"] = counters["real_updates"] + counters["planning_updates"]
        if record:
            trace.append({"transition": transition.to_batch(), "real_batches": sampled,
                          "real_results": step_results, "real_updates": len(step_results),
                          "planning_updates": planned["updates"], "sync": synced, "cutoff": cutoff,
                          "flush_records": flushed, "eligible_records": counters["eligible_records"]})
        episode_reward += reward
        episode_length += 1
        if terminated or truncated or cutoff:
            if not cutoff:
                rewards.append(episode_reward)
                lengths.append(episode_length)
                terminated_episodes += int(terminated)
                truncated_episodes += int(truncated)
            if time < steps:
                observation, _ = env.reset()
                episode_reward, episode_length = 0., 0
        else:
            observation = next_state
    return {**counters, "trace": trace, "real_losses": np.array(real_losses),
            "planning_losses": np.array(planning_losses), "training_rewards": np.array(rewards),
            "episode_lengths": np.array(lengths), "terminated_episodes": terminated_episodes,
            "truncated_episodes": truncated_episodes, "pending": pending, "n_step_replay": n_step_replay,
            "n_step_rng": n_step_rng,
            "partial_episode": {"steps": episode_length, "reward": episode_reward} if cutoff else None}


def _positive_integer(value, name, *, minimum=1):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _validate_run(steps, batch_size, minimum, frequency, gradients, sync, decay, n,
                  replay, world, planning_steps, rng):
    for value, name in ((steps, "steps"), (batch_size, "batch_size"), (minimum, "minimum buffer size"),
                        (frequency, "train frequency"), (gradients, "gradient steps"),
                        (sync, "target sync steps"), (decay, "epsilon decay steps"), (n, "n")):
        _positive_integer(value, name)
    _positive_integer(planning_steps, "planning steps", minimum=0)
    if max(minimum, batch_size) > replay.capacity:
        raise ValueError("warm-up and batch_size must not exceed replay capacity")
    if planning_steps and (world is None or rng is None):
        raise ValueError("planning requires a World Model and independent RNG")


def _validate_window(window):
    if not window:
        raise ValueError("window must be nonempty")
    previous = None
    for row in window:
        for name in ("observation", "action", "reward", "next_observation", "terminated", "truncated"):
            if getattr(row, name).shape != (1,):
                raise ValueError("window fields must have online shape (1,)")
        if previous is not None:
            if previous.terminated.item() or previous.truncated.item():
                raise ValueError("window cannot cross an episode boundary")
            if not torch.equal(previous.next_observation, row.observation):
                raise ValueError("window must be contiguous")
        previous = row


def _validate_experiences(experiences, gamma):
    if not experiences:
        raise ValueError("n-step minibatch must be nonempty")
    for row in experiences:
        _positive_integer(row.horizon, "horizon")
        if row.gamma != gamma:
            raise ValueError("n-step reward sums require the same gamma used at collection")
        for name in ("observation", "action", "reward_sum", "next_observation", "terminated", "truncated"):
            if getattr(row, name).ndim != 0:
                raise ValueError("n-step experience fields must be scalar")
        for value in (row.observation, row.action, row.next_observation):
            if value.is_floating_point() or value.is_complex() or value.dtype == torch.bool:
                raise TypeError("n-step state/action IDs must have integer dtype")
        if not row.reward_sum.is_floating_point():
            raise TypeError("reward sum must have floating dtype")
        if row.terminated.dtype != torch.bool or row.truncated.dtype != torch.bool:
            raise TypeError("n-step boundary flags must have bool dtype")


def train_and_evaluate(*, steps=3000, seed=7, planning_steps=1, n=1, baseline=False,
                       batch_size=32, minimum_buffer_size=64, replay_capacity=10000,
                       train_frequency=1, gradient_steps=1, target_sync_steps=250,
                       epsilon_decay_steps=2000, learning_rate=1e-4, gamma=.9, max_steps=2000):
    """Same real-step budgets, deterministic MLP, one CPU thread restored on exit."""
    if baseline and (planning_steps or n != 1):
        raise ValueError("baseline requires n=1 and planning_steps=0")
    torch.manual_seed(seed)
    previous_threads = torch.get_num_threads()
    env = make_dlfs4_gridworld_env(max_episode_steps=max_steps)
    evaluation_env = make_dlfs4_gridworld_env(max_episode_steps=20)
    started = perf_counter()
    try:
        torch.set_num_threads(1)
        learner = create_learner(learning_rate=learning_rate, gamma=gamma)
        replay = ReplayBuffer(replay_capacity, seed=seed + 1)
        world = None if baseline else TabularWorldModel()
        planning_rng = np.random.default_rng(seed + 2)
        before_mode = learner.online_model.training
        learner.online_model.eval()
        before = evaluate_greedy(learner.online_model, evaluation_env, seed=seed + 10_000)
        learner.online_model.train(before_mode)
        optimal = value_iteration(env.unwrapped.mdp, gamma=gamma, theta=1e-12)
        if not optimal.converged:
            raise RuntimeError("DP reference did not converge")
        dp_episode = collect_episode(evaluation_env, lambda state: int(np.argmax(optimal.policy[state])),
                                     seed=seed + 10_000)
        result = run_steps(env, learner, replay, steps=steps, world_model=world, planning_steps=planning_steps,
                           planning_rng=planning_rng, batch_size=batch_size, minimum_buffer_size=minimum_buffer_size,
                           train_frequency=train_frequency, gradient_steps=gradient_steps,
                           target_sync_steps=target_sync_steps, epsilon_decay_steps=epsilon_decay_steps,
                           n=n, seed=seed)
        mode = learner.online_model.training
        learner.online_model.eval()
        with torch.no_grad():
            after = evaluate_greedy(learner.online_model, evaluation_env, seed=seed + 10_000)
            q, = learner.online_model(torch.arange(len(DLFS4_STATES))).values.q
            q_values = q.cpu().numpy().copy()
        learner.online_model.train(mode)
        nonterminal = np.setdiff1d(np.arange(len(DLFS4_STATES)), env.unwrapped.mdp.terminal_states)
        result.update({"seed": seed, "n": n, "planning_steps": planning_steps, "baseline": baseline,
                       "gamma": gamma, "learning_rate": learning_rate, "batch_size": batch_size,
                       "minimum_buffer_size": minimum_buffer_size, "replay_capacity": replay_capacity,
                       "train_frequency": train_frequency, "gradient_steps": gradient_steps,
                       "target_sync_steps": target_sync_steps, "epsilon_decay_steps": epsilon_decay_steps,
                       "before": before, "after": after, "world_model": world, "replay": replay,
                       "planning_rng": planning_rng, "behavior_rng": torch.get_rng_state().clone(),
                       "world_pairs": 0 if world is None else len(world), "q_values": q_values,
                       "dp_values": optimal.values, "dp_steps": len(dp_episode),
                       "max_q_rmse": float(np.sqrt(np.mean((q_values.max(-1)[nonterminal] -
                                                            optimal.values[nonterminal]) ** 2))),
                       "elapsed_seconds": perf_counter() - started})
        return learner, result
    finally:
        env.close()
        evaluation_env.close()
        torch.set_num_threads(previous_threads)


def comparison(*, steps=3000, seeds=(7, 11, 19)):
    results = []
    for seed in seeds:
        for baseline, budget, n in [(True, 0, 1), (False, 0, 1), (False, 1, 1), (False, 5, 1),
                                    (False, 1, 3), (False, 1, 5)]:
            _, result = train_and_evaluate(steps=steps, seed=seed, planning_steps=budget, n=n, baseline=baseline)
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    args = parser.parse_args()
    for row in comparison(steps=args.steps, seeds=args.seeds):
        print(f"seed={row['seed']} baseline={row['baseline']} real_n={row['n']} plan={row['planning_steps']} "
              f"env={row['environment_steps']} real={row['real_updates']} planning={row['planning_updates']} "
              f"transitions={row['real_transitions']} observed={row['world_observations']} total={row['total_updates']} "
              f"requested={row['requested_planning_updates']} syncs={row['target_syncs']} pairs={row['world_pairs']} "
              f"flush={row['flush_records']} RMSE={row['max_q_rmse']:.6f} seconds={row['elapsed_seconds']:.3f}")
        print(f"mean_episode_reward={np.mean(row['training_rewards']) if row['training_rewards'].size else None} "
              f"mean_episode_length={np.mean(row['episode_lengths']) if row['episode_lengths'].size else None} "
              f"greedy_goal={row['after']['terminated']} greedy_steps={row['after']['steps']} DP_steps={row['dp_steps']} "
              f"real_loss={np.mean(row['real_losses']) if row['real_losses'].size else None} planning_loss="
              f"{np.mean(row['planning_losses']) if row['planning_losses'].size else None}", flush=True)
        with np.printoptions(precision=4, suppress=True):
            print("Q:\n", row["q_values"])


if __name__ == "__main__":
    main()
