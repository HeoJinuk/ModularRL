"""Diagnose FrozenLake Notebook 14 without changing its learning equations.

Run from the checkout: python -m experiments.dyna_dqn_diagnosis --help.
Raw results/checkpoints belong outside Git. The report records the runs actually
performed, including unsuccessful policies; this is not a hyperparameter search.
"""

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from modular_rl import Model, representations as R, estimators as E, selectors as S
from modular_rl import buffers as B, targets as T, updates as U, learners as L
from modular_rl import world_models as W, planners as P
from modular_rl.data import Transition


@dataclass(frozen=True)
class Config:
    """Notebook settings plus explicit experimental controls, not library config."""

    seed: int = 7
    steps: int = 7527
    episodes: int | None = None  # original Notebook stops after 1000 episodes
    collect_world: bool = True
    planning_steps: int = 1
    gamma: float = .99
    learning_rate: float = .001
    epsilon_decay_steps: int = 6000
    replay_capacity: int = 5000
    warmup: int = 64
    batch_size: int = 32
    target_sync_steps: int = 100
    planning_start: int = 1
    architecture: str = "mlp"
    matched_planning: int | None = None
    checkpoint_interval: int = 250
    evaluation_episodes: int = 100


def build_learner(config):
    """Reproduce Notebook initialization, including behavior RNG after init.

    The independent linear ablation copies ALL initial MLP Q values and restores
    its post-initialization Torch RNG while changing the parameterization.
    Default MLP and Adam initialization are unchanged.
    """
    torch.manual_seed(config.seed)
    representation = nn.Sequential(nn.Embedding.from_pretrained(torch.eye(16), freeze=True),
                                   R.MLP(16, hidden_dims=(64, 64)))
    model = Model(representation, E.DiscreteQ(64, 4), S.EpsilonGreedy(1.))
    if config.architecture == "independent":
        rng = torch.get_rng_state()
        with torch.no_grad():
            initial = torch.stack([model(state).values.q[0] for state in range(16)])
        network = nn.Linear(16, 4, bias=False)
        with torch.no_grad():
            network.weight.copy_(initial.T)
        model = Model(nn.Embedding.from_pretrained(torch.eye(16), freeze=True),
                      E.DiscreteQ(16, 4, network=network), S.EpsilonGreedy(1.))
        torch.set_rng_state(rng)
    target = deepcopy(model)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)
    learner = L.DQN(model, target, optimizer, T.TD0(config.gamma),
                    U.TDLoss(nn.SmoothL1Loss(reduction="none")))
    learner.sync_target()
    return learner


@torch.no_grad()
def action_values(model):
    """Use scalar forwards as actual Model.act does, preserving near-tie rounding."""
    return torch.stack([model(state).values.q[0] for state in range(16)]).cpu().numpy().copy()


def evaluate(policy, *, seed, episodes=100):
    """Separate deterministic environment; never touch the training RNG/model."""
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, max_episode_steps=100)
    rewards, lengths, paths, terminateds, truncateds = [], [], [], [], []
    try:
        for index in range(episodes):
            state, _ = env.reset(seed=seed + 1000 + index)
            total, path = 0., [int(state)]
            for length in range(1, 101):
                state, reward, terminated, truncated, _ = env.step(int(policy[state]))
                total += reward
                path.append(int(state))
                if terminated or truncated:
                    break
            rewards.append(float(total)); lengths.append(length); paths.append(path)
            terminateds.append(bool(terminated)); truncateds.append(bool(truncated))
    finally:
        env.close()
    path = paths[0]
    repeated = next((int(state) for index, state in enumerate(path) if state in path[:index]), None)
    return dict(successes=sum(r > 0 for r in rewards), episodes=episodes,
                mean_reward=float(np.mean(rewards)), mean_length=float(np.mean(lengths)),
                terminated=sum(terminateds), truncated=sum(truncateds),
                first_path=path, first_repeated_state=repeated)


def _stats():
    return dict(updates=0, samples=0, loss_sum=0., td_sum=0., td_abs_sum=0.,
                td_abs_max=0., positive_samples=0)


def _record(stats, output, batch):
    error = output.td_error.detach().cpu().numpy().astype(np.float64)
    loss = output.loss.item()
    if not np.isfinite(error).all() or not np.isfinite(loss):
        raise FloatingPointError("Nonfinite TD error/loss during diagnosis")
    stats["updates"] += 1; stats["samples"] += error.size
    stats["loss_sum"] += loss; stats["td_sum"] += float(error.sum())
    stats["td_abs_sum"] += float(np.abs(error).sum())
    stats["td_abs_max"] = max(stats["td_abs_max"], float(np.abs(error).max()))
    stats["positive_samples"] += int((batch.reward > 0).sum())


def _means(stats):
    return dict(updates=stats["updates"], samples=stats["samples"],
                loss_mean=stats["loss_sum"] / max(1, stats["updates"]),
                td_mean=stats["td_sum"] / max(1, stats["samples"]),
                td_abs_mean=stats["td_abs_sum"] / max(1, stats["samples"]),
                td_abs_max=stats["td_abs_max"], positive_samples=stats["positive_samples"],
                positive_fraction=stats["positive_samples"] / max(1, stats["samples"]))


def _world_profile(world, env):
    if world is None:
        return dict(pairs=0, positive_pairs=0, records=[], incorrect_pairs=0)
    records, incorrect = [], 0
    for row in world.state_dict()["outcomes"]:
        _, state, reward, terminated = env.unwrapped.P[row.observation][row.action][0]
        incorrect += int((row.next_observation, row.reward.item(), row.terminated) != (state, reward, terminated))
        records.append([row.observation, row.action, row.reward.item(), row.next_observation, row.terminated])
    return dict(pairs=len(records), positive_pairs=sum(row[2] > 0 for row in records),
                records=records, incorrect_pairs=incorrect)


def _checkpoint(learner, config, step, counters, world, env, replay_rewards, interval):
    online, target = action_values(learner.online_model), action_values(learner.target_model)
    if not np.isfinite(online).all() or not np.isfinite(target).all():
        raise FloatingPointError("Nonfinite Q values during diagnosis")
    residuals = []
    for state in range(16):
        for action in range(4):
            # Known P is used only for diagnosis, never passed to world/learner.
            expected = sum(p * (r + config.gamma * (not terminal) * target[next_state].max())
                           for p, next_state, r, terminal in env.unwrapped.P[state][action])
            if env.unwrapped.desc.flat[state] not in (b"H", b"G"):
                residuals.append(float(expected - online[state, action]))
    return dict(step=step, **counters, q=online.tolist(), q_start=online[0].tolist(),
                max_q=online.max(-1).tolist(), q_abs_max=float(np.abs(online).max()),
                online_target_abs_max=float(np.abs(online-target).max()),
                bellman_target_residual_abs_max=float(np.abs(residuals).max()),
                policy=online.argmax(-1).tolist(),
                evaluation=evaluate(online.argmax(-1), seed=config.seed, episodes=config.evaluation_episodes),
                replay_positive=sum(replay_rewards), replay_size=len(replay_rewards),
                world=_world_profile(world, env), real=_means(interval["real"]), planning=_means(interval["planning"]))


def fingerprint(value):
    """Stable exact hashes for on/off logging, A0/A1 and saved experiment checks."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.detach().cpu().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode()); digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key, child in item.items():
                digest.update(repr(key).encode()); visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item: visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def _validate(config):
    if config.planning_steps < 0 or (config.planning_steps and not config.collect_world):
        raise ValueError("Planning requires a world and nonnegative planning_steps")
    if not 1 <= config.batch_size <= config.warmup <= config.replay_capacity:
        raise ValueError("Require 1 <= batch_size <= warmup <= replay_capacity")
    if config.steps < 1 or config.checkpoint_interval < 1 or config.target_sync_steps < 1:
        raise ValueError("Step budgets, checkpoint interval and target sync must be positive")
    if config.architecture not in ("mlp", "independent"):
        raise ValueError("architecture must be mlp or independent")
    if config.matched_planning is not None and (config.episodes is not None or config.planning_steps):
        raise ValueError("Matched control uses fixed steps and replay-only learning")
    if config.matched_planning is not None and config.steps < config.warmup:
        raise ValueError("Matched replay requires at least warmup environment steps")
    if config.epsilon_decay_steps < 1 or config.planning_start < 1:
        raise ValueError("Epsilon decay and planning start must be positive")
    if config.episodes is not None and config.episodes < 1:
        raise ValueError("episodes must be positive")


def run(config=Config(), *, diagnostics=True):
    """Return JSON diagnostics and exact state for tests, without changing core APIs.

    Order matches Notebook 14: act -> real replay update -> observe -> planning
    -> checkpoint BEFORE scheduled hard sync -> boundary/reset. No collector
    cutoff is invented as an environment flag. A partial episode is reported.
    Matched controls distribute the exact optimizer quota over replay-eligible
    environment steps; unavailable early planning updates cannot be time matched.
    """
    _validate(config)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    learner = build_learner(config)
    model = learner.online_model
    buffer = B.Replay(config.replay_capacity, seed=config.seed)
    world = W.Tabular() if config.collect_world else None
    planning_rng = np.random.default_rng(config.seed + 3)
    planner = None if world is None else P.Dyna(world, learner, batch_size=config.batch_size, rng=planning_rng)
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, max_episode_steps=1000)
    initial_q = action_values(model)
    initial_rng = torch.get_rng_state().clone()
    replay_rewards = deque(maxlen=config.replay_capacity)
    counts = dict(real_updates=0, planning_updates=0, optimizer_steps=0, target_syncs=0,
                  completed_episodes=0, goals=0, terminated=0, truncated=0)
    visits, state_actions = np.zeros(16, dtype=np.int64), np.zeros((16, 4), dtype=np.int64)
    total_stats = {source: _stats() for source in ("real", "planning")}
    interval = deepcopy(total_stats)
    checkpoints, episodes, trace = [], [], []
    eligible = max(0, config.steps - config.warmup + 1)
    quota = eligible + (config.matched_planning or 0) * config.steps
    try:
        observation, _ = env.reset(seed=config.seed)
        visits[observation] += 1
        reward_sum, length, step = 0., 0, 0
        while True:
            model.action_selector.epsilon = max(.05, 1.-step/config.epsilon_decay_steps)
            selected = model.act(observation)
            action = selected.action.item()
            next_observation, reward, terminated, truncated, _ = env.step(action)
            reward = float(reward)
            buffer.add(observation, selected.action, reward, next_observation, terminated, truncated)
            replay_rewards.append(int(reward > 0))
            state_actions[observation, action] += 1; visits[next_observation] += 1
            step += 1
            if len(buffer) >= config.warmup:
                gradients = 1 if config.matched_planning is None else (
                    (step-config.warmup+1)*quota//eligible - counts["real_updates"])
                for _ in range(gradients):
                    batch = buffer.sample(config.batch_size)
                    result = learner.learn(batch)
                    counts["real_updates"] += 1
                    if diagnostics:
                        _record(total_stats["real"], result, batch); _record(interval["real"], result, batch)
            if world is not None:
                world.observe(Transition(observation, action, reward, next_observation, terminated, truncated))
                planned = planner.plan(config.planning_steps if step >= config.planning_start else 0,
                                       record_batches=diagnostics)
                counts["planning_updates"] += planned["updates"]
                if diagnostics:
                    for result, batch in zip(planned["results"], planned["batches"]):
                        _record(total_stats["planning"], result, batch)
                        _record(interval["planning"], result, batch)
            counts["optimizer_steps"] = counts["real_updates"] + counts["planning_updates"]
            counts["goals"] += int(reward > 0)
            counts["terminated"] += int(terminated); counts["truncated"] += int(truncated)
            reward_sum += reward; length += 1
            trace.append([observation, action, reward, next_observation, int(terminated), int(truncated)])
            if terminated or truncated:
                counts["completed_episodes"] += 1
                episodes.append(dict(step=step, reward=reward_sum, length=length,
                                     terminated=bool(terminated), truncated=bool(truncated)))
            finished = (counts["completed_episodes"] >= config.episodes if config.episodes is not None
                        else step >= config.steps)
            if diagnostics and (step % config.checkpoint_interval == 0 or finished):
                checkpoints.append(_checkpoint(learner, config, step, counts, world, env, replay_rewards, interval))
                interval = {source: _stats() for source in ("real", "planning")}
            if step % config.target_sync_steps == 0:
                learner.sync_target(); counts["target_syncs"] += 1
            if finished:
                break
            if terminated or truncated:
                observation, _ = env.reset(); visits[observation] += 1
                reward_sum, length = 0., 0
            else:
                observation = next_observation
        with torch.no_grad():
            batched_q = model(torch.arange(16)).values.q[0].cpu().numpy().copy()
        final_q = action_values(model)
        world_profile = _world_profile(world, env)
        state = dict(online=deepcopy(model.state_dict()), target=deepcopy(learner.target_model.state_dict()),
                     optimizer=deepcopy(learner.optimizer.state_dict()), torch_rng=torch.get_rng_state().clone(),
                     replay_rng=deepcopy(buffer._rng.bit_generator.state),
                     planning_rng=deepcopy(planning_rng.bit_generator.state),
                     environment_rng=deepcopy(env.np_random.bit_generator.state), trace=np.array(trace))
        result = dict(config=asdict(config), steps=step, **counts, initial_q=initial_q.tolist(),
                      initial_behavior_rng_hash=fingerprint(initial_rng), q=final_q.tolist(), q_batched=batched_q.tolist(),
                      evaluation=evaluate(final_q.argmax(-1), seed=config.seed, episodes=config.evaluation_episodes),
                      state_visits=visits.tolist(), state_action_visits=state_actions.tolist(),
                      observed_pairs=int((state_actions>0).sum()), world=world_profile,
                      positive_transitions=counts["goals"], replay_positive=sum(replay_rewards), replay_size=len(buffer),
                      real=_means(total_stats["real"]), planning=_means(total_stats["planning"]),
                      checkpoints=checkpoints, episodes=episodes,
                      partial_episode=None if terminated or truncated else dict(reward=reward_sum, length=length),
                      hashes={key: fingerprint(value) for key, value in state.items()})
        return result, state
    finally:
        env.close()
        torch.set_num_threads(previous_threads)


def chain_propagation(*, blocks=80, seed=7):
    """Known two-transition chain, no exploration: reward must propagate backward.

    Both methods use the same observed-only world. Idle planning has no sync
    schedule of its own: this controlled demonstration explicitly copies the
    target after each block of four calls. It does not simulate environment
    steps or claim that this is the FrozenLake schedule.
    """
    torch.manual_seed(seed)
    table_model = Model(R.Identity(), E.TabularQ(3, 1))
    tabular = L.QLearning(table_model, T.TD0(.9), U.TabularQ(.3))
    network = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        network.weight.zero_(); network.weight[0, 2] = 50.
    model = Model(nn.Embedding.from_pretrained(torch.eye(3), freeze=True),
                  E.DiscreteQ(3, 1, network=network))
    neural = L.DQN(model, deepcopy(model), torch.optim.SGD(model.parameters(), lr=.1),
                   T.TD0(.9), U.TDLoss())
    neural.sync_target()
    table_model.estimator.q_table[2] = 50.
    world = W.Tabular()
    world.observe(Transition(0, 0, 0., 1, False, False))
    world.observe(Transition(1, 0, 1., 2, True, False))
    table_planner = P.Dyna(world, tabular, seed=seed)
    neural_planner = P.Dyna(world, neural, seed=seed)
    trajectory = []
    for block in range(blocks):
        table_planner.plan(4); neural_planner.plan(4)
        with torch.no_grad():
            q = model(torch.tensor([0, 1])).values.q[0][:, 0].tolist()
        trajectory.append(dict(updates=(block+1)*4, tabular=table_model.estimator.q_table[:2, 0].tolist(), neural=q))
        neural.sync_target()
    return dict(gamma=.9, expected=[.9, 1.], trajectory=trajectory,
                terminal_q=50., note="Target sync is explicit; terminal Q=50 is never bootstrapped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=7527)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--planning", type=int, default=1)
    parser.add_argument("--no-world", action="store_true")
    parser.add_argument("--matched-planning", type=int)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--target-sync", type=int, default=100)
    parser.add_argument("--planning-start", type=int, default=1)
    parser.add_argument("--epsilon-decay", type=int, default=6000)
    parser.add_argument("--architecture", choices=("mlp", "independent"), default="mlp")
    parser.add_argument("--interval", type=int, default=250)
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = Config(seed=args.seed, steps=args.steps, episodes=args.episodes, planning_steps=args.planning,
                    collect_world=not args.no_world, matched_planning=args.matched_planning,
                    learning_rate=args.learning_rate, target_sync_steps=args.target_sync,
                    planning_start=args.planning_start, epsilon_decay_steps=args.epsilon_decay,
                    architecture=args.architecture, checkpoint_interval=args.interval)
    result, _ = run(config, diagnostics=not args.no_diagnostics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(seed=args.seed, planning=args.planning, steps=result["steps"], goals=result["goals"],
                          updates=result["optimizer_steps"], evaluation=result["evaluation"], output=str(args.output)), sort_keys=True))


if __name__ == "__main__":
    main()
