"""Run python -m examples.cartpole_dqn; requires the optional examples extra."""

import argparse
from copy import deepcopy

import gymnasium as gym
import torch
from torch import nn

from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.buffer import ReplayBuffer
from modular_rl.component.estimator import DiscreteQEstimator
from modular_rl.component.representation import MLPRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TDLossUpdateRule
from modular_rl.data import Transition
from modular_rl.learner import DQNLearner
from modular_rl.model import Model


def evaluate_policy(model, env, episodes, seed):
    """Evaluate greedy actions with separate reset seeds and restore online mode."""
    training = model.training
    model.eval()
    returns = []
    terminated_count = truncated_count = 0
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            while True:
                action = model.act(observation, deterministic=True).action.item()
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward
                if terminated or truncated:
                    terminated_count += int(terminated)
                    truncated_count += int(truncated)
                    break
            returns.append(episode_return)
    finally:
        model.train(training)
    return {"episodes": episodes, "mean_reward": sum(returns) / episodes,
            "terminated": terminated_count, "truncated": truncated_count}


def train_and_evaluate(
    episodes=400, evaluation_episodes=20, seed=7, max_episode_steps=500,
    batch_size=64, warmup_steps=1000, target_sync_steps=250, epsilon_decay_steps=10_000,
    log_every=25,
):
    """Explicit basic DQN; exploration, sampling and synchronization stay here.

    CPU, MLP (128, 128), replay capacity 10,000, Adam lr=1e-4, gamma=.99,
    SmoothL1 TD loss, epsilon linearly decreases from 1 to .05 over 10,000
    environment steps. No reward shaping, clipping, Double DQN or early stop.
    Small CPU networks use one Torch thread, restored when this function exits.
    """
    if min(episodes, evaluation_episodes, max_episode_steps, batch_size,
           warmup_steps, target_sync_steps, epsilon_decay_steps) < 1:
        raise ValueError("episode counts, batch size and step intervals must be positive")
    if batch_size > 10_000:
        raise ValueError("batch_size must not exceed replay capacity 10,000")
    torch.manual_seed(seed)
    previous_threads = torch.get_num_threads()
    env = gym.make("CartPole-v1", max_episode_steps=max_episode_steps)
    try:
        torch.set_num_threads(1)
        representation = MLPRepresentation(env.observation_space.shape[0], hidden_dims=(128, 128))
        model = Model(representation, DiscreteQEstimator(representation.output_dim, env.action_space.n),
                      EpsilonGreedyActionSelector(1.0))
        target_model = deepcopy(model)  # Independent parameters and buffers.
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        learner = DQNLearner(model, target_model, optimizer, TD0TargetGenerator(gamma=0.99),
                             TDLossUpdateRule(nn.SmoothL1Loss(reduction="none")))
        learner.sync_target()
        replay = ReplayBuffer(capacity=10_000, seed=seed)
        before = evaluate_policy(model, env, evaluation_episodes, seed + 10_000)
        initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
        episode_returns = []
        steps = updates = syncs = terminated_count = truncated_count = 0
        latest = None

        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            while True:
                fraction = min(1.0, steps / epsilon_decay_steps)
                model.action_selector.epsilon = 1.0 - 0.95 * fraction
                selected = model.act(observation)
                next_observation, reward, terminated, truncated, _ = env.step(selected.action.item())
                # No autoreset wrapper: next_observation is the actual final observation.
                replay.add(Transition(observation, selected.action, float(reward), next_observation,
                                      terminated, truncated, selected.log_prob))
                steps += 1
                if steps >= warmup_steps and len(replay) >= batch_size:
                    latest = learner.learn(replay.sample(batch_size))
                    updates += 1
                if steps % target_sync_steps == 0:
                    learner.sync_target()
                    syncs += 1
                observation = next_observation
                episode_return += reward
                if terminated or truncated:
                    terminated_count += int(terminated)
                    truncated_count += int(truncated)
                    break
            episode_returns.append(episode_return)
            if log_every and ((episode + 1) % log_every == 0 or episode + 1 == episodes):
                loss = "n/a" if latest is None else f"{latest.loss.item():.5f}"
                td_mean = "n/a" if latest is None else f"{latest.td_error.mean().item():.5f}"
                td_abs = "n/a" if latest is None else f"{latest.td_error.abs().mean().item():.5f}"
                recent = episode_returns[-log_every:]
                print(f"episode={episode + 1}/{episodes} reward={episode_return:.1f} "
                      f"recent_mean_reward={sum(recent) / len(recent):.1f} "
                      f"epsilon={model.action_selector.epsilon:.3f} loss={loss} "
                      f"td_mean={td_mean} td_abs_mean={td_abs}", flush=True)

        after = evaluate_policy(model, env, evaluation_episodes, seed + 10_000)
        parameter_change = max((parameter.detach() - initial).abs().max().item()
                               for parameter, initial in zip(model.parameters(), initial_parameters))
        training = {"episodes": episodes, "steps": steps, "updates": updates, "target_syncs": syncs,
                    "mean_reward": sum(episode_returns) / episodes,
                    "terminated": terminated_count, "truncated": truncated_count,
                    "max_parameter_change": parameter_change,
                    "last_loss": None if latest is None else latest.loss.item(),
                    "last_td_abs_mean": None if latest is None else latest.td_error.abs().mean().item()}
        return model, {"before": before, "after": after, "training": training}
    finally:
        env.close()
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description="Basic DQN on CartPole-v1")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--target-sync-steps", type=int, default=250)
    parser.add_argument("--epsilon-decay-steps", type=int, default=10_000)
    args = parser.parse_args()
    _, results = train_and_evaluate(args.episodes, args.eval_episodes, args.seed, args.max_steps,
                                    args.batch_size, args.warmup_steps, args.target_sync_steps,
                                    args.epsilon_decay_steps)
    print(f"CartPole-v1 seed={args.seed}")
    for phase in ("before", "after"):
        metrics = results[phase]
        print(f"{phase} greedy: evaluation_mean_reward={metrics['mean_reward']:.2f} "
              f"episodes={metrics['episodes']} terminated={metrics['terminated']} "
              f"truncated={metrics['truncated']}")
    metrics = results["training"]
    print(f"training: episodes={metrics['episodes']} steps={metrics['steps']} "
          f"updates={metrics['updates']} target_syncs={metrics['target_syncs']} "
          f"terminated={metrics['terminated']} truncated={metrics['truncated']}")
    print(f"network: max_parameter_change={metrics['max_parameter_change']:.6f} "
          f"last_loss={metrics['last_loss']} last_td_abs_mean={metrics['last_td_abs_mean']}")


if __name__ == "__main__":
    main()
