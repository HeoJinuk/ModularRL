"""Run python -m examples.frozenlake_tabular_q; requires the examples extra."""

import argparse

import gymnasium as gym
import torch

from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.buffer import ReplayBuffer
from modular_rl.component.estimator import TabularQEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import Transition
from modular_rl.learner import TabularQLearner
from modular_rl.model import Model


def evaluate_policy(model, env, episodes, seed):
    """Evaluate a deterministic greedy policy without collecting or learning."""
    successes = terminated_count = truncated_count = 0
    lengths = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        length = 0
        while True:
            action = model.act(observation, deterministic=True).action.item()
            observation, reward, terminated, truncated, _ = env.step(action)
            length += 1
            if terminated or truncated:
                successes += int(reward > 0)
                terminated_count += int(terminated)
                truncated_count += int(truncated)
                lengths.append(length)
                break
    return {"episodes": episodes, "success_rate": successes / episodes,
            "mean_length": sum(lengths) / episodes, "terminated": terminated_count,
            "truncated": truncated_count}


def train_and_evaluate(
    episodes=1000, evaluation_episodes=100, seed=7, max_episode_steps=100, log_every=100
):
    """An explicit environment loop; buffer sampling stays outside Learner."""
    if episodes < 1 or evaluation_episodes < 1 or max_episode_steps < 1:
        raise ValueError("episode counts and max_episode_steps must be positive")
    torch.manual_seed(seed)  # ActionSelector samples with PyTorch RNG.
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False,
                   max_episode_steps=max_episode_steps)
    try:
        model = Model(IdentityRepresentation(), TabularQEstimator(env.observation_space.n, env.action_space.n),
                      EpsilonGreedyActionSelector(1.0))
        learner = TabularQLearner(model, TD0TargetGenerator(gamma=0.99), TabularQUpdateRule(alpha=0.5))
        replay = ReplayBuffer(capacity=2000, seed=seed)
        initial_q = model.estimator.q_table.clone()
        before = evaluate_policy(model, env, evaluation_episodes, seed + 10_000)
        episode_returns, episode_lengths = [], []
        terminated_count = truncated_count = 0

        for episode in range(episodes):
            model.action_selector.epsilon = max(0.05, 1.0 - episode / episodes)
            observation, _ = env.reset(seed=seed + episode)
            episode_return, length = 0.0, 0
            while True:
                selected = model.act(observation)
                next_observation, reward, terminated, truncated, _ = env.step(selected.action.item())
                # No autoreset wrapper: step returns this transition's actual final observation.
                replay.add(Transition(observation, selected.action, float(reward), next_observation,
                                      terminated, truncated, selected.log_prob))
                batch = replay.sample(min(32, len(replay)))
                learner.learn(batch)
                observation = next_observation
                episode_return += reward
                length += 1
                if terminated or truncated:
                    terminated_count += int(terminated)
                    truncated_count += int(truncated)
                    break
            episode_returns.append(episode_return)
            episode_lengths.append(length)
            if log_every and ((episode + 1) % log_every == 0 or episode + 1 == episodes):
                recent = episode_returns[-log_every:]
                print(f"episode={episode + 1}/{episodes} epsilon={model.action_selector.epsilon:.3f} "
                      f"recent_success={sum(recent) / len(recent):.1%} length={length}", flush=True)

        after = evaluate_policy(model, env, evaluation_episodes, seed + 10_000)
        delta = model.estimator.q_table - initial_q
        training = {"episodes": episodes, "steps": sum(episode_lengths),
                    "successes": sum(reward > 0 for reward in episode_returns),
                    "mean_length": sum(episode_lengths) / episodes,
                    "terminated": terminated_count, "truncated": truncated_count,
                    "changed_q_entries": int(torch.count_nonzero(delta)),
                    "max_q_change": delta.abs().max().item()}
        return model, {"before": before, "after": after, "training": training}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Tabular Q-learning on non-slippery FrozenLake-v1")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()
    model, results = train_and_evaluate(args.episodes, args.eval_episodes, args.seed, args.max_steps)
    print(f"FrozenLake-v1 is_slippery=False seed={args.seed}")
    for phase in ("before", "after"):
        metrics = results[phase]
        print(f"{phase} greedy: success={metrics['success_rate']:.1%} "
              f"episodes={metrics['episodes']} mean_length={metrics['mean_length']:.2f} "
              f"terminated={metrics['terminated']} truncated={metrics['truncated']}")
    training = results["training"]
    print(f"training: episodes={training['episodes']} steps={training['steps']} "
          f"successes={training['successes']} terminated={training['terminated']} "
          f"truncated={training['truncated']}")
    print(f"Q-table: changed_entries={training['changed_q_entries']}/64 "
          f"max_change={training['max_q_change']:.6f}")
    print("Greedy actions (4x4; 0=left, 1=down, 2=right, 3=up):")
    print(model.estimator.q_table.argmax(-1).reshape(4, 4))


if __name__ == "__main__":
    main()
