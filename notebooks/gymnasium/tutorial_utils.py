"""Tutorial-only plotting, evaluation and FrozenLake's known-kernel conversion.

No training loop, return formula, learner wrapper or world-model fitting lives
here. Notebook cells own construction, collection and learning order.
"""
import json

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from modular_rl.data import TabularMDP, TransitionOutcome


def frozenlake_mdp(env):
    """Explicitly adapt Gymnasium's known P for DP/reference diagnostics only."""
    base = env.unwrapped
    transitions = tuple(tuple(tuple(TransitionOutcome(float(p), int(s), float(r), bool(d))
                                    for p, s, r, d in base.P[state][action])
                              for action in range(base.action_space.n))
                        for state in range(base.observation_space.n))
    terminal = tuple(int(i) for i, tile in enumerate(base.desc.flatten()) if tile in (b'H', b'G'))
    return TabularMDP(base.observation_space.n, base.action_space.n, transitions, terminal)


def evaluate(env, action, *, episodes=20, seed=1007):
    """Separate evaluation environment; never call learn, add or optimizer."""
    rewards, lengths, terminateds, truncateds, paths = [], [], [], [], []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed+episode)
        total, length, path = 0., 0, [observation]
        while True:
            observation, reward, terminated, truncated, _ = env.step(int(action(observation)))
            total += reward; length += 1; path.append(observation)
            if terminated or truncated:
                break
        rewards.append(total); lengths.append(length)
        terminateds.append(terminated); truncateds.append(truncated); paths.append(path)
    return dict(rewards=rewards, lengths=lengths, terminated=terminateds, truncated=truncateds, paths=paths)


def history():
    return dict(rewards=[], lengths=[], terminated=[], truncated=[])


def record(log, reward, length, terminated, truncated):
    log['rewards'].append(float(reward)); log['lengths'].append(int(length))
    log['terminated'].append(bool(terminated)); log['truncated'].append(bool(truncated))


def moving_average(values, window=20):
    values = np.asarray(values, dtype=float)
    window = min(window, len(values))
    return np.convolve(values, np.ones(window)/window, mode='valid') if window else values


def frozen_report(desc, log, table, evaluation, *, greedy_actions, label):
    """Plot collected behavior and separate evaluation; tables have explicit meaning."""
    rewards = np.asarray(log['rewards'])
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes[0, 0].plot(rewards); axes[0, 0].set_title('Behavior episode reward')
    axes[0, 1].plot(log['lengths']); axes[0, 1].set_title('Behavior episode length')
    axes[1, 0].plot(moving_average(rewards > 0)); axes[1, 0].set_title('Behavior success rate (moving mean)')
    axes[1, 1].bar(['terminated', 'truncated'], [np.mean(log[k]) for k in ('terminated', 'truncated')])
    axes[1, 1].set_ylim(0, 1); axes[1, 1].set_title('Boundary flags (independent)')
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.tight_layout(); plt.show()
    table = np.asarray(table)
    values = table.max(-1) if table.ndim == 2 else table
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    image = axes[0].imshow(values.reshape(desc.shape), cmap='viridis'); fig.colorbar(image, ax=axes[0])
    axes[0].set_title(label)
    arrows = ['←', '↓', '→', '↑']  # Gymnasium FrozenLake action IDs
    axes[1].imshow(np.zeros(desc.shape), cmap='Greys', vmin=0, vmax=1)
    for state, tile in enumerate(desc.flatten()):
        y, x = divmod(state, desc.shape[1])
        text = tile.decode() if tile in (b'H', b'G') else arrows[int(greedy_actions[state])]
        axes[1].text(x, y, text, ha='center', va='center', fontsize=15)
    path = evaluation['paths'][0]
    axes[1].plot([int(s) % desc.shape[1] for s in path], [int(s)//desc.shape[1] for s in path], 'o-', alpha=.5)
    axes[1].set_title('Greedy arrows / evaluation episode path')
    fig.tight_layout(); plt.show()
    print(label, np.round(table, 4))
    result = dict(evaluation_mean_reward=float(np.mean(evaluation['rewards'])),
                  evaluation_success=float(np.mean(np.asarray(evaluation['rewards']) > 0)),
                  episodes=len(rewards), environment_steps=int(sum(log['lengths'])),
                  terminated_rate=float(np.mean(log['terminated'])), truncated_rate=float(np.mean(log['truncated'])))
    print('RESULT', json.dumps(result, sort_keys=True))
    return result


def neural_report(log, losses, epsilons, updates, before, after):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes[0, 0].plot(log['rewards'], alpha=.4, label='episode return')
    mean = moving_average(log['rewards']); axes[0, 0].plot(np.arange(len(mean))+len(log['rewards'])-len(mean), mean, label='moving mean')
    axes[0, 0].legend()
    axes[0, 1].bar(['before', 'after'], [np.mean(before['rewards']), np.mean(after['rewards'])]); axes[0, 1].set_title('Separate greedy evaluation')
    axes[0, 2].plot(losses); axes[0, 2].set_title('Pre-update loss')
    axes[1, 0].plot(epsilons); axes[1, 0].set_title('Epsilon by episode')
    axes[1, 1].plot(np.cumsum(log['lengths'])); axes[1, 1].set_title('Environment steps')
    axes[1, 2].plot(updates); axes[1, 2].set_title('Gradient updates')
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.tight_layout(); plt.show()
    result = dict(before=float(np.mean(before['rewards'])), after=float(np.mean(after['rewards'])),
                  episodes=len(log['rewards']), environment_steps=int(sum(log['lengths'])),
                  gradient_updates=int(updates[-1]), last_loss=float(losses[-1]))
    print('RESULT', json.dumps(result, sort_keys=True))
    return result
