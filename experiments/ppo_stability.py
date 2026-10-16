"""Compare optional PPO safeguards at a fixed CartPole transition budget.

This explicit experiment owns collection, diagnostics and evaluation. Gradient
instrumentation records the learner's pre-clip norm and optimizer's post-clip
norm without changing gradients, settings, or action/shuffle random streams.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
import numpy as np
import torch
from torch import nn

from modular_rl import Model, representations as R, estimators as E, selectors as S
from modular_rl import buffers as B, targets as T, updates as U, learners as L
from modular_rl.environment import final_observations, validate_vector_env


@contextmanager
def gradient_norms(optimizer):
    """Record norms around actual learner clipping; restore instrumentation."""
    before, after = [], []
    clip = torch.nn.utils.clip_grad_norm_
    def record_clip(*args, **kwargs):
        norm = clip(*args, **kwargs)
        before.append(norm.item())
        return norm
    def record_step(optimizer, args, kwargs):
        parameters = {id(p): p for group in optimizer.param_groups for p in group['params']}
        norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in parameters.values()
                                                    if p.requires_grad and p.grad is not None])).item()
        after.append(norm)
        if len(before) < len(after):
            before.append(norm)
    hook = optimizer.register_step_pre_hook(record_step)
    torch.nn.utils.clip_grad_norm_ = record_clip
    try:
        yield before, after
    finally:
        torch.nn.utils.clip_grad_norm_ = clip
        hook.remove()


def evaluate(model, seed, episodes):
    rewards = []
    with gym.make('CartPole-v1') as env:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + 1000 + episode)
            total = 0.
            while True:
                observation, reward, terminated, truncated, _ = env.step(
                    model.act(observation, deterministic=True).action.item())
                total += reward
                if terminated or truncated:
                    break
            rewards.append(total)
    return float(np.mean(rewards))


def run(seed, *, steps=81920, max_grad_norm=None, target_kl=None, eval_episodes=20):
    n_envs, horizon = 4, 128
    if steps <= 0 or steps % (n_envs * horizon):
        raise ValueError('steps must be a positive multiple of 512')
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = Model(R.MLP(4, (64,), activation=nn.Tanh),
                  E.ActorCritic(E.CategoricalPolicy(64, 2), E.StateValue(64)), S.Categorical()).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    learner = L.PPO(model, optimizer, T.GAE(.99, .95), U.PPO(.2, .5, .01),
                    epochs=4, minibatch_size=64, seed=seed,
                    max_grad_norm=max_grad_norm, target_kl=target_kl)
    before = evaluate(model, seed, eval_episodes)
    buffer = B.Rollout(horizon, n_envs)
    optimizer_steps, completed_epochs, early_stops, processed = 0, 0, 0, 0
    kl_sum, clip_sum = 0., 0.
    env = AsyncVectorEnv([lambda: gym.make('CartPole-v1') for _ in range(n_envs)],
                         context='spawn', copy=True, autoreset_mode=AutoresetMode.SAME_STEP)
    try:
        validate_vector_env(env)
        observation, _ = env.reset(seed=[seed + index for index in range(n_envs)])
        with gradient_norms(optimizer) as (grad_before, grad_after):
            for _ in range(steps // (n_envs * horizon)):
                for _ in range(horizon):
                    with torch.no_grad():
                        estimate = model(observation)
                        action = model.action_selector.select(estimate)
                    returned, reward, terminated, truncated, info = env.step(action.action.numpy())
                    actual = final_observations(returned, terminated, truncated, info)
                    buffer.add(observation, action.action, reward, actual, terminated, truncated,
                               value=estimate.values.v, log_prob=action.log_prob)
                    observation = returned
                result = learner.learn(buffer.get())
                count = result.epochs_completed * horizon * n_envs
                kl_sum += result.approx_kl.item() * count
                clip_sum += result.clip_fraction.item() * count
                processed += count
                optimizer_steps += result.optimizer_steps
                completed_epochs += result.epochs_completed
                early_stops += int(result.early_stopped)
                buffer.clear()
    except BaseException:
        env.close(terminate=True)
        raise
    finally:
        env.close()
    return dict(seed=seed, max_grad_norm=max_grad_norm, target_kl=target_kl,
                before=before, after=evaluate(model, seed, eval_episodes), environment_steps=steps,
                optimizer_steps=optimizer_steps, epochs_completed=completed_epochs, early_stops=early_stops,
                rollouts=steps // 512, sample_uses=processed, approx_kl=kl_sum / processed,
                clip_fraction=clip_sum / processed, grad_norm_before_mean=float(np.mean(grad_before)),
                grad_norm_before_max=float(np.max(grad_before)), grad_norm_after_mean=float(np.mean(grad_after)),
                grad_norm_after_max=float(np.max(grad_after)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 11, 19, 23, 42])
    parser.add_argument('--steps', type=int, default=81920)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    records = []
    settings = [('baseline', None, None), ('clip', .5, None), ('kl', None, .02), ('clip_kl', .5, .02)]
    for label, max_norm, kl in settings:
        for seed in args.seeds:
            result = run(seed, steps=args.steps, max_grad_norm=max_norm, target_kl=kl,
                         eval_episodes=args.eval_episodes)
            records.append(dict(setting=label, **result))
            print(json.dumps(records[-1], sort_keys=True, allow_nan=False), flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(dict(config=dict(n_envs=4, horizon=128, epochs=4,
                    minibatch_size=64, learning_rate=.0003, gamma=.99, lam=.95,
                    clip_epsilon=.2, value_coef=.5, entropy_coef=.01, normalize_advantage=True,
                    eval_episodes=args.eval_episodes), results=records), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
