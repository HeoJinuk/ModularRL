"""Online n-step Q(sigma), using Sutton & Barto (2nd ed.) Eq. 7.17's boundary."""

import argparse
from collections import deque
from numbers import Real

import numpy as np
import torch
from torch import Tensor

from examples import gridworld_n_step_sarsa as sampled
from examples import gridworld_tree_backup as tree
from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import TabularQEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import Transition
from modular_rl.dp import action_values, policy_evaluation
from modular_rl.environment import DLFS4_STATES, make_dlfs4_gridworld_env
from modular_rl.model import Model


@torch.no_grad()
def learn_q_sigma(model, window, target_generator, update_rule, *, target_policy, sigma):
    """Compute Eq. 7.17 from one Q/pi snapshot, then update only the window head.

    window contains contiguous owned B=1 TransitionBatch records. sigma is a
    fixed scalar or m frozen values [sigma_(t+1), ..., sigma_(t+m)], where m is
    len(window). The last value is a validated placeholder: the horizon residual
    G_h:h-Q(S_h,A_h) is zero. No endpoint action/log is required or sampled.
    Callbacks belong to collection, never this function. Intermediate b comes
    from original sampling-time log_prob, before batch floating-dtype conversion.

    target_policy is a fixed (S,A) Tensor or existing discrete Selector. Q and
    dynamic pi share one estimate before any write. Natural termination starts
    at the last reward; truncation uses the final observation's expected Q.
    Each reward backup reuses TD0TargetGenerator. There is no second IS product.
    Numeric failures leave this head unchanged; earlier successful calls remain.
    Returned target/td_error are detached (1,) in table dtype/device; coefficients
    and importance_ratios are float64 (m-1,) in forward intermediate-time order.
    The caller pops window/metadata after success; each flush reads current Q.
    """
    sampled._validate_window(model, window)
    table = model.estimator.q_table
    tree._validate_actions(window, table)
    sigmas = _validate_sigma(sigma, len(window), table.device)
    tree._off_policy['_validate_logs'](window, None, table)
    if isinstance(target_policy, Tensor):
        tree._off_policy['_validate_policy'](target_policy, table)
    elif not callable(getattr(target_policy, 'probabilities', None)):
        raise TypeError('target_policy must be a probability Tensor or discrete Selector')
    batches = [row.to(table.device, dtype=table.dtype) for row in window]
    first, last = batches[0], batches[-1]
    next_states = torch.cat([row.next_observation for row in batches]).long()
    needed = next_states[:-1] if last.terminated.item() else next_states
    ratios, coefficients = [], []
    if needed.numel():
        estimate = model.forward(needed)
        q = estimate.values.q[0]
        probabilities = tree._probabilities(target_policy, needed, estimate)
        expected_values = (probabilities * q).sum(-1)
        if not torch.isfinite(q).all() or not torch.isfinite(expected_values).all():
            raise FloatingPointError('Non-finite Q/expectation; head was not updated')
        for time, row in enumerate(list(window)[1:]):
            pi = probabilities[time, row.action.item()]
            rho, coefficient = _coefficient(pi, row.log_prob, sigmas[time])
            ratios.append(rho)
            coefficients.append(coefficient)

    continuation = torch.zeros_like(last.reward) if last.terminated.item() else expected_values[-1:]
    target = target_generator.generate(last.reward, continuation, last.terminated)
    if not torch.isfinite(target).all():
        raise FloatingPointError('Non-finite horizon return; head was not updated')
    for time in range(len(batches)-2, -1, -1):
        pi = probabilities[time, batches[time+1].action.item()]
        sampled_q = q[time, batches[time+1].action.item()].reshape(1)
        expected = expected_values[time:time+1]
        coefficient = coefficients[time].to(table.dtype)
        if coefficient.item() == 0:
            continuation = expected
        elif coefficient.item() == 1 and pi.item() == 1:
            continuation = target  # Exact Tree Backup deterministic branch.
        else:
            residual = target - sampled_q
            correction = coefficient * residual
            _validate_product(coefficient, residual, correction)
            continuation = expected + correction
        row = batches[time]
        target = target_generator.generate(row.reward, continuation, row.terminated)
        if not torch.isfinite(target).all():
            raise FloatingPointError('Non-finite recursive target; head was not updated')

    states, actions = first.observation.long(), first.action.long()
    current = table[states, actions]
    tree._validate_update(current, target, update_rule.alpha)
    _validate_product(target-current, update_rule.alpha, update_rule.alpha*(target-current))
    error = update_rule.update(table, states, actions, target)
    empty = torch.empty(0, dtype=torch.float64, device=table.device)
    return {'target': target, 'td_error': error,
            'importance_ratios': torch.stack(ratios) if ratios else empty,
            'coefficients': torch.stack(coefficients) if coefficients else empty.clone()}


def _validate_sigma(sigma, size, device):
    if callable(sigma):
        raise TypeError('Decide sigma at action sampling time; pass frozen values to learn')
    values = torch.as_tensor(sigma, device=device)
    if values.dtype == torch.bool or values.is_complex():
        raise TypeError('sigma must be real, not bool or complex')
    values = values.double() if isinstance(sigma, Tensor) else torch.as_tensor(sigma, dtype=torch.float64, device=device)
    if values.ndim == 0:
        values = values.expand(size)
    if values.shape != (size,):
        raise ValueError('sigma must be scalar or m values [sigma_(t+1), ..., sigma_(t+m)]')
    if not torch.isfinite(values).all() or torch.any(values < 0) or torch.any(values > 1):
        raise ValueError('sigma must be finite and in [0,1]')
    return values


def _coefficient(pi, log_b, sigma):
    # pi=0 is a real zero branch, not exp(-inf)-related numeric underflow.
    if pi.item() == 0:
        rho = torch.zeros((), dtype=torch.float64, device=pi.device)
    else:
        log_ratio = pi.double().log() - log_b.detach().to(pi.device).double().reshape(())
        rho = log_ratio.exp()
        if not torch.isfinite(rho) or rho.item() == 0:
            raise FloatingPointError('Positive pi/b overflows or underflows; no write committed')
    coefficient = sigma*rho + (1-sigma)*pi.double()
    cast = coefficient.to(pi.dtype)
    if not torch.isfinite(cast) or (coefficient.item() > 0 and cast.item() == 0):
        raise FloatingPointError('Mixture coefficient is not representable in Q dtype')
    return rho, coefficient


def _validate_product(left, right, product):
    if not torch.isfinite(product).all() or torch.any((left != 0) & (right != 0) & (product == 0)):
        raise FloatingPointError('Recursive correction/update overflow or underflow; no write committed')


def run_episode(env, model, target_generator, update_rule, *, target_policy, sigma=.5, n=3, seed=None):
    """step -> snapshot -> eligible head update -> next action and frozen sigma.

    sigma is a scalar, repeating time pattern, or callback(state, action, timestep).
    A callback runs once per real selected action, before that action is executed.
    pending_sigmas aligns with pending observations/actions (including the head).
    learn receives its tail plus an ignored zero horizon placeholder. Flush makes
    no step/act/provider calls. No boundary bootstrap-only action is selected.
    Model/window/metadata/current action, policy/provider config, n/gamma/alpha,
    Torch RNG and environment/TimeLimit state are caller-owned checkpoint state.
    """
    sampled._validate_n(n)
    pattern = None
    if not callable(sigma):
        pattern = torch.as_tensor(sigma)
        if pattern.ndim > 1 or pattern.numel() == 0:
            raise ValueError('collection sigma must be a scalar or nonempty time pattern')
        pattern = _validate_sigma(sigma, pattern.numel(), 'cpu').tolist()

    def choose_sigma(state, selected, timestep):
        if pattern is not None:
            return pattern[timestep % len(pattern)]
        value = sigma(state, selected.action.item(), timestep)
        return _validate_sigma(value, 1, 'cpu')[0].item()

    observation, _ = env.reset(seed=seed)
    selected = model.act(observation)
    selected_sigma = choose_sigma(observation, selected, 0)
    pending, pending_sigmas = deque(), deque()
    rewards, steps, updates, coefficients, ratios, decided = 0., 0, 0, [], [], []

    def update_head():
        result = learn_q_sigma(model, pending, target_generator, update_rule,
                              target_policy=target_policy, sigma=[*list(pending_sigmas)[1:], 0.])
        coefficients.extend(result['coefficients'].cpu().tolist())
        ratios.extend(result['importance_ratios'].cpu().tolist())
        pending.popleft()
        pending_sigmas.popleft()

    while True:
        action = selected.action.item()
        log_b = None if selected.log_prob is None else selected.log_prob.detach().cpu().clone()
        next_observation, reward, terminated, truncated, _ = env.step(action)
        pending.append(Transition(observation, action, reward, next_observation, terminated, truncated, log_b).to_batch())
        pending_sigmas.append(selected_sigma)
        decided.append(selected_sigma)
        if len(pending) == n:
            update_head()
            updates += 1
        rewards += reward
        steps += 1
        if terminated or truncated:
            while pending:
                update_head()
                updates += 1
            break
        observation = next_observation
        selected = model.act(observation)
        selected_sigma = choose_sigma(observation, selected, steps)
    return {'reward': rewards, 'steps': steps, 'updates': updates, 'terminated': terminated,
            'truncated': truncated, 'sigmas': np.asarray(decided),
            'coefficients': np.asarray(coefficients), 'importance_ratios': np.asarray(ratios)}


def train_and_evaluate(target_mode='different', n=3, sigma=.5, episodes=200, seed=7,
                       gamma=.9, alpha=.05, max_steps=2000, log_every=50):
    """Uniform behavior, fixed target; report DP Qpi RMSE and per-branch statistics."""
    if target_mode not in ('matched', 'different') or episodes < 1:
        raise ValueError('target_mode must be matched/different and episodes positive')
    torch.manual_seed(seed)
    model = Model(IdentityRepresentation(), TabularQEstimator(len(DLFS4_STATES), 4),
                  EpsilonGreedyActionSelector(1.)).double()
    policy = torch.full_like(model.estimator.q_table, .25)
    if target_mode == 'different':
        policy[:] = policy.new_tensor([.3, .2, .2, .3])
    target, rule = TD0TargetGenerator(gamma), TabularQUpdateRule(alpha)
    env = make_dlfs4_gridworld_env(max_episode_steps=max_steps)
    rewards, lengths, steps, updates, terminals, truncations = [], [], 0, 0, 0, 0
    coefficients, ratios, sigmas = [], [], []
    try:
        reference = policy_evaluation(env.unwrapped.mdp, policy.numpy(), gamma=gamma, theta=1e-12)
        if not reference.converged:
            raise RuntimeError('DP reference did not converge')
        reference_q = action_values(env.unwrapped.mdp, reference.values, gamma)
        nonterminal = np.setdiff1d(np.arange(len(DLFS4_STATES)), env.unwrapped.mdp.terminal_states)

        def rmse():
            return float(np.sqrt(np.mean((model.estimator.q_table.numpy()[nonterminal]-reference_q[nonterminal])**2)))

        history = [{'episodes': 0, 'steps': 0, 'q_rmse': rmse()}]
        for number in range(1, episodes+1):
            result = run_episode(env, model, target, rule, target_policy=policy, sigma=sigma, n=n,
                                 seed=seed if number == 1 else None)
            rewards.append(result['reward'])
            lengths.append(result['steps'])
            steps += result['steps']
            updates += result['updates']
            terminals += int(result['terminated'])
            truncations += int(result['truncated'])
            coefficients.extend(result['coefficients'])
            ratios.extend(result['importance_ratios'])
            sigmas.extend(result['sigmas'])
            if number == episodes or number % max(1, log_every) == 0:
                history.append({'episodes': number, 'steps': steps, 'q_rmse': rmse()})
                if log_every:
                    print(f'Q(sigma) {target_mode} n={n} sigma={sigma} seed={seed} episode={number} '
                          f'steps={steps} reward={result["reward"]:.1f} length={result["steps"]} Qpi_RMSE={rmse():.6f}', flush=True)
    finally:
        env.close()
    return model, {'algorithm': 'q_sigma', 'target_mode': target_mode, 'n': n, 'sigma': sigma, 'seed': seed,
                   'episodes': episodes, 'steps': steps, 'updates': updates, 'gamma': gamma, 'alpha': alpha,
                   'target_policy': policy.numpy().copy(), 'q_values': model.estimator.q_table.numpy().copy(),
                   'dp_values': reference.values.copy(), 'dp_q_values': reference_q.copy(), 'history': history,
                   'q_rmse': history[-1]['q_rmse'], 'training_rewards': np.asarray(rewards),
                   'episode_lengths': np.asarray(lengths), 'terminated_episodes': terminals,
                   'truncated_episodes': truncations, 'coefficients': np.asarray(coefficients),
                   'importance_ratios': np.asarray(ratios), 'sampled_sigmas': np.asarray(sigmas)}


def run_comparison(seed=7, episodes=200, ns=(1,3,5), sigmas=(0.,.25,.5,.75,1.,(0.,1.)),
                   *, reference_results=None, **kwargs):
    """Reuse existing SARSA/Tree experiments, or already executed matching results.

    Passing Notebook's tree_results avoids repeating those runs. Their budget,
    seed, horizon, gamma/alpha/max_steps must match the caller's experiments;
    truncation counts are retained, not equated across different action timings.
    """
    results = tree.run_comparison(seed, episodes, ns, **kwargs) if reference_results is None else list(reference_results)
    for result in results:
        if (result['seed'] != seed or result['episodes'] != episodes or result['n'] not in ns
                or result['gamma'] != kwargs.get('gamma', .9) or result['alpha'] != kwargs.get('alpha', .05)):
            raise ValueError('reference_results must match seed/budget/horizon/gamma/alpha')
    for n in ns:
        for mode in ('matched','different'):
            for sigma in sigmas:
                _, result = train_and_evaluate(mode, n, sigma, episodes, seed, **kwargs)
                reference = next(r for r in results if r['algorithm']=='tree_backup' and r['n']==n and r['target_mode']==mode)
                for key in ('training_rewards','episode_lengths','dp_q_values'):
                    np.testing.assert_array_equal(result[key], reference[key])
                if isinstance(sigma, Real) and sigma == 0:
                    np.testing.assert_array_equal(result['q_values'], reference['q_values'])
                if n == 1:
                    np.testing.assert_array_equal(result['q_values'], reference['q_values'])
                results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7,11,19])
    parser.add_argument('--ns', type=int, nargs='+', default=[1,3,5])
    parser.add_argument('--sigmas', type=float, nargs='+', default=[0.,.25,.5,.75,1.])
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--gamma', type=float, default=.9)
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--log-every', type=int, default=50)
    args = parser.parse_args()
    results = []
    for seed in args.seeds:
        group = run_comparison(seed, args.episodes, args.ns, (*args.sigmas, (0.,1.)),
                               alpha=args.alpha, gamma=args.gamma, max_steps=args.max_steps, log_every=args.log_every)
        results.extend(group)
        for r in group:
            c, rho = r.get('coefficients', []), r.get('importance_ratios', [])
            print(f'algorithm={r["algorithm"]} target={r["target_mode"]} n={r["n"]} sigma={r.get("sigma","-")} '
                  f'seed={seed} episodes={r["episodes"]} steps={r["steps"]} updates={r["updates"]} '
                  f'reward={np.mean(r["training_rewards"]):.4f} length={np.mean(r["episode_lengths"]):.2f} '
                  f'terminated={r["terminated_episodes"]} truncated={r["truncated_episodes"]} '
                  f'Qpi_RMSE={r["q_rmse"]:.6f} c_mean={np.mean(c) if len(c) else 0:.4f} '
                  f'rho_mean={np.mean(rho) if len(rho) else 0:.4f} rho_max={np.max(rho) if len(rho) else 0:.4f}')
            print('Target pi:\n', r['target_policy'])
            print('Qpi estimate:\n', r['q_values'])
    for n in args.ns:
        for mode in ('matched','different'):
            for sigma in (*args.sigmas, (0.,1.)):
                errors = [r['q_rmse'] for r in results if r['algorithm']=='q_sigma' and r['n']==n
                          and r['target_mode']==mode and r['sigma']==sigma]
                print(f'n={n} target={mode} sigma={sigma} RMSE mean={np.mean(errors):.6f} std={np.std(errors):.6f}')


if __name__ == '__main__':
    main()
