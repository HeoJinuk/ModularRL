"""Returns, online timing, actual actions, tail bootstrap and checkpoint state."""

from collections import deque
from dataclasses import asdict, replace

import gymnasium as gym
import numpy as np
import pytest
import torch

from examples import gridworld_n_step_sarsa as example
from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import TabularQEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import TabularMDP, Transition, TransitionBatch, TransitionOutcome as Outcome
from modular_rl.environment import TabularEnv
from modular_rl.learner import TabularSARSALearner
from modular_rl.model import Model


def components(dtype=torch.float64, gamma=.9, alpha=1, states=8):
    model = Model(IdentityRepresentation(), TabularQEstimator(states, 2), EpsilonGreedyActionSelector(.2)).to(dtype=dtype)
    return model, TD0TargetGenerator(gamma), TabularQUpdateRule(alpha)


def window(rewards, *, terminated=False, truncated=False):
    return deque(Transition(i, i % 2, float(reward), i+1,
                            terminated and i == len(rewards)-1, truncated and i == len(rewards)-1).to_batch()
                 for i, reward in enumerate(rewards))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("gamma,expected", [(0., 1.), (.9, 8.146), (1., 10.)])
def test_three_step_target_reuses_td0_in_reverse_order_with_actual_action(dtype, gamma, expected, monkeypatch):
    model, target, rule = components(dtype, gamma)
    model.estimator.q_table[3] = torch.tensor([100., 4.])
    pending = window([1, 2, 3])
    generate, calls, rng, pointer = target.generate, [], torch.get_rng_state().clone(), model.estimator.q_table.data_ptr()

    def backup(reward, continuation, terminated):
        assert not torch.is_grad_enabled() and reward.dtype == continuation.dtype == dtype
        calls.append(reward.item())
        return generate(reward, continuation, terminated)

    monkeypatch.setattr(target, "generate", backup)
    monkeypatch.setattr(model, "act", lambda *args, **kwargs: pytest.fail("Target computation must not sample an action"))
    error = example.learn_n_step_sarsa(model, pending, target, rule, next_action=1)
    assert error.item() == pytest.approx(expected) and model.estimator.q_table[0, 0].item() == pytest.approx(expected)
    assert calls == [3., 2., 1.] and len(pending) == 3 and error.shape == (1,)
    assert not error.requires_grad and error.grad_fn is None and error.dtype == dtype
    assert model.estimator.q_table.data_ptr() == pointer and torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("terminated,truncated,expected", [
    (True, False, [2.8, 2.]), (True, True, [2.8, 2.]), (False, True, [6.04, 5.6]),
])
def test_short_episode_flush_uses_actual_horizon_and_termination_dominates(terminated, truncated, expected, monkeypatch):
    model, target, rule = components()
    model.estimator.q_table[2] = torch.tensor([100., 4.])
    pending, targets, update = window([1, 2], terminated=terminated, truncated=truncated), [], rule.update

    def write(table, states, actions, value):
        targets.append(value.item())
        assert states.item() == len(targets)-1
        return update(table, states, actions, value)

    monkeypatch.setattr(rule, "update", write)
    if terminated:
        monkeypatch.setattr(model, "forward", lambda *args: pytest.fail("No terminal bootstrap evaluation"))
    while pending:
        example.learn_n_step_sarsa(model, pending, target, rule, next_action=None if terminated else 1)
        pending.popleft()
    assert targets == pytest.approx(expected)


def test_truncation_tails_re_evaluate_current_q_after_each_write_without_resampling(monkeypatch):
    model, target, rule = components(alpha=.5)
    model.estimator.q_table[0, 0] = 4.
    pending = deque(Transition(0, 0, r, 0, False, i == 1).to_batch() for i, r in enumerate((1., 2.)))
    targets, bootstraps, forward, update = [], [], model.forward, rule.update

    def estimate(observation):
        result = forward(observation)
        bootstraps.append(result.values.q[0][0, 0].item())
        return result

    def write(table, states, actions, value):
        targets.append(value.item())
        return update(table, states, actions, value)

    monkeypatch.setattr(model, "forward", estimate)
    monkeypatch.setattr(model, "act", lambda *args, **kwargs: pytest.fail("Final selected action must be retained"))
    monkeypatch.setattr(rule, "update", write)
    errors = []
    while pending:
        errors.append(example.learn_n_step_sarsa(model, pending, target, rule, next_action=0).item())
        pending.popleft()
    assert bootstraps == pytest.approx([4., 5.02])
    assert targets == pytest.approx([6.04, 6.518])
    assert errors == pytest.approx([2.04, 1.498]) and model.estimator.q_table[0, 0].item() == pytest.approx(5.769)


def chain(length, boundary):
    terminal = boundary != "truncated"
    rows = []
    for state in range(length+1):
        outcome = Outcome(1., min(state+1, length), float(state+1) if state < length else 0., terminal and state >= length-1)
        rows.append(((outcome,), (outcome,)))
    mdp = TabularMDP(length+1, 2, tuple(rows), (length,) if terminal else ())
    cap = length+1 if boundary == "terminated" else length
    return gym.wrappers.TimeLimit(TabularEnv(mdp, start_state=0), max_episode_steps=cap)


@pytest.mark.parametrize("n", [1, 3, 5])
@pytest.mark.parametrize("length", [2, 3, 5, 7])
@pytest.mark.parametrize("boundary", ["terminated", "truncated", "both"])
def test_online_update_timing_action_reuse_and_every_transition_updated_once(n, length, boundary, monkeypatch):
    env = chain(length, boundary)
    model, target, rule = components(states=length+1)
    model.estimator.q_table[length] = torch.tensor([4., 7.])
    events, selections, executed, updates = [], [], [], []
    act, step, write = model.act, env.step, rule.update

    def select(observation, *, deterministic=False):
        output = act(observation, deterministic=deterministic)
        selections.append(output.action.item())
        events.append(("act", observation))
        return output

    def advance(action):
        executed.append(action)
        events.append(("step", len(executed)-1))
        return step(action)

    def update(table, states, actions, value):
        tau = states.item()
        updates.append((len(executed)-1, tau, actions.item()))
        events.append(("update", tau))
        return write(table, states, actions, value)

    monkeypatch.setattr(model, "act", select)
    monkeypatch.setattr(env, "step", advance)
    monkeypatch.setattr(rule, "update", update)
    torch.manual_seed(11)
    try:
        result = example.run_episode(env, model, target, rule, n=n, seed=11)
    finally:
        env.close()
    assert result["steps"] == result["updates"] == length
    assert [tau for _, tau, _ in updates] == list(range(length))
    assert [(t, tau) for t, tau, _ in updates] == [(min(tau+n-1, length-1), tau) for tau in range(length)]
    assert executed == selections[:length] and [action for _, _, action in updates] == executed
    assert len(selections) == length + int(boundary == "truncated")
    for tau in range(max(0, length-n+1)):
        time = tau+n-1
        if time+1 < length:
            assert events.index(("act", time+1)) < events.index(("update", tau)) < events.index(("step", time+1))
    last_step = events.index(("step", length-1))
    assert all(kind != "step" for kind, _ in events[last_step+1:])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("cap", [1, 2, 8])
def test_one_step_matches_existing_sarsa_full_trace_rng_targets_errors_and_table(dtype, cap):
    mdp = TabularMDP(3, 2, (((Outcome(.3, 0, 1.), Outcome(.7, 1, 2.)), (Outcome(1., 1, 0.),)),
                           ((Outcome(1., 2, 3., True),), (Outcome(1., 2, 4., True),)),
                           ((Outcome(1., 2, 0., True),), (Outcome(1., 2, 0., True),))), (2,))

    def run(existing):
        model, target, rule = components(dtype, alpha=.5, states=3)
        model.estimator.q_table.copy_(torch.tensor([[4., 2.], [10., 2.], [0., 0.]], dtype=dtype))
        env = gym.wrappers.TimeLimit(TabularEnv(mdp, start_state=0), max_episode_steps=cap)
        actions, transitions, targets, errors = [], [], [], []
        act, step, update = model.act, env.step, rule.update

        def select(state, *, deterministic=False):
            result = act(state, deterministic=deterministic)
            actions.append((state, result.action.item(), result.log_prob.item()))
            return result

        def advance(action):
            result = step(action)
            transitions.append((action, result))
            return result

        def write(table, states, chosen, value):
            targets.append(value.clone())
            result = update(table, states, chosen, value)
            errors.append(result.clone())
            return result

        model.act, env.step, rule.update = select, advance, write
        torch.manual_seed(19)
        try:
            if existing:
                learner = TabularSARSALearner(model, target, rule)
                observation, _ = env.reset(seed=19)
                selected = model.act(observation)
                while True:
                    action = selected.action.item()
                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    next_selected = None if terminated else model.act(next_observation)
                    learner.learn(Transition(observation, action, reward, next_observation, terminated, truncated,
                                             selected.log_prob).to_batch(), next_action=None if next_selected is None else next_selected.action.item())
                    if terminated or truncated:
                        break
                    observation, selected = next_observation, next_selected
            else:
                example.run_episode(env, model, target, rule, n=1, seed=19)
        finally:
            env.close()
        return actions, transitions, torch.cat(targets), torch.cat(errors), model.estimator.q_table, torch.get_rng_state()

    first, reference = run(False), run(True)
    assert first[:2] == reference[:2]
    for left, right in zip(first[2:], reference[2:]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_mid_episode_checkpoint_restores_pending_window_selected_action_and_rng(tmp_path):
    model, target, rule = components(alpha=.5, states=2)
    torch.manual_seed(23)
    selected, pending = model.act(0), deque()

    def advance(model, pending, selected, rewards, *, ending):
        errors, actions = [], []
        for index, reward in enumerate(rewards):
            terminated = ending and index == len(rewards)-1
            next_selected = None if terminated else model.act(0)
            pending.append(Transition(0, selected.action, reward, 0, terminated, False, selected.log_prob).to_batch())
            actions.append(selected.action.item())
            if len(pending) == 3:
                errors.append(example.learn_n_step_sarsa(model, pending, target, rule,
                                                         next_action=None if next_selected is None else next_selected.action).item())
                pending.popleft()
            if terminated:
                while pending:
                    errors.append(example.learn_n_step_sarsa(model, pending, target, rule).item())
                    pending.popleft()
            selected = next_selected
        return selected, errors, actions

    selected, _, _ = advance(model, pending, selected, [1., 2., 3., 4.], ending=False)
    path = tmp_path / "n-step.pt"
    torch.save({"model": model.state_dict(), "pending": [asdict(batch) for batch in pending],
                "selected_action": selected.action, "selected_log_prob": selected.log_prob,
                "rng": torch.get_rng_state(), "n": 3, "gamma": target.gamma, "alpha": rule.alpha}, path)
    _, expected_errors, expected_actions = advance(model, pending, selected, [5., 6., 7.], ending=True)
    saved = torch.load(path, weights_only=True)
    restored, restored_target, restored_rule = components(alpha=saved["alpha"], gamma=saved["gamma"], states=2)
    restored.load_state_dict(saved["model"])
    restored_pending = deque(TransitionBatch(**values) for values in saved["pending"])
    from modular_rl.data import ActionOutput
    restored_selected = ActionOutput(saved["selected_action"], saved["selected_log_prob"])
    torch.set_rng_state(saved["rng"])
    # Use the restored component settings for the continuation.
    target, rule = restored_target, restored_rule
    _, actual_errors, actual_actions = advance(restored, restored_pending, restored_selected, [5., 6., 7.], ending=True)
    assert actual_errors == expected_errors and actual_actions == expected_actions
    torch.testing.assert_close(restored.estimator.q_table, model.estimator.q_table, rtol=0, atol=0)
    assert not restored_pending


@pytest.mark.parametrize("n", [0, -1, True, 1.5])
def test_invalid_horizon_fails_before_environment_reset(n):
    model, target, rule = components()
    with pytest.raises(ValueError, match="positive integer"):
        example.run_episode(None, model, target, rule, n=n)


def test_window_contract_and_actual_bootstrap_action_are_explicit():
    model, target, rule = components()
    with pytest.raises(ValueError, match="at least one"):
        example.learn_n_step_sarsa(model, deque(), target, rule)
    for action, error in ((None, ValueError), (1.2, TypeError), (True, TypeError), ([0, 1], ValueError), (2, IndexError)):
        with pytest.raises(error):
            example.learn_n_step_sarsa(model, window([1]), target, rule, next_action=action)
    for batch, error in ((replace(window([1])[0], observation=torch.tensor([0.])), TypeError),
                         (replace(window([1])[0], reward=torch.tensor([1])), TypeError),
                         (replace(window([1])[0], terminated=torch.tensor([0])), TypeError),
                         (replace(window([1])[0], reward=torch.ones(2)), ValueError)):
        with pytest.raises(error):
            example.learn_n_step_sarsa(model, [batch], target, rule, next_action=0)
    with pytest.raises(ValueError, match="boundary"):
        example.learn_n_step_sarsa(model, [Transition(0, 0, 1., 1, False, True).to_batch(),
                                           Transition(1, 0, 2., 2, True, False).to_batch()], target, rule)
    with pytest.raises(ValueError, match="contiguous"):
        example.learn_n_step_sarsa(model, [Transition(0, 0, 1., 1, False, False).to_batch(),
                                           Transition(2, 0, 2., 3, True, False).to_batch()], target, rule)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_table_with_cpu_window_preserves_dtype_device_and_rng(dtype):
    model, target, rule = components(dtype)
    model.cuda()
    model.estimator.q_table[3, 1] = 4.
    cpu, cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    error = example.learn_n_step_sarsa(model, window([1, 2, 3]), target, rule, next_action=1)
    assert error.item() == pytest.approx(8.146) and error.dtype == dtype and error.device.type == "cuda"
    assert torch.equal(cpu, torch.get_rng_state()) and torch.equal(cuda, torch.cuda.get_rng_state())
