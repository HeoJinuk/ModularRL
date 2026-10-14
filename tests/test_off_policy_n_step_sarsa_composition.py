"""Sampled endpoint ratios, numerical atomicity and online schedule contracts."""

from collections import deque
import copy
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from examples import gridworld_n_step_sarsa as on
from examples import gridworld_off_policy_n_step_sarsa as off
from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import TabularQEstimator, TabularVEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import ActionOutput, Transition, TransitionBatch
from modular_rl.environment import make_dlfs4_gridworld_env
from modular_rl.model import Model

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]


def components(dtype=torch.float64, device="cpu", gamma=.9, alpha=.5):
    model = Model(IdentityRepresentation(), TabularQEstimator(8, 2), EpsilonGreedyActionSelector(1.)).to(device=device, dtype=dtype)
    return model, TD0TargetGenerator(gamma), TabularQUpdateRule(alpha), torch.full_like(model.estimator.q_table, .5)


def pending(rewards, behavior=None, *, terminated=False, truncated=False, dtype=torch.float64):
    behavior = [.5] * len(rewards) if behavior is None else behavior
    return deque(Transition(i, 0, float(r), i+1, terminated and i == len(rewards)-1,
                            truncated and i == len(rewards)-1, torch.tensor(b, dtype=dtype).log()).to_batch()
                 for i, (r, b) in enumerate(zip(rewards, behavior)))


def learn(parts, window, *, b=.5, action=0):
    model, target, rule, policy = parts
    return off.learn_off_policy_n_step_sarsa(model, window, target, rule, target_policy=policy,
                                            bootstrap_action=action, bootstrap_log_prob=torch.tensor(b, dtype=policy.dtype).log())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("case,expected", [("A", (2.8, 2., -1.2, 2.8)), ("B", (8.146, 3., 6.146, 11.219)),
                                         ("C", (2.8, 2., .8, 2.8)), ("D", (6.04, 6., 4.04, 14.12))])
def test_required_numeric_examples(case, expected, dtype, device, monkeypatch):
    parts = components(dtype, device)
    model, target, rule, policy = parts
    table = model.estimator.q_table
    table[0, 0] = 4 if case == "A" else 2
    if case == "A":
        table[1, 0] = 2
        rows, b = pending([1], [.01], dtype=dtype), .25
    elif case == "B":
        table[3, 0] = 4
        policy[2] = policy.new_tensor([.4, .6])
        policy[3] = policy.new_tensor([.6, .4])
        rows, b = pending([1, 2, 3], [.01, .25, .8], dtype=dtype), .2
    else:
        table[2, 0] = 4
        rows, b = pending([1, 2], [.01, .25], terminated=case == "C", truncated=case == "D", dtype=dtype), .2
        policy[2] = policy.new_tensor([.6, .4])
    before, pointer, rng, calls, original = table.clone(), table.data_ptr(), torch.get_rng_state().clone(), [], rule.update

    def update(q, states, actions, target):
        assert torch.equal(q, before) and not torch.is_grad_enabled()
        calls.append(target.clone())
        return original(q, states, actions, target)

    monkeypatch.setattr(rule, "update", update)
    monkeypatch.setattr(model, "act", lambda *a, **k: pytest.fail("learning must not sample"))
    if case == "C":
        monkeypatch.setattr(model, "forward", lambda *a: pytest.fail("terminal state must not bootstrap"))
    result = learn(parts, rows, b=b)
    g, rho, raw, q = expected
    assert result["target"].item() == pytest.approx(g, rel=2e-6)
    assert result["importance_ratio"].item() == pytest.approx(rho, rel=2e-6)
    assert result["raw_td_error"].item() == pytest.approx(raw, rel=2e-6)
    assert result["td_error"].item() == pytest.approx(rho*raw, rel=2e-6)
    assert table[0, 0].item() == pytest.approx(q, rel=2e-6)
    assert len(calls) == 1 and table.data_ptr() == pointer and torch.equal(rng, torch.get_rng_state())
    for key, value in result.items():
        assert value.shape == (1,) and not value.requires_grad and value.grad_fn is None
        assert value.dtype == (torch.float64 if key == "importance_ratio" else dtype)
        assert value.device == table.device


@pytest.mark.parametrize("gamma", [0., .9, 1.])
def test_td0_components_fold_in_reverse_order_with_no_new_rule_interface(gamma, monkeypatch):
    parts = components(gamma=gamma)
    model, target, rule, policy = parts
    model.estimator.q_table[3, 0] = 4
    calls, generate = [], target.generate

    def fold(reward, continuation, terminated):
        calls.append(reward.item())
        return generate(reward, continuation, terminated)

    monkeypatch.setattr(target, "generate", fold)
    result = learn(parts, pending([1, 2, 3]))
    assert result["target"].item() == pytest.approx(1+gamma*2+gamma**2*3+gamma**3*4)
    assert calls == [3., 2., 1.]


@pytest.mark.parametrize("flags", [(True, False), (True, True)])
def test_terminal_empty_product_is_one_and_no_bootstrap_arguments_required(flags):
    parts = components()
    model, target, rule, policy = parts
    result = off.learn_off_policy_n_step_sarsa(model, pending([4], terminated=flags[0], truncated=flags[1]),
                                              target, rule, target_policy=policy)
    assert result["importance_ratio"].item() == 1 and model.estimator.q_table[0, 0] == 2


def test_zero_target_probability_skips_write_but_window_can_advance():
    parts = components()
    model, target, rule, policy = parts
    model.estimator.q_table[0, 0] = 4
    policy[1] = policy.new_tensor([0., 1.])
    rows = pending([1, 2], terminated=True)
    before = model.estimator.q_table.clone()
    result = learn(parts, rows)
    assert result["importance_ratio"].item() == result["td_error"].item() == 0
    assert torch.equal(before, model.estimator.q_table)
    rows.popleft()
    assert learn(parts, rows)["importance_ratio"].item() == 1
    assert model.estimator.q_table[1, 0] == 1


@pytest.mark.parametrize("log_b", [None, float("inf"), float("nan"), float("-inf"), .01])
def test_invalid_behavior_support_fails_before_write(log_b):
    parts = components()
    rows = pending([1])
    rows[0] = replace(rows[0], log_prob=None if log_b is None else torch.tensor([log_b], dtype=torch.float64))
    before = parts[0].estimator.q_table.clone()
    with pytest.raises(ValueError, match="log_prob"):
        learn(parts, rows)
    assert torch.equal(before, parts[0].estimator.q_table)


@pytest.mark.parametrize("failure", ["rho_overflow", "rho_underflow", "cast_overflow", "cast_underflow", "target", "effective", "error", "step"])
def test_numerical_failure_is_atomic_for_current_window(failure):
    parts = components(torch.float32)
    model, target, rule, policy = parts
    rows = pending([1], dtype=torch.float64)
    log_b = torch.tensor(-1000. if failure == "rho_overflow" else -100. if failure == "cast_overflow" else -.6931471805599453,
                         dtype=torch.float64)
    if failure in ("rho_underflow", "cast_underflow"):
        policy = torch.full((8, 2), .5, dtype=torch.float64)
        policy[1] = torch.tensor([1e-320 if failure == "rho_underflow" else 1e-100, 1.], dtype=torch.float64)
        rows = pending([1, 2] if failure == "rho_underflow" else [1], dtype=torch.float64)
        if len(rows) == 2:
            policy[2] = policy[1]
    if failure == "target":
        rows = pending([float("inf")])
    if failure == "effective":
        rows = pending([2e38])
        log_b = torch.tensor(-2., dtype=torch.float64)
    if failure == "error":
        model.estimator.q_table[0, 0] = -3e38
        rows = pending([3e38])
    if failure == "step":
        model.estimator.q_table[0, 0] = 3e38
        rows = pending([3e38])
        log_b = torch.tensor(-2., dtype=torch.float64)
        model.estimator.q_table[1, 0] = 1e38
    before = model.estimator.q_table.clone()
    with pytest.raises(FloatingPointError):
        off.learn_off_policy_n_step_sarsa(model, rows, target, rule, target_policy=policy,
                                        bootstrap_action=0, bootstrap_log_prob=log_b)
    assert torch.equal(before, model.estimator.q_table)


def test_sampling_time_logs_survive_changed_q_epsilon_and_no_probability_query(monkeypatch):
    parts = components()
    model, target, rule, policy = parts
    model.action_selector.epsilon = .2
    model.estimator.q_table[1] = torch.tensor([10., -10.])
    torch.manual_seed(7)
    selected = model.act(1)
    rows = pending([1, 2])
    rows[1] = replace(rows[1], action=selected.action.reshape(1), log_prob=selected.log_prob.reshape(1).clone())
    recorded = rows[1].log_prob.clone()
    model.estimator.q_table[1] = torch.tensor([-10., 10.])
    model.action_selector.epsilon = 1
    monkeypatch.setattr(model.action_selector, "probabilities", lambda *a: pytest.fail("Past b must not be recomputed"))
    result = learn(parts, rows)
    assert result["importance_ratio"].item() == pytest.approx(.5/recorded.exp().item())
    assert torch.equal(rows[1].log_prob, recorded)


def test_flush_repeated_state_uses_current_q_per_call_but_frozen_target_within_call(monkeypatch):
    parts = components()
    model, target, rule, policy = parts
    model.estimator.q_table[0, 0] = 4
    rows = deque(Transition(0, 0, r, 0, False, i == 1, torch.tensor(.5, dtype=torch.float64).log()).to_batch()
                 for i, r in enumerate([1., 2.]))
    bootstraps, original = [], model.forward

    def forward(obs):
        result = original(obs)
        bootstraps.append(result.values.q[0][0, 0].item())
        return result

    monkeypatch.setattr(model, "forward", forward)
    first = learn(parts, rows)
    rows.popleft()
    second = learn(parts, rows)
    assert bootstraps == pytest.approx([4., 5.02])
    assert first["target"].item() == pytest.approx(6.04)
    assert second["target"].item() == pytest.approx(6.518)


class ScriptedEnv:
    def __init__(self, length, flags):
        self.length, self.flags, self.trace = length, flags, []

    def reset(self, *, seed=None):
        self.time = 0
        return 0, {}

    def step(self, action):
        self.trace.append((self.time, action))
        self.time += 1
        terminated, truncated = self.flags if self.time == self.length else (False, False)
        return self.time % 3, float(self.time), terminated, truncated, {}


@pytest.mark.parametrize("n", [1, 3, 5])
@pytest.mark.parametrize("length", [1, 3, 5, 7])
@pytest.mark.parametrize("flags", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bit_exact_on_policy_equivalence_including_timing_actions_targets_errors_rng(n, length, flags, dtype, monkeypatch):
    traces = []
    for module in (on, off):
        model, target, rule, policy = components(dtype)
        env, actions, writes = ScriptedEnv(length, flags), [], []
        act, update = model.act, rule.update

        def select(state, **kwargs):
            selected = act(state, **kwargs)
            actions.append((env.time, state, selected.action.item(), selected.log_prob.item()))
            return selected

        def write(table, states, chosen, value):
            error = update(table, states, chosen, value)
            writes.append((env.time, states.item(), chosen.item(), value.clone(), error.clone()))
            return error

        monkeypatch.setattr(model, "act", select)
        monkeypatch.setattr(rule, "update", write)
        torch.manual_seed(19)
        kwargs = {"target_policy": policy} if module is off else {}
        result = module.run_episode(env, model, target, rule, n=n, **kwargs)
        traces.append((actions, writes, model.estimator.q_table.clone(), torch.get_rng_state().clone(), result))
        assert result["steps"] == result["updates"] == length
        assert len(actions) == length + int(not flags[0])  # Truncation endpoint sampled once, never executed.
        expected_times = [*range(n, length+1), *([length] * (n-1))] if length >= n else [length]*length
        assert [w[0] for w in writes] == expected_times
    assert traces[0][0] == traces[1][0]
    for left, right in zip(traces[0][1], traces[1][1]):
        assert left[:3] == right[:3] and torch.equal(left[3], right[3]) and torch.equal(left[4], right[4])
    assert torch.equal(traces[0][2], traces[1][2]) and torch.equal(traces[0][3], traces[1][3])
    np.testing.assert_array_equal(traces[1][4]["importance_ratios"], np.ones(length))


def test_checkpoint_restores_model_window_endpoint_policy_and_rng(tmp_path):
    parts = components()
    model, target, rule, policy = parts
    policy[:] = policy.new_tensor([.8, .2])
    model.estimator.q_table.copy_(torch.arange(16).reshape(8, 2))
    rows = pending([1, 2], [.3, .7])
    torch.manual_seed(11)
    endpoint = model.act(2)
    checkpoint = {"model": copy.deepcopy(model.state_dict()), "window": [asdict(b) for b in rows],
                  "action": endpoint.action, "log_prob": endpoint.log_prob, "policy": policy.clone(),
                  "rng": torch.get_rng_state(), "gamma": target.gamma, "alpha": rule.alpha}
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    original = learn(parts, rows, action=endpoint.action)
    assert original["importance_ratio"].item() != 1
    next_action = model.act(3)
    state = torch.load(path, weights_only=True)
    restored = components(gamma=state["gamma"], alpha=state["alpha"])
    restored[0].load_state_dict(state["model"])
    restored_rows = deque(TransitionBatch(**b) for b in state["window"])
    result = off.learn_off_policy_n_step_sarsa(restored[0], restored_rows, restored[1], restored[2],
                                             target_policy=state["policy"], bootstrap_action=state["action"],
                                             bootstrap_log_prob=state["log_prob"])
    for key in original:
        assert torch.equal(result[key], original[key])
    assert torch.equal(model.estimator.q_table, restored[0].estimator.q_table)
    torch.set_rng_state(state["rng"])
    resumed_action = restored[0].act(3)
    assert torch.equal(next_action.action, resumed_action.action) and torch.equal(next_action.log_prob, resumed_action.log_prob)


@pytest.mark.parametrize("problem", ["model", "policy_shape", "policy_sum", "policy_negative", "policy_nan", "empty", "boundary", "next_action", "next_log", "index", "log_shape", "reward_dtype"])
def test_important_contract_errors_leave_table_unchanged(problem):
    model, target, rule, policy = components()
    rows, kwargs = pending([1]), {"bootstrap_action": 0, "bootstrap_log_prob": torch.tensor(.5).log()}
    if problem == "model":
        model = Model(IdentityRepresentation(), TabularVEstimator(8))
    elif problem == "policy_shape":
        policy = policy[:1]
    elif problem == "policy_sum":
        policy[0] = .1
    elif problem == "policy_negative":
        policy[0] = torch.tensor([-.1, 1.1])
    elif problem == "policy_nan":
        policy[0, 0] = float("nan")
    elif problem == "empty":
        rows.clear()
    elif problem == "boundary":
        rows = pending([1, 2])
        rows[0] = replace(rows[0], truncated=torch.tensor([True]))
    elif problem == "next_action":
        kwargs["bootstrap_action"] = None
    elif problem == "next_log":
        kwargs["bootstrap_log_prob"] = None
    elif problem == "index":
        rows[0] = replace(rows[0], action=torch.tensor([-1]))
    elif problem == "log_shape":
        rows[0] = replace(rows[0], log_prob=torch.zeros(1, 1))
    elif problem == "reward_dtype":
        rows[0] = replace(rows[0], reward=torch.tensor([1]))
    with pytest.raises((TypeError, ValueError, IndexError)):
        off.learn_off_policy_n_step_sarsa(model, rows, target, rule, target_policy=policy, **kwargs)


def test_saved_logs_keep_float64_precision_when_q_is_float32():
    parts = components(torch.float32)
    parts = (*parts[:3], parts[3].double())
    rows = pending([1, 2], [.5, .123456789123], dtype=torch.float64)
    result = learn(parts, rows)
    assert result["importance_ratio"].item() == pytest.approx(.5/.123456789123, rel=1e-14)
