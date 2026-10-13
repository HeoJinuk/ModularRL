"""Cross-bootstrap arithmetic, explicit sum policy, independence and restart."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from examples import gridworld_double_q_learning as example
from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import DiscreteQEstimator, TabularQEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import Transition, TransitionBatch
from modular_rl.model import Model


def components(dtype=torch.float64, n_actions=2, gamma=.9, alpha=.5):
    models = tuple(Model(IdentityRepresentation(), TabularQEstimator(4, n_actions)).to(dtype=dtype) for _ in range(2))
    if n_actions == 2:
        models[0].estimator.q_table[1, 0] = 4
        models[1].estimator.q_table[1, 0] = 6
        models[0].estimator.q_table[2] = torch.tensor([10., 2.])
        models[1].estimator.q_table[2] = torch.tensor([1., 7.])
    return models, TD0TargetGenerator(gamma), TabularQUpdateRule(alpha)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("first,bootstrap,error,value", [(True, 1., -2.1, 2.95), (False, 2., -3.2, 4.4)])
def test_exact_cross_bootstrap_updates_only_the_selected_table_in_place(first, bootstrap, error, value, dtype, monkeypatch):
    models, target, rule = components(dtype)
    tables = [model.estimator.q_table for model in models]
    before, pointers = [table.clone() for table in tables], [table.data_ptr() for table in tables]
    assert tables[0].untyped_storage().data_ptr() != tables[1].untyped_storage().data_ptr()
    selected, other = (0, 1) if first else (1, 0)
    calls, generate, update = [], target.generate, rule.update

    def td(reward, next_value, terminated):
        assert not torch.is_grad_enabled()
        assert next_value.item() == bootstrap
        calls.append("target")
        for old, table in zip(before, tables):
            torch.testing.assert_close(old, table, rtol=0, atol=0)
        return generate(reward, next_value, terminated)

    def write(table, states, actions, targets):
        assert table is tables[selected]
        assert states.dtype == actions.dtype == torch.long and targets.dtype == dtype
        calls.append("update")
        return update(table, states, actions, targets)

    monkeypatch.setattr(target, "generate", td)
    monkeypatch.setattr(rule, "update", write)
    rng = torch.get_rng_state().clone()
    result = example.learn_double_q(*models, Transition(np.int32(1), np.int32(0), 1., np.int32(2), False, False).to_batch(),
                                    target, rule, update_first=first)
    assert result.shape == (1,) and result.dtype == dtype and result.item() == pytest.approx(error)
    assert tables[selected][1, 0].item() == pytest.approx(value)
    torch.testing.assert_close(tables[other], before[other], rtol=0, atol=0)
    assert [table.data_ptr() for table in tables] == pointers
    assert not result.requires_grad and result.grad_fn is None and calls == ["target", "update"]
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("first", [True, False])
@pytest.mark.parametrize("terminated,truncated", [(True, False), (True, True), (False, True)])
def test_termination_and_truncation_cross_bootstrap(first, terminated, truncated):
    models, target, rule = components()
    current, bootstrap = (4., 1.) if first else (6., 2.)
    expected = 1. + (0. if terminated else .9 * bootstrap) - current
    error = example.learn_double_q(*models, Transition(1, 0, 1., 2, terminated, truncated).to_batch(),
                                   target, rule, update_first=first)
    assert error.item() == pytest.approx(expected)


@pytest.mark.parametrize("first", [True, False])
def test_batch_uses_one_update_table_and_freezes_action_selection_before_writes(first):
    models, target, rule = components(alpha=1)
    selected, other = models if first else models[::-1]
    unchanged = other.estimator.q_table.clone()
    terminal_action = 0 if first else 1
    batch = TransitionBatch(torch.tensor([2, 1]), torch.tensor([terminal_action, 0]), torch.tensor([0., 1.]),
                            torch.tensor([3, 2]), torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool))
    result = example.learn_double_q(*models, batch, target, rule, update_first=first)
    expected = [-10., -2.1] if first else [-7., -3.2]
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))
    torch.testing.assert_close(other.estimator.q_table, unchanged, rtol=0, atol=0)
    # A sequential B=1 call observes the changed argmax on the next transition.
    online, online_target, online_rule = components(alpha=1)
    example.learn_double_q(*online, Transition(2, terminal_action, 0., 3, True, False).to_batch(),
                           online_target, online_rule, update_first=first)
    example.learn_double_q(*online, Transition(1, 0, 1., 2, False, False).to_batch(),
                           online_target, online_rule, update_first=first)
    assert online[0 if first else 1].estimator.q_table[1, 0].item() == pytest.approx(7.3 if first else 10.)


def test_duplicate_pairs_keep_existing_input_order_updates_with_fixed_cross_targets():
    models, target, rule = components(gamma=1)
    models[0].estimator.q_table[1] = torch.tensor([4., 0.])
    models[1].estimator.q_table[1] = torch.tensor([6., 0.])
    batch = TransitionBatch(torch.tensor([1, 1]), torch.tensor([0, 0]), torch.tensor([1., 3.]),
                            torch.tensor([1, 1]), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
    before = models[1].estimator.q_table.clone()
    result = example.learn_double_q(*models, batch, target, rule, update_first=True)
    torch.testing.assert_close(result, torch.tensor([3., 3.5], dtype=torch.float64))
    assert models[0].estimator.q_table[1, 0] == 7.25
    torch.testing.assert_close(models[1].estimator.q_table, before, rtol=0, atol=0)


def test_cross_argmax_ties_use_first_index_rather_than_behavior_sum_or_other_max():
    models, target, rule = components(gamma=1, alpha=1)
    models[0].estimator.q_table[2] = torch.tensor([5., 5.])
    selector = EpsilonGreedyActionSelector(0)
    assert example.act_double_q(*models, 2, selector, deterministic=True).action.item() == 1
    error = example.learn_double_q(*models, Transition(0, 0, 0., 2, False, False).to_batch(), target, rule, update_first=True)
    assert error.item() == 1 and models[0].estimator.q_table[0, 0] == 1


def test_sum_policy_can_prefer_an_action_neither_estimator_prefers_and_reuses_selector(monkeypatch):
    models, _, _ = components(n_actions=3)
    models[0].estimator.q_table[2] = torch.tensor([6., 0., 5.])
    models[1].estimator.q_table[2] = torch.tensor([0., 6., 5.])
    selector = EpsilonGreedyActionSelector(.2).double()
    summed = example.combined_estimate(*models, 2)
    assert len(summed.values.q) == 1
    torch.testing.assert_close(summed.values.q[0], torch.tensor([6., 6., 10.], dtype=torch.float64))
    rng, before = torch.get_rng_state().clone(), [m.estimator.q_table.clone() for m in models]
    probabilities = selector.probabilities(summed)
    torch.testing.assert_close(probabilities, torch.tensor([.2/3, .2/3, .8+.2/3], dtype=torch.float64))
    assert torch.equal(rng, torch.get_rng_state())
    select, estimates = selector.select, []

    def sample(estimate, *, deterministic=False):
        estimates.append(estimate)
        return select(estimate, deterministic=deterministic)

    monkeypatch.setattr(selector, "select", sample)
    monkeypatch.setattr(Model, "act", lambda *args, **kwargs: pytest.fail("The public selector must receive the explicit Q sum"))
    selected = example.act_double_q(*models, np.int32(2), selector, deterministic=True)
    assert selected.action.item() == 2 and selected.log_prob is None and len(estimates) == 1
    for table, old in zip((m.estimator.q_table for m in models), before):
        torch.testing.assert_close(table, old, rtol=0, atol=0)


@pytest.mark.parametrize("epsilon", [0., .1, 1.])
def test_initial_ties_are_uniform_and_deterministic_selection_uses_first(epsilon):
    models, _, _ = components(n_actions=3)
    selector = EpsilonGreedyActionSelector(epsilon).double()
    estimate = example.combined_estimate(*models, torch.tensor([0, 1], dtype=torch.int32))
    torch.testing.assert_close(selector.probabilities(estimate), torch.full((2, 3), 1/3, dtype=torch.float64))
    action = example.act_double_q(*models, 0, selector, deterministic=True)
    assert action.action.shape == () and action.action.dtype == torch.long and action.action.item() == 0
    assert action.log_prob is None


def test_sample_distribution_log_probability_and_rng_reproduction_of_batched_actions():
    models, _, _ = components(n_actions=3)
    models[0].estimator.q_table[2] = torch.tensor([6., 0., 5.])
    models[1].estimator.q_table[2] = torch.tensor([0., 6., 5.])
    selector = EpsilonGreedyActionSelector(.2).double()
    observations = np.full(20000, 2, dtype=np.int32)
    torch.manual_seed(17)
    first = example.act_double_q(*models, observations, selector)
    torch.manual_seed(17)
    second = example.act_double_q(*models, observations, selector)
    assert first.action.shape == first.log_prob.shape == (20000,)
    assert torch.equal(first.action, second.action) and torch.equal(first.log_prob, second.log_prob)
    measured = torch.bincount(first.action, minlength=3).double() / len(first.action)
    expected = torch.tensor([.2/3, .2/3, .8+.2/3], dtype=torch.float64)
    torch.testing.assert_close(measured, expected, rtol=0, atol=.015)
    torch.testing.assert_close(first.log_prob, expected[first.action].log())
    assert not first.log_prob.requires_grad


def test_fair_update_coin_uses_independent_explicit_rng_and_reproduces():
    torch_rng = torch.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    first, second = np.random.default_rng(17), np.random.default_rng(17)
    selections = np.array([example.choose_update_first(first) for _ in range(20000)])
    repeated = np.array([example.choose_update_first(second) for _ in range(20000)])
    np.testing.assert_array_equal(selections, repeated)
    assert abs(selections.mean() - .5) < .015
    assert torch.equal(torch_rng, torch.get_rng_state())
    np.testing.assert_array_equal(numpy_rng[1], np.random.get_state()[1])
    assert numpy_rng[0] == np.random.get_state()[0] and numpy_rng[2:] == np.random.get_state()[2:]


def test_checkpoint_restores_both_models_selector_and_rng_for_identical_suffix_learning(tmp_path):
    models, target, rule = components()
    selector, rng = EpsilonGreedyActionSelector(.4).double(), np.random.default_rng(19)
    torch.manual_seed(19)

    def suffix(models, selector, rng, count):
        results = []
        for _ in range(count):
            action = example.act_double_q(*models, 1, selector)
            first = example.choose_update_first(rng)
            error = example.learn_double_q(*models, Transition(1, action.action, 1., 2, False, False, action.log_prob).to_batch(),
                                           target, rule, update_first=first)
            results.append((action.action.item(), first, error.item(), action.log_prob.item()))
        return results

    suffix(models, selector, rng, 5)
    checkpoint = tmp_path / "double-q.pt"
    torch.save({"model1": models[0].state_dict(), "model2": models[1].state_dict(), "selector": selector.state_dict(),
                "torch_rng": torch.get_rng_state(), "update_rng": rng.bit_generator.state,
                "gamma": target.gamma, "alpha": rule.alpha}, checkpoint)
    expected = suffix(models, selector, rng, 25)
    saved = torch.load(checkpoint, weights_only=True)
    restored, _, _ = components()
    for model, key in zip(restored, ("model1", "model2")):
        model.load_state_dict(saved[key])
    restored_selector, restored_rng = EpsilonGreedyActionSelector(0).double(), np.random.default_rng()
    restored_selector.load_state_dict(saved["selector"])
    restored_rng.bit_generator.state = saved["update_rng"]
    torch.set_rng_state(saved["torch_rng"])
    assert suffix(restored, restored_selector, restored_rng, 25) == expected
    for result, reference in zip(restored, models):
        torch.testing.assert_close(result.estimator.q_table, reference.estimator.q_table, rtol=0, atol=0)
    assert restored[0].estimator.q_table.untyped_storage().data_ptr() != restored[1].estimator.q_table.untyped_storage().data_ptr()


@pytest.mark.parametrize("offset_views", [False, True])
def test_shared_storage_is_rejected_even_when_data_pointers_differ(offset_views):
    models, target, rule = components()
    if offset_views:
        storage = torch.zeros(8, 2, dtype=torch.float64)
        models[0].estimator.q_table, models[1].estimator.q_table = storage[:4], storage[4:]
        assert models[0].estimator.q_table.data_ptr() != models[1].estimator.q_table.data_ptr()
    else:
        models[1].estimator.q_table = models[0].estimator.q_table
    for operation in (lambda: example.combined_estimate(*models, 1),
                      lambda: example.learn_double_q(*models, Transition(1, 0, 1., 2, False, False).to_batch(), target, rule, update_first=True)):
        with pytest.raises(ValueError, match="independent storage"):
            operation()


@pytest.mark.parametrize("mismatch", ["dtype", "shape", "estimator"])
def test_incompatible_models_fail_before_any_update(mismatch):
    models, target, rule = components()
    if mismatch == "dtype":
        models[1].float()
    else:
        wrong = TabularQEstimator(4, 3) if mismatch == "shape" else DiscreteQEstimator(1, 2)
        models = (models[0], Model(IdentityRepresentation(), wrong).double())
    with pytest.raises((ValueError, TypeError), match="Q tables|TabularQEstimator"):
        example.learn_double_q(*models, Transition(1, 0, 1., 2, False, False).to_batch(), target, rule, update_first=True)


@pytest.mark.parametrize("name,value,error", [
    ("observation", torch.tensor([1.]), TypeError), ("action", torch.tensor([0.]), TypeError),
    ("next_observation", torch.tensor([2.]), TypeError), ("reward", torch.tensor([1]), TypeError),
    ("terminated", torch.tensor([0]), TypeError), ("truncated", torch.tensor([False, False]), ValueError),
    ("observation", torch.tensor(1), ValueError), ("observation", torch.tensor([], dtype=torch.long), ValueError),
])
def test_batch_validation_rejects_ambiguous_shapes_or_rounding_indices(name, value, error):
    models, target, rule = components()
    with pytest.raises(error):
        example.learn_double_q(*models, replace(Transition(1, 0, 1., 2, False, False).to_batch(), **{name: value}),
                               target, rule, update_first=True)


@pytest.mark.parametrize("update_first", [1, None, torch.tensor([True, False])])
def test_per_row_update_choices_are_not_silently_accepted(update_first):
    models, target, rule = components()
    with pytest.raises(TypeError, match="one bool"):
        example.learn_double_q(*models, Transition(1, 0, 1., 2, False, False).to_batch(), target, rule, update_first=update_first)


def test_reward_graph_is_not_changed_but_no_graph_is_retained_by_learning():
    models, target, rule = components()
    reward = torch.tensor([1.], requires_grad=True)
    batch = replace(Transition(1, 0, 1., 2, False, False).to_batch(), reward=reward)
    result = example.learn_double_q(*models, batch, target, rule, update_first=np.bool_(True))
    assert result.item() == pytest.approx(-2.1) and not result.requires_grad
    assert batch.reward is reward and reward.grad is None and reward.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("first", [True, False])
def test_cuda_tables_and_cpu_batch_preserve_cross_dtype_device_and_rng(dtype, first):
    models, target, rule = components(dtype)
    for model in models:
        model.cuda()
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    error = example.learn_double_q(*models, Transition(1, 0, 1., 2, False, True).to_batch(), target, rule, update_first=first)
    assert error.dtype == dtype and error.device.type == "cuda"
    assert error.item() == pytest.approx(-2.1 if first else -3.2)
    selector = EpsilonGreedyActionSelector(.1)
    selected = example.act_double_q(*models, 2, selector, deterministic=True)
    assert selected.action.device.type == "cuda" and selected.log_prob is None
    torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
