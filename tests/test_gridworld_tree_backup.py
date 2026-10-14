"""Fixed-policy Gymnasium comparison, repeatability, file loading and CLI."""

from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

from examples import gridworld_tree_backup as example


@pytest.mark.parametrize('n',[1,3,5])
def test_short_fixed_policy_runs_compare_same_trajectories_with_own_dp_q(n):
    results=example.run_comparison(seed=11,episodes=3,ns=(n,),log_every=0)
    assert len(results)==5
    for result in results:
        assert result['steps']==result['updates']==result['episode_lengths'].sum()
        assert result['q_values'].shape==result['target_policy'].shape==result['dp_q_values'].shape==(11,4)
        assert np.isfinite(result['q_values']).all() and np.isfinite(result['q_rmse'])
        assert result['terminated_episodes']+result['truncated_episodes']==3
    for tree,reference in ((results[3],results[0]),(results[4],results[2])):
        np.testing.assert_array_equal(tree['training_rewards'],reference['training_rewards'])
        np.testing.assert_array_equal(tree['episode_lengths'],reference['episode_lengths'])
        np.testing.assert_array_equal(tree['dp_q_values'],reference['dp_q_values'])
        assert 'importance_ratios' not in tree and 'ess' not in tree
    assert not np.array_equal(results[3]['dp_q_values'],results[4]['dp_q_values'])


def test_seed_reproduces_fixed_policy_q_rewards_lengths_and_history():
    _,one=example.train_and_evaluate(episodes=4,seed=7,log_every=2)
    model,two=example.train_and_evaluate(episodes=4,seed=7,log_every=2)
    for key in ('q_values','training_rewards','episode_lengths','target_policy'):np.testing.assert_array_equal(one[key],two[key])
    assert one['history']==two['history']
    two['q_values'].fill(100)
    assert not np.all(model.estimator.q_table.numpy()==100)


def test_time_limit_flushes_without_a_bootstrap_action_and_closes_environment(monkeypatch):
    closed,make=[],example.make_dlfs4_gridworld_env
    def create(**kwargs):
        env=make(**kwargs);close=env.close
        def finish():closed.append(env);close()
        monkeypatch.setattr(env,'close',finish);return env
    monkeypatch.setattr(example,'make_dlfs4_gridworld_env',create)
    _,result=example.train_and_evaluate(n=5,episodes=3,max_steps=2,log_every=0)
    assert result['steps']==result['updates']==6 and result['truncated_episodes']==3 and len(closed)==1


def test_file_loads_from_notebook_directory_without_installed_examples(monkeypatch):
    root=Path(__file__).resolve().parents[1];monkeypatch.chdir(root/'notebooks'/'dlfs4')
    api=runpy.run_path(str(root/'examples'/'gridworld_tree_backup.py'))
    assert all(callable(api[key]) for key in ('learn_tree_backup','run_episode','train_and_evaluate','run_comparison'))


def test_cli_reports_n_policy_rmse_and_existing_comparators():
    result=subprocess.run([sys.executable,'-m','examples.gridworld_tree_backup','--episodes','1','--seeds','7','--ns','1','--log-every','0'],
                          capture_output=True,text=True,check=True,timeout=30,cwd=Path(__file__).resolve().parents[1])
    for label in ('algorithm=tree_backup','algorithm=off_policy_sarsa','algorithm=on_policy_sarsa','target=matched',
                  'target=different','Qpi_RMSE','RMSE mean=','Target pi:','Qpi estimate:'):
        assert label in result.stdout
