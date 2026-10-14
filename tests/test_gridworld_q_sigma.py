"""Gymnasium comparison uses the same fixed-policy DP reference and real steps."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from examples import gridworld_q_sigma as qs
from examples import gridworld_tree_backup as tree


@pytest.mark.parametrize('n',[1,3,5])
def test_short_comparison_reuses_references_and_sigma_zero_tree_values(n):
    refs=tree.run_comparison(seed=11,episodes=2,ns=(n,),log_every=0)
    results=qs.run_comparison(seed=11,episodes=2,ns=(n,),sigmas=(0.,.5,1.,(0.,1.)),reference_results=refs,log_every=0)
    assert len(results)==13 and len(refs)==5
    for result in results:
        assert result['steps']==result['updates']==result['episode_lengths'].sum()
        assert result['q_values'].shape==result['target_policy'].shape==result['dp_q_values'].shape==(11,4)
        assert np.isfinite(result['q_values']).all() and np.isfinite(result['q_rmse'])
    for result in results[5:]:
        assert np.isfinite(result['importance_ratios']).all() and np.isfinite(result['coefficients']).all()
        assert result['sampled_sigmas'].shape==(result['steps'],)
        if result['target_mode']=='matched':np.testing.assert_allclose(result['importance_ratios'],1,atol=1e-14)


def test_seed_reproduces_q_sigma_patterns_statistics_and_histories():
    _,one=qs.train_and_evaluate(episodes=3,sigma=(.2,.7,.4),seed=7,log_every=0)
    model,two=qs.train_and_evaluate(episodes=3,sigma=(.2,.7,.4),seed=7,log_every=0)
    for key in ('q_values','target_policy','training_rewards','episode_lengths','coefficients','importance_ratios','sampled_sigmas'):
        np.testing.assert_array_equal(one[key],two[key])
    assert one['history']==two['history']
    two['q_values'].fill(100);assert not np.all(model.estimator.q_table.numpy()==100)


@pytest.mark.parametrize('sigma',[0.,.5,1.,(0.,1.)])
def test_truncation_flush_has_no_extra_rng_and_matches_tree_trajectories(sigma,monkeypatch):
    closed=[];make=qs.make_dlfs4_gridworld_env
    def create(**kwargs):
        env=make(**kwargs);close=env.close
        def finish():closed.append(env);close()
        monkeypatch.setattr(env,'close',finish);return env
    monkeypatch.setattr(qs,'make_dlfs4_gridworld_env',create)
    _,result=qs.train_and_evaluate(n=5,episodes=3,sigma=sigma,max_steps=2,log_every=0)
    _,reference=tree.train_and_evaluate(n=5,episodes=3,max_steps=2,log_every=0)
    assert result['steps']==result['updates']==6 and result['truncated_episodes']==3 and len(closed)==1
    for key in ('training_rewards','episode_lengths'):np.testing.assert_array_equal(result[key],reference[key])


def test_namespace_loading_in_notebook_cwd_without_sys_path_changes():
    root=Path(__file__).resolve().parents[1]
    code=f"""
import importlib.machinery,importlib.util,sys
before=list(sys.path)
spec=importlib.machinery.ModuleSpec('examples',loader=None,is_package=True)
spec.submodule_search_locations=[{str(root/'examples')!r}]
sys.modules['examples']=importlib.util.module_from_spec(spec)
from examples import gridworld_q_sigma as qs
assert sys.path==before
assert all(callable(getattr(qs,key)) for key in ('learn_q_sigma','run_episode','train_and_evaluate','run_comparison'))
"""
    subprocess.run([sys.executable,'-c',code],cwd=root/'notebooks/dlfs4',capture_output=True,text=True,check=True,timeout=30)


def test_cli_reports_comparators_statistics_and_q_tables():
    result=subprocess.run([sys.executable,'-m','examples.gridworld_q_sigma','--episodes','1','--seeds','7','--ns','1',
                           '--sigmas','0','.5','1','--log-every','0'],cwd=Path(__file__).resolve().parents[1],
                          capture_output=True,text=True,check=True,timeout=45)
    for label in ('algorithm=q_sigma','algorithm=tree_backup','algorithm=off_policy_sarsa','algorithm=on_policy_sarsa',
                  'target=matched','target=different','Qpi_RMSE','RMSE mean=','c_mean=','rho_mean=','Target pi:','Qpi estimate:'):
        assert label in result.stdout
