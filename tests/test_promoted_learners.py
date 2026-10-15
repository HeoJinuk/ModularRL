"""Promoted algorithms versus frozen dc2ecdc calculations and online schedules."""
import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from modular_rl import Model, representations as R, estimators as E, selectors as S
from modular_rl import learners as L, targets as T, updates as U, buffers as B
from modular_rl.data import Transition
from modular_rl.environment import make_dlfs4_gridworld_env
from tests._legacy_td import expected_sarsa as expected_ref, double_q as double_ref


class RecordingTarget(T.TD0):
    def __init__(self):
        super().__init__(.9)
        self.targets=[]
    def generate(self,*args):
        value=super().generate(*args)
        self.targets.append(value.clone())
        return value


class RecordingRule(U.TabularQ):
    def __init__(self):
        super().__init__(.1)
        self.trace=[]
    def update(self,table,states,actions,target):
        result=super().update(table,states,actions,target)
        self.trace.append((target.clone(),result.clone(),table.clone(),torch.get_rng_state().clone()))
        return result


def model(dtype=torch.float64):
    result=Model(R.Identity(),E.TabularQ(12,4),S.EpsilonGreedy(.2)).to(dtype=dtype)
    with torch.no_grad():
        result.estimator.q_table.copy_(torch.arange(48).reshape(12,4).to(dtype)/100)
    return result


def equal(left,right):
    if isinstance(left,torch.Tensor):
        assert torch.equal(left,right)
        assert left.requires_grad==right.requires_grad
    elif isinstance(left,np.ndarray): np.testing.assert_array_equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for k in left: equal(left[k],right[k])
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right)
        for a,b in zip(left,right): equal(a,b)
    else: assert left==right


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('truncated',[False,True])
def test_expected_and_double_frozen_batch_equivalence(dtype,truncated):
    batch=B.Replay(4,seed=3)
    for state,action,reward,next_state,term in [(0,1,1.,2,False),(0,1,2.,3,False),(4,2,3.,5,True)]:
        batch.add(state,action,reward,next_state,term,truncated)
    batch=batch.sample(3)
    for algorithm in ('expected','double'):
        traces=[]
        for legacy in (True,False):
            torch.manual_seed(7)
            first,second=model(dtype),model(dtype)
            second.estimator.q_table.mul_(2)
            target,rule=RecordingTarget(),RecordingRule()
            if algorithm=='expected':
                learner=L.ExpectedSARSA(first,target,rule)
                result=(expected_ref.learn_expected_sarsa(first,batch,target,rule,policy_selector=first.action_selector)
                        if legacy else learner.learn(batch))
            else:
                learner=L.DoubleQ(first,second,target,rule,action_selector=first.action_selector)
                a=(double_ref.act_double_q(first,second,torch.tensor([0,1]),first.action_selector)
                   if legacy else learner.act(torch.tensor([0,1])))
                result=[]
                for update_first in (True,False,True):
                    result.append(double_ref.learn_double_q(first,second,batch,target,rule,update_first=update_first)
                                  if legacy else learner.learn(batch,update_first=update_first))
                result+=(a.action,a.log_prob)
            traces.append((result,rule.trace,target.targets,first.state_dict(),second.state_dict(),torch.get_rng_state()))
        equal(*traces)


WINDOWS=[('n_step_sarsa','gridworld_n_step_sarsa'),
         ('off_policy_n_step_sarsa','gridworld_off_policy_n_step_sarsa'),
         ('tree_backup','gridworld_tree_backup'),('q_sigma','gridworld_q_sigma')]

@pytest.mark.parametrize('algorithm,example',WINDOWS)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('max_steps',[3,1000])
def test_full_online_trace_and_flush_equal_frozen_reference(algorithm,example,dtype,max_steps):
    frozen=importlib.import_module('tests._legacy_td.'+algorithm)
    promoted=importlib.import_module('examples.'+example)
    traces=[]
    for module in (frozen,promoted):
        torch.manual_seed(7)
        current=model(dtype)
        # Uniform behavior makes all recorded logs independent of later Q changes.
        current.action_selector.epsilon=1.
        target,rule=RecordingTarget(),RecordingRule()
        policy=torch.full_like(current.estimator.q_table,.25)
        policy[:]=policy.new_tensor([.3,.2,.2,.3])
        kwargs={} if algorithm=='n_step_sarsa' else {'target_policy':policy}
        if algorithm=='q_sigma':kwargs['sigma']=[0.,.5,1.]
        env=make_dlfs4_gridworld_env(max_episode_steps=max_steps)
        try:
            episodes=[module.run_episode(env,current,target,rule,n=3,seed=7 if i==0 else None,**kwargs) for i in range(3)]
            traces.append((episodes,rule.trace,target.targets,current.state_dict(),
                           torch.get_rng_state(),env.unwrapped.np_random.bit_generator.state))
            assert len(rule.trace)==sum(row['steps'] for row in episodes)
        finally: env.close()
    equal(*traces)


@pytest.mark.parametrize('name,class_name',[
    ('ExpectedSARSA','ExpectedSARSALearner'),('DoubleQ','DoubleQLearner'),
    ('NStepSARSA','NStepSARSALearner'),('OffPolicyNStepSARSA','OffPolicyNStepSARSALearner'),
    ('TreeBackup','TreeBackupLearner'),('QSigma','QSigmaLearner')])
def test_promoted_public_alias(name,class_name):
    from modular_rl import learner
    assert getattr(L,name) is getattr(learner,class_name)


def test_positional_n_step_dqn_and_legacy_constructor_are_identical():
    import copy
    online=Model(R.MLP(2,(4,)),E.DiscreteQ(4,2))
    target=copy.deepcopy(online)
    first=L.DQN(online,target,torch.optim.SGD(online.parameters(),lr=.01),T.TD0(.9),U.TDLoss())
    other=copy.deepcopy(online)
    second=L.DQN(other,copy.deepcopy(other),torch.optim.SGD(other.parameters(),lr=.01),T.NStep(.9),U.TDLoss())
    replay=B.SequenceReplay(5,seed=7)
    replay.add(np.array([0.,1.],dtype=np.float32),0,1.,np.array([1.,2.],dtype=np.float32),False,False)
    replay.add(np.array([1.,2.],dtype=np.float32),1,2.,np.array([2.,3.],dtype=np.float32),False,True)
    batch=replay.sample(1,length=2)
    a,b=first.learn(batch),second.learn(batch)
    equal((a.loss,a.td_error,online.state_dict()),(b.loss,b.td_error,other.state_dict()))
    # A positional n-step generator still permits the one-step planning path.
    one=B.Replay(1);one.add(np.array([0.,1.],dtype=np.float32),0,1.,np.array([1.,2.],dtype=np.float32),False,True)
    a,b=first.learn(one.sample(1)),second.learn(one.sample(1))
    equal((a.loss,a.td_error,online.state_dict()),(b.loss,b.td_error,other.state_dict()))


def test_installed_wheel_needs_no_examples(tmp_path):
    """Run all six new APIs using only an installed wheel, outside the checkout."""
    root=Path(__file__).resolve().parents[1]
    wheels=tmp_path/'wheels';wheels.mkdir()
    result=subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation',
        '--no-cache-dir','--disable-pip-version-check','--wheel-dir',str(wheels),str(root)],
        cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    package=tmp_path/'installed'
    result=subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-cache-dir',
        '--disable-pip-version-check','--target',str(package),str(next(wheels.glob('*.whl')))],
        cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    code=r'''
import sys
sys.path.insert(0,sys.argv[1])
sys.modules['examples']=None
import torch, modular_rl
from modular_rl import Model, representations as R, estimators as E, selectors as S
from modular_rl import learners as L, targets as T, updates as U
from modular_rl.data import Transition
assert modular_rl.__file__.startswith(sys.argv[1])
def model():return Model(R.Identity(),E.TabularQ(4,2),S.EpsilonGreedy(.1))
batch=Transition(0,0,1.,1,True,False,torch.tensor(-.6931471805599453,dtype=torch.float64)).to_batch()
policy=torch.full((4,2),.5)
for cls in (L.ExpectedSARSA,L.NStepSARSA,L.OffPolicyNStepSARSA,L.TreeBackup,L.QSigma):
    current=model();kwargs={} if cls in (L.ExpectedSARSA,L.NStepSARSA) else {'target_policy':policy}
    learner=cls(current,T.TD0(.9),U.TabularQ(.5),**kwargs)
    result=learner.learn(batch) if cls is L.ExpectedSARSA else learner.learn([batch],**({'sigma':.5} if cls is L.QSigma else {}))
    assert current.estimator.q_table[0,0].item()==.5
learner=L.DoubleQ(model(),model(),T.TD0(.9),U.TabularQ(.5),action_selector=S.EpsilonGreedy(.1))
learner.act(0);learner.learn(batch,update_first=True)
assert learner.model1.estimator.q_table[0,0].item()==.5

# New planning and neural APIs also work with examples deliberately unavailable.
from copy import deepcopy
from modular_rl import world_models as W, planners as P
world=W.Tabular();world.observe(Transition(0,0,1.,1,True,True))
q=Model(R.Identity(),E.TabularQ(4,2))
planning=P.Dyna(world,L.QLearning(q,T.TD0(.9),U.TabularQ(.5)),seed=7)
assert planning.plan(2)['updates']==2 and q.estimator.q_table[0,0].item()==.75
world2=W.Tabular();world2.load_state_dict(world.state_dict())
assert len(world2)==1 and not world2.sample(planning.rng).truncated
online=Model(R.MLP(2,(4,)),E.DiscreteQ(4,2))
double=L.DoubleDQN(online,deepcopy(online),torch.optim.Adam(online.parameters()),T.NStep(.9),U.TDLoss())
double.sync_target()
neural_batch=Transition(torch.tensor([1.,2.]),0,1.,torch.tensor([2.,3.]),False,True).to_batch()
result=double.learn(neural_batch)
assert result.loss.shape==() and not result.loss.requires_grad
ac_model=Model(R.MLP(2,(4,)),E.ActorCritic(E.CategoricalPolicy(4,2),E.StateValue(4)),S.Categorical())
ac=L.ActorCritic(ac_model,torch.optim.Adam(ac_model.parameters()),T.TD0(.9),U.ActorCritic())
result=ac.learn(neural_batch)
assert result.td_error.shape==(1,) and not result.total_loss.requires_grad
'''
    result=subprocess.run([sys.executable,'-I','-c',code,str(package)],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_double_q_learning_without_selector_and_default_selector_identity():
    one=Model(R.Identity(),E.TabularQ(3,2))
    two=Model(R.Identity(),E.TabularQ(3,2))
    learner=L.DoubleQ(one,two,T.TD0(.9),U.TabularQ(.5))
    error=learner.learn(Transition(0,1,2.,1,True,False).to_batch(),update_first=True)
    assert error.item()==2. and one.estimator.q_table[0,1].item()==1.
    with pytest.raises(ValueError,match='action_selector'): learner.act(0)
    selector=S.EpsilonGreedy(.1)
    one.action_selector=selector
    learner=L.DoubleQ(one,two,T.TD0(.9),U.TabularQ(.5))
    assert learner.action_selector is selector


def test_dqn_rejects_ambiguous_n_step_injection():
    import copy
    online=Model(R.MLP(2,(4,)),E.DiscreteQ(4,2))
    with pytest.raises(ValueError,match='once'):
        L.DQN(online,copy.deepcopy(online),torch.optim.SGD(online.parameters(),lr=.01),
              T.NStep(.9),U.TDLoss(),n_step_target_generator=T.NStep(.9))
