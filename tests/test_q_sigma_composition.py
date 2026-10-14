"""Eq. 7.17's explicit boundary, control variates, frozen metadata and scheduling."""

from collections import deque
import copy
from dataclasses import asdict, replace

import pytest
import torch

from examples import gridworld_q_sigma as qs
from examples import gridworld_n_step_sarsa as sampled
from examples import gridworld_tree_backup as tree
from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import TabularQEstimator, TabularVEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import Transition, TransitionBatch
from modular_rl.model import Model

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))]


def components(dtype=torch.float64, device='cpu', gamma=.9, alpha=.5):
    model = Model(IdentityRepresentation(), TabularQEstimator(8,2), EpsilonGreedyActionSelector(.2)).to(device=device,dtype=dtype)
    return model, TD0TargetGenerator(gamma), TabularQUpdateRule(alpha), torch.full_like(model.estimator.q_table,.5)


def window(rewards, actions=None, *, flags=(False,False), states=None, logs=None):
    actions = [0]*len(rewards) if actions is None else actions
    states = list(range(len(rewards)+1)) if states is None else states
    logs = [torch.tensor(-.6931471805599453,dtype=torch.float64)]*len(rewards) if logs is None else logs
    return deque(Transition(states[i],a,float(r),states[i+1],flags[0] and i==len(rewards)-1,
                            flags[1] and i==len(rewards)-1,logs[i]).to_batch()
                 for i,(r,a) in enumerate(zip(rewards,actions)))


def numeric_case(dtype=torch.float64,device='cpu'):
    model,target,rule,policy=components(dtype,device)
    q=model.estimator.q_table; q[0,0]=2
    q[1]=q.new_tensor([10.,2.]);policy[1]=policy.new_tensor([.25,.75])
    q[2]=q.new_tensor([4.,8.]);policy[2]=policy.new_tensor([.6,.4])
    q[3]=q.new_tensor([6.,2.])
    rows=window([1,2,3],[0,0,1],logs=[torch.tensor(-.1,dtype=torch.float64),torch.tensor(.5,dtype=torch.float64).log(),torch.tensor(.25,dtype=torch.float64).log()])
    return model,target,rule,policy,rows


def reference_return(q,policy,rows,sigmas,gamma):
    """Independent Python scalar recursion; no implementation helper or TD0 call."""
    last=rows[-1]
    g=float(last.reward.item())
    if not last.terminated.item():
        state=last.next_observation.item()
        g+=gamma*sum(float(policy[state,a])*float(q[state,a]) for a in range(q.shape[1]))
    for i in range(len(rows)-2,-1,-1):
        state,action=rows[i+1].observation.item(),rows[i+1].action.item()
        pi=float(policy[state,action]);b=float(rows[i+1].log_prob.double().exp().item())
        c=sigmas[i]*pi/b+(1-sigmas[i])*pi
        value=sum(float(policy[state,a])*float(q[state,a]) for a in range(q.shape[1]))
        g=float(rows[i].reward.item())+gamma*(value+c*(g-float(q[state,action])))
    return g


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_required_n3_snapshot_return_coefficients_and_original_update(dtype,device,monkeypatch):
    model,target,rule,policy,rows=numeric_case(dtype,device)
    before=model.estimator.q_table.clone();rng=torch.get_rng_state().clone();ptr=model.estimator.q_table.data_ptr()
    backups,estimates,writes=[],[],[];generate,forward,update=target.generate,model.forward,rule.update
    def fold(r,v,t):
        value=generate(r,v,t);backups.append(value.item());return value
    def estimate(states):
        estimates.append(states.clone());return forward(states)
    def write(q,s,a,y):
        assert torch.equal(q,before) and not torch.is_grad_enabled()
        writes.append(y.clone());return update(q,s,a,y)
    monkeypatch.setattr(target,'generate',fold);monkeypatch.setattr(model,'forward',estimate);monkeypatch.setattr(rule,'update',write)
    monkeypatch.setattr(model,'act',lambda *a,**k:pytest.fail('learn must not sample'))
    result=qs.learn_q_sigma(model,rows,target,rule,target_policy=policy,sigma=[.5,.25,.75])
    tolerance=2e-6 if dtype==torch.float32 else 1e-12
    assert backups==pytest.approx([6.6,6.158,3.303325],rel=tolerance)
    assert result['target'].item()==pytest.approx(3.303325,rel=tolerance)
    assert model.estimator.q_table[0,0].item()==pytest.approx(2.6516625,rel=tolerance)
    assert result['td_error'].item()==pytest.approx(1.303325,rel=tolerance)
    assert result['importance_ratios'].tolist()==pytest.approx([.5,1.6],rel=tolerance)
    assert result['coefficients'].tolist()==pytest.approx([.375,.7],rel=tolerance)
    assert len(estimates)==len(writes)==1 and estimates[0].tolist()==[1,2,3]
    for name in ('target','td_error'):
        assert result[name].shape==(1,) and result[name].dtype==dtype and result[name].device==before.device
        assert not result[name].requires_grad and result[name].grad_fn is None
    assert result['coefficients'].dtype==torch.float64 and model.estimator.q_table.data_ptr()==ptr
    assert len(rows)==3 and torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('sigmas',[[0,0,0],[1,1,1],[0,1,0],[1,0,1],[.2,.7,.4],[.5,.5,.5]])
def test_time_indices_match_independent_reference_not_final_target_interpolation(sigmas):
    model,target,rule,policy,rows=numeric_case();before=model.estimator.q_table.clone()
    expected=reference_return(before,policy,rows,sigmas,.9)
    result=qs.learn_q_sigma(model,rows,target,rule,target_policy=policy,sigma=sigmas)
    assert result['target'].item()==pytest.approx(expected,abs=1e-12)
    if sigmas==[.5,.5,.5]:
        ends=[reference_return(before,policy,rows,[v]*3,.9) for v in (0,1)]
        assert abs(expected-sum(ends)/2)>1e-3


@pytest.mark.parametrize('sigma',[0.,.5,1.])
def test_n1_expected_horizon_is_46_and_has_no_ratio_or_sigma_dependence(sigma):
    model,target,rule,policy,rows=numeric_case()
    result=qs.learn_q_sigma(model,deque([rows[0]]),target,rule,target_policy=policy,sigma=sigma)
    assert result['target'].item()==pytest.approx(4.6) and result['importance_ratios'].numel()==0
    assert model.estimator.q_table[0,0].item()==pytest.approx(3.3)


@pytest.mark.parametrize('endpoint_sigma',[0.,.25,.75,1.])
def test_last_horizon_sigma_is_ignored(endpoint_sigma):
    model,target,rule,policy,rows=numeric_case()
    result=qs.learn_q_sigma(model,rows,target,rule,target_policy=policy,sigma=[.5,.25,endpoint_sigma])
    assert result['target'].item()==pytest.approx(3.303325)


@pytest.mark.parametrize('n',[1,3,5])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('flags',[(False,False),(True,False),(False,True),(True,True)])
def test_sigma_zero_is_bit_exact_tree_backup_with_same_rng(n,dtype,flags):
    one,target,rule,policy=components(dtype);two,t,r,_=components(dtype)
    one.estimator.q_table.copy_(torch.arange(16).reshape(8,2)/3);two.load_state_dict(one.state_dict())
    policy[1]=policy.new_tensor([1.,0.])  # Test Tree's exact deterministic shortcut.
    rows=window(range(1,n+1),[i%2 for i in range(n)],flags=flags)
    rng=torch.get_rng_state().clone()
    result=qs.learn_q_sigma(one,rows,target,rule,target_policy=policy,sigma=0.)
    expected=tree.learn_tree_backup(two,rows,t,r,target_policy=policy)
    assert torch.equal(result['td_error'],expected) and torch.equal(one.estimator.q_table,two.estimator.q_table)
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('n',[1,3,5])
def test_sigma_one_is_control_variate_not_unconditional_sampled_sarsa(n):
    one,target,rule,policy=components();two,t,r,_=components()
    one.estimator.q_table.copy_(torch.arange(16).reshape(8,2)/3);two.load_state_dict(one.state_dict())
    rows=window(range(1,n+1),[1]*n)
    expected=reference_return(one.estimator.q_table,policy,rows,[1.]*n,.9)
    result=qs.learn_q_sigma(one,rows,target,rule,target_policy=policy,sigma=1.)
    sampled_error=sampled.learn_n_step_sarsa(two,rows,t,r,next_action=1)
    assert result['target'].item()==pytest.approx(expected,abs=1e-12)
    assert not torch.equal(result['td_error'],sampled_error)


def test_sigma_one_matches_sampled_return_on_deterministic_supported_path():
    one,target,rule,policy=components();two,t,r,_=components()
    one.estimator.q_table.copy_(torch.arange(16).reshape(8,2)/3);two.load_state_dict(one.state_dict())
    policy[:,0]=1;policy[:,1]=0
    rows=window([1,2,3],logs=[torch.tensor(0.)]*3)
    result=qs.learn_q_sigma(one,rows,target,rule,target_policy=policy,sigma=1.)
    expected=sampled.learn_n_step_sarsa(two,rows,t,r,next_action=0)
    torch.testing.assert_close(result['td_error'],expected,rtol=0,atol=1e-14)


@pytest.mark.parametrize('sigma',[0.,.5,1.])
def test_zero_pi_drops_only_sampled_branch_even_with_extreme_behavior_log(sigma):
    model,target,rule,policy=components();model.estimator.q_table[1]=torch.tensor([100.,2.]);policy[1]=policy.new_tensor([0.,1.])
    rows=window([1,999],flags=(True,False),logs=[torch.tensor(-.1),torch.tensor(-1000.)])
    result=qs.learn_q_sigma(model,rows,target,rule,target_policy=policy,sigma=sigma)
    assert result['target'].item()==pytest.approx(2.8) and result['importance_ratios'].item()==0
    assert model.estimator.q_table[0,0].item()==pytest.approx(1.4)


@pytest.mark.parametrize('gamma',[0.,.9,1.])
@pytest.mark.parametrize('alpha',[.01,.5,1.])
def test_gamma_alpha_boundaries_and_frozen_self_transition_flush(gamma,alpha):
    model,target,rule,policy=components(gamma=gamma,alpha=alpha);model.estimator.q_table[0]=torch.tensor([4.,10.])
    rows=window([1,2],states=[0,0,0],flags=(False,True));snap=model.estimator.q_table.clone()
    expected=reference_return(snap,policy,rows,[.5,0],gamma)
    result=qs.learn_q_sigma(model,rows,target,rule,target_policy=policy,sigma=[.5,0])
    assert result['target'].item()==pytest.approx(expected)
    assert model.estimator.q_table[0,0].item()==pytest.approx(4+alpha*(expected-4))
    rows.popleft();value=model.estimator.q_table[0].mean().item()
    tail=qs.learn_q_sigma(model,rows,target,rule,target_policy=policy,sigma=1.)
    assert tail['target'].item()==pytest.approx(2+gamma*value)


def test_dynamic_policy_reads_same_single_snapshot_and_never_consumes_rng(monkeypatch):
    model,target,rule,_=components();model.estimator.q_table.copy_(torch.arange(16).reshape(8,2)/3)
    seen=[];forward,probs=model.forward,model.action_selector.probabilities
    def estimate(states):
        out=forward(states);seen.append(out);return out
    def probabilities(out):
        assert out is seen[0];return probs(out)
    monkeypatch.setattr(model,'forward',estimate);monkeypatch.setattr(model.action_selector,'probabilities',probabilities)
    rng=torch.get_rng_state().clone()
    qs.learn_q_sigma(model,window([1,2,3]),target,rule,target_policy=model.action_selector,sigma=.5)
    assert len(seen)==1 and torch.equal(rng,torch.get_rng_state())


class ScriptedEnv:
    def __init__(self,length,flags):self.length,self.flags=length,flags
    def reset(self,*,seed=None):self.time=0;return 0,{}
    def step(self,action):
        self.time+=1;last=self.time==self.length
        return self.time%3,float(self.time),self.flags[0] and last,self.flags[1] and last,{}


@pytest.mark.parametrize('n',[1,3,5])
@pytest.mark.parametrize('length',[1,3,5,7])
@pytest.mark.parametrize('flags',[(True,False),(False,True),(True,True)])
def test_online_schedule_freezes_sigma_at_selection_and_flushes_once(n,length,flags,monkeypatch):
    model,target,rule,policy=components();env=ScriptedEnv(length,flags)
    choices,writes,seen=[],[],[];act,update,learn=model.act,rule.update,qs.learn_q_sigma
    def select(state):
        out=act(state);choices.append((env.time,state,out.action.item(),out.log_prob.clone(),len(writes)));return out
    def provider(state,action,time):
        assert (time,state,action)==choices[-1][:3]
        seen.append((time,model.estimator.q_table.clone()));return (time%3)/2
    def write(q,s,a,y):
        writes.append(env.time);return update(q,s,a,y)
    def compute(m,rows,t,r,**kwargs):
        for i,row in enumerate(rows):
            torch.testing.assert_close(row.log_prob.reshape(()),choices[env.time-len(rows)+i][3])
        times=range(env.time-len(rows)+1,env.time)
        assert kwargs['sigma']==[(time%3)/2 for time in times]+[0.]
        return learn(m,rows,t,r,**kwargs)
    monkeypatch.setattr(model,'act',select);monkeypatch.setattr(rule,'update',write);monkeypatch.setattr(qs,'learn_q_sigma',compute)
    result=qs.run_episode(env,model,target,rule,target_policy=policy,sigma=provider,n=n)
    expected=[*range(n,length+1),*([length]*(n-1))] if length>=n else [length]*length
    assert writes==expected and result['updates']==result['steps']==length
    assert len(seen)==len(choices)==length and result['sigmas'].tolist()==[(i%3)/2 for i in range(length)]
    assert [choice[-1] for choice in choices]==[max(0,time-n+1) for time in range(length)]


def test_stored_behavior_and_sigma_are_not_recomputed_after_policy_change():
    model,target,rule,policy,rows=numeric_case();saved=[b.log_prob.clone() for b in rows]
    model.action_selector.epsilon=0.;model.estimator.q_table[5]=99
    result=qs.learn_q_sigma(model,rows,target,rule,target_policy=policy,sigma=[.5,.25,.75])
    assert result['target'].item()==pytest.approx(3.303325)
    assert all(torch.equal(b.log_prob,value) for b,value in zip(rows,saved))


def test_checkpoint_restores_pending_sigma_current_action_policy_model_rng(tmp_path):
    model,target,rule,policy=components();model.estimator.q_table.copy_(torch.arange(16).reshape(8,2)/5)
    pending=window([1,2]);sigmas=deque([.2,.7]);torch.manual_seed(19);chosen=model.act(2)
    checkpoint={'model':copy.deepcopy(model.state_dict()),'policy':policy.clone(),'pending':[asdict(b) for b in pending],
                'pending_sigmas':list(sigmas),'current_sigma':.4,'action':chosen.action,'log_prob':chosen.log_prob,
                'rng':torch.get_rng_state(),'n':3,'gamma':.9,'alpha':.5}
    path=tmp_path/'q_sigma.pt';torch.save(checkpoint,path)
    def finish(m,rows,metadata,t,r,p,action,log_b,decided):
        rows.append(Transition(2,action,3.,3,False,True,log_b).to_batch());metadata.append(decided);errors=[]
        while rows:
            errors.append(qs.learn_q_sigma(m,rows,t,r,target_policy=p,sigma=[*list(metadata)[1:],0.])['td_error'])
            rows.popleft();metadata.popleft()
        return torch.cat(errors)
    errors=finish(model,pending,sigmas,target,rule,policy,chosen.action,chosen.log_prob,.4);next_action=model.act(4)
    saved=torch.load(path,weights_only=True);other,t,r,_=components(gamma=saved['gamma'],alpha=saved['alpha'])
    other.load_state_dict(saved['model'])
    resumed=finish(other,deque(TransitionBatch(**b) for b in saved['pending']),deque(saved['pending_sigmas']),t,r,saved['policy'],saved['action'],saved['log_prob'],saved['current_sigma'])
    assert torch.equal(errors,resumed) and torch.equal(model.estimator.q_table,other.estimator.q_table)
    torch.set_rng_state(saved['rng']);out=other.act(4)
    assert torch.equal(out.action,next_action.action) and torch.equal(out.log_prob,next_action.log_prob)


@pytest.mark.parametrize('problem',['sigma_nan','sigma_inf','sigma_negative','sigma_large','sigma_shape','sigma_bool','sigma_callback',
                                    'log_none','log_positive','log_nan','log_zero_support','log_shape','ratio_overflow',
                                    'coefficient_overflow','product_underflow','reward_inf','state','action','boundary','policy_shape','policy_sum','model'])
def test_errors_are_explicit_and_atomic(problem):
    model,target,rule,policy,rows=numeric_case();sigma=.5
    if problem=='sigma_nan':sigma=float('nan')
    elif problem=='sigma_inf':sigma=float('inf')
    elif problem=='sigma_negative':sigma=-.1
    elif problem=='sigma_large':sigma=1.1
    elif problem=='sigma_shape':sigma=[.5]
    elif problem=='sigma_bool':sigma=True
    elif problem=='sigma_callback':sigma=lambda *a:.5
    elif problem=='log_none':rows[1]=replace(rows[1],log_prob=None)
    elif problem=='log_positive':rows[1]=replace(rows[1],log_prob=torch.tensor([.1]))
    elif problem=='log_nan':rows[1]=replace(rows[1],log_prob=torch.tensor([float('nan')]))
    elif problem=='log_zero_support':rows[1]=replace(rows[1],log_prob=torch.tensor([float('-inf')]))
    elif problem=='log_shape':rows[1]=replace(rows[1],log_prob=torch.zeros(1,1))
    elif problem=='ratio_overflow':rows[1]=replace(rows[1],log_prob=torch.tensor([-1000.]))
    elif problem=='coefficient_overflow':
        model=model.float();policy=policy.float();rows[1]=replace(rows[1],log_prob=torch.tensor([-100.]))
    elif problem=='product_underflow':
        model=model.float();policy=policy.float();policy[1]=policy.new_tensor([1.,0.]);policy[2]=policy.new_tensor([1.,0.])
        model.estimator.q_table[0,0]=1e-30;model.estimator.q_table[1,0]=1e-30;rows=window([0.,0.],flags=(True,False),logs=[torch.tensor(0.),torch.tensor(-1.)]);target=TD0TargetGenerator(0.);rule=TabularQUpdateRule(1e-30)
    elif problem=='reward_inf':rows[-1]=replace(rows[-1],reward=torch.tensor([float('inf')]))
    elif problem=='state':rows[1]=replace(rows[1],observation=torch.tensor([-1]))
    elif problem=='action':rows[1]=replace(rows[1],action=torch.tensor([3]))
    elif problem=='boundary':rows[0]=replace(rows[0],truncated=torch.tensor([True]))
    elif problem=='policy_shape':policy=policy[:1]
    elif problem=='policy_sum':policy[0]=.1
    elif problem=='model':model=Model(IdentityRepresentation(),TabularVEstimator(8))
    before=model.estimator.q_table.clone() if hasattr(model.estimator,'q_table') else None
    with pytest.raises((TypeError,ValueError,IndexError,FloatingPointError)):
        qs.learn_q_sigma(model,rows,target,rule,target_policy=policy,sigma=sigma)
    if before is not None:assert torch.equal(model.estimator.q_table,before)


def test_head_probability_is_excluded_and_no_global_ratio_is_applied():
    one,t,r,policy,rows=numeric_case();two,t2,r2,_,other=numeric_case()
    other[0]=replace(other[0],log_prob=torch.tensor([-1000.],dtype=torch.float64))
    result=qs.learn_q_sigma(one,rows,t,r,target_policy=policy,sigma=[.5,.25,.75])
    altered=qs.learn_q_sigma(two,other,t2,r2,target_policy=policy,sigma=[.5,.25,.75])
    assert torch.equal(result['target'],altered['target']) and torch.equal(one.estimator.q_table,two.estimator.q_table)
    assert result['td_error'].item()==pytest.approx(1.303325)  # Not rho-product times delta.


def test_table_dtype_update_overflow_and_recursive_underflow_are_atomic():
    model,target,rule,policy=components(torch.float32)
    model.estimator.q_table[0,0]=-3e38
    rows=window([3e38],flags=(True,False));before=model.estimator.q_table.clone()
    with pytest.raises(FloatingPointError):qs.learn_q_sigma(model,rows,target,rule,target_policy=policy,sigma=1.)
    assert torch.equal(before,model.estimator.q_table)
    model,target,rule,policy=components(torch.float32,gamma=1.)
    policy[1]=policy.new_tensor([1e-30,1.]);model.estimator.q_table[1,0]=1e-30
    rows=window([0.,0.],flags=(True,False),logs=[torch.tensor(0.),torch.tensor(0.)]);before=model.estimator.q_table.clone()
    with pytest.raises(FloatingPointError):qs.learn_q_sigma(model,rows,target,rule,target_policy=policy,sigma=1.)
    assert torch.equal(before,model.estimator.q_table)


def test_input_autograd_graphs_do_not_escape_learning():
    model,target,rule,policy,rows=numeric_case()
    model.estimator.q_table.requires_grad_();policy.requires_grad_()
    for i,row in enumerate(rows):
        rows[i]=replace(row,reward=row.reward.requires_grad_(),log_prob=row.log_prob.requires_grad_())
    result=qs.learn_q_sigma(model,rows,target,rule,target_policy=policy,sigma=[.5,.25,.75])
    assert all(not value.requires_grad and value.grad_fn is None for value in result.values())
    assert model.estimator.q_table.grad is None and policy.grad is None


def test_required_window_sigma_one_versus_sampled_sarsa_numeric_difference():
    one,t,r,policy,rows=numeric_case();two,t2,r2,_,_=numeric_case()
    result=qs.learn_q_sigma(one,rows,t,r,target_policy=policy,sigma=1.)
    sampled_error=sampled.learn_n_step_sarsa(two,rows,t2,r2,next_action=0)
    assert result['target'].item()==pytest.approx(2.3608)
    assert 2+sampled_error.item()==pytest.approx(9.604)
    assert two.estimator.q_table[0,0].item()==pytest.approx(5.802)
    assert result['target'].item()-(2+sampled_error.item())==pytest.approx(-7.2432)
