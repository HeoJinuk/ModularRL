"""Recursive targets, branch probabilities, snapshots and online timing."""

from collections import deque
import copy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from examples import gridworld_expected_sarsa as expected
from examples import gridworld_n_step_sarsa as sampled
from examples import gridworld_tree_backup as tree
from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import TabularQEstimator, TabularVEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import Transition, TransitionBatch
from modular_rl.model import Model

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]


def components(dtype=torch.float64, device="cpu", gamma=.9, alpha=.5):
    model = Model(IdentityRepresentation(), TabularQEstimator(8, 2), EpsilonGreedyActionSelector(.2)).to(device=device, dtype=dtype)
    return model, TD0TargetGenerator(gamma), TabularQUpdateRule(alpha), torch.full_like(model.estimator.q_table, .5)


def window(rewards, actions=None, *, terminated=False, truncated=False):
    actions = [0]*len(rewards) if actions is None else actions
    return deque(Transition(i, a, float(r), i+1, terminated and i==len(rewards)-1,
                            truncated and i==len(rewards)-1).to_batch() for i,(r,a) in enumerate(zip(rewards,actions)))


def numeric_case(case, dtype=torch.float64, device="cpu"):
    model, target, rule, policy = components(dtype, device)
    q = model.estimator.q_table
    q[0,0] = 2
    q[1] = q.new_tensor([10.,2.]); policy[1] = policy.new_tensor([.25,.75])
    q[2] = q.new_tensor([4.,8.]); policy[2] = policy.new_tensor([.6,.4])
    q[3] = q.new_tensor([6.,2.])
    if case == "A":
        rows = window([1])
    elif case == "B":
        rows = window([1,2,3], [0,0,1])
    else:
        rows = window([1,2], terminated=case=="C", truncated=case=="D")
        q[2] = q.new_tensor([6.,2.]); policy[2] = .5
    return model,target,rule,policy,rows


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32,torch.float64])
@pytest.mark.parametrize("case,g,error,value", [("A",4.6,2.6,3.3),("B",3.8206,1.8206,2.9103),
                                                ("C",2.8,.8,2.4),("D",3.61,1.61,2.805)])
def test_required_numeric_cases_and_original_rule(case,g,error,value,dtype,device,monkeypatch):
    model,target,rule,policy,rows = numeric_case(case,dtype,device)
    updates, backups, estimates = [], [], []
    update,generate,forward = rule.update,target.generate,model.forward
    before,pointer,rng = model.estimator.q_table.clone(),model.estimator.q_table.data_ptr(),torch.get_rng_state().clone()

    def write(table,states,actions,y):
        assert torch.equal(table,before) and not torch.is_grad_enabled()
        updates.append(y.clone()); return update(table,states,actions,y)

    def fold(reward,continuation,terminated):
        result = generate(reward,continuation,terminated)
        backups.append(result.item()); return result

    def estimate(states):
        estimates.append(states.clone()); return forward(states)

    monkeypatch.setattr(rule,"update",write); monkeypatch.setattr(target,"generate",fold)
    monkeypatch.setattr(model,"forward",estimate)
    monkeypatch.setattr(model,"act",lambda *a,**k: pytest.fail("Tree Backup must not sample"))
    result = tree.learn_tree_backup(model,rows,target,rule,target_policy=policy)
    tolerance = 2e-6 if dtype==torch.float32 else 1e-12
    assert updates[0].item()==pytest.approx(g,rel=tolerance)
    assert result.item()==pytest.approx(error,rel=tolerance)
    assert model.estimator.q_table[0,0].item()==pytest.approx(value,rel=tolerance)
    assert backups==pytest.approx({"A":[4.6],"B":[6.6,6.536,3.8206],"C":[2.,2.8],"D":[5.6,3.61]}[case],rel=tolerance)
    assert len(estimates)==1 and estimates[0].numel()==len(rows)-int(case=="C")
    assert result.shape==(1,) and result.dtype==dtype and result.device==model.estimator.q_table.device
    assert not result.requires_grad and result.grad_fn is None and len(rows)==(3 if case=="B" else 1 if case=="A" else 2)
    assert model.estimator.q_table.data_ptr()==pointer and torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize("flags,targets", [((True,False),[2.8,2.]),((True,True),[2.8,2.]),((False,True),[3.61,5.6])])
def test_short_boundary_flush_and_no_terminal_endpoint_lookup(flags,targets,monkeypatch):
    model,target,rule,policy,rows = numeric_case("D")
    rows[-1]=replace(rows[-1],terminated=torch.tensor([flags[0]]),truncated=torch.tensor([flags[1]]))
    written,original,forward = [],rule.update,model.forward

    def write(q,s,a,y):
        written.append(y.item()); return original(q,s,a,y)

    def estimate(states):
        assert not flags[0] or 2 not in states.tolist()
        return forward(states)

    monkeypatch.setattr(rule,"update",write); monkeypatch.setattr(model,"forward",estimate)
    monkeypatch.setattr(model,"act",lambda *a,**k: pytest.fail("No flush or bootstrap actions"))
    while rows:
        tree.learn_tree_backup(model,rows,target,rule,target_policy=policy); rows.popleft()
    assert written==pytest.approx(targets)


@pytest.mark.parametrize("gamma", [0.,.9,1.])
@pytest.mark.parametrize("alpha", [.01,.5,1.])
def test_gamma_alpha_boundaries(gamma,alpha):
    model,target,rule,policy=components(gamma=gamma,alpha=alpha)
    model.estimator.q_table[1]=torch.tensor([2.,6.]);model.estimator.q_table[0,0]=2
    error=tree.learn_tree_backup(model,window([1]),target,rule,target_policy=policy)
    assert error.item()==pytest.approx(1+gamma*4-2)
    assert model.estimator.q_table[0,0].item()==pytest.approx(2+alpha*(1+gamma*4-2))


def test_zero_probability_cuts_only_sampled_branch_and_behavior_logs_are_unused():
    model,target,rule,policy=components()
    model.estimator.q_table[1]=torch.tensor([100.,2.]);policy[1]=policy.new_tensor([0.,1.])
    rows=window([1,999],terminated=True)
    rows[0]=replace(rows[0],log_prob=torch.tensor([float("nan")]))
    rows[1]=replace(rows[1],log_prob=torch.tensor([float("-inf")]))
    error=tree.learn_tree_backup(model,rows,target,rule,target_policy=policy)
    assert error.item()==pytest.approx(2.8) and model.estimator.q_table[0,0].item()==pytest.approx(1.4)


@pytest.mark.parametrize("n",[1,3,5])
def test_deterministic_matching_path_equals_sampled_sarsa_but_stochastic_policy_does_not(n):
    model,target,rule,policy=components()
    model.estimator.q_table.copy_(torch.arange(16,dtype=torch.float64).reshape(8,2)/3)
    policy[:,0]=0;policy[:,1]=1
    rows=window(range(1,n+1),[1]*n)
    reference,ref_target,ref_rule,_=components();reference.load_state_dict(model.state_dict())
    error=tree.learn_tree_backup(model,rows,target,rule,target_policy=policy)
    ref=sampled.learn_n_step_sarsa(reference,rows,ref_target,ref_rule,next_action=1)
    assert torch.equal(error,ref) and torch.equal(model.estimator.q_table,reference.estimator.q_table)
    if n>1:
        model,target,rule,policy=components();reference,ref_target,ref_rule,_=components()
        model.estimator.q_table.copy_(torch.arange(16).reshape(8,2));reference.load_state_dict(model.state_dict())
        assert not torch.equal(tree.learn_tree_backup(model,rows,target,rule,target_policy=policy),
                               sampled.learn_n_step_sarsa(reference,rows,ref_target,ref_rule,next_action=1))


def test_repeated_states_use_one_snapshot_per_call_and_latest_q_in_next_flush(monkeypatch):
    model,target,rule,policy=components()
    model.estimator.q_table[0]=torch.tensor([4.,10.])
    rows=deque(Transition(0,0,r,0,False,i==1).to_batch() for i,r in enumerate([1.,2.]))
    snapshots,targets,forward,update=[],[],model.forward,rule.update

    def estimate(states):
        result=forward(states);snapshots.append(result.values.q[0].clone());return result

    def write(q,s,a,y):
        targets.append(y.item());return update(q,s,a,y)

    monkeypatch.setattr(model,"forward",estimate);monkeypatch.setattr(rule,"update",write)
    tree.learn_tree_backup(model,rows,target,rule,target_policy=policy);rows.popleft()
    tree.learn_tree_backup(model,rows,target,rule,target_policy=policy)
    assert snapshots[0].tolist()==[[4.,10.],[4.,10.]]
    assert snapshots[1][0,0].item()==pytest.approx(6.6175)
    assert targets==pytest.approx([9.235,9.477875])
    assert model.estimator.q_table[0,0].item()==pytest.approx(8.0476875)


class ScriptedEnv:
    def __init__(self,length,flags):self.length,self.flags=length,flags
    def reset(self,*,seed=None):self.time=0;return 0,{}
    def step(self,action):
        self.time+=1
        ended=self.time==self.length
        return self.time%3,float(self.time),self.flags[0] and ended,self.flags[1] and ended,{}


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
@pytest.mark.parametrize("flags",[(True,False),(False,True),(True,True)])
def test_dynamic_one_step_matches_expected_sarsa_actions_rewards_targets_errors_table_rng(dtype,flags,monkeypatch):
    traces=[]
    for algorithm in ("expected","tree"):
        model,target,rule,_=components(dtype)
        model.estimator.q_table.copy_(torch.arange(16).reshape(8,2)/3)
        env,events,writes=ScriptedEnv(7,flags),[],[]
        act,update=model.act,rule.update

        def select(state):
            out=act(state);events.append((env.time,state,out.action.item(),out.log_prob.item()));return out

        def write(q,s,a,y):
            error=update(q,s,a,y);writes.append((env.time,s.item(),a.item(),y.clone(),error.clone()));return error

        monkeypatch.setattr(model,"act",select);monkeypatch.setattr(rule,"update",write)
        torch.manual_seed(11)
        if algorithm=="tree":
            tree.run_episode(env,model,target,rule,target_policy=model.action_selector,n=1)
        else:
            state,_=env.reset();chosen=model.act(state)
            while True:
                next_state,reward,term,trunc,_=env.step(chosen.action.item())
                batch=Transition(state,chosen.action,reward,next_state,term,trunc,chosen.log_prob).to_batch()
                expected.learn_expected_sarsa(model,batch,target,rule,policy_selector=model.action_selector)
                if term or trunc:break
                state=next_state;chosen=model.act(state)
        traces.append((events,writes,model.estimator.q_table.clone(),torch.get_rng_state().clone()))
    assert traces[0][0]==traces[1][0] and len(traces[1][0])==7
    for left,right in zip(traces[0][1],traces[1][1]):
        assert left[:3]==right[:3] and torch.equal(left[3],right[3]) and torch.equal(left[4],right[4])
    assert torch.equal(traces[0][2],traces[1][2]) and torch.equal(traces[0][3],traces[1][3])


@pytest.mark.parametrize("n",[1,3,5])
@pytest.mark.parametrize("length",[1,3,5,7])
@pytest.mark.parametrize("flags",[(True,False),(False,True),(True,True)])
def test_online_schedule_updates_each_transition_once_and_never_samples_endpoint(n,length,flags,monkeypatch):
    model,target,rule,policy=components()
    env,selected,written=ScriptedEnv(length,flags),[],[]
    act,update=model.act,rule.update

    def select(state):
        selected.append((env.time,len(written)));return act(state)

    def write(q,s,a,y):
        written.append(env.time);return update(q,s,a,y)

    monkeypatch.setattr(model,"act",select);monkeypatch.setattr(rule,"update",write)
    result=tree.run_episode(env,model,target,rule,target_policy=policy,n=n)
    assert result["steps"]==result["updates"]==length and len(selected)==length
    expected_times=[*range(n,length+1),*([length]*(n-1))] if length>=n else [length]*length
    assert written==expected_times
    assert selected==[(time,max(0,time-n+1)) for time in range(length)]


def test_selector_probabilities_share_one_q_estimate_with_tie_breaking_and_no_rng(monkeypatch):
    model,target,rule,_=components()
    rows=window([1,2,3]);seen=[];probabilities=model.action_selector.probabilities
    def probs(estimate):
        seen.append(estimate.values.q[0].clone());return probabilities(estimate)
    monkeypatch.setattr(model.action_selector,"probabilities",probs)
    rng=torch.get_rng_state().clone()
    tree.learn_tree_backup(model,rows,target,rule,target_policy=model.action_selector)
    assert len(seen)==1 and seen[0].shape==(3,2) and torch.equal(rng,torch.get_rng_state())
    assert torch.equal(probabilities(model.forward(torch.tensor([1]))),torch.tensor([[.5,.5]],dtype=torch.float64))


def test_checkpoint_restores_pending_fixed_policy_current_action_model_and_rng(tmp_path):
    model,target,rule,policy=components();policy[:]=policy.new_tensor([.7,.3])
    model.estimator.q_table.copy_(torch.arange(16).reshape(8,2)/5)
    pending=window([1,2]);torch.manual_seed(19);chosen=model.act(2)
    state={"model":copy.deepcopy(model.state_dict()),"policy":policy.clone(),"pending":[asdict(b) for b in pending],
           "action":chosen.action,"log_prob":chosen.log_prob,"rng":torch.get_rng_state(),"n":3,"gamma":.9,"alpha":.5}
    path=tmp_path/'tree.pt';torch.save(state,path)
    def finish(model,rows,target,rule,policy,action):
        rows.append(Transition(2,action,3.,3,True,False).to_batch());errors=[]
        while rows:
            errors.append(tree.learn_tree_backup(model,rows,target,rule,target_policy=policy));rows.popleft()
        return torch.cat(errors)
    errors=finish(model,pending,target,rule,policy,chosen.action);next_action=model.act(4)
    restored=torch.load(path,weights_only=True);other,t,r,_=components(gamma=restored["gamma"],alpha=restored["alpha"])
    other.load_state_dict(restored["model"])
    result=finish(other,deque(TransitionBatch(**b) for b in restored["pending"]),t,r,restored["policy"],restored["action"])
    assert torch.equal(errors,result) and torch.equal(model.estimator.q_table,other.estimator.q_table)
    torch.set_rng_state(restored["rng"]);resumed=other.act(4)
    assert torch.equal(resumed.action,next_action.action) and torch.equal(resumed.log_prob,next_action.log_prob)


@pytest.mark.parametrize("problem",["empty","model","shape","action","boundary","policy_shape","policy_negative","policy_nan","policy_sum","policy_device","selector_shape","selector_dtype","selector_nan","selector_sum","target_inf"])
def test_important_contract_errors_do_not_write(problem):
    model,target,rule,policy=components();rows=window([1]);before=model.estimator.q_table.clone()
    if problem=="empty":rows.clear()
    elif problem=="model":model=Model(IdentityRepresentation(),TabularVEstimator(8))
    elif problem=="shape":rows[0]=replace(rows[0],reward=torch.ones(1,1))
    elif problem=="action":rows[0]=replace(rows[0],action=torch.tensor([-1]))
    elif problem=="boundary":rows=window([1,2]);rows[0]=replace(rows[0],truncated=torch.tensor([True]))
    elif problem=="policy_shape":policy=policy[:1]
    elif problem=="policy_negative":policy[0]=policy.new_tensor([-.1,1.1])
    elif problem=="policy_nan":policy[0,0]=float("nan")
    elif problem=="policy_sum":policy[0]=.1
    elif problem=="policy_device":policy=policy.to('meta')
    elif problem.startswith('selector_'):
        value=torch.ones((1,3)) if problem=='selector_shape' else torch.ones((1,2),dtype=torch.float32 if problem=='selector_dtype' else torch.float64)
        if problem=='selector_nan':value[0,0]=float('nan')
        policy=SimpleNamespace(probabilities=lambda estimate:value)
    elif problem=="target_inf":rows[0]=replace(rows[0],reward=torch.tensor([float('inf')]))
    with pytest.raises((TypeError,ValueError,IndexError,FloatingPointError)):
        tree.learn_tree_backup(model,rows,target,rule,target_policy=policy)
    if hasattr(model.estimator,'q_table'):assert torch.equal(before,model.estimator.q_table)


def test_fixed_policy_float64_explicitly_casts_to_float32_q_and_is_unchanged():
    model,target,rule,policy=components(torch.float32)
    policy=policy.double();original=policy.clone()
    error=tree.learn_tree_backup(model,window([1]),target,rule,target_policy=policy)
    assert error.dtype==torch.float32 and torch.equal(policy,original)
