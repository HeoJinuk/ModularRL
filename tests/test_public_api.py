"""Concise imports are aliases, preserving the original types and state."""
import importlib
import pickle

import pytest
import torch

from modular_rl import Model, representations as R, estimators as E, selectors as S

ALIASES = {
    'representations': ('modular_rl.component.representation', {'Identity':'IdentityRepresentation','MLP':'MLPRepresentation'}),
    'estimators': ('modular_rl.component.estimator', {name:name+'Estimator' for name in (
        'TabularV','TabularQ','DiscreteQ','ContinuousQ','ActorCritic','StateValue',
        'DiscreteAdvantage','CategoricalPolicy','GaussianPolicy','DeterministicPolicy')}),
    'selectors': ('modular_rl.component.action_selector', {'Greedy':'GreedyActionSelector',
        'EpsilonGreedy':'EpsilonGreedyActionSelector','Categorical':'CategoricalActionSelector'}),
    'buffers': ('modular_rl.component.buffer', {'Replay':'ReplayBuffer','SequenceReplay':'SequenceReplayBuffer'}),
    'targets': ('modular_rl.component.target_generator', {'TD0':'TD0TargetGenerator',
        'NStep':'NStepTargetGenerator','MCReturn':'MonteCarloReturnGenerator'}),
    'updates': ('modular_rl.component.update_rule', {'TabularV':'TabularVUpdateRule',
        'TabularQ':'TabularQUpdateRule','TDLoss':'TDLossUpdateRule'}),
    'learners': ('modular_rl.learner', {'QLearning':'TabularQLearner','SARSA':'TabularSARSALearner',
        'DQN':'DQNLearner','MCPrediction':'MCPredictionLearner','MCControl':'TabularMCControlLearner',
        'OffPolicyMCPrediction':'OffPolicyMCPredictionLearner','OffPolicyMCControl':'OffPolicyMCControlLearner',
        'TD0Prediction':'TabularTD0PredictionLearner'}),
}

@pytest.mark.parametrize('namespace,old_path,short,long_name', [
    (namespace,old_path,short,long_name) for namespace,(old_path,names) in ALIASES.items()
    for short,long_name in names.items()])
def test_alias_is_the_original_class(namespace, old_path, short, long_name):
    public = importlib.import_module('modular_rl.'+namespace)
    original = getattr(importlib.import_module(old_path), long_name)
    assert getattr(public, short) is original
    assert original.__name__ == long_name
    assert short in public.__all__
    assert importlib.import_module(original.__module__) is not public


def test_alias_preserves_parameters_buffers_rng_forward_and_pickle():
    from modular_rl.component.representation import MLPRepresentation
    from modular_rl.component.estimator import DiscreteQEstimator
    from modular_rl.component.action_selector import EpsilonGreedyActionSelector
    from modular_rl.model import Model as OriginalModel
    assert Model is OriginalModel
    torch.manual_seed(17)
    old = OriginalModel(MLPRepresentation(3, (4,)), DiscreteQEstimator(4, 2), EpsilonGreedyActionSelector(.2))
    old_rng = torch.get_rng_state()
    torch.manual_seed(17)
    new = Model(R.MLP(3, (4,)), E.DiscreteQ(4, 2), S.EpsilonGreedy(.2))
    assert torch.equal(old_rng, torch.get_rng_state())
    for getter in ('state_dict','named_parameters','named_buffers'):
        left, right = dict(getattr(old,getter)()), dict(getattr(new,getter)())
        assert left.keys() == right.keys()
        for name in left: assert torch.equal(left[name],right[name])
    observation = torch.ones(5,3)
    assert torch.equal(old(observation).values.q[0],new(observation).values.q[0])
    torch.manual_seed(9); first=old.act(observation); old_rng=torch.get_rng_state()
    torch.manual_seed(9); second=new.act(observation)
    assert torch.equal(first.action,second.action) and torch.equal(first.log_prob,second.log_prob)
    assert torch.equal(old_rng,torch.get_rng_state())
    restored=pickle.loads(pickle.dumps(new))
    assert type(restored.estimator) is DiscreteQEstimator
    assert torch.equal(restored(observation).values.q[0],new(observation).values.q[0])


def test_documented_q_learning_composition():
    from modular_rl import buffers as B, learners as L, targets as T, updates as U
    model=Model(R.Identity(),E.TabularQ(16,4),S.EpsilonGreedy(.1))
    replay=B.Replay(10000,seed=7)
    replay.add(observation=0,action=1,reward=1.,next_observation=1,terminated=False,truncated=False)
    assert L.QLearning(model,T.TD0(.99),U.TabularQ(.5)).learn(replay.sample(1)).item()==1
    assert model.estimator.q_table[0,1].item()==.5
