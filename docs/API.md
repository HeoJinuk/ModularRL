# 공개 API와 Learner 계약

`R/E/S/B/T/U/L`은 사용자가 선택하는 import 별칭이다. 공개 모듈 이름은 복수형이며,
짧은 이름은 기존 클래스 자체를 가리킨다. 이름 변경용 상속이나 Wrapper는 없다.

```python
from modular_rl import Model
from modular_rl import representations as R, estimators as E, selectors as S
from modular_rl import buffers as B, targets as T, updates as U, learners as L

model = Model(R.Identity(), E.TabularQ(16, 4), S.EpsilonGreedy(.1))
learner = L.QLearning(model, T.TD0(.99), U.TabularQ(.5))
replay = B.Replay(10000, seed=7)
replay.add(observation=0, action=1, reward=1., next_observation=1,
           terminated=False, truncated=False)
error = learner.learn(replay.sample(1))
```

`import modular_rl.estimators as E`도 같다. `S`를 Selector 모듈에 쓰는 코드에서는 상태 수를
`N_STATES`처럼 구분한다. 기존 내부 경로와 긴 이름을 계속 사용할 수 있으며,
`S.EpsilonGreedy is EpsilonGreedyActionSelector`다. 원래 클래스의 `__name__`, `__module__`,
Parameter/Buffer 등록, state_dict key, Pickle 경로와 RNG 사용은 바뀌지 않는다.
Model의 생성자, forward/act/evaluate_q/encode와 데이터 출력 필드도 유지한다.

## 컴포넌트 이름

|공개 모듈|이름|
|---|---|
|representations|Identity, MLP|
|estimators|TabularV, TabularQ, StateValue, DiscreteQ, ContinuousQ, DiscreteAdvantage, ActorCritic, CategoricalPolicy, GaussianPolicy, DeterministicPolicy|
|selectors|Greedy, EpsilonGreedy, Categorical|
|buffers|Replay, SequenceReplay|
|targets|TD0, NStep, MCReturn|
|updates|TabularV, TabularQ, TDLoss|

Estimator는 추정 결과를 출력한다. Selector는 그 결과에서 행동을 선택하고 sampling-time
log_prob을 반환한다. `probabilities(estimate)`는 행동을 샘플링하거나 RNG를 소비하지 않는다.
TargetGenerator는 주어진 값의 수식만 계산한다. Model 호출·Bootstrap 선택·Gradient 경계는
Learner가 맡는다. UpdateRule은 테이블 갱신 또는 Loss 계산을 담당한다.

기존 Learner의 짧은 이름은 QLearning, SARSA, DQN, TD0Prediction, MCPrediction, MCControl, OffPolicyMCPrediction, OffPolicyMCControl이다.
