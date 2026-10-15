# 공개 API와 Learner 계약

`R/E/S/B/T/U/L`은 사용자가 선택하는 import 별칭이다. 공개 모듈 이름은 복수형이며,
짧은 이름은 기존 클래스 자체를 가리킨다. 이름 변경용 상속이나 Wrapper는 없다.

```python
from modular_rl import Model
from modular_rl import representations as R, estimators as E, selectors as S
from modular_rl import buffers as B, targets as T, updates as U, learners as L

model = Model(R.Identity(), E.TabularQ(16, 4), S.EpsilonGreedy(.1))
learner = L.ExpectedSARSA(model, T.TD0(.99), U.TabularQ(.5))
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

## 사용할 수 있는 Learner

아래의 `model, target, rule`은 알고리즘에 맞는 Model/TargetGenerator/UpdateRule이다.
별도 언급이 없으면 생성자 앞 세 인자는 이 순서다. 공통 BaseLearner는 없다.

|이름|생성자 추가 인자 / learn 입력|결과|
|---|---|---|
|QLearning|learn(batch)|signed TD error `(B,)`|
|SARSA|learn(batch, *, next_action=...)|signed TD error `(B,)`|
|ExpectedSARSA|learn(batch); Model 자신의 Selector가 probabilities 제공|signed TD error `(B,)`|
|DoubleQ|model1, model2, target, rule, *, action_selector=None; learn(batch, *, update_first=bool)|signed TD error `(B,)`|
|NStepSARSA|learn(window, *, next_action=None)|head signed TD error `(1,)`|
|OffPolicyNStepSARSA|생성자 `target_policy=`; learn(window, *, bootstrap_action=None, bootstrap_log_prob=None)|td_error, raw_td_error, target, effective_target, importance_ratio dict|
|TreeBackup|생성자 `target_policy=`; learn(window)|head signed TD error `(1,)`|
|QSigma|생성자 `target_policy=`; learn(window, *, sigma=...)|target, td_error, importance_ratios, coefficients dict|
|TD0Prediction|Tabular V Model; learn(batch)|signed TD error `(B,)`|
|MCPrediction|model, return_generator, visit_mode="first"; learn(episode)|Returns `(T,)`|
|MCControl|model, return_generator, visit_mode="first"; learn(episode)|Returns `(T,)`|
|OffPolicyMCPrediction|model, return_generator, target_policy, method="weighted", visit_mode="first"; learn(episode)|Returns `(T,)`|
|OffPolicyMCControl|model, return_generator; learn(episode)|Returns `(T,)`|
|DQN|online, target_model, optimizer, target_generator, rule; learn(batch or sequence)|detached TDLossOutput|

QLearning/SARSA/ExpectedSARSA/DoubleQ와 네 Window Learner는 E.TabularQ Model을 사용한다.
TD0Prediction/MCPrediction/OffPolicyMCPrediction은 E.TabularV, MCControl/OffPolicyMCControl은
E.TabularQ를 사용한다. DQN은 하나의 Scalar Discrete Q를 출력하는 Neural Model을 사용한다.

Tabular Learner는 원본 테이블을 직접 갱신하고 Optimizer를 사용하지 않는다. DQN은 Online
prediction의 Gradient로 학습하고 Optimizer를 소유한다. Target 평가는 no_grad/eval이며
`sync_target()`과 `optimize()`의 기존 경계를 유지한다. 환경과 Buffer는 어느 Learner에도
전달하지 않는다. TargetGenerator/UpdateRule의 공개 API는 그대로다.

## 복수 Model과 시간순 데이터

```python
model1 = Model(R.Identity(), E.TabularQ(16, 4))
model2 = Model(R.Identity(), E.TabularQ(16, 4))
learner = L.DoubleQ(model1, model2, T.TD0(.99), U.TabularQ(.1),
                    action_selector=S.EpsilonGreedy(.1))
selected = learner.act(0)  # 기존 Selector로 Q1+Q2에서 선택
# 어느 테이블을 갱신할지 결정하는 별도 RNG는 호출자가 관리한다.
error = learner.learn(batch, update_first=True)
```

DoubleQ의 두 Model은 shape/dtype/device가 같고 storage가 독립적이어야 한다.
`estimate(observation)`는 명시적인 Q 합을 출력하며 세 번째 테이블을 만들지 않는다.
`action_selector`를 생략하면 model1의 Selector를 사용한다. Selector가 없어도 학습은 가능하고,
그때 act()는 명확하게 실패한다. learn()은 행동 선택이나 동전 RNG를 실행하지 않는다.

Online multi-step의 window는 연속적인 owned `TransitionBatch` `(1,)` 행의 list/deque다.
Learner는 head 하나만 갱신하고 window를 pop하지 않는다. 다음 행동의 사전 선택·재사용,
window 진행과 terminal/truncation Tail Flush는 호출자의 명시적 책임이다.
무작위 SequenceReplay를 Online SARSA window로 자동 해석하지 않는다.

```python
learner = L.NStepSARSA(model, T.TD0(.99), U.TabularQ(.1))
error = learner.learn(window, next_action=selected_endpoint.action)
# 성공 후 caller가 window.popleft(); 같은 endpoint 행동으로 다음 flush 호출
```

기존 Online 계산은 TD0를 보상 역순으로 조합한다. Reward Sum과 Bootstrap을 분리한 NStep의
산술 재배열로 인해 기존 float32 결과를 바꾸지 않는다. SARSA의 실제 endpoint 행동과
Off-policy SARSA의 endpoint 행동 확률은 필수 정보다. 후자는 head를 제외하고 중간 행동과
비종료 endpoint의 pi/b를 곱한 effective target을 사용한다. 기록된 log_prob의 원래 정밀도를
보존하며 현재 Behavior 정책으로 재계산하지 않는다. Coverage는 수집자 책임이다.

TreeBackup/QSigma는 fixed `(S,A)` Target Tensor 또는 discrete Selector를 받는다.
Fixed 정책을 바꾸거나 외부에서 Tensor를 수정하지 않는 것은 호출자 책임이다. Selector를 쓰면
한 estimate의 Q와 정책 확률로 계산한다. 둘 모두 endpoint 행동을 샘플링하지 않는다.
QSigma의 sigma는 수집 때 확정한 scalar 또는 m개 `[sigma_(t+1),...,sigma_(t+m)]` 값이다.
마지막 값은 검증되지만 horizon residual이 0이므로 Return에 영향을 주지 않는다.
본문 식 (7.17)의 비종료 경계는 `R_h+gamma*Vpi(S_h)`, 자연 종료는 `R_T`다.
σ=0은 TreeBackup, n=1은 Expected SARSA이고, σ=1은 Control Variate를 포함한다.
의사코드와의 경계 차이를 공식 정오표에서 인정된 오류로 단정하지 않는다.
별도 전체 Importance Ratio를 다시 곱하지 않는다. 기존 수치 실패의 무기록 계약을 유지한다.

## Neural replay와 Dyna

```python
learner = L.DQN(online, target_model, optimizer, T.NStep(.99), U.TDLoss())
learner.sync_target()
sequence = B.SequenceReplay(10000, seed=7)
# 시간순 add 후:
result = learner.learn(sequence.sample(batch_size=32, length=3))
```

DQN의 네 번째 인자로 TD0와 NStep을 모두 허용한다. NStep을 주면 그 객체를 sequence 계산에
사용하며 같은 gamma의 TD0를 1-step 경로용으로 구성한다. 기존 `TD0` + keyword-only
`n_step_target_generator=` 호출도 유지한다. 두 gamma는 같아야 한다.
Real n-step과 Planning 1-step은 같은 Learner/Online/Target/Optimizer를 공유할 수 있다.
Target Sync 주기는 외부 환경 Step 기준 스케줄이다.

여섯 승격 알고리즘은 설치한 `modular_rl`만으로 계산할 수 있다. 기존 examples의 learn 함수는
호환 위임 함수이며, 수학의 독립 참조는 tests/_legacy_td에 dc2ecdc 기준으로 보존한다.
Dyna의 TabularWorldModel/plan과 실험 루프는 계속 examples 지원이다. WorldModel/Planner의
공용 API, Runner/Agent, Prioritized Replay, Neural SARSA/Actor-Critic/SAC Learner는 아직 없다.
새 이름을 위해 기존 계산이나 Experiment 예산을 변경하지 않는다.
