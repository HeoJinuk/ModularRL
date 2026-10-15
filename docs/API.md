# 공개 API와 Learner 계약

`R/E/S/B/T/U/L/W/P`은 사용자가 선택하는 import 별칭이다. 공개 모듈 이름은 복수형이며,
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
`S.EpsilonGreedy is EpsilonGreedyActionSelector`다. 기존 라이브러리 클래스의 `__name__`, `__module__`,
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
|updates|TabularV, TabularQ, TDLoss, ActorCritic|
|world_models|Tabular|
|planners|Dyna|

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
|DQN / DoubleDQN|online, target_model, optimizer, target_generator, rule; learn(batch or sequence)|detached TDLossOutput|
|ActorCritic|model, optimizer, target_generator, rule; learn(fresh B=1 batch)|detached ActorCriticOutput|

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

학습·Planning 계산은 설치한 `modular_rl`만으로 실행한다. 기존 examples의 learn/plan 함수와
TabularWorldModel import는 호환 경로로 유지한다. World Model의 state_dict 필드와 pair 순서는
기존과 같고, 새 Pickle의 클래스 경로는 라이브러리 경로다. 예전 examples 경로의 객체 Pickle은
그 예제가 있는 checkout에서만 호환된다. 이식 가능한 저장은 state_dict를 사용한다.
수학의 독립 참조는 tests/_legacy_td(dc2ecdc), tests/_legacy_planning.py(32634ce)에 보존한다.
Runner/Agent, Prioritized Replay, 연속 관측 Learned World Model, Neural SARSA/SAC는 아직 없다.

## World Model과 Dyna Planner

```python
from modular_rl import world_models as W, planners as P
from modular_rl.data import Transition

world = W.Tabular()
world.observe(Transition(0, 1, 1., 2, False, True))
planner = P.Dyna(world, learner, batch_size=1, seed=7)
result = planner.plan(steps=5)
# result: requested_updates, updates, results, batches
```

`W.Tabular`는 `component.world_model.TabularWorldModel`의 alias다. 관측 `(state,action)`별
최신 결정적 전이를 owned CPU snapshot으로 저장한다. 정수 상태·행동 ID, floating scalar reward와
실제 자연 종료를 보존한다. 외부 truncation과 behavior log_prob은 예측하지 않는다. 미관측 pair를
예측하지 않으며 반복 관측은 기존 pair의 삽입 순서를 유지한 채 덮어쓴다. `sample(rng)`은 관측
pair를 균등 샘플링한다. `reset`, `len`, `state_dict`, `load_state_dict`가 기존 계약을 유지한다.

`P.Dyna`는 `component.planner.DynaPlanner`의 alias다. 생성자는
`(world_model, learner, *, batch_size=1, seed=None, rng=None)`이다. seed와 외부 NumPy Generator
rng 중 하나를 사용한다. World 경계는 `len(world)`와 `sample(rng)->Transition`, Learner 경계는
`learn(TransitionBatch)`다. 정식 Base Class나 Factory는 추가하지 않는다.

`plan(steps=1, *, record_batches=False)`의 `updates`는 실제 learn 호출 수다. 기본 B=1은 가상
전이당 한 번의 Tabular Q 갱신이고, DQN의 B개 복원 추출은 한 번의 Gradient Update다. `results`는
Learner의 반환값을 순서대로 보존한다. `batches`는 명시적 QA 옵션에서만 기록한다. 빈 World나
steps=0이면 학습과 RNG 소비가 없다. 환경 step/observe/Replay/행동 선택/ε 변경/Target Sync를
수행하지 않는다. Real과 Planning에 같은 Learner를 전달한다. Target Sync는 계속 외부 스케줄이다.
World/Model/Optimizer, planner.rng.bit_generator.state와 수집 상태를 별도로 저장한다.
현재 Tabular World Model은 연속 CartPole 관측을 예측하는 모델이 아니다.

## Double DQN

```python
learner = L.DoubleDQN(online, target_model, optimizer, T.TD0(.99), U.TDLoss())
learner.sync_target()
result = learner.learn(batch)
# T.NStep(.99) + SequenceBatch도 동일 생성자/learn 경로를 사용한다.
```

긴 이름은 `learner.DoubleDQNLearner`다. DQNLearner를 상속해 optimizer, optimize, sync_target,
입력 검증, n-step 보상 계산과 마지막 유효 관측 선택을 재사용한다. 차이는 Bootstrap뿐이다.
Online argmax로 행동을 선택하고 Target의 해당 Q를 gather한다. Online=[1,4], Target=[10,2]면
DQN은 10, Double DQN은 2를 사용한다. 두 평가는 no_grad이고 Target은 동결/eval이다.
Online Bootstrap은 임시 eval로 BatchNorm 통계와 Dropout RNG를 보존한 뒤 모든 하위 모듈의
원래 모드를 복원한다. 현재 상태 prediction은 호출자의 모드를 따른다. 기본 DQN의 연산 순서와
Target max 경로는 유지한다. 복수 Critic/Distributional Q는 임의로 선택하지 않고 실패한다.
이것은 Double DQN이며 Dueling Network를 뜻하지 않는다.

## 1-step Actor-Critic

```python
model = Model(R.MLP(4, hidden_dims=(64,)),
              E.ActorCritic(E.CategoricalPolicy(64, 2), E.StateValue(64)), S.Categorical())
optimizer = torch.optim.Adam(model.parameters(), lr=.001)
learner = L.ActorCritic(model, optimizer, T.TD0(.99),
                        U.ActorCritic(value_coef=.5, entropy_coef=.01))
selected = model.act(observation)
# env.step(selected.action.item())의 실제 결과로 Transition을 만든 직후:
result = learner.learn(transition.to_batch())
```

긴 이름은 `ActorCriticLearner`, `ActorCriticUpdateRule`이며 기존 긴 경로로도 import한다.
첫 공식 범위는 Categorical 이산 Policy와 Scalar V, **새로 수집한 B=1의 즉시 On-policy 갱신**이다.
B>1은 거절한다. Batch만으로 정책의 최신성을 검사할 수 없으므로 Replay 재사용 금지는 호출자
책임이다. 정책·가치·공유 Representation에는 같은 Optimizer를 사용한다. Learner는 환경·Replay나
Target Network를 소유하지 않는다. 실제 최종 관측의 V에서 no_grad TD0 target을 만들고 자연 종료만
Bootstrap을 막는다. 다음 상태 평가는 임시 eval 후 모드를 복원한다. 현재 상태 forward는 호출자
모드를 사용한다. 상태를 바꾸는/확률적인 사용자 Network는 행동 수집과 재평가의 정책 일치를
호출자가 보장해야 한다. 예제는 Dropout/BatchNorm 없는 MLP를 사용한다.

UpdateRule의 공개 메서드는 `compute(log_prob, value, target, entropy)`다. 모두 같은 비어 있지
않은 floating `(B,)` Tensor이며 shape/dtype/device를 검사해 Broadcasting을 막는다.
`delta=target-value`, `actor_loss=mean(-log_prob*delta.detach())`,
`critic_loss=mean(.5*delta**2)`, `total_loss=actor_loss+value_coef*critic_loss-entropy_coef*mean(entropy)`다.
계수는 유한한 0 이상의 실수다. Rule 자체는 Critic/Target graph를 유지하며, Learner가 no_grad
Target을 공급하고 zero_grad/backward/step한다. 수집 당시 detached log_prob을 학습에 직접 쓰지
않고 현재 Categorical에서 선택한 행동의 log_prob을 재계산한다.

`data.ActorCriticOutput`은 frozen dataclass다. actor_loss, critic_loss, entropy, total_loss는
scalar Tensor이고 signed td_error는 `(B,)`다. dtype/device는 모델을 따른다. Rule 출력은 graph를
유지하고 Learner 출력은 전부 detach된 업데이트 직전 지표다. Model과 Optimizer state_dict,
행동·환경 RNG를 함께 복원한다. 연속 행동, A3C, PPO, GAE는 이번 범위에 포함하지 않는다.
Vector Rollout의 A2C는 아래 별도 Learner를 사용한다.

## Vector Rollout and A2C

```python
buffer = B.Rollout(horizon=5, n_envs=4)
buffer.add(observation, action, reward, actual_next_observation, terminated, truncated,
           value=collected_value, log_prob=collected_log_prob)
# 위 add를 정책 파라미터 변경 없이 5번 호출한다.
rollout = buffer.get()
learner = L.A2C(model, optimizer, T.RolloutReturn(.99),
                U.ActorCritic(value_coef=.5, entropy_coef=.01))
result = learner.learn(rollout)
buffer.clear()
```

긴 이름은 `RolloutBuffer`, `RolloutReturnGenerator`, `A2CLearner`이며
`data.RolloutBatch`를 사용한다. Replay/SequenceReplay 및 기존 Learner의 API는 그대로다.

|필드|RolloutBatch shape / dtype|
|---|---|
|observation / next_observation|`(T,N,*obs_shape)`, 원래 numeric dtype|
|action|`(T,N)`, integer; Learner가 long으로 정규화|
|reward|`(T,N)`, floating|
|terminated / truncated|`(T,N)`, bool, 독립 보존|
|value / log_prob|`(T,N)`, 수집 시점 floating Snapshot, 필수|

Buffer는 detached owned CPU 복사본을 저장하고 순서·shape·dtype을 확인한다. 경계가 아닌 환경은
이전 next_observation과 다음 observation이 이어져야 한다. 경계에서는 Reset 관측으로 넘어갈 수 있다.
get은 정확히 horizon개 vector step을 stack하고 독립 결과를 반환한다. 자동 clear·무작위 sample·
오래된 경험 덮어쓰기는 없다. len은 vector step 수이며 개별 환경 전이 수는 len*N이다.
`RolloutBatch.to(device, dtype=...)`는 floating 필드만 변환하고 integer/bool을 보존한다.
Buffer나 Batch는 Return·환경·Optimizer를 소유하지 않는다.

`T.RolloutReturn(gamma).generate(reward, next_value, terminated, truncated)`의 모든 입력과 출력은
동일한 `(T,N)` shape/device다. reward/next_value는 같은 floating dtype, 두 Mask는 bool이다.
next_value는 각 전이의 **실제** 다음 관측 V다. Generator는 역순으로 Return을 계산한다:
계속되는 구간은 다음 Return, truncated-only는 해당 final V, 자연 종료는 0으로 Bootstrap한다.
마지막 계속 구간은 next_value[-1]을 사용한다. Truncation 후 새 Episode의 보상은 연결하지 않는다.
Generator 자체는 graph를 보존하며 no_grad/detach는 Learner 책임이다.

`L.A2C(model, optimizer, target_generator, update_rule).learn(rollout)`은 T*N을 flatten하여
Categorical Policy와 scalar V를 재평가하고, eval/no_grad의 next V로 Return을 고정한다.
기존 U.ActorCritic의 mean Loss와 단일 zero_grad/backward/step을 사용한다.
반환 `ActorCriticOutput`의 네 지표는 detached scalar, signed `td_error=return-current_value`는
detached `(T,N)`이며 Model dtype/device를 따른다. 기존 Rule의 `(B,)` 출력과 B=1 AC 반환은 바꾸지 않는다.

수집 도중·수집 이후 첫 learn 전까지 파라미터를 고정하고 Rollout을 한 번만 학습한다.
Learner는 현재 V/log_prob과 수집 Snapshot을 rtol=1e-5, atol=1e-6으로 비교해 관측 가능한 변경을
거절한다. 이는 모든 파라미터 변경이나 데이터 출처를 인증하는 장치가 아니다. 호출자는 정책 최신성을
보장해야 한다. Dropout/BatchNorm에는 일관된 eval Forward를 권장한다(eval에서도 gradient는 계산된다).
저장 log_prob과 value는 Loss의 gradient 원천이 아니다.

### SAME_STEP 수집 경계

`environment.validate_vector_env(env)`를 수집 전에 호출한다. 동일한 Box 관측·0-based Discrete
행동과 SAME_STEP만 지원하고 NEXT_STEP/DISABLED는 거절한다. Gymnasium >=1.1이 필요하며
1.4.0에서 검증했다. 환경 constructor는 workers 간 동일 space를 검사한다.

`environment.final_observations(returned, terminated, truncated, info)`는 실제 다음 관측 배열의
복사본을 만든다. 경계에는 올바른 `final_obs`/`_final_obs`가 필수이며 누락 시 Reset 관측으로 대체하지
않고 실패한다. returned는 다음 행동에, helper 결과는 Buffer에 전달한다. Buffer는 info를 해석하지 않는다.
이는 [Gymnasium SAME_STEP 계약](https://farama.org/Vector-Autoreset-Mode)을 따른다.

수집자는 기본 copy=True를 유지하거나 env.step 전에 이전 관측을 별도로 복사해야 한다.
copy=False의 공유 관측이 이미 덮어써진 뒤 Buffer에 넣으면 이전 상태를 복구할 수 없다.
환경은 수집자가 try/finally로 닫는다. Worker 예외 시 `env.close(terminate=True)`로 살아 있는
Worker도 종료한 뒤 예외를 다시 발생시킨다. Async 환경 실행과 A2C의 동기식 업데이트를 구분한다.
Model/Optimizer와 환경·Torch RNG, 수집 진행 상태가 checkpoint의 책임이며 Rollout만으로
학습을 재개하지 않는다. GAE/PPO·연속 행동·Recurrent state·부분 Rollout·NEXT_STEP은 아직 없다.
