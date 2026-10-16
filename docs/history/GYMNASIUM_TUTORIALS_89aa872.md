# Historical record at 89aa872

This is a preserved development record, not the current API specification. See [current API](../API.md) and [architecture](../ARCHITECTURE.md).

# Gymnasium 알고리즘 Notebook과 API 사용 가이드

현재 Conda 환경에 Notebook 의존성을 설치한 뒤 실행한다.

```sh
python -m pip install -e '.[notebooks]'
python -m jupyter lab notebooks/gymnasium
```

기본값은 실제 학습 예산을 사용하는 **FULL**이다. API·Shape 연결만 확인하려면
`MODULARRL_FAST=1`로 Jupyter를 실행한다. 각 Notebook은 독립 Kernel에서 Model,
Learner와 RNG를 새로 구성한다. 같은 Kernel에서 처음부터 Run All해도 상태를 다시
초기화한다. FAST의 성공률은 학습 성능 판정 기준이 아니다.

## 짧은 공개 API

```python
from modular_rl import Model
from modular_rl import representations as R, estimators as E, selectors as S
from modular_rl import buffers as B, targets as T, updates as U, learners as L
model = Model(R.Identity(), E.TabularQ(16, 4), S.EpsilonGreedy(.1))
learner = L.ExpectedSARSA(model, T.TD0(.99), U.TabularQ(.5))
```

기존 긴 이름과 동일한 클래스다. [공개 API 계약](../API.md)에 모든 Learner의 생성자·입력·결과와
DoubleQ.act, ordered window, Real n-step/Planning 1-step을 정리했다. 상태/행동 수 변수는
N_STATES/N_ACTIONS로 구분해 Selector 모듈 S와 충돌하지 않게 한다. DLFS4 04–08의 소스는
변경하지 않고 레거시 회귀 기준으로 유지한다.

## Notebook 목록과 지원 범위

|Notebook|환경|학습·계산 계약|
|---|---|---|
|00_api_basics|FrozenLake|직접 step, add/sample/learn; 기존·간편 Replay 호출과 Sequence|
|01_dp|FrozenLake|알려진 kernel의 Policy Evaluation, Policy/Value Iteration|
|02_mc_prediction|FrozenLake|First/Every V; 고정 Target의 Ordinary/Weighted IS|
|03_mc_control|FrozenLake|On-policy Sample Average Q, Off-policy Weighted-IS Q|
|04_td_prediction|FrozenLake|Online B=1 V Prediction|
|05_q_learning|FrozenLake|Tabular max-Q와 무작위 Replay|
|06_sarsa|FrozenLake|갱신 전에 다음 행동을 한 번 선택하고 그대로 실행|
|07_expected_sarsa|FrozenLake|L.ExpectedSARSA, 같은 Selector의 probabilities|
|08_double_q|FrozenLake|독립 Model 두 개, 명시적인 Q 합과 업데이트 동전|
|09_n_step_sarsa|FrozenLake|Online Window, 실제 선택한 Horizon 행동|
|10_off_policy_n_step|FrozenLake|고정 Target Q Prediction, 수집 당시 Behavior 확률|
|11_tree_backup|FrozenLake|고정 Target Q Prediction, Horizon 행동 추가 샘플링 없음|
|12_q_sigma|FrozenLake|고정 Target Prediction, 교재 본문 식 (7.17) 경계|
|13_dyna_q|FrozenLake|관측 pair 모델, 실제·가상 경험에 같은 Tabular Learner|
|14_dyna_dqn|FrozenLake|정수 ID와 One-hot MLP; Planning에도 같은 DQN/Optimizer|
|15_dqn|CartPole|1-step Replay, 명시적인 Warm-up·Epsilon·Target 스케줄|
|16_n_step_dqn|CartPole|SequenceReplay -> NStepTargetGenerator -> DQNLearner.learn|
|17_double_dqn|CartPole|DQN/Double DQN의 환경 step·Gradient Update·Sync 예산 통제|
|18_actor_critic|CartPole|Categorical Policy/Scalar V, B=1 즉시 On-policy 학습|
|19_a2c|CartPole|Async SAME_STEP, 시간 우선 Rollout, 경계별 Multi-step Return, T*N당 1회 학습|
|20_gae|CartPole|실제 Rollout, lambda=0/.5/.95/1 Advantage, TD0/Return 동등성, Truncation Trace|
|21_ppo|CartPole|Async SAME_STEP, GAE, 고정 Snapshot, Multi-epoch Minibatch PPO, KL/Clip 진단|

19는 SAME_STEP을 지원하는 Gymnasium >=1.1이 필요하다(이번 검증 1.4.0).
기존 환경이 1.0이면 `python -m pip install 'gymnasium>=1.1'`로 새 Vector 실습 의존성을 준비한다.

FrozenLake는 `map_name="4x4", is_slippery=False`다. 행동 ID는 Left/Down/Right/Up이다.
알려진 P는 DP와 참조값 진단에만 사용하며, Dyna 모델에는 관측한 전이만 기록한다.
Prediction은 지정 정책을 평가한다. V에서 유도한 Greedy 화살표는 별도 진단으로 표시한다.
CartPole은 Floating Observation과 Scalar Discrete Action을 사용한다.

DP, MC/TD Prediction·Control, Q-learning, SARSA, DQN, Expected SARSA, Double Q,
Online n-step SARSA, Off-policy n-step SARSA, Tree Backup, Q(σ)는 라이브러리 API다.
이들의 학습 계산은 Notebook에서 공식 Learner를 구성해 호출한다. Dyna의
World Model/Planner도 `W.Tabular`/`P.Dyna`의 설치형 API다. Double DQN과 1-step Actor-Critic도
공식 Learner다. A2C도 B.Rollout/T.RolloutReturn/U.ActorCritic을 조합한 공식 Learner다.
실험 수집·스케줄은 외부 루프에 둔다. Return 수식은 복사하지 않는다.
Tree Backup/Q(σ)는 여기서 고정 정책의
평가다. Q(σ)는 교재 본문 식 (7.17)의 경계 정의를 따르며 의사코드와의 차이를 기록한다.
공식 정오표에서 인정된 오류라고 단정하지 않는다.

연속 관측을 예측하는 Learned World Model, CartPole Dyna-DQN, Prioritized Replay,
All-action Planning은 구현된 API가 아니다.

## Replay 입력과 소유권

```python
buffer.add(transition)
buffer.add(transition=transition)
buffer.add(state, action, reward, next_state, terminated, truncated)
buffer.add(observation=state, action=action, reward=reward,
           next_observation=next_state, terminated=terminated,
           truncated=truncated, log_prob=selected.log_prob)
```

모두 같은 검증·Snapshot 경로를 사용한다. NumPy/Tensor dtype을 보존하고 Graph가 없는
owned CPU Tensor로 저장한다. log_prob은 선택적이지만 한 Buffer에서 유무를 섞지 않는다.
Transition은 Episode와 Online B=1 갱신에도 필요하므로 유지한다. 누락·중복·잘못된 혼합
호출은 저장소와 RNG를 변경하기 전에 거절한다. 기존 Replay의 Sampling RNG는 유지한다.

## Sequence 학습

```python
sequence_buffer = B.SequenceReplay(10000, seed=7)
sequence_buffer.add(state, action, reward, next_state, terminated, truncated)
if sequence_buffer.num_sequences(length=3) >= 32:
    batch = sequence_buffer.sample(32, length=3)
    result = learner.learn(batch)  # DQNLearner
```

Capacity는 전이 수다. 단일 순차 환경의 시간 순서와 관측 연속성을 확인하고, 유효한 시작점을
균등·비복원 샘플링한다. 열린 짧은 Tail은 후보가 아니다. 실제 terminated/truncated 경계가
있을 때만 짧은 Tail을 확정한다. Ring 상에서 덮어쓴 전이나 다음 Episode를 이어 붙이지 않는다.
`num_sequences(length)`는 RNG를 소비하지 않고 후보 수를 알려준다. `clear()`는 저장·Schema를
초기화하지만 RNG를 되감지 않는다.

|SequenceBatch 필드|Shape와 dtype|
|---|---|
|observation, next_observation|(B,T,*observation_shape), 원본 dtype|
|action|(B,T) 정수 또는 (B,T,A) Floating|
|reward, 선택적 log_prob|(B,T), 원본 dtype|
|terminated, truncated, valid|(B,T), bool|
|horizon|(B,), long|

valid는 행마다 비어 있지 않은 Prefix이고 horizon은 그 길이다. Padding은 0이며 경험으로
취급하지 않는다. `.to(device, dtype=...)`는 Floating 필드만 cast하고 정수·bool을 보존한다.
Buffer는 Target을 계산하지 않는다. 실제 마지막 관측을 Reset 전에 제공하는 것은 수집자의 책임이다.

```python
learner = L.DQN(online, target, optimizer, T.NStep(.99), U.TDLoss())
```

두 Generator의 gamma는 생성 시 일치해야 한다. 기존 다섯 인자 DQN 생성도 지원하며,
n-step Generator를 생략하면 같은 gamma로 기본 구성한다.

`NStepTargetGenerator.generate(rewards, next_value, terminated, valid)`는 (B,T)의
보상·종료·Mask와 (B,)의 외부 Bootstrap Value를 받는다. 동일 Device, 같은 Floating dtype을
요구하며 자동 Broadcasting하지 않는다. 실제 Horizon m에 대해 다음을 계산한다.

`sum(k=0..m-1, gamma**k * reward[k]) + gamma**m * (1-last_terminated) * next_value`

마지막 자연 종료만 Bootstrap을 제거한다. Truncation은 Bootstrap을 유지하고 Padding은
NaN/Inf를 포함해 무시한다. Gradient를 보존하며 no_grad/detach는 호출자 책임이다.
단위 테스트에서 n=1의 TD0 일치, gamma=.9·보상 [1,2,3]·next_value=4의 **8.146**,
Short Horizon·종료·dtype·Gradient를 확인한다.

DQN은 첫 State/Action의 Online Q를 gather하고 마지막 유효 next_observation에서
**현재 Target Network**를 평가한다. n-step Target을 만든 뒤 기존 optimize를 호출한다.
Bootstrap Q를 영구 저장하지 않는다. 같은 Learner/Optimizer로 Planning 1-step도 학습할 수 있고,
Target 동기화는 호출자가 결정한다. 기존 예제의 NStepExperience/보상합 경로는 회귀 기준으로 유지한다.

이 경로는 Behavior가 수집한 중간 보상과 max-Q 끝점을 사용하는 **uncorrected n-step backup**이다.
Importance Sampling을 수행하거나 무작위 Replay를 On-policy SARSA로 바꾸지 않는다.
Tree Backup/Q(σ)의 중간 Q·정책·σ·Importance 정보와 Online 행동 타이밍도 자동 통합하지 않는다.

## 실제 사용에서 발견한 사항

- 간편 add는 Transition 중첩을 없애고 환경 step 결과와 직접 대응한다.
- 1-step/Sequence DQN은 learn과 Optimizer를 공유한다.
- MC의 Episode, Online B=1의 Transition.to_batch는 여전히 명시적이다. 각 방법의 데이터 의미가 드러난다.
- Expected SARSA/Double Q/multi-step은 공식 Learner를 직접 구성한다. 시간순 Window,
  실제 endpoint 행동, sigma/log_prob 기록과 Flush는 계속 호출자 책임이다. BaseLearner는 없다.
- On-policy의 다음 행동 선택·Flush·σ 메타데이터는 알고리즘별 책임이다.
- 환경·탐색·동기화 스케줄과 Buffer RNG/Window Checkpoint는 호출자 책임이다.
  SequenceReplayBuffer의 공개 Checkpoint API는 이번 범위에 없다.
- Sequence 후보 검색은 query/sample마다 O(capacity)다. 큰 Replay에서는 성능 검토가 필요하다.
  간편 add는 긴 Runtime Signature 대신 문서의 Transition 필드 순서를 따른다.
- 각 Notebook 마지막에는 사용자가 수정할 수 있는 API 평가 질문 10개가 있다.

실행 중 두 가지 경계 변환을 확인했다. FrozenLake가 정수 보상을 반환할 수 있어 수집 시
`float(reward)`를 명시했다. Discrete Space의 `.n`이 NumPy 정수일 수 있으므로 Python int를
요구하는 기존 MLP 생성자에는 `int(space.n)`을 전달한다. 기존 공개 API는 바꾸지 않았다.

## 검증 결과

수학 단위 테스트, FAST API 연결 확인, FULL 학습 성능은 서로 구분한다.

신규 17개 모두 독립 Kernel에서 **같은 Kernel의 Run All 두 번**을 통과했다. 텍스트·수치·
그래프 PNG가 일치했다. Jupyter stdout의 메시지 분할 경계는 합쳐서 실제 텍스트를 비교한다.
추가 임시 검사 셀로 Episode 기록, Loss/Epsilon/Update 곡선, 평가 Return, RMSE, V/Q 테이블의
전체 정밀도 원자료도 비교했다. 이 검사 셀은 전달 Notebook에 남기지 않는다. PNG는 실행당 총 35개다.

05/15/16에는 실제 FULL 출력, 다른 14개에는 검증한 FAST 출력을 저장했다.
각 출력에 실행 Mode를 표시한다. 모든 Notebook의 FULL 실행을 검증했다는 의미는 아니다.
기본 FULL 모드로 실행하면 FAST 저장 결과를 실제 학습 예산의 결과로 바꿀 수 있다.

|Notebook|FAST 상태|Run All 횟수|실행당 PNG|저장 Mode|
|---|---|---|---|---|
|00_api_basics.ipynb|PASS|2|2|FAST|
|01_dp.ipynb|PASS|2|2|FAST|
|02_mc_prediction.ipynb|PASS|2|3|FAST|
|03_mc_control.ipynb|PASS|2|4|FAST|
|04_td_prediction.ipynb|PASS|2|2|FAST|
|05_q_learning.ipynb|PASS|2|2|FULL|
|06_sarsa.ipynb|PASS|2|2|FAST|
|07_expected_sarsa.ipynb|PASS|2|2|FAST|
|08_double_q.ipynb|PASS|2|2|FAST|
|09_n_step_sarsa.ipynb|PASS|2|2|FAST|
|10_off_policy_n_step.ipynb|PASS|2|2|FAST|
|11_tree_backup.ipynb|PASS|2|2|FAST|
|12_q_sigma.ipynb|PASS|2|2|FAST|
|13_dyna_q.ipynb|PASS|2|2|FAST|
|14_dyna_dqn.ipynb|PASS|2|2|FAST|
|15_dqn.ipynb|PASS|2|1|FULL|
|16_n_step_dqn.ipynb|PASS|2|1|FULL|

### FULL 측정 — Seed 7, CPU, 별도 Greedy 평가

|Notebook|학습 예산|평가 결과|
|---|---|---|
|05 Q-learning|1,500 Episode / 10,252 Step|100/100 성공, 평균 보상 1.0|
|15 CartPole DQN|400 Episode / 35,232 Step / 34,233 Gradient Update|평균 Return 9.2 → 186.55 (20회)|
|16 CartPole n=3 DQN|400 Episode / 61,342 Step / 60,343 Gradient Update|평균 Return 9.2 → 287.0 (20회)|

같은 Episode 예산에서도 환경 Step 수는 다르다. 단일 Seed의 측정으로 일반적인 우열이나
수렴을 보장하지 않는다. CartPole의 최종 평균은 500 Step 한도보다 낮다.
n-step 실행은 약 758초, 1-step은 약 102초였다. n-step은 더 많은 경험을 수집했고
순서 있는 Replay 추출도 수행한다. 최종 소스의 1-step FULL 재실행에서도 수치와 PNG가 일치했다.

기존 DLFS4 04–08 다섯 Notebook도 각각 같은 Kernel의 Run All 두 번을 통과했다.
실행된 소스와 텍스트·수치·PNG는 Phase 1의 실행 결과와도 같았다. 기존 그림은 총 40개다.
원본 Notebook과 보호 대상 예제·사용자 practice/체크포인트 파일의 시작 시점 해시를 보존했다.
기존 예제 Smoke Test 18개(quickstart, FrozenLake, CartPole와 DP/MC/TD·Control·multi-step·Dyna)도 통과했다.

### 최종 pytest 집합

최신 수집 목록은 2,257개다. 다음처럼 중복 없이 나눈 검증 합계는
**2,108 passed / 149 skipped**이며 Skip은 CUDA 미지원 항목이다.

|범위|Passed|Skipped|
|---|---|---|
|Notebook 제외 Core 전체|2,086|149|
|기존 DLFS4 Notebook|5|0|
|신규 Gymnasium Notebook|17|0|
|합계|2,108|149|

실행 명령과 로그:

- `python -m pytest -q --ignore tests/test_gymnasium_notebooks.py`: 기존 Notebook 포함
  2,087 passed / 145 skipped, 3,100.31초. 이 장시간 실행의 수집 이후 B=1 동등성
  파라미터 8개를 추가했으므로, 최신 Core 전체를 아래 명령으로 다시 검증했다.
- `python -m pytest -q --ignore tests/test_dlfs4_notebooks.py --ignore tests/test_gymnasium_notebooks.py`:
  최신 Core 2,086 passed / 149 skipped, 111.29초.
- `python -m pytest -q tests/test_gymnasium_notebooks.py`:
  원자료 비교까지 포함한 최종 실행 17 passed, 163.35초.

작업 시 로그는 `/private/tmp/modularrl-sequence-full-01.log`,
`/private/tmp/modularrl-sequence-core-final.log`, `/private/tmp/modularrl-gym-fast-04.log`에 보존했다.
이 임시 로그·Kernel 상태·학습 데이터는 Git에 포함하지 않는다.

## 공개 API 승격 후 재검증 (2026-10-15)

Gymnasium 17개를 FAST로 같은 Kernel에서 두 번 실행해 수치·Model state_dict·RNG·그래프를
비교했다. 별도의 작업 전/후 독립 실행 17쌍에서도 전체 정밀도의 학습 상태와 그래프가 일치했다.
기존에 FULL 출력을 저장했던 05/15/16은 FULL을, 다른 Notebook은 FAST 출력을 유지했다.
아래의 FULL 여섯 실험은 저장 모드와 별도로 실제 다시 실행한 결과다. 모두 Seed=7이다.

|FULL 실험|Episode|환경 Step|업데이트|평가|
|---|---:|---:|---:|---|
|FrozenLake Q-learning|1500|10252|10252|100/100 성공, 평균 Reward 1.0|
|FrozenLake SARSA|1500|15272|15272|100/100 성공, 평균 Reward 1.0|
|FrozenLake Expected SARSA|1500|10010|10010|100/100 성공, 평균 Reward 1.0|
|FrozenLake Double Q|1500|10208|10208|100/100 성공, 평균 Reward 1.0|
|CartPole DQN|400|35232|34233|평균 Reward 9.2 → 186.55 (20 Episode)|
|CartPole n=3 DQN|400|61342|60343|평균 Reward 9.2 → 287.0 (20 Episode)|

Q-learning은 Step마다 Replay에서 하나를 샘플링하고 나머지 세 Tabular Control은 B=1 Online이다.
동일한 Episode 수가 동일한 경험이나 Step 예산을 뜻하지 않는다. 두 CartPole도 Step·업데이트 수가
다르므로 이 수치만으로 알고리즘의 우위를 주장하지 않는다. 기존 FULL 05/15/16의 PNG도 일치했다.

기존 DLFS4 Notebook은 소스를 바꾸지 않는다. Notebook cwd에 프로젝트 루트가 없어도 예제의
파일 기반 로딩이 동작해야 한다. Off-policy n-step/Tree Backup 예제의 원래 runpy 연결은 그
호환 경로로 유지한다. 설치형 modular_rl의 학습 계산에는 examples/runpy 의존성이 없다.

최종 전체 회귀: `python -m pytest -q` — **2,171 passed / 149 skipped**, 55분 7초.
149개는 CUDA 미사용 환경의 Device 검사다. DLFS4 04–08 다섯 개와 Gymnasium 17개를
각각 같은 Kernel에서 두 번 실행한 결과를 포함한다. 기존 예제 18개 Smoke Test도 통과했다.

## Planning / Double DQN / Actor-Critic 전환

13/14는 examples import 없이 `W.Tabular()`와 `P.Dyna(...)`를 직접 구성한다.
13의 B=1은 가상 전이당 Q 갱신 하나이고 14의 B개 복원 추출은 DQN Gradient Update 하나다.
Planning이 Real Replay·환경 step·Target Sync 일정을 바꾸지 않는다. World는 정수 관측 기반이므로
14는 FrozenLake Dyna-DQN이며 연속 관측 CartPole Dyna-DQN으로 해석하지 않는다.

17은 Seed 7, MLP(128,128), Adam(1e-4), gamma=.99, Replay 10000, SmoothL1을 사용한다.
FULL에서는 DQN과 Double DQN 각각 **실제 환경 40000 step / 39001 Gradient Update / 160 Sync**,
Warm-up 1000, Batch 64, Sync 주기 250이다. 평가용 환경과 20 Episode의 Seed 집합도 같다.
행동이 다르므로 경험 자체가 같다는 뜻은 아니다. 같은 예산 안에서 완료한 Episode 수는 달라진다.
FAST는 각 400 step, Warm-up 32, Batch 16이다. n-step Double DQN의 수학은 SequenceBatch
단위 테스트로 검증하며 Notebook에 보상합이나 Bootstrap을 재구현하지 않는다.

18은 공유 Tanh MLP(64), 단일 Adam(.001), gamma=.99, value_coef=.5, entropy_coef=.01이다.
FULL 80000 step, FAST 400 step의 각 경험을 B=1로 즉시 한 번 사용한다. 정책·가치 Loss와
Entropy·TD Error, 학습 전후 별도 Greedy Return을 표시한다. Replay 재사용과 Target Network가
없으므로 DQN과 비교할 때 환경 step뿐 아니라 학습 호출 수·Batch 크기·재사용도 구분한다.
저장 log_prob은 학습에 직접 쓰지 않고 현재 Policy에서 재계산한다.

기존 개발 단계별 수치와 당시 지원 범위는 위의 날짜별 검증 기록이다. 현재 API의 정확한 계약은
[API.md](../API.md)를 따른다. 새로운 Notebook도 같은 Kernel에서 처음부터 Run All하면 Model,
Optimizer, 환경과 RNG를 초기화한다. 성능은 단일 Seed 관찰이며 확률적 성능을 pytest 합격 조건으로
쓰지 않는다. A2C는 아래 19번의 별도 Rollout 계약을 사용하며 A3C/PPO/GAE와 연속 행동 AC는 아직 없다.

### 이번 FULL 실행 결과 (Seed 7)

평가 환경은 학습 환경과 분리하고 동일한 초기 Seed 집합을 사용했다. CartPole 평가는 각각
20 Episode다. 아래는 실제 실행 결과이며 확률적 학습 성능을 단위 테스트 합격 기준으로 삼지 않는다.

|실험|환경 Step|학습 호출|Episode|평가 평균 Return 전 → 후|
|---|---:|---:|---:|---:|
|17: DQN 통제군|40000|39001|426|9.2 → 162.10|
|17: Double DQN|40000|39001|395|9.2 → 197.45|
|18: 1-step Actor-Critic|80000|80000|619|9.2 → 500.00|

DQN 두 실행은 Target Sync 160회, Batch 64로 과거 데이터를 반복 사용했다. Actor-Critic은
새로운 B=1 경험을 한 번씩만 사용했고 Target Network가 없다. Episode 예산·데이터 사용 방식이
다르므로 세 수치로 일반적인 알고리즘 우열을 주장하지 않는다. 새 Notebook에는 FULL 출력을
저장하고 13/14는 기존 FAST 학습 그래프를 유지한다. FAST 두 번 실행은 별도 수치·RNG 검사다.

### Dyna와 기존 DQN의 수치 회귀

Dyna 13/14는 작업 전 Notebook과 전환 후 Notebook을 독립 실행해 FAST의 기존 학습 수치,
Model/RNG와 PNG가 일치했다. 학습 전 Greedy 평가만 추가했으며 학습 RNG는 바꾸지 않았다.
동결한 32634ce World Model/Planning 수식으로 FULL도 별도 실행해 13/14의 학습 값·Model·RNG·PNG를
대조했다. Optimizer와 Target Sync 일정의 동등성은 별도 결정적 학습 루프 테스트에서 검증한다.

|FULL Dyna 실험|Episode|실제 Step|실제 학습|Planning 학습|Greedy 평가|
|---|---:|---:|---:|---:|---|
|13 Tabular Dyna-Q|1000|6832|6832|34160|0 → 100/100 성공|
|14 FrozenLake Dyna-DQN|1000|7527|7464|7527|0 → 0/100 성공|

14는 학습 중 558/1000 Episode에서 Goal에 도달했지만 마지막 Greedy Policy는 실패했다.
이는 이동 전 동결 참조에서도 같은 결과다. API 동등성은 통과했으며 이 설정의 최종 정책 성능은
미해결 제약으로 남긴다. 성능을 높이기 위해 이번 호환성 검증 중 Network/보상/학습 예산을 바꾸지 않았다.
Dyna-DQN은 FrozenLake의 정수 상태 실험이며 CartPole Neural World Model 구현을 뜻하지 않는다.

기존 15 DQN FULL은 400 Episode, 35232 step, 34233 업데이트와 평가 9.2 → 186.55를 재현했다.
기존 검증 기록의 전체 정밀도 학습 값·Model state_dict·RNG도 동일했다.

기존 16 n=3 DQN FULL도 400 Episode, 61342 step, 60343 업데이트와 평가 9.2 → 287.0을
재현했다. 기존 FULL 15/16 모두 전체 정밀도 학습 수치·Model state_dict·RNG와 PNG가 동일했다.

### 최종 검증

`python -m pytest -q --basetemp=/private/tmp/modularrl-planning-full-tests`는
**2245 passed / 165 skipped**, 4333.49초(72분 13초)로 완료됐다. 165개는 CUDA 미지원 검사다.
Gymnasium 19개와 DLFS4 04–08의 5개 Notebook을 각각 같은 Kernel에서 Run All 두 번 실행한
수치·출력·PNG 검증을 포함한다. 기존 예제 18개의 Smoke Test도 통과했다. 설치 wheel만 쓰고
`examples`를 차단한 독립 Python 프로세스에서 기존 승격 API 및 W.Tabular/P.Dyna/DoubleDQN/
ActorCritic의 실제 학습을 검증했다. Model/Estimator/Buffer/TargetGenerator 공개 계약은 유지했다.
작업 전부터 있던 Notebook 수정과 미추적 연습·체크포인트 파일은 보존하고 커밋에서 제외했다.

## 19: Async Rollout A2C

직접 AsyncVectorEnv를 만들고 SAME_STEP을 검사한 뒤 N개 실제 Step을 반복한다.
`final_observations` 결과는 전이의 next_observation으로, 반환된 Reset 관측은 다음 행동으로 쓴다.
`B.Rollout(horizon=5,n_envs=4)`는 시간 우선 Snapshot을 보관하고, get/learn/clear로 재사용한다.
학습은 `T.RolloutReturn`과 기존 `U.ActorCritic`을 쓰며 Return을 Notebook에서 재구현하지 않는다.
마지막 Rollout의 Return과 Advantage, Episode 곡선, Loss/Entropy, Greedy 평가를 표시한다.

기본은 FULL 80000 환경 전이 / 20000 vector step / 4000 Optimizer Step이다.
FAST는 400 환경 전이 / 20 업데이트다. 기존 18의 FULL 80000회 B=1 갱신과 달리
한 번에 20개 새 경험을 평균하되 각 경험의 사용 횟수는 둘 다 한 번이다. N=4/T=5의
이번 Seed 7 실행은 1231 Episode와 Greedy 평가 9.20→465.55를 기록했다. 동일 예산의
18 별도 재실행은 기록 619 Episode, 9.20→500.00이었다. 기존 18의 619에는 마지막 수집 cutoff를
truncated로 기록한 193-Step 구간 하나가 포함된다. A2C는 열린 구간 flag를 바꾸지 않는다.
동일한 경험이나 일반적 우열을 뜻하지 않는다.

정상/예외 경로에서 Worker를 모두 종료한다. 정책은 Rollout 동안 고정하고 저장 log_prob으로
직접 역전파하지 않는다. 일관된 eval 모드는 Gradient를 끄지 않는다. NEXT_STEP, GAE/PPO,
연속 행동, Recurrent state와 부분 Rollout은 아직 없다. 최신 계약은 API/ARCHITECTURE의 A2C 절을 따른다.

이번 전체 회귀는 **2337 passed / 191 skipped**, 56분 19초다. Gymnasium 20개와 DLFS4 5개의
두 차례 Run All 수치·RNG·텍스트·PNG 검증을 포함한다. CUDA 미사용 조건부 검사는 Skip했다.
기존 예제 18개 Smoke Test와 examples 없는 격리 wheel의 Async→Rollout→A2C 학습도 통과했다.

## 20–21: GAE와 PPO

20은 GAE를 독립적인 Return/Advantage 추정기로 사용한다. TimeLimit=3의 실제 Async Rollout에서
자연 종료와 Truncation을 분리하고 lambda=0은 TD0 residual, lambda=1은 고정 Value Snapshot이
연속적인 경우 RolloutReturn과 같음을 검증한다. 시간순 그래프는 Truncation 이후 새 Episode의
보상이 이전 Episode의 Trace에 들어가지 않는 모습을 보여준다. 마지막 셀은 같은 fresh Rollout을
GAE A2C에 한 번 전달한다. Generator는 Gradient를 보존하고 Learner가 no_grad 경계를 관리한다.

21은 별도 PPOBuffer 없이 B.Rollout을 재사용한다. 수집 당시 Policy/log_prob과 현재 학습 Policy를
구분한다. 최초 learn 시작 때만 Snapshot을 비교하고, Epoch 안에서는 Policy 변화가 정상이다.
GAE/Return/old_value/old_log_prob은 학습 시작 시 고정하며 Advantage 정규화는 전체 T*N에 한 번만
적용한다. 상수/단일 표본 Advantage는 0이 되므로 Actor 신호가 사라질 수 있음을 명시한다.

각 Epoch는 전용 shuffle_rng로 전체 표본을 섞어 한 번씩 사용한다. 작은 마지막 Minibatch도
포함하고 지표는 표본 수 가중 평균이다. 근사 KL은 mean(ratio-1-log_ratio), Clip Fraction은
mean(abs(ratio-1)>epsilon)이다. Model/Optimizer와 Shuffle RNG를 함께 복원하고, 환경과 Torch 행동
RNG는 별도로 관리한다. KL 조기 종료·Value/Gradient Clipping은 없다.

ActorCritic은 B=1 즉시 업데이트, A2C는 T*N당 한 Step, PPO는 여러 Epoch의 Minibatch Step이다.
FULL 비교에서는 환경 전이 수와 함께 Optimizer Step 및 표본 재사용 횟수를 보고한다. 기존 25개
Notebook 원본과 출력은 보존하고 두 새 Notebook만 추가한다. 성능 500은 pytest 조건이 아니다.

### CartPole FULL 실행 기록 — 같은 81,920 환경 전이 예산

CPU float32, Torch thread=1, Tanh MLP(64), Categorical+V, Adam, gamma=.99,
value_coef=.5, entropy_coef=.01을 사용했다. Worker Seed는 seed+i이고 각 별도 Greedy 평가는
20 Episode, seed=학습 seed+1000+i다. 같은 Seed에서도 두 알고리즘은 같은 경험을 뜻하지 않는다.

|Seed|학습 전|PPO 평가 후|A2C 평가 후|PPO 완료 Episode|A2C 완료 Episode|
|---|---:|---:|---:|---:|---:|
|7|9.20|415.25|466.80|1218|1240|
|11|10.60|483.70|282.50|1152|938|
|19|9.35|427.65|409.75|1162|1012|
|23|9.20|147.50|400.05|1285|1035|
|42|9.15|197.50|176.00|1087|1176|

5개 Seed 평균±population 표준편차(ddof=0)는 PPO **334.32±135.05**, A2C **347.02±104.41**이다.
각 Seed 내 20 Episode 평균을 다시 요약한 통계이며 100 Episode를 독립 학습 표본처럼 합치지 않는다.

|항목|PPO|A2C|
|---|---:|---:|
|N / T|4 / 128|4 / 5|
|학습률|.0003|.001|
|GAE lambda|.95|미사용, RolloutReturn|
|수집 Rollout|160|4096|
|Minibatch 표본 수|64|20 (전체 Rollout)|
|Epoch / 표본 재사용|4|1|
|Optimizer Step|5120|4096|
|환경 전이|81920|81920|
|총 학습 표본 사용|327680|81920|

학습률과 Horizon 등 설정도 다르며 초기 후보를 최적화한 비교가 아니다. 이 예산/설정에서는 PPO가
일관되게 우세하지 않았고 Seed 편차가 컸다. 500점 달성이나 알고리즘 우위를 테스트 조건으로 쓰지 않는다.
기존 19 원본은 수정하지 않고 임시 Kernel 사본에서 예산만 81,920으로 맞춰 재실행했다.
21의 Seed 7 FULL은 최종 그래프 소스로 다시 실행해 415.25를 재현했다. 새 20/21에는 실제 FULL
출력과 그래프를 저장했다. 20은 고정 8×4 시연이며 FULL/FAST에서 같은 짧은 수집 예산을 사용한다.
lambda=1과 RolloutReturn의 최대 오차는 2.3841858e-7(float32)였다.


### GAE/PPO 최종 검증

- 전체 `python -m pytest -q`: **2427 passed / 225 skipped**, 3217.47초(53분 37초).
  Skip은 CUDA 미지원 조건부 검사이며 Notebook/환경 검증은 실행했다.
- DLFS4 5개 + Gymnasium 22개, 총 **27개 Notebook**을 각각 같은 Kernel에서 두 번 실행해
  텍스트·수치·RNG(해당 검사)·Model 상태와 PNG를 비교했다. 기존 25개 원본은 바이트 단위로 보존했다.
- 선택적 Advantage 검증 호출을 정리한 최종 소스의 관련 단위/API/실제 Async 회귀:
  **219 passed / 68 skipped**. 최종 신규 Notebook 별도 재검사도 **2 passed**였다.
- 기존 예제 **18개 Smoke Test**(quickstart, DP/MC/TD/Control/n-step/Dyna, FrozenLake, CartPole)를 통과했다.
- GAE lambda=0/1 동등성, .95 수치 [8.577305,6.991,4.2], 종료/Truncation 및 두 flag,
  환경별 다른 경계, float32/64, Gradient 보존/차단과 다중 Step finite-difference gradcheck를 확인했다.
- PPO의 양/음 Advantage와 Clip 방향, 범위 안/밖 Gradient, old log_prob/Advantage detach,
  Critic/Entropy/Shared Representation Gradient와 비유한 입력·Ratio·Loss 실패를 확인했다.
  수치 예의 policy_loss=-.56666667, value_loss=2, entropy=.4, total_loss=.42933333을 검증했다.
- T=1/N=1, lambda=0, normalization=False, 1 Epoch/1 Minibatch에서 기존 AC와 Critic/Entropy,
  전체 Gradient·Parameter·Optimizer Momentum 갱신을 비교했다. Actor/Total Loss 숫자 차이는 위에 설명했다.
- 20개 표본/Minibatch 6/4 Epoch의 16 Step, 모든 표본 1회/Epoch, 작은 마지막 Batch,
  고정 Snapshot/Target, 전역 1회 정규화, Epoch 중 Ratio 변화와 가중 지표 평균을 확인했다.
- Model/Optimizer/전용 Shuffle RNG checkpoint 재개가 동일한 다음 갱신을 재현했고,
  Shuffle은 Torch 행동 RNG를 소비하지 않았다. 초기 정책 변경/학습 후 재전달 거절도 확인했다.
- 최종 wheel의 Python 모듈 **98개**가 현재 소스와 동일하다. 임시 경로에 설치한 `python -I`에서
  저장소와 examples 없이 실제 spawn Async→Rollout→GAE→PPO 학습과 모든 Worker 종료를 통과했다.
  wheel SHA256: `681bd2ac5e8e9db4fb8e2c3c415f056e13a2f5c4a02f7231a477b939601d92e3`.

CPU에서 실제 학습/Gradient를 검증했으며 CUDA 실행은 이 장비에서 확인하지 못했다.
현재 PPO는 Categorical Discrete Policy와 scalar V다. 연속 행동·Recurrent Policy·AMP,
Value Clipping·KL 조기 종료·완전한 Rollout 소유권 인증·환경 중간 상태의 범용 Checkpoint는 남은 확장이다.
