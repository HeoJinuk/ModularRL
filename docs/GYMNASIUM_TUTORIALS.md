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

기존 긴 이름과 동일한 클래스다. [공개 API 계약](API.md)에 모든 Learner의 생성자·입력·결과와
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

FrozenLake는 `map_name="4x4", is_slippery=False`다. 행동 ID는 Left/Down/Right/Up이다.
알려진 P는 DP와 참조값 진단에만 사용하며, Dyna 모델에는 관측한 전이만 기록한다.
Prediction은 지정 정책을 평가한다. V에서 유도한 Greedy 화살표는 별도 진단으로 표시한다.
CartPole은 Floating Observation과 Scalar Discrete Action을 사용한다.

DP, MC/TD Prediction·Control, Q-learning, SARSA, DQN, Expected SARSA, Double Q,
Online n-step SARSA, Off-policy n-step SARSA, Tree Backup, Q(σ)는 라이브러리 API다.
이들의 학습 계산은 Notebook에서 공식 Learner를 구성해 호출한다. Dyna의
TabularWorldModel/plan과 실험 수집·스케줄은 **examples 지원**이다. Return 수식은 복사하지 않는다. Tree Backup/Q(σ)는 여기서 고정 정책의
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
