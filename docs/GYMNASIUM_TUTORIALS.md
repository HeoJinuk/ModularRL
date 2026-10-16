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
|17_double_dqn|CartPole|DQN/Double DQN의 환경 step·Gradient Update·Sync 예산 통제|
|18_actor_critic|CartPole|Categorical Policy/Scalar V, B=1 즉시 On-policy 학습|
|19_a2c|CartPole|Async SAME_STEP, 시간 우선 Rollout, 경계별 Multi-step Return, T*N당 1회 학습|
|20_gae|CartPole|실제 Rollout, lambda=0/.5/.95/1 Advantage, TD0/Return 동등성, Truncation Trace|
|21_ppo|CartPole|Async SAME_STEP, GAE, 고정 Snapshot, Multi-epoch Minibatch PPO, KL/Clip 진단|

19–21은 SAME_STEP을 지원하는 Gymnasium >=1.1이 필요하다(이번 검증 1.4.0).
기존 환경이 1.0이면 `python -m pip install 'gymnasium>=1.1'`로 새 Vector 실습 의존성을 준비한다.

FrozenLake는 `map_name="4x4", is_slippery=False`다. 행동 ID는 Left/Down/Right/Up이다.
알려진 P는 DP와 참조값 진단에만 사용하며, Dyna 모델에는 관측한 전이만 기록한다.
Prediction은 지정 정책을 평가한다. V에서 유도한 Greedy 화살표는 별도 진단으로 표시한다.
CartPole은 Floating Observation과 Scalar Discrete Action을 사용한다.

DP, MC/TD Prediction·Control, Q-learning, SARSA, DQN, Expected SARSA, Double Q,
Online n-step SARSA, Off-policy n-step SARSA, Tree Backup, Q(σ)는 라이브러리 API다.
이들의 학습 계산은 Notebook에서 공식 Learner를 구성해 호출한다. Dyna의
World Model/Planner도 `W.Tabular`/`P.Dyna`의 설치형 API다. Double DQN과 1-step Actor-Critic도
공식 Learner다. A2C는 B.Rollout/T.RolloutReturn 또는 T.GAE/U.ActorCritic을 조합하고,
PPO는 B.Rollout/T.GAE/U.PPO로 다중 Epoch 학습을 수행한다.
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

## PPO 안정화 옵션과 재현성

21은 기본 PPO의 명시적인 구성·학습 루프를 유지한다. 다음 선택 옵션을 추가할 수 있다.

```python
learner = L.PPO(model, optimizer, T.GAE(.99, .95), U.PPO(),
                epochs=4, minibatch_size=64, seed=7,
                max_grad_norm=.5, target_kl=.02)
```

두 기본값은 None이다. Gradient Norm은 Optimizer가 관리하는 학습 가능 파라미터 전체에서
중복 없이 계산한다. Backward 다음, Step 전에 Clip하며 비유한 Norm은 오류다.
KL은 다음 Epoch가 남아 있을 때 Epoch 종료 후 전체 Rollout에서 eval/no_grad로 측정한다.
mean(exp(log_ratio)-1-log_ratio)가 target_kl을 넘으면 다음 Epoch를 중단한다.
Model 모드는 복원하고 Shuffle/행동 RNG를 소비하지 않는다. 마지막 Epoch 뒤에는 불필요한 판정을 하지 않는다.

PPOOutput의 optimizer_steps/epochs_completed는 실제 실행 횟수이며, early_stopped는
남은 Epoch를 중단했는지를 나타낸다. 여섯 지표는 실제 처리 표본 수로 가중 평균한다.
수집 value/log_prob과 GAE/Return은 모든 Epoch에서 고정한다. 정규화는 전체 표본에 한 번만 적용한다.
Checkpoint는 Model/Optimizer/Shuffle RNG와 두 옵션을 포함한 동일한 설정을 함께 복원한다.

동일한 81,920 전이 예산으로 4가지 설정을 비교하는 재현 가능한 실험:

```sh
python -m experiments.ppo_stability --seeds 7 11 19 23 42 --output experiments/results/ppo_stability.json
```

기본 학습률·Network·Seed·예산을 설정별로 바꾸지 않는다. 결과와 검증 상태는
[PPO 안정화 실험](PPO_STABILITY.md), [공개 준비 점검](PUBLIC_RELEASE.md)에 기록한다.
이전 FULL 결과와 개발 단계별 기록은 [이력 문서](history/GYMNASIUM_TUTORIALS_89aa872.md)에 보존했다.
