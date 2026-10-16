# ModularRL 아키텍처와 데이터 계약

ModularRL은 작은 컴포넌트를 조합하는 교육·연구용 강화학습 라이브러리다.
현재 공개 생성자와 사용 예는 [API.md](API.md), 실행 안내는
[GYMNASIUM_TUTORIALS.md](GYMNASIUM_TUTORIALS.md)에 있다.
개발 단계별 수치 결과와 이전 설계 기록은 [89aa872 이력](history/ARCHITECTURE_89aa872.md)에 보존한다.
이 문서는 현재 계약을 설명하며 과거 단계의 미구현 목록을 현재 상태로 해석하지 않는다.

## 책임과 데이터 흐름

```text
Observation -> InputAdapter -> Representation -> Estimator -> EstimatorOutput
                                                        -> ActionSelector -> ActionOutput
Experience -> Buffer/Episode/Rollout -> Learner -> TargetGenerator + UpdateRule
                                              -> Optimizer (Neural only)
Real Experience -> TabularWorldModel -> DynaPlanner -> same Learner
```

Model은 추론 컴포넌트다. 환경, Loss, Optimizer나 Replay를 소유하지 않는다.
Learner는 Model 호출, Bootstrap 선택, Gradient 경계와 갱신을 조합한다.
Buffer와 환경 수집·reset·탐색·Target Sync 스케줄은 외부 루프 책임이다.
Runner/Agent는 목표 논리 구성이지 현재 구현된 통합 실행 계층이 아니다.
공통 BaseLearner나 Registry/Factory는 두지 않는다.

`R/E/S/B/T/U/L/W/P`는 기존 클래스 자체의 짧은 alias다. 긴 이름·내부 import 경로,
Parameter/Buffer 등록과 state_dict key는 유지한다. 라이브러리는 examples를 import하지 않는다.
가독성을 위해 공개 계산 메서드에 수식이 드러나게 하고 중요한 검증만 짧은 private 메서드로 분리한다.

## Model과 Estimator

- `Model.forward(observation)`은 EstimatorOutput을 반환한다.
- `Model.act(observation, deterministic=False)`는 no_grad에서 ActionOutput을 반환한다.
  train/eval 모드는 호출자가 설정한다. 환경 경계의 Python/NumPy 변환도 호출자 책임이다.
- `Model.encode(observation)`는 공통 Features를 제공한다.
- `Model.evaluate_q(observation, actions)`는 행동 조건부 Critic의 선택적 경로다.

InputAdapter는 NumPy/Tensor/scalar를 Tensor로 준비한다. 정수는 기본 long이며 Tabular Index를
Float로 바꾸지 않는다. 정수로 표현한 연속 관측은 명시적인 dtype 설정을 사용한다.
Representation은 Tensor -> Tensor다. Identity와 `MLP(input_dim, hidden_dims=(128,), activation=nn.ReLU)`를
지원하며 MLP.output_dim은 마지막 Hidden Dimension이다. Estimator는 NumPy 변환이나 Device 이동을 하지 않는다.

EstimatorOutput의 최상위 필드는 policy, values다. ValueOutput은 v, q(tuple), advantage를 보존한다.
Policy는 Distribution 또는 행동 Tensor를 담을 수 있고, Value payload는 원본 추정 정보를 유지한다.
Distributional payload를 수용하는 것과 Distributional 알고리즘 지원은 별개다.

|Estimator|입력|출력 또는 Network 계약|
|---|---|---|
|TabularV / TabularQ|정수 state `(...)`|v `(...)` / q[0] `(..., A)`; Lookup Table|
|StateValue|Float Features `(..., F)`|Network `(..., 1)` -> v `(...)`|
|DiscreteQ / DiscreteAdvantage|Float Features `(..., F)`|Network `(..., A)` -> q[0] / advantage|
|ContinuousQ.evaluate|Features `(..., F)`, Actions `(..., A)`|결합 `(..., F+A)` -> Network `(..., 1)` -> q[0] `(...)`|
|CategoricalPolicy|Float Features `(..., F)`|Network Logits `(..., A)` -> Categorical|
|GaussianPolicy|Float Features `(..., F)`|Network Mean `(..., A)` -> Independent Normal|
|DeterministicPolicy|Float Features `(..., F)`|Network Action Tensor `(..., A)`|

Neural Estimator는 keyword-only `network=nn.Module` 주입을 지원한다. 생략하면 기존 Linear,
ContinuousQ는 기본 joint MLP를 쓴다. Network를 정식 submodule로 등록하므로 parameters(),
state_dict(), to(device)에 포함된다. 출력 Dimension은 명시적으로 검사한다.

```python
head = E.DiscreteQ(64, 4)
custom = E.DiscreteQ(64, 4, network=nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4)))
```

Gaussian은 상태 독립적인 학습 가능 log_std를 유지한다. batch_shape는 `(...)`, event_shape는 `(A,)`이며
log_prob은 행동 차원을 합산한다. 분산이 해당 dtype에서 안전하게 표현되지 않으면 오류다.
State-dependent scale, Action Bound와 SAC용 Tanh-squashed Gaussian은 별도 향후 확장이다.
ActorCritic은 기존 Policy/Value Estimator의 출력을 변환 없이 결합하는 가벼운 Wrapper다.
행동이 필요한 ContinuousQ를 단순 ActorCritic.forward에 넣지 않는다. 현재는 Policy와 Critic Model을 나누고
Critic의 evaluate_q를 호출한다. Advantage Head는 GAE나 자동으로 참된 Aπ를 의미하지 않는다.

## ActionSelector

`select(estimate, deterministic=False) -> ActionOutput(action, log_prob)`이며 action은 Tensor다.
Selector는 Estimator 구현 클래스가 아니라 의미 있는 출력 필드를 사용한다. NumPy 변환은 하지 않는다.
Greedy/EpsilonGreedy는 정확히 하나의 Scalar Discrete Q Tensor를 요구하고 복수 Critic을 임의로 고르지 않는다.
앞쪽 Batch Dimension을 유지하며 PyTorch RNG로 샘플링한다.

최댓값 행동 집합 M, 행동 수 A에 대해 EpsilonGreedy는 다음 확률을 사용한다.

`P(a|s) = epsilon/A + (1-epsilon)*I(a in M)/|M|`

epsilon은 [0,1]이다. Greedy는 epsilon=0과 같은 균등 Tie Sampling을 사용한다.
확률 조회 probabilities(estimate)는 RNG를 소비하지 않는다. 확률적 선택의 log_prob은 실제 선택 확률이다.
deterministic=True는 최소 인덱스 최댓값 행동과 log_prob=None을 반환한다.
Categorical Selector는 policy를 샘플링하고 log_prob을 계산하며 Actor-Critic의 V를 사용하지 않는다.
Estimator 자체는 행동을 샘플링하지 않는다. 연속 행동 Selector는 아직 없다.

## 경험 자료와 소유권

Transition은 observation, action, reward, next_observation, terminated, truncated와 선택적 log_prob을 갖는다.
log_prob은 수집 시점의 log b(a|s)다. 현재 Q/epsilon으로 과거 확률을 다시 계산하지 않는다.
terminated와 truncated를 합치지 않는다. 실제 마지막 관측 제공은 수집자 책임이다.

|자료|시간 의미와 Shape|
|---|---|
|TransitionBatch|Replay/일괄 갱신 `(B,...)`; reward와 flags `(B,)`|
|Episode|시간순 Transition; MC는 자연 종료된 Episode 전체 사용|
|SequenceBatch|샘플별 순서 `(B,T,...)`; valid Prefix와 horizon `(B,)`|
|RolloutBatch|시간 우선 Vector 수집 `(T,N,...)`; value/log_prob 필수|

Discrete action은 Scalar 또는 Batch `(B,)`, Continuous action은 Batch `(B,A)`다.
Observation의 Feature Shape를 고정하지 않는다. 저장 경계는 graph 없는 owned CPU Tensor로 복사해
환경의 NumPy 재사용이나 반환 Batch 수정이 원본을 바꾸지 못하게 한다. 원본 정수/Float dtype을 보존한다.
Batch.to(device, dtype=...)는 Floating 필드만 cast하고 정수·bool을 보존한다.

Replay는 capacity Ring과 독립 NumPy RNG를 사용한다. add/sample/clear/len만 책임지고 시간 순서를 보장하지 않는다.
한 Buffer에서 log_prob 유무를 섞지 않는다. SequenceReplay는 연속적인 단일 환경의 전이를 확인하고
Episode 경계를 넘지 않는 유효 시작점을 균등·비복원 샘플링한다. 열린 짧은 Tail은 후보가 아니다.
Rollout은 정확히 horizon개 Vector Step을 모으고 get으로 독립 Batch를 반환한다. 자동 clear/덮어쓰기는 없다.
Episode는 연속성과 종료 경계를 검사하고 종료 이후 전이를 거절한다.

## TargetGenerator와 UpdateRule

TargetGenerator는 외부 Tensor의 수식만 계산하며 Model/Buffer/Optimizer를 참조하지 않는다.
자동 Broadcasting을 막기 위해 약속된 shape/dtype/device를 검사하고 graph를 보존한다.
호출자가 Bootstrap 값을 정하고 no_grad/detach를 적용한다.

|Generator|계산|
|---|---|
|TD0|`reward + gamma*(1-terminated)*next_value`; 동일 shape|
|NStep|유효 보상 합 + 마지막 비종료 관측의 할인 Bootstrap; `(B,T)` -> `(B,)`|
|MCReturn|Episode 보상을 역순 누적; `(T,)`, Bootstrap 없음|
|RolloutReturn|역순 `(T,N)` Return; 자연 종료 0, Truncation final V, 계속 구간 다음 Return|
|GAE|TD residual과 경계별 역순 Advantage; returns=advantage+value, `(T,N)`|

GAE에서 terminated는 Bootstrap을 제거하고 terminated 또는 truncated는 Trace를 끊는다.
두 flag가 True면 자연 종료가 우선한다. lambda=0은 TD0 residual이고 lambda=1은
계속 구간의 next_value[t]==value[t+1] 조건에서 RolloutReturn과 같다.
Truncation의 Bootstrap 허용과 다음 Episode로의 Trace 연결은 다른 개념이다.

UpdateRule에는 공통 Base Class가 없다. TabularV/Q는 원본 테이블에 no_grad로
`delta=target-current; current+=alpha*delta`를 입력 순서대로 적용한다.
중복 상태/쌍도 순차 갱신하며 Target은 시작 전에 복사해 Table Storage alias에도 고정한다.
반환 signed TD error는 graph가 없다. Neural TDLoss는 같은 shape의 prediction/target에서
미축약 MSE 또는 Custom Loss를 계산하고 선택적 weights를 곱해 평균한다.
Prediction/Target graph를 유지하며 Optimizer/backward는 호출하지 않는다.

## Tabular Learner와 Monte Carlo

QLearning은 다음 Q의 max, SARSA는 외부에서 실제 선택한 Next Action의 Q,
ExpectedSARSA는 Model 자신의 Selector probabilities와 Q의 합 ΣπQ를 Bootstrap으로 사용한다.
다음 상태의 값과 Target은 전체 Batch 갱신 전에 계산한다. B>1 Frozen Target과
시간순 B=1 Online 학습을 구분한다. SARSA는 선택한 Next Action을 다음 실제 Step에서 재사용한다.
DoubleQ는 독립 Table 두 개에서 행동 선택/평가를 분리하며 갱신 동전 RNG는 외부 책임이다.
TD0Prediction도 같은 TD0/TabularV Rule을 조합하고 행동 선택이나 MDP를 소유하지 않는다.

MC Prediction/Control은 정방향 First/Every 상태 또는 (state,action) 방문을 선택한다.
역순 Return 계산 중 처음 만난 방문을 First-visit로 해석하지 않는다.
Count를 증가시키고 `(G-current)/count`로 Sample Average를 갱신한다.
열린 Episode와 truncated-only는 거절하며 최종 next_observation을 추가 방문으로 세지 않는다.

Off-policy Prediction의 suffix rho는 현재 행동부터 마지막 행동까지 pi/b의 곱이다.
Ordinary는 sum(rho*G)/N, Weighted는 sum(rho*G)/sum(rho)다. Weight=0도 Count에 포함하고
누적 Weight=0이면 Weighted V를 유지한다. Target Policy는 고정하고 기록된 log b를 사용한다.
Coverage(pi>0 => b>0)는 호출자 책임이며 선택 행동의 log_prob만으로 미관측 support를 검사할 수 없다.
Off-policy Control은 역순 W=1에서 C/Q를 먼저 갱신하고, 갱신된 Greedy 행동과 기록 행동이 다르면 중단한다.
같으면 이전 시점 W를 기록 log b로 계산한다. 비유한 수치 실패는 작업 복사본으로 부분 갱신을 막는다.

MC Count/Weight 통계는 Learner 상태로 Model.state_dict에 포함되지 않는다.
Model과 해당 통계, 정책/gamma/visit 설정, 환경·행동 RNG를 함께 복원한다.
collect_episode는 기존 정수 콜백과 ActionOutput 콜백을 모두 받으며 선택 시점 log_prob을 step 전에 복사한다.
직접 reset/step으로 Transition을 모아 Episode로 묶는 방식도 같다.

### Online multi-step

SARSA/OffPolicyNStepSARSA/TreeBackup/QSigma는 시간순 owned B=1 Window의 head 하나만 갱신한다.
Window 이동과 Tail Flush는 호출자 책임이며 SequenceReplay를 자동으로 Online Window로 해석하지 않는다.
실제 SARSA Horizon 행동과 Off-policy 행동 확률은 외부에서 선택·보존한다.

Q(σ)는 교재 본문 식 (7.17)의 비종료 경계 G_h:h=Q(S_h,A_h)를 사용한다.
끝점 residual이 0이므로 `R_h+gamma*Vpi(S_h)`에서 시작하며 끝점 행동을 추가 샘플링하지 않는다.
자연 종료는 R_T, Truncation은 실제 final 관측의 Vpi로 Bootstrap한다.
실제 중간 행동의 σ/ρ만 Return에 영향을 준다. σ=0은 Tree Backup, n=1은 Expected SARSA다.
σ=1은 Control Variate를 포함하며 Sampled SARSA와 표본별 동일성을 가정하지 않는다.
재귀계수 `sigma*rho+(1-sigma)*pi` 뒤 전체 IS Ratio를 다시 곱하지 않는다.
본문·의사코드·업데이트 설명의 해석 차이는 [기존 수학 검증 기록](history/ARCHITECTURE_89aa872.md)에 남겼다.
공식 정오표에서 확인되지 않은 차이를 공식 인정된 오류라고 단정하지 않는다.

## DQN, Double DQN과 Dyna

DQN은 Online 단일 Scalar Discrete Q `(B,A)`에서 실제 행동을 gather하고 no_grad Target max로
Target을 계산한 뒤 TDLoss와 zero_grad/backward/step을 실행한다. 반환 Loss/TD error는 detach된 갱신 직전 값이다.
Double DQN은 Online argmax를 Target에서 평가하는 Bootstrap만 다르다.
Target은 동결/eval이며 Online과 구조·dtype·device는 같고 저장 공간은 독립적이다.
sync_target은 Parameter와 필요한 Buffer를 복사한다. 초기 복사와 이후 주기는 호출자가 결정한다.
Neural n-step은 SequenceBatch의 첫 State/Action과 마지막 유효 관측을 사용한다.
중간 Behavior 보상의 uncorrected backup이며 IS를 수행한 SARSA는 아니다.

W.Tabular는 관측 pair의 최신 결정적 전이를 저장하는 경험 모델이다. 알려진 TabularMDP나 추론 Model과 다르다.
외부 Truncation/log_prob은 예측하지 않고 미관측 pair를 생성하지 않는다.
P.Dyna는 World.sample(rng)->Transition과 Learner.learn(batch)를 조합한다. 실제·가상 갱신에 같은
Learner/Optimizer를 쓰며 환경 step, observe, Replay, 탐색이나 Target Sync는 수행하지 않는다.
빈 World/0회 Planning은 RNG를 소비하지 않는다. 연속 관측 Learned World Model은 아직 없다.

## Actor-Critic, A2C와 PPO

현재 On-policy Neural Learner는 Categorical Policy와 Scalar V를 사용한다.
1-step ActorCritic은 fresh B=1 전이를 즉시 갱신한다. A2C는 정책을 고정해 수집한 T*N을
한 번 재평가하고 RolloutReturn 또는 GAE와 ActorCritic Rule로 단일 Optimizer Step을 수행한다.
PPO는 같은 Rollout에서 GAE/Returns와 old value/log_prob을 한 번 고정하고 여러 Epoch/Minibatch를 학습한다.
첫 학습 시 Snapshot 비교는 관측 가능한 정책 변화를 검출할 뿐 데이터 소유권을 인증하지 않는다.

U.PPO는 clipped surrogate, value_coef*0.5*MSE와 entropy 보너스를 계산한다.
old_log_prob/advantage는 detach하고 Critic Return graph는 호출자가 제어한다.
Learner의 Advantage 정규화는 전체 T*N에서 한 번, population variance와 sqrt(var+1e-8)을 사용한다.
상수/단일 Advantage는 0이 되어 Actor surrogate 신호가 사라지지만 Critic/Entropy 신호는 남는다.

`L.PPO(..., max_grad_norm=None, target_kl=None)`의 기본값은 기존 계산을 보존한다.
Clipping은 backward 뒤 Step 전에 Optimizer의 학습 가능한 unique Parameter 전체 Norm에 적용한다.
비유한 Norm이면 해당 Step 전에 실패한다. KL은 Epoch 뒤 다음 Epoch가 남았을 때 전체 Rollout을
no_grad/eval로 재평가하고 `mean(exp(log_ratio)-1-log_ratio)>target_kl`이면 중단한다.
old_log_prob은 수집 Snapshot이며 진단은 BatchNorm/Dropout 상태·Shuffle/행동 RNG를 보존한다.

PPOOutput의 여섯 detached 지표는 실제 완료 Minibatch의 표본 수 가중 평균이다.
최종 정책 재평가 KL과 이 평균 approx_kl을 혼동하지 않는다. optimizer_steps/epochs_completed는 실제 횟수,
early_stopped는 남은 Epoch를 중단했는지다. U.PPO 출력의 세 실행 통계는 기본 0/0/False다.
Model/Optimizer, shuffle_rng와 옵션 포함 설정, 별도 환경·행동 RNG를 함께 저장한다.
[PPO 실험·검증](PPO_STABILITY.md)에 비교 조건과 결과를 기록한다.

### Vector 수집

AsyncVectorEnv는 SAME_STEP/copy=True를 사용한다. validate_vector_env와 final_observations는
실제 final_obs를 보존하고 Reset 관측으로 대체하지 않는다. 반환 관측은 다음 행동에,
실제 마지막 관측은 Buffer에 사용한다. NEXT_STEP/DISABLED는 지원하지 않는다.
Gymnasium >=1.1이 필요하며 1.4.0에서 검증했다. 환경은 try/finally로 닫고 Worker 실패도 정리한다.
수집 도중 학습하지 않으며 Async 환경 실행과 동기식 Learner 업데이트를 구분한다.
Dropout/BatchNorm 사용 시 수집·학습 재평가의 일치를 위해 일관된 eval Forward를 권장한다.
eval도 미분 가능하며 호출자의 원래 모드는 임시 평가 후 복원한다.

## 알려진 MDP와 DP

TabularMDP는 transitions[state][action]의 TransitionOutcome(probability,next_state,reward,terminated)를 보존한다.
확률합을 검사하고 DP는 전체 Outcome 기대값을 CPU NumPy float64로 계산한다. V `(S,)`, Policy `(S,A)`다.
Policy Evaluation/Value Iteration은 동기식 Sweep이고 Policy Improvement는 정확한 최대 Q 동률에 균등 확률을 준다.
Policy Iteration은 평가를 재사용하며 정책 행렬의 안정성을 확인한다. 평가 실패면 Improvement하지 않는다.
반복 한도는 DPResult.converged=False로 보고한다. gamma=1의 Improper Policy 수렴은 보장하지 않는다.

Sutton/DLFS4 kernel과 정책 확률을 TabularEnv가 샘플링하므로 DP/MC/TD를 같은 환경에서 비교할 수 있다.
DP는 알려진 모델, MC는 완료 Episode Return, TD는 관측 전이의 Bootstrap을 사용한다.
Sutton RMSE는 14개 비종료 상태에서 계산한다. MC/TD 비교는 동일 완료 Episode를 독립 Model에 전달하고
TD는 B=1로 순차 처리한다. 이는 종료 전에 학습하는 Streaming Runner와 구분한다.
확률적 RMSE의 단조 감소나 한 Seed의 알고리즘 우위를 테스트 조건으로 삼지 않는다.

## 현재 한계와 검증

연속 행동 Learner/SAC/TD3/DDPG/A3C, Prioritized Replay, Learned Neural World Model,
Recurrent Policy, 통합 Runner/Agent와 범용 Checkpoint는 구현하지 않는다.
Sequence 후보 검색은 O(capacity)이며 대규모 Replay 최적화는 별도 과제다.
수학이 다른 Learner를 복잡한 공통 Bootstrap 인터페이스로 숨기지 않는다.

전체 pytest에는 결정적 수치·Gradient·경계·Checkpoint·Alias·Notebook 회귀가 포함된다.
기본 CPU CI와 수동 Notebook 회귀를 분리하고 배포 파일은 examples 없이 설치·검증한다.
실행 환경, 라이선스, 배포/보안 감사와 검증 결과는 [공개 준비 보고](PUBLIC_RELEASE.md)에 기록한다.
