# ModularRL 개발 맥락

## 목적과 범위
- 연구 및 교육용 강화학습 라이브러리. Sutton & Barto의 Tabular RL부터 DQN, A2C, PPO, SAC, Dyna, World Model 계열까지 확장 가능한 구조를 지향한다. 목표 알고리즘을 모두 구현된 것으로 간주하지 않는다.
- 초심자가 이해할 수 있는 단순한 Base Class와 연구자가 자유롭게 교체할 수 있는 명시적인 입출력 계약을 유지한다.
- GUI는 현재 범위 밖이다. 컴포넌트의 독립성과 교체 가능성은 핵심 요구사항이다.
- 새로운 추상 계층, 불필요한 Registry 또는 GUI 코드를 도입하지 않는다.

## 현재 저장소의 범위
- 현재는 Model·경험 데이터(Transition/TransitionBatch/Episode)·ReplayBuffer·TD(0)/MC Return·Tabular TD 갱신·Neural TD Loss·TabularQLearner·DQNLearner·DP Phase 1·MC Prediction Phase 2·TD(0) Prediction Phase 3의 참조 구현이다. FrozenLake Q-learning, CartPole DQN, Sutton Gridworld DP/MC/TD 비교를 실행할 수 있다. Runner, Agent, SARSA, MC Control, 다른 Neural Learner, WorldModel/Planner는 아직 포함하지 않는다.
- 아래 Model 계약과 파일 배치는 이미 구현되어 있다. 기존 구현을 먼저 검토하고 중복 계층을 만들지 않는다.
- 정수 입력은 기본적으로 torch.long으로 정규화한다. 정수로 저장된 연속 관측은 명시적인 InputAdapter dtype 설정을 사용한다.
- Distributional 결과는 출력 payload에 담을 수 있으나, 해당 estimator/selector 및 학습 알고리즘이 구현된 것은 아니다.
- Model.act는 no_grad를 사용하며 train/eval 모드는 호출자가 설정한다. action은 Tensor이고, 향후 Runner가 환경 경계에서 Python/NumPy로 변환한다.
- 기존 공개 import와 Base Class 호환 alias를 보존한다. Model.evaluate_q는 선택적인 action-conditioned critic 확장 경로다.

## 구성
- Runner = Environment + Agent
- Agent = Model + Learner
- Model = InputAdapter + Representation + Estimator + ActionSelector
- Learner = Buffer + TargetGenerator + UpdateRule + 선택적 WorldModel/Planner
- 위 구성은 목표 학습 시스템의 논리적 구성이다. TabularQ/DQN/TD0 Learner는 외부 TransitionBatch, MC Learner는 외부 Episode를 받으며 Buffer나 환경을 소유·샘플링하지 않는다.

## 목표 Model 계약
- InputAdapter: NumPy, PyTorch Tensor, scalar 입력을 Tensor로 변환한다. Tabular State Index의 정수 dtype을 유지한다.
- Representation: Tensor -> Tensor. NumPy 변환을 하지 않는다. Identity와 MLP부터 지원한다.
- MLPRepresentation: `input_dim`, `hidden_dims=(128,)`, `activation=nn.ReLU`. `output_dim`은 마지막 hidden dimension이다.
- Estimator: Tensor features -> EstimatorOutput.
- EstimatorOutput: `policy`와 `values`를 구분한다.
- ValueOutput: `v`, `q`(tuple), `advantage`. Scalar 및 Distributional 결과를 수용하도록 설계한다.
- ActorCriticEstimator: PolicyEstimator와 ValueEstimator를 결합하는 Wrapper.
- ActionSelector: EstimatorOutput -> ActionOutput(`action`, `log_prob`). Estimator 구현 클래스에 의존하지 않고 의미 있는 출력 필드를 사용한다.
- Model.forward(observation) -> EstimatorOutput.
- Model.act(observation) -> ActionOutput.
- Model은 NumPy와 Tensor 입력을 모두 지원한다.

## 목표 파일 배치
아래 경로는 Python 패키지 루트 기준이다.
- `data/outputs.py`, `data/experience.py`, `data/updates.py`, `data/mdp.py`, `data/episode.py`
- `model/rl_model.py`, `model/input_adapter.py`
- `component/representation/{base,identity,mlp}.py`
- `component/estimator/{base,actor_critic}.py`
- `component/estimator/policy/`, `component/estimator/value/`
- `component/action_selector/`
- `component/buffer/`, `component/target_generator/{base,td0,monte_carlo}.py`
- `component/update_rule/{tabular,td_loss}.py`
- `learner/{__init__,tabular_q,dqn,mc_prediction,tabular_td0}.py`
- `dp/{__init__,algorithms}.py`, `environment/{__init__,gridworld,episode_sampler}.py`

## TargetGenerator 계약
- TD0TargetGenerator는 외부에서 계산한 reward, next_value, terminated Tensor로 target 수식만 계산한다. Model, Buffer, Optimizer를 참조하지 않는다.
- 세 입력의 shape·device는 동일해야 한다. reward/next_value는 같은 floating dtype, terminated는 bool이며 자동 Broadcasting이나 입력 변환을 하지 않는다.
- next_value 선택과 no_grad/detach는 호출자(Learner)의 책임이다. terminated는 Bootstrap을 중단하지만 일반적인 Time-limit truncated는 Bootstrap이 가능하다. Episode Boundary와 Bootstrap Mask를 구분한다.

## UpdateRule 계약
- 공통 BaseUpdateRule은 두지 않는다. TabularV/QUpdateRule은 Tensor 테이블을 직접 갱신하고, TDLossUpdateRule은 외부 prediction/target의 signed TD error와 scalar loss를 계산한다.
- TD error는 `target - current_value`다. Tabular는 no_grad에서 원본 테이블을 in-place로 입력 순서대로 갱신하며, target은 갱신 전에 복사해 고정한다. 중복 인덱스를 일괄 Advanced Indexing 연산으로 대체하지 않는다.
- Neural은 prediction/target의 graph를 유지한다. loss_fn은 입력과 동일 shape의 미축약 loss를 반환하며, weights는 정규화 없이 샘플별 loss에 곱한 뒤 평균한다.
- Model 호출, Bootstrap 선택, Optimizer의 zero_grad/backward/step과 Target Model 갱신은 Learner 책임이다. 로그·Priority용 TD error detach도 Learner가 수행한다.

## TabularQLearner 계약
- `TabularQLearner(model, target_generator, update_rule).learn(batch)`는 TabularQEstimator Model, TD0TargetGenerator와 TabularQUpdateRule을 조합한다. ActionSelector는 선택적이며 학습에 사용하지 않는다.
- 각 입력 필드는 `(B,)`다. 정수 state/action indices를 long으로 정규화하고 Batch를 현재 q_table의 device·floating dtype에 맞춘다. Representation은 table state index를 보존해야 한다.
- no_grad에서 다음 상태 Q의 action별 최댓값과 Batch target을 먼저 계산한 뒤 기존 UpdateRule로 갱신한다. 반환 signed TD error는 행 순서를 유지하고 graph가 없다. 수식을 Learner에 중복 구현하지 않는다.
- Buffer 저장·샘플링, 환경 종료·reset과 실제 final observation 제공은 호출자 책임이다. Tabular에는 Optimizer가 없으며 기본 FrozenLake 예제는 Gymnasium의 선택적 examples 의존성을 사용한다.

## DQNLearner 계약
- `DQNLearner(online_model, target_model, optimizer, target_generator, update_rule).learn(batch)`는 단일 Scalar Discrete Q `(B, A)`에서 선택 행동의 prediction `(B,)`을 gather하고, no_grad Target max Q로 TD0 target을 만든 뒤 기존 TDLossUpdateRule과 Optimizer를 실행한다. 반환 TDLossOutput의 signed td_error와 scalar loss는 detach된 업데이트 직전 값이다.
- Online/Target은 같은 구조·device·floating dtype이며 Parameter와 Buffer 저장 공간을 공유하지 않는다. Optimizer는 Online Parameter만 포함한다. Target은 동결·eval로 운용하고 Online train/eval은 호출자가 설정한다.
- 초기 복사와 이후 hard copy는 `sync_target()`으로 명시적으로 호출한다. Parameter와 nonpersistent Buffer도 복사하며 호환되지 않는 구조는 실패한다. 동기화 주기·탐색 감소·Warm-up은 예제/향후 Runner 책임이다.
- TargetGenerator/UpdateRule의 gradient 계약과 기존 Model API는 유지한다. Optimizer checkpoint는 향후 Learner checkpoint의 책임이다. Double DQN, n-step, Gradient Clipping/AMP는 아직 구현하지 않는다.

## Tabular DP 계약
- TabularMDP는 알려진 환경의 `transitions[state][action]`에 TransitionOutcome(probability, next_state, reward, terminated)를 보존한다. 추론용 Model, 경험 Transition, 학습된 WorldModel과 역할이 다르며 확률합은 생성 시 검사한다.
- DP는 경험을 샘플링하지 않고 전체 결과의 기대값을 계산한다. CPU NumPy float64의 V `(S,)`와 Policy `(S, A)`를 사용한다. 기존 Tensor 기반 Model/Learner에 값을 옮기는 것은 호출자의 명시적 변환 책임이다.
- Policy Evaluation과 Value Iteration은 동기식 sweep을 사용한다. Policy Improvement는 정확히 같은 최대 Q 행동에 균등한 확률을 부여하며 Policy Iteration은 같은 정책 행렬인지 확인해 안정성을 판단한다.
- 반복 한도는 DPResult.converged=False로 보고한다. Policy Iteration은 평가 실패 시 Improvement를 진행하지 않으며 마지막 평가 Policy와 Value를 함께 반환한다. gamma=1에서 Proper Policy를 자동으로 판별하거나 수렴을 보장하지 않는다.
- `environment/gridworld.py`의 알려진 kernel과 상태별 Policy 확률은 MC/TD Prediction에 재사용한다. DP는 NumPy 기대값 계산을 유지하며 Episode Sampler와 Tensor Learner로 강제 통합하지 않는다. Dyna Planner는 아직 없다.

## MC Prediction 계약
- Episode는 기존 Transition의 순서·연속성·경계를 보존하고 NumPy/Tensor 입력을 dtype 유지·graph 없는 owned CPU Tensor로 복사한다. 끝난 Episode 뒤의 전이나 불연속 전이는 거절한다. (T,B) 범용 Trajectory는 만들지 않는다.
- MonteCarloReturnGenerator는 floating rewards `(T,)`에서 역순으로 전체 Return을 계산한다. Bootstrap 없이 gradient를 보존하며 기존 TD BaseTargetGenerator 상속을 강제하지 않는다.
- MCPredictionLearner(model, return_generator, visit_mode="first")는 TabularVEstimator table에 정방향 first/every visit을 선택하고 count 증가 후 `(G-V)/count`로 sample average를 갱신한다. learn 반환값은 전체 시간 순서 Returns `(T,)`, graph 없음이다.
- terminated=True만 MC 학습을 허용한다. truncated-only와 열린 Episode는 거절하고 양쪽 flag가 True면 실제 종료가 확인된 것으로 처리한다. 최종 next_observation은 추가 방문이 아니다.
- visit_counts는 Model.state_dict에 포함되지 않는 Learner의 long Tensor다. 재개 시 V-table과 Counts를 같은 gamma·visit_mode·정책으로 함께 복원한다. Sampler RNG와 시작 상태 RNG도 별도 학습 상태다.
- TabularEpisodeSampler는 알려진 MDP와 고정 Policy로 외부 Episode를 수집한다. terminal 시작은 거절하고 max_steps 중단은 truncated로 표시한다. 기본 MC 예제는 잘린 Episode를 버리고 계속하는 대신 실험을 명시적으로 중단한다.

## TD(0) Prediction 계약
- `TabularTD0PredictionLearner(model, target_generator, update_rule).learn(batch)`는 TabularVEstimator Model과 기존 TD0TargetGenerator/TabularVUpdateRule을 조합한다. State/reward/종료 fields `(B,)`를 table device·dtype에 맞추며 action, policy, MDP, Buffer를 학습에 사용하지 않는다.
- no_grad에서 모든 next V와 Target을 먼저 계산하고 원본 V-table을 갱신한다. B>1은 갱신 전 Target을 고정하며 중복 상태는 기존 Rule의 순차 갱신을 따른다. 전이별 Online TD는 시간 순서대로 B=1 learn 호출을 사용한다. 반환 signed TD error는 `(B,)`, graph 없음이다.
- terminated만 Bootstrap을 중단한다. 실제 final observation 제공은 수집자 책임이다. 비교 예제는 같은 완료 Episode를 MC와 TD에 전달하고 TD가 이미 수집된 Episode를 순차 처리한다. Streaming Runner는 아직 없으며 비교 중 cutoff는 모든 Learner 갱신 전에 보고한다.

## 가독성과 입력 검증
- Readability First: 공개 메서드는 수식과 알고리즘의 핵심 흐름이 쉽게 읽히도록 작성한다.
- 입력 검증은 필요한 Component의 짧은 `_validate()` 등의 메서드로 분리한다. 데이터 추출과 검증의 책임을 구분하되 메서드를 불필요하게 늘리지 않는다.
- 내부 타입과 입출력 계약을 신뢰한다. 필요한 입력의 부재와 의미적으로 잘못된 연결 등 중요한 계약 위반만 명확하게 검사한다.
- 동일한 공개 호출에서 같은 입력을 중복 검증하지 않는다. 설정값 검증은 설정 시 수행한다.
- 검증용 추상 클래스나 Validation Framework를 추가하지 않고, 모든 Base Class에 공통 검증 메서드를 강제하지 않는다.
- 검증 분리를 이유로 기능을 제거하지 않는다. 공개 API, 입력 오류 조건, 동률 처리, 행동 확률과 log_prob 등 기존 알고리즘의 동작을 보존한다.
- Docstring은 실제 책임과 검증 범위를 설명한다. 리팩터링 후 의미적 동작과 기존 회귀 테스트를 확인한다.

## 변경 절차
- 현재 로컬 저장소의 소스, 호출부, 테스트와 적용되는 AGENTS.md를 먼저 확인한다.
- 파일 수정 전에 기존 설계와 위 계약의 차이 및 변경 영향을 보고한다. 목표 계약을 현재 구현의 보장으로 간주하지 않는다.
- 기존 기능과 호환성을 최대한 보존한다. 반환 타입이나 파일 경로를 변경할 때 Runner, Learner, TargetGenerator, UpdateRule, 예제, 공개 import 및 checkpoint 영향을 함께 검토한다.
- 구현 수정 후 관련 단위 테스트와 기존 회귀 테스트를 실행하고 결과 및 실행하지 못한 이유를 보고한다.
- 이번 요청은 분석과 AGENTS.md 정리까지다. 사용자 승인 전 구현 파일을 수정하지 않는다.
- 기존 루트 AGENTS.md가 있으면 덮어쓰지 않고 병합안을 먼저 제시한다.

## 검증
- 프로젝트 루트에서 `python -m pytest -q`로 계약 단위 테스트와 MLP 호환성 회귀 테스트를 모두 실행한다.
- 예제 확인: `python -m examples.quickstart`.
- FrozenLake 확인: examples extra 설치 후 `python -m examples.frozenlake_tabular_q`. 희소 보상 성공률을 단위 테스트 합격 조건으로 삼지 않는다.
- CartPole 확인: `python -m examples.cartpole_dqn`. 전체 학습 성능은 예제로 보고하고 pytest에서는 결정적인 계산과 짧은 환경 Smoke Test를 검증한다.
- Gridworld DP 확인: `python -m examples.gridworld_dp`. Gymnasium 없이 Sutton 참조 V와 Policy/Value Iteration의 최적 V를 검증한다.
- Gridworld MC 확인: `python -m examples.gridworld_mc`. 같은 Uniform Policy에서 First/Every-visit을 비교하고 14개 비종료 상태의 DP 대비 RMSE를 보고한다. RMSE의 단조 감소를 pytest 조건으로 사용하지 않는다.
- Gridworld TD 비교: `python -m examples.gridworld_td --seeds 7 11 19`. 동일 경험에서 MC 두 모드와 TD alpha=.01/.05/.1을 비교한다. Episode/step별 RMSE와 Seed별 평균·표준편차를 보고하며 특정 알고리즘의 우위를 pytest 조건으로 삼지 않는다.

## Git 작업 원칙
- 작업 전 branch와 `git status`를 확인하고 기존 수정·미추적·stage 상태를 파악한다. 사용자의 기존 변경을 삭제하거나 덮어쓰지 않는다.
- 작업 후 전체 pytest와 관련 예제를 실행하고 통과한 변경을 검토한다. 실패하면 원인을 보고하고 임의로 커밋하지 않는다.
- 이번 작업과 관련된 파일 경로만 명시적으로 stage한다. `git add .`, `git add -A`를 사용하지 않고 관련 없는 기존 변경을 포함하지 않는다.
- 비밀번호·인증서·.env·대용량 데이터와 cache·임시 작업 파일은 커밋하지 않는다.
- 의미 있는 Conventional Commit 메시지를 사용하고 commit hash와 최종 Git 상태를 보고한다. 사용자 변경과 충돌하면 커밋 전에 확인한다.
- 별도 요청 없이 push하지 않으며 `--force`, `reset --hard` 등 파괴적 명령을 사용하지 않는다.
