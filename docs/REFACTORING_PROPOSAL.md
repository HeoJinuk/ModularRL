# ModularRL 전체 아키텍처 감사 및 최소 리팩터링 제안

작성일: 2026-10-15. 조사 기준: `main`, `9ccb7f7b0314acc42934c1533a445878fa74b928`.

**상태: 권장안 승인 후 Phase 1 구현 반영.** 최초 감사는 기준 Commit의 코드를 분석한 문서 작업이었다.
이번 Phase 1에서 `DQNLearner.optimize(prediction, target)`와 기존 learn/n-step 호출부 위임을 구현했다.
아래 1–8절의 현재 구조·중복 분석은 별도 표시가 없으면 감사 당시 기준이며, 9절 API와 13절 상태는
이번 구현을 반영한다. Replay/Window/World Model/Runner와 다른 후속 제안은 구현하지 않았다.

## 1. 종합 판단과 조사 범위

현재 구조는 작은 컴포넌트 조합이라는 목표를 상당 부분 달성했다. Expected SARSA부터 Q(σ), Dyna-DQN, Real n-step + Planning 1-step까지 핵심 패키지를 변경하지 않고 예제 함수로 구현했다. 출력 계약, 수학적 Target 계산, 표 직접 갱신, Neural Loss 계산을 다시 설계할 근거는 없다.

가장 우선할 개선은 **DQN의 Target 구성과 Optimizer 실행 사이에 작은 재사용 경계를 추가하는 것**이다. 현재 n-step DQN은 기본 Learner의 고정 1-step Target을 우회하기 위해 Loss/Optimizer 절차를 반복한다. 반면 모든 알고리즘을 한 Learner나 Target Context로 합치는 것은 계산·실행 시점의 차이를 숨긴다.

그다음 과제는 예제의 비공개 함수 의존, 안정적인 Replay/실험 재개 계약, World Model의 질의와 검색제어 분리다. 모두 현재 코드의 오류라는 뜻은 아니다. 관련 연구 기능을 제품화하기 전에 해결할 확장 경계이며, 첫 리팩터링에 한꺼번에 포함하지 않는다.

### 조사한 실제 구조

```text
modular_rl/                       # Python 파일 61개
  model/                          # Model, InputAdapter
  component/
    representation/               # Base, Identity, MLP
    estimator/                    # Base 역할, ActorCritic, policy/, value/
    action_selector/              # Base, Greedy, EpsilonGreedy, Categorical
    buffer/                       # BaseBuffer, ReplayBuffer
    target_generator/             # BaseTargetGenerator, TD0, MonteCarlo
    update_rule/                  # Tabular V/Q, TD Loss
  data/                           # outputs, experience, episode, updates, mdp
  learner/                        # Tabular Q/SARSA/TD0, DQN, MC 4종
  dp/                             # NumPy Bellman 계산
  environment/                    # Gymnasium TabularEnv, 수집, 기존 Sampler
examples/                         # Python 파일 18개; 알고리즘 조합·실행 루프
tests/                            # 계약·수학·통합·예제·Notebook 회귀
notebooks/dlfs4/                   # 04 DP, 05 MC, 06 TD, 07 Dyna, 08 Dyna-DQN
```

`agent/`, `runner/`, `OnlineRunner`, `EvaluationRunner`, 설치 패키지의 WorldModel/Planner는 **없다**. World Model과 Planning 함수는 예제에 있다. `pyproject.toml`은 `modular_rl*`만 설치한다. 예제 모듈을 재사용하는 현재 실습은 저장소 checkout을 전제로 한다.

AGENTS.md, README.md, ARCHITECTURE.md의 현재 범위 설명에는 과거 단계의 표현도 남아 있다. 예를 들어 설치 패키지에 Off-policy SARSA가 없다는 설명과 예제의 Off-policy n-step SARSA가 존재한다는 사실은 구분해야 한다. ARCHITECTURE.md의 초기 “TD(n)은 아직 구현하지 않는다”는 문구도 뒤의 예제 조합 구현을 부정하는 현재 전체 기능 목록으로 읽으면 안 된다. 이번 감사는 실제 파일과 호출부를 기준으로 한다. 문서의 범위·시점 정리는 이후 별도 작은 문서 변경 후보다.

### 근거 표기

아래 링크는 기준 Commit의 실제 파일이며, 함께 적은 클래스·함수는 존재를 확인했다. 문서에만 제안하는 이름은 반드시 “제안”으로 표시한다.

| 근거 | 실제 파일·확인한 정의 |
|---|---|
| M | [rl_model.py](../modular_rl/model/rl_model.py): `Model.encode/forward/act/evaluate_q`; [input_adapter.py](../modular_rl/model/input_adapter.py): `InputAdapter` |
| E | [estimator/](../modular_rl/component/estimator/): `TabularVEstimator`, `TabularQEstimator`, Neural heads, `ActorCriticEstimator` |
| A | [action_selector/](../modular_rl/component/action_selector/): `GreedyActionSelector`, `EpsilonGreedyActionSelector`, `CategoricalActionSelector` |
| D | [experience.py](../modular_rl/data/experience.py), [episode.py](../modular_rl/data/episode.py), [outputs.py](../modular_rl/data/outputs.py), [updates.py](../modular_rl/data/updates.py), [mdp.py](../modular_rl/data/mdp.py) |
| B | [replay.py](../modular_rl/component/buffer/replay.py): `ReplayBuffer.add/sample/clear/__len__` |
| T | [td0.py](../modular_rl/component/target_generator/td0.py): `TD0TargetGenerator.generate`; [monte_carlo.py](../modular_rl/component/target_generator/monte_carlo.py): `MonteCarloReturnGenerator.generate` |
| U | [tabular.py](../modular_rl/component/update_rule/tabular.py): `TabularVUpdateRule.update`, `TabularQUpdateRule.update`; [td_loss.py](../modular_rl/component/update_rule/td_loss.py): `TDLossUpdateRule.compute` |
| L | [learner/](../modular_rl/learner/): 각 `learn`; 특히 [dqn.py](../modular_rl/learner/dqn.py): `DQNLearner.learn/sync_target` |
| P | [algorithms.py](../modular_rl/dp/algorithms.py): `action_values`, `policy_evaluation`, `policy_improvement`, `policy_iteration`, `value_iteration`, `uniform_policy` |
| G | [tabular_env.py](../modular_rl/environment/tabular_env.py): `TabularEnv`; [episode_collection.py](../modular_rl/environment/episode_collection.py): `collect_episode` |
| X | [gridworld_dyna_dqn.py](../examples/gridworld_dyna_dqn.py): `NStepExperience`, `learn_real_n_step`, `synthetic_batch`, `plan`, `run_steps` |
| W | [gridworld_dyna_q.py](../examples/gridworld_dyna_q.py): `TabularWorldModel`, `learn_real_n_step`, `plan`, `run_steps` |

## 2. 책임과 현재 공개 API

`Learner = Buffer + TargetGenerator + UpdateRule + …`는 시스템의 논리적 구성이다. 실제 Learner가 Buffer·환경을 소유한다는 의미가 아니다.

| 컴포넌트 | 현재 API와 입출력 | 상태·책임 | 이번 권장 변경 |
|---|---|---|---|
| InputAdapter | `__call__`로 NumPy/Tensor/scalar → Tensor | 변환 설정; 정수 기본 long, floating은 Model dtype | 유지 |
| Representation | `forward(Tensor) -> Tensor` | 등록된 Network/Parameter; NumPy 변환 없음 | 유지 |
| Estimator | `forward(features) -> EstimatorOutput` | 추정 함수·등록된 Parameter/Buffer | 유지 |
| ContinuousQEstimator | `evaluate(features, actions)` | State-action 결합; `forward`만으로 평가 불가 | 유지 |
| Model | `encode`, `forward`, `act`, `evaluate_q`; 상속된 `.to`, `parameters`, `state_dict` | Adapter→Representation→Estimator; act만 no_grad | 유지 |
| ActionSelector | `select(estimate, *, deterministic=False)`; 이산 구현의 `probabilities(estimate)` | 행동 선택·실제 log probability; epsilon은 Buffer | 유지 |
| ReplayBuffer | `add(transition)`, `sample(batch_size)`, `clear()`, `len` | CPU owned snapshots, ring, 독립 NumPy RNG | 현재 API 유지; 재개 API는 조건부 후속 |
| TD0TargetGenerator | `generate(reward, next_value, terminated)` | gamma; 외부 continuation으로 수식 계산 | 유지 |
| MC Return | `generate(rewards)` | gamma; 시간순 rewards → 역순 return | 유지; TD Base 상속 강제 안 함 |
| Tabular Rule | `update(table, states, [actions,] target)` | alpha; no_grad, 원본 표 순차 갱신 | 유지 |
| TD Loss Rule | `compute(prediction, target, *, weights=None)` | loss_fn; signed error·scalar loss, graph 보존 | 유지 |
| Tabular Q/TD0 Learner | `learn(batch) -> Tensor` | Model/TG/Rule 참조; 고정 max-Q/next-V Bootstrap | 유지 |
| SARSA Learner | `learn(batch, *, next_action=None) -> Tensor` | 실제 다음 행동의 Q; 행동 선택은 외부 | 유지 |
| MC Learners | `learn(episode) -> Tensor` | V/Q와 방문 수/IS 누적 통계; 반환은 전체 Returns | 유지 |
| DQNLearner | 기준: `learn(batch) -> TDLossOutput`, `sync_target()` | Online/Target/Optimizer/TG/Rule; Target 동결·eval | **Phase 1 구현:** `optimize(prediction, target)` 추가 |
| TabularMDP / DP | 알려진 kernel; `policy_evaluation`, `policy_iteration`, `value_iteration` 등 | CPU NumPy float64 기대값 계산 | 유지 |
| Gymnasium 환경 | `reset`, `step`, `render`, `close` | 실제 상태·환경 RNG; TimeLimit은 외부 wrapper | 유지 |
| Episode 수집 | `collect_episode(env, select_action, seed=None, options=None)` | int/ActionOutput 콜백; owned 행동 확률 기록 | 유지 |
| 예제 World Model | `observe`, `sample(rng)`, `reset`, `len`, `state_dict/load_state_dict` | 관측 쌍별 최신 결정적 outcome | 현재 유지; `predict`는 후속 제안 |
| 예제 Planning | Tabular/Neural 각각의 `plan(...)` | synthetic 선택→Batch→학습; 환경 미참조 | 현재 함수 유지 |
| Runner / Agent | 없음 | 현재는 예제의 명시적 실행 루프 | 신규 클래스 제안하지 않음 |

같은 의미의 호출은 유지하되 다른 책임을 이름만 통일하지 않는다. `UpdateRule.update`는 표 변경, `TDLossUpdateRule.compute`는 계산이며, 둘을 `learn`으로 합칠 이유가 없다. 상태 없는 TargetGenerator에 `reset`을 강제하지 않는다.

## 3. 알고리즘 × 컴포넌트 매트릭스

표의 `I/V/Q/NQ`는 Identity + Tabular V / Tabular Q / MLP + Neural Discrete Q를 뜻한다. `ε/G`는 EpsilonGreedy/Greedy, `T0/MC`는 기존 TD0/MC Return Generator, `UV/UQ/UL`은 기존 V/Q Rule과 TD Loss Rule이다. “없음”은 숨겨진 공통 구현이 있다는 뜻이 아니다.

| 알고리즘 | Model / Estimator | 행동 선택 | 저장 / 샘플링 | Target / Rule | Learner | 실제 조합 함수·실행 위치 |
|---|---|---|---|---|---|---|
| DP Evaluation / Value Iteration / Policy Iteration | Model 없음; NumPy V와 알려진 TabularMDP | 정책 행렬; greedy tie 균등 | 전체 kernel 열거 | Bellman 기대값/최댓값; Rule 없음 | 없음 | `dp/algorithms.py`: `policy_evaluation`, `value_iteration`, `policy_iteration`; `gridworld_dp.solve_gridworld` |
| On-policy MC Prediction | I/V 2개 | 외부 uniform callback | 완료 Episode, Replay 없음 | MC / Learner 내부 sample average | `MCPredictionLearner(first/every)` | `gridworld_mc.train_and_compare` |
| On-policy MC Control | I/Q | ε; Episode 동안 Q 고정 | 완료 Episode | MC / pair sample average | `TabularMCControlLearner` | `gridworld_mc_control.train_and_evaluate` |
| Off-policy MC Prediction | I/V; 별도 Behavior I/Q | Behavior ε; 고정 target π(S,A) | Episode의 수집 당시 log b | MC / ordinary·weighted 누적 IS | `OffPolicyMCPredictionLearner` | `gridworld_off_policy_mc.train_and_compare` |
| Off-policy MC Control | I/Q | Behavior ε; 갱신된 Q의 최소-ID greedy target | Episode + log b | MC / 역순 weighted Q 갱신 | `OffPolicyMCControlLearner` | `gridworld_off_policy_mc_control.train_and_evaluate` |
| TD(0) Prediction | I/V | 외부 uniform callback | 동일 완료 Episode를 B=1 순차 전달 | T0(next V) / UV | `TabularTD0PredictionLearner` | `gridworld_td.train_and_compare` |
| Q-learning | I/Q | ε; 학습은 selector 미사용 | B=1 또는 Replay | T0(max Q) / UQ | `TabularQLearner` | `gridworld_sarsa.train_and_evaluate`; `frozenlake_tabular_q.train_and_evaluate` |
| SARSA | I/Q | 실제 다음 행동을 갱신 전에 선택·재사용 | B=1 + 외부 next_action | T0(gather Q) / UQ | `TabularSARSALearner` | `gridworld_sarsa.train_and_evaluate` |
| Expected SARSA | I/Q | 같은 ε 객체로 행동·확률 조회 | B=1; Batch도 지원 | T0(ΣπQ) / UQ | 신규 Learner 없음 | `gridworld_expected_sarsa.learn_expected_sarsa`, `train_and_evaluate` |
| Double Q-learning | 독립 I/Q 2개 | Q1+Q2의 명시적 임시 출력 → ε | 외부 Batch; 갱신 표 선택 RNG | 한 표 argmax, 다른 표 gather → T0 / UQ | 없음 | `gridworld_double_q_learning.combined_estimate`, `act_double_q`, `learn_double_q` |
| Online n-step SARSA | I/Q | 실제 endpoint 행동을 갱신 전에 선택 | owned ordered B=1 window | 실제 행동 Q에서 T0 역순 합성 / UQ | 없음 | `gridworld_n_step_sarsa.learn_n_step_sarsa`, `run_episode` |
| Off-policy n-step SARSA | I/Q | Behavior ε; 고정 target π | window + 기록 log b + 끝점 action/log | T0 역순, 해당 구간 ρ, effective target / UQ | 없음 | `gridworld_off_policy_n_step_sarsa.learn_off_policy_n_step_sarsa`, `importance_ratio`, `run_episode` |
| Tree Backup | I/Q | fixed π 또는 ε의 확률; 다음 실제 행동은 갱신 후 | ordered window | 같은 Q/π snapshot의 분기 기대값·T0 재귀 / UQ | 없음 | `gridworld_tree_backup.learn_tree_backup`, `run_episode` |
| Q(σ) | I/Q | fixed/dynamic π; Behavior ε | window + sampling-time log b + 정렬된 σ | T0와 control-variate 재귀 / UQ | 없음 | `gridworld_q_sigma.learn_q_sigma`, `run_episode` |
| DQN | NQ; 독립 Target 복제 | ε; 학습 중 selector 미사용 | 실제 Transition Replay | T0(Target max Q) / UL | `DQNLearner` | `cartpole_dqn.train_and_evaluate` |
| Dyna-Q | 공유 I/Q; World는 별도 TabularWorldModel | ε | real window; World 최신 쌍 표; Replay 없음 | Real T0 역순 max-Q, Plan T0 / UQ | 1-step은 TabularQLearner; real n-step은 함수 | `gridworld_dyna_q.learn_real_n_step`, `plan`, `run_steps` |
| Dyna-DQN (n=1) | 공유 NQ/Target/Optimizer; 별도 World | ε | 실제 Replay; synthetic는 sample로 즉시 Batch | Real/Plan T0 / UL | **같은** DQNLearner | `gridworld_dyna_dqn.synthetic_batch`, `plan`, `run_steps` |
| Real n-step + Planning 1-step DQN | 같은 NQ/Target/Optimizer | ε | 실제 Replay + pending window + 별도 NStepExperience deque | Real reward_sum + γ^m Target max Q; Plan T0 / UL | Real은 `learn_real_n_step`; Plan은 동일 DQNLearner.learn | `gridworld_dyna_dqn.make_n_step_experience`, `learn_real_n_step`, `plan`, `run_steps` |

### 반복 실행 책임과 본체 변경이 불필요했던 이유

| 묶음 | 예제에서 반복하는 실행 코드 | 핵심 패키지를 변경하지 않아도 된 이유 |
|---|---|---|
| DP | 수렴 설정·표 출력·최적 행동 표시 | TabularMDP가 joint outcome을 보존하여 전체 기대값 계산 가능 |
| MC Prediction/Control | seed·완료 Episode 수집·경계 확인·평가·통계 | Episode/Return과 표 추정 계약으로 충분; 통계의 수학은 Learner별로 다름 |
| Q/SARSA/Expected/Double | reset/step, B=1 생성, target-before-write, 별도 greedy 평가 | 외부 next_value를 받는 TD0와 raw table Rule을 직접 호출 가능 |
| n-step/Off-policy/Tree/Q(σ) | pending append, eligible head, 성공 후 pop, boundary flush | 외부 continuation을 T0에 넣고 기존 Rule에 최종 target 전달 가능; 시간 상태는 외부 보유 |
| Dyna-Q | real 업데이트, observe, plan, tail flush, reset | 실제/가상 Transition 모두 기존 Tabular QLearner가 수용 |
| DQN/Dyna-DQN | Replay warm-up·샘플링·학습 빈도·ε·sync·평가 | Batch source와 무관한 DQNLearner.learn; n-step만 Target 경계 부족으로 Optimizer 처리 반복 |

## 4. 중복 구현: 의미가 같은 부분과 다른 부분

### 4.1 우선 개선할 Neural Optimizer 경계

`modular_rl/learner/dqn.py:DQNLearner.learn`과 `examples/gridworld_dyna_dqn.py:learn_real_n_step`을 비교했다.

| 처리 | 공통성 | 달라야 하는 부분 |
|---|---|---|
| Batch 준비 | 학습 device·floating dtype, action long | TransitionBatch와 NStepExperience의 필드·실제 m·gamma provenance가 다름 |
| Online prediction | Online forward → 선택 action gather, gradient 유지 | 관측 조립·일반 관측 vs 예제 tabular ID·출력 검증의 적용 지점 |
| Target evaluation | no_grad, Target eval, 단일 max Q | 1-step reward와 n-step reward_sum·γ^(m−1) continuation |
| Target 생성 | 기존 TD0.generate 사용 | horizon·terminated·sampling metadata를 해석하는 조합 책임 |
| Loss·실제 갱신·반환 | **동일한 5개 statement** | 없음: compute, zero_grad, backward, step, detached TDLossOutput |

마지막 다섯 문장은 같은 Learner/Optimizer를 사용하면서도 반복된다. `DQNLearner.learn`은 target을 외부에서 받을 경로가 없어 n-step 함수가 전체 학습 호출을 재사용할 수 없다. 가장 작고 확실한 추출 대상이다. 전체 prediction/target 계산까지 중복이 사라진다고 과장하지 않는다.

### 4.2 Window: 저장 책임은 같지만 실행 순서는 다르다

| 구현 | action 선택·metadata | eligible update / flush |
|---|---|---|
| n-step SARSA | 다음 실제 action을 **갱신 전** 선택; truncated endpoint action도 한 번 선택 | head 갱신 후 pop; tail마다 최신 Q 재평가, endpoint 재샘플 안 함 |
| Off-policy n-step SARSA | 위 action과 sampling-time log b를 함께 보관 | 현재 행동 이후의 IS 구간에 endpoint 비율 포함; first action 비율과 혼동 금지 |
| Tree Backup | 끝점 행동 불필요; Q와 π를 같은 estimate에서 조회 | update 후 다음 실제 행동 선택; flush 중 행동 선택 안 함 |
| Q(σ) | 행동 선택 시 σ provider 한 번 호출; log b와 σ 정렬 | 실제 중간 행동의 계수만 사용; 끝점 σ 영향 없음; flush 중 action/provider 호출 안 함 |
| Tabular Dyna real n-step | max-Q continuation; world observe·plan은 별도 | plan 뒤 tail을 **실제로 갱신**하며 각 tail에서 최신 Q 사용 |
| Neural Dyna real n-step | raw pending에서 reward_sum/m/gamma를 요약 | plan 뒤 tail을 **Metadata Replay에 기록**; flush 자체가 추가 optimizer step은 아님 |

공통화 가능한 것은 owned B=1 record 보관, 연속성·중간 경계 검사, 실제 horizon, head 접근, 성공 후 제거, boundary에서 짧은 window를 순서대로 노출하는 동작이다. `act → learn → pop → flush`를 통합하는 범용 실행기는 권장하지 않는다. 특히 모든 tail target을 한 번에 계산하면 현재 Tabular flush의 순차 의미가 바뀐다.

초기 단계에는 Window 클래스도 추가하지 않는다. 후속 공통화가 필요하면 **데이터만 다루는 작은 함수**부터 추출한다. 학습·행동 선택·provider 호출 시점은 예제 루프에 남긴다.

### 4.3 Target 계산은 세 책임으로 구분한다

| 책임 | 현재 위치 | 유지할 경계 |
|---|---|---|
| Q/V 평가·max/gather/ΣπQ·snapshot·no_grad | Learner 또는 조합 함수 | Model을 참조할 수 있는 상위 코드 |
| reward/continuation의 Bellman 계산 | TD0.generate; MC.generate | Tensor 수식, shape/dtype/device 일치, Model/Buffer/Optimizer 미참조 |
| 표 변경 또는 Loss/Optimizer | Rule; Learner | Tabular 순차 쓰기와 Neural 평균 Loss를 구분 |

Tree Backup/Q(σ)는 endpoint 선택만 바꾸는 TD(n)이 아니다. 중간 Q/π/실제 행동/σ/ρ가 필요하다. 이를 `next_value_provider` 하나로 숨기거나, Model과 모든 Window 상태를 TD0TargetGenerator에 전달하지 않는다. 동일한 `reward + gamma * continuation` 호출은 수학적 재사용이며 제거할 중복이 아니다.

Importance Sampling도 곱셈 모양만 보고 하나의 공통 Return으로 합치지 않는다.

| 구현 | 보정 범위·순서 |
|---|---|
| Off-policy MC Prediction | 현재 행동부터 종료까지 suffix ρ. Ordinary는 선택 방문 수로, Weighted는 weight 합으로 나눔. ρ=0도 Ordinary count에 포함, Weighted 분모 0은 V 유지 |
| Off-policy MC Control | W=1에서 현재 Q/C를 먼저 갱신한 뒤 greedy 일치 여부 검사; 통과하면 이전 시점용 W에 1/b 반영 |
| Off-policy n-step SARSA | head 다음부터 실제 sampled endpoint까지 ρ. B=1 effective target을 기존 Rule에 전달; Neural Loss weight와 자동 동등하지 않음 |
| Q(σ) | 실제 중간 행동의 σρ+(1−σ)π만 재귀에 반영; 별도 전체 ρ를 다시 곱하지 않음 |

모든 Off-policy 경로의 coverage 조건 `π(a|s)>0 ⇒ b(a|s)>0`는 호출자 책임이다. 선택된 행동의 log b만으로 미관측 행동의 support까지 검증할 수 없다. 현재 log b의 유효성과 수치 overflow/underflow 검사는 유지할 가치가 있다. MC Learner의 작업 복사본을 통한 Episode 단위 원자성과 n-step 함수의 한 호출 단위 실패 전 무갱신도 구분한다.

### 4.4 Planning

두 `plan`은 model sample→학습 반복이라는 흐름을 공유하지만 계약은 다르다.

| 항목 | Tabular `gridworld_dyna_q.plan` | Neural `gridworld_dyna_dqn.plan` |
|---|---|---|
| 1회 계획 단위 | 1 synthetic transition, 표 갱신 | B개의 synthetic 표본, 1 gradient step |
| 샘플 구성 | `sample(rng).to_batch()` | `synthetic_batch`: sample B회, replacement 허용 |
| 반환 | signed error Tensor `(updates,)` | 요청/실제 갱신 수, TDLossOutput 목록 등 dict |
| 의존 | `learner.model.estimator.q_table` | `learner.online_model`, 일반 DQNLearner.learn |
| 빈 모델 / budget 0 | 0회 | 0회 |

Tabular plan에 DQNLearner를 그대로 넘길 수 없다. 이것만으로 BaseLearner/Planner가 필요하다고 결론 내리지 않는다. 공통화 이득은 작은 budget 반복 수준이며 현재 두 함수는 읽기 쉽다. 향후 sample-batch 함수와 learn-batch 함수가 실제로 여러 곳에서 교체될 때만 조합 경계를 공통화한다.

### 4.5 예제 간 의존

현재 의존은 다음과 같다.

```text
Q(σ) --일반 import--> Tree Backup --runpy--> Off-policy n-step --runpy--> n-step SARSA
Q(σ) --일반 import--> n-step SARSA
Dyna-Q --비공개 _validate_n/_validate_window import--> n-step SARSA
Dyna-DQN --일반 import--> Dyna-Q.TabularWorldModel
여러 Control 예제 --일반 import--> MC Control.evaluate_greedy
06 Notebook --runpy--> Expected/Double/n-step/Off-policy/Tree 예제
```

Q(σ)와 Tree는 로드된 dict 안의 private validator에도 접근한다. 경로 변경·함수명 변경·설치 형태 변경이 다른 예제로 전파되고, 정적 분석과 공유 함수의 책임 파악이 어렵다. 현재 runpy 사용이 RNG를 잘못 소비한다는 증거는 없으며 그런 버그를 가정하지 않는다.

공유 가치가 있는 부분은 Model에 의존하지 않는 ordered data 검증과 fixed-policy/log probability의 기본 계약이다. Return별 IS 구간·수치 실패 원자성·Q 검사까지 묶지 않는다. Greedy 평가·출력·실험 설정은 작은 명시적 반복을 허용한다. 모든 예제 함수를 설치 패키지로 옮기는 것은 해결책이 아니다.

## 5. Model·Estimator 감사

### 유지해야 할 장점

- Model은 InputAdapter→Representation→Estimator를 순서대로 호출한다. `act`만 no_grad를 적용하고 train/eval은 호출자가 결정한다. Loss/Buffer/Optimizer가 섞이지 않는다. [M]
- `EstimatorOutput(policy, values)`와 `ValueOutput(v, q: tuple, advantage)`는 원본 정보를 보존한다. 단일 Q와 복수 Q의 표현이 같고, Greedy/DQN은 여러 Q를 임의로 하나 선택하지 않는다. [D, A, L]
- Neural heads는 주입된 nn.Module을 등록하므로 parameters/state_dict/to(device)에 포함된다. 기본 Linear/MLP도 유지된다. NumPy 변환·device 이동은 Estimator 안에 없다. [E]
- Double Q는 독립 Model 2개와 명시적 Q 합산·교차 평가로 표현된다. tuple이 있다는 이유로 자동 평균/최소/합산 정책을 도입하지 않는다. [gridworld_double_q_learning.py](../examples/gridworld_double_q_learning.py): `combined_estimate`, `learn_double_q`.
- ActorCriticEstimator는 같은 features를 Policy/Value estimator에 전달하는 가벼운 wrapper다. representation/loss/optimizer를 중복하지 않는다. [actor_critic.py](../modular_rl/component/estimator/actor_critic.py)
- Gaussian은 `Independent(Normal(...), 1)`이므로 action dimension이 event이고 다차원 log_prob가 합산된다. 학습 가능한 state-independent log_std와 Mean Network 주입을 유지한다. 수치적으로 불가능한 scale 검사에는 실제 의미가 있다. [gaussian.py](../modular_rl/component/estimator/policy/gaussian.py)

### 구현 범위와 제약

| 항목 | 실제 상태·판단 |
|---|---|
| Distributional RL | payload는 원본을 담을 수 있으나 현재 Q Selector/DQN은 scalar discrete Q 전용. 지원된 알고리즘으로 표시하지 않음 |
| Continuous policy | Gaussian/Deterministic head는 있으나 대응 Continuous ActionSelector·SAC Learner 없음. 임시 Selector로 대체하지 않음 |
| Continuous critic | `Model.evaluate_q` 가능; ActorCriticEstimator.forward는 actions가 없어 ContinuousQEstimator를 그대로 결합할 수 없음. 현재 분리 actor/critic Model 방식 유지 |
| Target Network | Q Model과 같은 구조의 독립 복제, DQNLearner가 동결/eval 및 sync 수행. 복사 주기는 외부. 소유권 적절 |
| 모델 전체 checkpoint | parameters/buffers는 저장되지만 architecture 재구성, InputAdapter 설정, train/eval, Optimizer·RNG는 별도 |
| Dimension validation | Neural heads와 MLP/Tabular 생성자의 정수 허용 기준에 작은 차이가 있음. 유효 입력 지원을 좁히는 일괄 정리는 우선순위 낮음 |
| World Model | `(state, action) → (next_state, reward, termination, uncertainty)`는 추정용 Q/Policy Model과 다른 계약. 같은 Model facade에 합치지 않음 |

## 6. 데이터·상태·RNG 소유권

### 6.1 현재 데이터 계약

| 자료형 | 의미·shape | 복사·graph·장치 |
|---|---|---|
| Transition | 실제/가상 **1-step**; obs 임의 shape, 이산 action scalar/연속 `(A,)`, reward/flags scalar, optional log_prob | 생성 자체는 원본 참조 유지. frozen이 Tensor/NumPy 내용까지 불변으로 만들지는 않음 |
| Transition.to_batch | 한 전이를 `(1, ...)`로 | detach+owned copy; Tensor device 유지, NumPy/Python은 CPU; 알고리즘 검증은 별도 |
| TransitionBatch | replay `(B, ...)`; discrete action/reward/flags `(B,)`, continuous action `(B,A)` | `.to`는 floating만 dtype 변환; integer/bool 보존. 변환 없는 Tensor는 storage 공유 가능 |
| Episode | 시간순 Transition, 연속성·한 에피소드 경계 | owned detached CPU snapshot; 데이터로는 open/truncated 허용, MC Learner는 자연 종료 요구 |
| ActionOutput | Tensor action, Tensor 또는 None log_prob | 선택 결과; `Model.act` no_grad, 환경에서 scalar/NumPy 변환은 외부 |
| EstimatorOutput | policy와 ValueOutput; q는 tuple | 추정 원본과 gradient 보존; 소비자가 의미 선택 |
| TDLossOutput | signed td_error, scalar loss | Rule.compute는 graph 유지; DQNLearner.learn 반환은 둘 다 detached |
| NStepExperience | obs/action/reward_sum/next/term/trunc owned scalar + 실제 horizon m + gamma | 예제 전용 CPU snapshot. Q bootstrap은 저장하지 않고 학습 시 현재 Target을 평가 |

`terminated`는 Bootstrap mask이고 `terminated or truncated`는 수집 경계다. 둘을 done 하나로 합치지 않는다. 양쪽 True이면 자연 종료가 Bootstrap을 차단한다. 자동 reset의 reset 관측 대신 실제 final observation을 제공하는 책임은 수집자에게 있다.

### 6.2 n-step을 Transition에 합치지 않는 이유

현재 ReplayBuffer.add는 Transition의 단일 reward와 필드를 검사·복사하므로 NStepExperience를 그대로 저장할 수 없다. 이것은 1-step Replay의 명확한 계약이며 당장 Generic[Any] Buffer로 바꿀 이유가 아니다. reward_sum을 Transition.reward로 위장하면 gamma·horizon·원래 1-step 의미가 사라진다.

**권장:** 당장은 별도 NStepExperience와 bounded deque 유지. 공용 n-step Replay가 실제로 필요할 때만 전용 자료형/Batch/Buffer를 추가한다. 표본별 실제 m을 반드시 보존한다.

```text
reward_sum = Σ(i=0..m−1) gamma^i r_(t+i+1)
target = reward_sum + gamma^m * max Q_target(final_observation, ·)   # 비종료만
```

현재 `gamma` 저장은 reward_sum 생성 조건을 확인하는 provenance다. gamma 변경 시 기존 합을 재해석하지 않고 records를 다시 만들어야 한다. horizon 대신 discount만 저장하면 실제 길이·집계 조건을 잃는다. 현재 상수 gamma에서는 **horizon + 생성 gamma를 유지하고 discount를 유도**한다. 가변 시간 간격·상태별 discount가 필요해질 때는 누적 discount 계약을 별도로 정하며 지금 추가하지 않는다.

미래 n-step Batch는 leading `(B,)`의 horizon(long)과 reward_sum/flags 등으로 명확히 정의할 수 있다. 이것도 이번 제안의 1단계 구현 범위는 아니다.

### 6.3 소유 상태와 재개

| 소유자 | 보존해야 하는 상태 | 현재 public checkpoint 범위 |
|---|---|---|
| Model | parameter/buffer, ε; 별도 architecture/Adapter 설정/모드 | nn.Module.state_dict; 모든 설정을 포괄하지 않음 |
| MC Learner | visit_counts, IS weight_sums/weighted_return_sums, 고정 π·method·visit_mode·gamma | 통계 속성 외부 저장; 공통 Learner.state_dict 없음 |
| Tabular Rule / Target | alpha / gamma | 설정값 별도 저장 |
| DQNLearner | Online/Target 상태, Optimizer moment/step | 각 객체의 state_dict를 호출자가 묶음 |
| ReplayBuffer | owned data, ring 위치, schema, 독립 RNG 상태 | **public state_dict/load_state_dict 없음** |
| TabularWorldModel | 관측 쌍별 최신 outcome와 삽입 순서 | 예제 state_dict/load_state_dict 있음; RNG는 호출자 소유 |
| 수집 루프 | pending window/σ, 현재 선택 action/log b, 관측·카운터·일정 | 외부 소유; 편의 run 함수에 중간 재개 API 없음 |
| 환경/TimeLimit | 현재 상태, np_random, 경과 step | 일반 Gymnasium 환경의 공통 mid-episode 복원 API 없음 |

Dyna-DQN 테스트는 신뢰된 로컬 Replay 객체를 `torch.save/load(..., weights_only=False)`로 왕복시키고 고정 경험 stream 재개를 검증한다. 이는 버전에 안정적인 공개 Replay checkpoint나 임의 실환경 재개를 보장하지 않는다. 장시간 연구 재개를 공용 기능으로 제공하기 전에 명시적 저장 형식이 필요하다.

### 6.4 RNG·Device

- EpsilonGreedy/Greedy/Categorical sampling은 Torch RNG를 사용한다. `probabilities` 조회와 deterministic 선택은 RNG를 소비하지 않는다. 행동 log b는 실제 선택 시점에 저장한다. [A, G]
- Replay는 전용 NumPy Generator, TabularEnv는 Gymnasium `np_random`, 예제 planning/metadata sampling은 호출자가 전달하는 NumPy Generator를 사용한다. 학습용 확률적 Network가 Torch RNG를 사용하면 행동 RNG와 별도 분리 계약이 추가로 필요하다. 현재 결정적 Dyna 테스트만으로 이를 해결했다고 간주하지 않는다.
- Replay/Episode는 detached owned CPU, Model의 입력 경계가 device/dtype을 맞춘다. Learner는 Batch를 Model/table에 맞추되 정수 state/action과 bool flags를 보존한다. InputAdapter의 정수→long 기본 규칙은 정수로 표현한 연속 센서값의 자동 float 해석을 보장하지 않는다.
- `clear()`는 Replay 데이터와 위치를 초기화하지만 RNG를 reseed하지 않는다. `world.reset()`도 환경 Episode reset과 무관한 모델 지식 삭제다. 매 Episode 호출하면 안 된다.

## 7. Buffer 저장과 Sampling 설계

| 용도 | 현재 구현 | 후속 최소 경계 |
|---|---|---|
| Uniform Replay | Transition ring + without-replacement 무작위 B개; sample Batch는 원본과 독립 | 그대로 유지 |
| n-step Replay | 예제 metadata deque + 외부 NumPy sampling; raw Replay와 분리 | 실제 재사용 필요 시 전용 typed 저장소; 1-step Transition API 보존 |
| Sequence Replay | 없음; ring 물리 순서나 sample 결과를 trajectory로 쓸 수 없음 | 별도 Buffer가 적절. episode/env ID·시간 인덱스·경계·valid/padding·길이 계약 필요 |
| Prioritized Replay | 없음; 안정적 sample ID/확률/priority update 없음 | 기존 sample을 바꾸지 않는 추가 조회 API와 ID 기반 feedback부터 설계 |
| World Sampling | 예제 World의 관측 쌍 균등 추출, 최신 예측 생성 | query 선택과 조건부 prediction을 분리할 수 있음 |
| All-action Planning | 없음 | 선택 state × 가능한 action 질의; 미관측 쌍 처리·budget 정의 필요 |

Storage와 Sampling을 구분한다고 현재 하나의 작은 ReplayBuffer를 즉시 둘로 나눌 필요는 없다. 두 책임의 변경 요구가 실제로 생길 때 분리한다.

Prioritized Replay의 향후 반환 metadata에는 안정적인 slot+generation ID, 선택 확률 P(i), 필요한 importance weight가 포함될 수 있다. 지연된 feedback이 덮어쓴 slot에 적용되지 않도록 해야 한다. Learner의 detached signed TD error와 metadata를 외부에서 연결한다. weights 계산·정규화는 Sampler/호출자 책임이며 현재 TD Loss Rule은 입력과 같은 shape의 weights를 정규화 없이 곱해 평균할 수 있다. 아직 새 API 명칭이나 데이터형을 확정하지 않는다.

Real과 Synthetic은 **source별 저장소 또는 호출 경로**로 먼저 구분한다. 현재 Replay는 log_prob 존재/부재를 한 저장소에서 혼합할 수 없다. synthetic에 가짜 log_prob=0을 넣지 않는다. 혼합이 필요하면 별도 provenance metadata를 가진 명시적 Batch 조합을 설계한다. 모든 Transition에 source/priority/confidence 필드를 넣지 않는다.

## 8. Target/Learner API 대안 비교

다음은 감사 당시 비교한 API 후보들이다. **대안 D만 Phase 1에서 구현했으며 A/B/C는 미구현**이다. 기존 TD0/TDLoss의 계약을 바꾸지 않는 것을 전제로 비교한다.

| 대안 | 사용 예 | 장점 | 비용·제약 | 판단 |
|---|---|---|---|---|
| A: learn에 외부 target | `learn(batch, target=y)` | 기존 호출 보존; 한 메서드로 기본/사용자 target | 자동 계산/외부 target의 두 경로, no_grad 책임 구분 필요. n-step record는 TransitionBatch가 아니므로 거짓 1-step 변환 또는 새 입력 계약 필요 | 가능하지만 이번 최소안으로 선택하지 않음 |
| B: TargetGenerator/context 교체 | `learn(batch, target_generator=g)` | 단순 next_value 수식 교체에는 자연스러움 | 현재 주입은 이미 가능하나 Learner가 max/1-step continuation을 고정. Window/중간 Q/π/σ/Model을 context에 넣으면 순수 수학 책임을 넘어섬 | 순수 TG 유지; 범용 context 도입 안 함 |
| C: target과 Batch update 분리 | `y=g.generate(...); learner.update(batch,y)` | Q forward/gather까지 공유, source별 target 외부 구성 | n-step metadata용 관측/action 전달 경계 추가 필요. target-first 호출로 현재 online→target 순서를 바꾸기 쉬움. Tensor형 `update(observation,action,y)`도 가능하지만 준비·검증 범위가 커짐 | 재사용 수요가 더 생기면 재검토 |
| D: Tensor 최적화만 분리 | `learner.optimize(prediction,y)` | 기존 learn 유지, 동일 5개 statement만 공유. Window/target 수식과 현재 forward 순서가 눈에 보임 | 호출자가 online prediction/gather와 no_grad target을 구성; 모든 데이터 준비 중복은 제거 못 함 | **이번 권장안** |

A/C에서도 target shape·floating dtype·device 일치, Target no_grad, terminated/truncated 해석을 명시해야 한다. 외부 target만 받은 메서드는 수식의 terminal 처리가 맞는지 검증할 수 없다. 의미 계약은 target 작성자 책임이며 이를 위해 매번 Model을 다시 평가하지 않는다.

B의 주입 가능성과 계산 재사용은 다르다. TD0는 이미 SARSA/Q-learning/TD Prediction에 재사용되며 Tree/Q(σ) 재귀에서도 사용된다. 모든 알고리즘을 하나의 generate 시그니처로 포괄할 추가 Base Class가 필요한 것은 아니다.

### 전체 리팩터링 수준 비교

| 평가 기준 | A. 좁은 추가 API, 현재 구조 유지 | B. Window/Experience/Planner까지 중간 재편 | C. 전체 Learner/Runner 재설계 |
|---|---|---|---|
| 가독성 | 현재 수식·흐름 유지 | 공유 책임은 명료해질 수 있으나 추적 단계 증가 | 알고리즘 흐름이 framework 뒤로 숨을 위험 큼 |
| 사용 API 단순성 | 메서드 하나 추가 | 여러 자료형·scheduler 계약 학습 필요 | 현재보다 사용 개념 수 크게 증가 |
| 기존 호환성 | additive, learn/반환 유지 | adapter/alias와 동작 보존 검증 필요 | 광범위 마이그레이션 |
| 신규 알고리즘 | 외부 target + 공유 optimizer 조합 개선 | 데이터·planning 확장에 유리 | 이론상 넓지만 실제 요구 근거 부족 |
| 중복 감소 | 의미가 확실한 optimizer 5문장 | 저장/검증/실행 일부까지 감소 | 줄 수는 줄어도 알고리즘 차이 숨길 수 있음 |
| 테스트 변경량 | 새 API·기존 trace·n-step 공유 확인 중심 | 많은 window/checkpoint/Notebook 영향 | 기존 대부분 재검증 |
| 성능 영향 | 같은 연산에 Python 호출 하나; benchmark로 확인 | copy/Batch 구성 변경에 따라 달라짐 | 현재 근거로 예측 어려움 |
| 유지보수 | 작은 추가 경계 | 공용 데이터형·정책 유지 책임 증가 | 추상화 자체 유지 비용 큼 |
| 연구 확장 | Real/Plan Target 독립성과 상태 공유에 충분 | Neural/GPR/PER 제품화에 유리 | 확장성을 미리 가정하는 위험 |
| 구현 위험 | 낮음; 순서·gradient 회귀는 필요 | 중간; RNG/flush/복원 의미에 민감 | 높음 |

**선택: 전체 수준 A + Target API 대안 D.** Batch forward/gather나 Window boilerplate를 전부 제거하는 장점은 포기한다. 수학을 바꾸지 않고 이미 확인된 중복을 해결하는 쪽을 우선한다. 수준 B의 일부는 새로운 요구와 별도 승인 때 적용한다.

## 9. 승인 후 구현한 API의 정확한 계약

### 9.1 구현된 시그니처

```python
# Phase 1 구현: modular_rl/learner/dqn.py의 기존 클래스에 추가
def optimize(self, prediction: Tensor, target: Tensor) -> TDLossOutput:
    ...
```

기존 생성자, `learn(batch)`와 `sync_target()`를 유지했다. 공통 BaseLearner나 별도 Optimizer wrapper는 추가하지 않았다. weights/options도 추가하지 않았다. PER를 실제 구현할 때 기존 TD Loss Rule의 weights를 연결하는 keyword-only 확장은 후속 검토 대상이다.

| 항목 | 구현 계약 |
|---|---|
| prediction | Online Model의 선택 행동 Q `(B,)`, B≥1; gradient 유지; 관리되는 Online Parameter로 이어져야 함 |
| target | `(B,)`, 동일 floating dtype/device; 호출자가 no_grad로 계산한 고정 target |
| 검증 | 새로운 DQN 호출 경계의 1D/Online device·dtype·target graph를 확인. 공통 shape/dtype/device 검증은 기존 Rule 사용; 반복 full finite scan·graph provenance 탐색은 안 함 |
| target graph | `target.requires_grad=True`는 명확히 거절하는 **semi-gradient DQN** 경계. 조용히 detach하지 않음 |
| 실행 | 기존 Rule.compute → 같은 Optimizer.zero_grad → loss.backward → optimizer.step |
| 반환 | 업데이트 직전 signed td_error `(B,)`, scalar loss; 둘 다 detach, 기존 learn과 같은 의미 |
| 하지 않는 일 | Model 호출, action gather/선택, Replay, Target 생성/복사, sync, ε 변경, 환경 step |
| 호출자 책임 | 해당 Online prediction 사용, forward 이후 optimize 전 파라미터를 변경하지 않기, 의미적으로 올바른 target·sampling metadata |

TargetGenerator와 TDLossUpdateRule의 **gradient 보존 계약은 그대로**다. target gradient를 연구하려는 호출자는 TDLossUpdateRule.compute와 별도의 명시적 Optimizer 절차를 직접 사용한다. DQNLearner의 신규 optimize는 기존 learn과 같은 semi-gradient 목적을 명시한다. 이 제한은 기존 공개 동작을 축소하는 변경이 아니다.

현재 `learn`의 Online forward→Target forward→generate→compute→zero_grad→backward→step 순서를 유지한다. online/target 평가 순서를 바꾸면 stochastic module의 RNG나 BatchNorm 상태 관찰 순서가 달라질 수 있다. 수치 결과가 비슷하다는 이유로 순서를 변경하지 않는다. Optimizer 실패까지 rollback하는 트랜잭션은 제안하지 않는다.

### 9.2 Before / After

```python
# 기준 Commit: learn_real_n_step 마지막 부분
result = learner.update_rule.compute(prediction, target)
learner.optimizer.zero_grad()
result.loss.backward()
learner.optimizer.step()
return TDLossOutput(result.td_error.detach(), result.loss.detach())

# Phase 1 적용 후: 동일한 prediction/target 계산 뒤
return learner.optimize(prediction, target)
```

기본 DQNLearner.learn도 같은 다섯 문장을 `self.optimize(prediction, target)`로 대체했다. 각 경로에서 5문장→1문장, 두 호출부 합계 순감소는 8문장이다. 새로운 메서드·짧은 검증이 생기므로 전체 파일 줄 수 감소를 약속하지 않는다. 핵심 이득은 Loss/Optimizer/반환 계약의 수정 지점을 한 곳으로 만드는 것이다.

Tabular는 이미 `update_rule.update(table, indices, target)`라는 외부 target 갱신 경계가 있다. 이를 모방한 Tabular Learner.update를 추가할 필요가 없다. MC의 sample average/IS 통계와 DP의 전체 기대값도 별도로 유지한다.

## 10. 선정안의 구체적인 조합 예시

아래는 **조합을 설명하는 의사코드**다. optimize는 Phase 1에서 구현했으나 `prepare_*`, `sample_*` 같은 설명용 함수와 Advanced Dyna 확장은 현재 API로 주장하지 않는다. 학습식·데이터 준비는 기존 예제 함수에 남기며 계산을 Notebook으로 복제하지 않는다.

### A. 기본 DQN

```python
batch = replay.sample(batch_size)
metrics = learner.learn(batch)       # 기존 API 그대로; 내부만 optimize에 위임
if env_steps % sync_interval == 0:
    learner.sync_target()
```

기본 사용자 코드 추가는 0줄이다. Buffer sampling/ε/warm-up/동기화 스케줄은 예제에 남는다.

### B. n-step DQN

```python
# learn_real_n_step 안: 현재와 같은 준비·평가 순서
states, actions, next_states, reward_sum, terminated, m = prepare_n_step(records)
q, = learner.online_model(states).values.q
prediction = q.gather(-1, actions[:, None]).squeeze(-1)
with torch.no_grad():
    learner.target_model.eval()
    next_q, = learner.target_model(next_states).values.q
    next_value = next_q.max(-1).values
    continuation = gamma ** (m - 1) * next_value
    target = learner.target_generator.generate(reward_sum, continuation, terminated)
return learner.optimize(prediction, target)  # Phase 1에서 구현한 API
```

TD0가 gamma를 한 번 더 곱하므로 실제 γ^m이다. gamma=0, m=1에서도 현재 수식과 순서를 보존한다. 축약된 record에 bootstrap Q를 저장하지 않는다. Q 값·단일 critic·shape 검사 책임은 기존 조합 함수에 남는다.

### C. Dyna-DQN: Real 1-step + Planning 1-step

```python
if real_update_due:
    real_metrics = learner.learn(real_replay.sample(batch_size))
world.observe(real_transition)
planning = plan(world, learner, steps=planning_budget,
                batch_size=batch_size, rng=planning_rng)
# plan은 synthetic_batch를 만들고 같은 learner.learn을 호출한다.
```

별도 DynaDQNLearner, Optimizer, Target Network가 필요 없다. 현재 구현도 이미 이 형태다.

### D. Real n-step + Planning 1-step

```python
pending.append(real_transition.to_batch())
if len(pending) >= n:
    real_n_step_replay.append(make_n_step_experience(tuple(pending), td0))
    pending.popleft()
if real_update_due and len(real_n_step_replay) >= batch_size:
    real_metrics = learn_real_n_step(learner, sample_records(real_n_step_replay))
world.observe(real_transition)
planning = plan(world, learner, steps=planning_budget,
                batch_size=batch_size, rng=planning_rng)
if terminated or truncated:
    flush_short_windows_to_metadata(pending)  # 현재 Neural Dyna처럼 기록만
if target_sync_due:
    learner.sync_target()                    # 실제 env step 기준; source별 sync 아님
```

위 readiness는 요약이다. 실제 구현의 raw Replay warm-up·train_frequency·gradient_steps·원본 snapshot 조건은 유지한다. Real n=1은 현재처럼 기본 learn을 사용한다. Planning은 real window/warm-up 지연과 독립이며 같은 Adam moment와 Target을 사용한다.

### E. Q(σ)

```python
# 행동 선택 시, 실제 실행 전에 한 번 고정
selected = behavior_model.act(state)
selected_sigma = sigma_provider(...)
# 실행 뒤 log b가 든 record와 σ를 나란히 보관
pending.append(record)
pending_sigmas.append(selected_sigma)
if head_ready:
    metrics = learn_q_sigma(model, window, td0, q_rule,
                           target_policy=pi,
                           sigma=intermediate_sigmas_with_unused_horizon_placeholder)
    pop_head_after_success()
```

여기서는 기존 `examples/gridworld_q_sigma.py:learn_q_sigma`를 그대로 사용한다. 계산 정의는 다음과 같다.

```text
비종료/truncated horizon: 마지막 G = R_h + gamma Vπ(S_h)
자연 종료:               마지막 G = R_T
실제 중간 행동 k:         c_k = σ_k ρ_k + (1−σ_k) π(A_k|S_k)
재귀 continuation:        Vπ(S_k) + c_k [G − Q(S_k,A_k)]
```

본문 식 (7.17)의 `G_h:h=Q(S_h,A_h)`를 기준으로 하므로 끝점 σ/ρ는 이번 Return에 영향이 없다. 별도 전체 ρ를 다시 곱하지 않는다. σ=0은 Tree Backup, n=1은 Expected SARSA다. σ=1은 control variate를 포함하며 Sampled SARSA와 표본별 동일성을 일반적으로 요구하지 않는다. §7.6 본문/의사코드와 업데이트 설명의 차이를 기존 문서·Notebook대로 보존하며 **공식적으로 인정된 오류라고 단정하지 않는다**.

### F. Advanced Dyna-Q

```python
# 후속 연구용 의사코드: neural/gpr adapter와 이 함수들은 아직 없음
fit_world_offline(world, offline_dataset)   # 모델 전용 학습, Q Optimizer와 분리
for transition in real_stream:
    store_real(transition)
    adapt_world_online(world, transition) # Tabular는 observe; Neural은 별도 data/update
    if real_update_due:
        update_real_with_shared_learner()
    if planning_due:
        plan_selected_queries(world, shared_learner, planning_budget)
    if sync_due:
        shared_learner.sync_target()
# idle 시에도 환경과 독립적인 planning 함수를 같은 learner로 호출 가능
```

Tabular 환경 모델은 기존 `observe`를 dataset에 반복 적용하여 사전 채울 수 있다. Neural/GPR 사전학습·온라인 적응이 현재 구현돼 있다는 뜻은 아니다. 한 Q 학습 상태에 대한 호출은 순차 실행하며 idle planning과 실시간 학습을 동시에 쓰는 동시성 framework를 제안하지 않는다.

## 11. World Model·Planning·연구 확장 경계

### 11.1 현재 환경 모델과 최소 후속 계약

`TabularWorldModel`은 latest deterministic outcome map이다. 환경의 전체 kernel도, 경험 이력 Buffer도, 확률적 dynamics 학습기도 아니다. 외부 TimeLimit truncation과 behavior log probability는 dynamics 예측에서 제거한다. 확률적 환경에 적용하면 분포를 학습하는 것이 아니라 마지막 관측 하나로 덮어쓴다. [W]

| 역할 | 현재 | 후속 제안·적용 조건 |
|---|---|---|
| 관측 반영 | `observe(Transition)` 즉시 쌍 덮어쓰기 | 이름 유지. Neural/GPR는 데이터 추가와 fitting을 구분 |
| 지정 쌍 예측 | 없음 | Sampler 교체가 필요하면 `predict(observations, actions, *, rng=...)` 같은 Batch 질의 경계 추가 검토 |
| 쌍 선택 + 예측 | `sample(rng) -> Transition` | 호환 유지. uniform legacy convenience로 남길 수 있음 |
| 초기화 | `reset()` | 학습 지식 삭제. 환경 Episode reset과 별개 |
| 단일 gradient 적응 | 없음 | 해당 Neural 구현의 `update(batch)` 또는 외부 학습 함수; 모든 모델에 강제 안 함 |
| Offline fitting | 없음 | 해당 모델의 `fit(dataset, ...)` 또는 외부 함수; 한 sample observe와 구분 |
| 복원 | Tabular outcome state_dict | Neural/GPR는 자체 parameter·data·optimizer·모델 version·RNG 추가 |

위 `predict`의 **planning adapter 출력**은 `(B,...)`의 detached owned TransitionBatch로 수렴시킬 수 있다. state/action의 dtype·관측 shape는 보존하고 학습 device 이동은 호출자가 담당한다. Tabular의 미관측 쌍은 명확히 실패하거나 검색 후보에서 제외한다. 이를 일반적인 미관측 상태 extrapolation과 동일시하지 않는다.

Neural/GPR/확률 모델의 원본 출력은 분포·joint covariance·ensemble 등일 수 있다. 이를 항상 평균 next_state/reward로 줄이지 않는다. 모델 고유 prediction과 “표본을 뽑아 학습 Batch로 만드는 adapter”를 구분하고, reward/next_state/termination의 joint 의미를 보존한다. 분포형 출력·uncertainty 공통 dataclass는 실제 모델 두 개를 연결할 때 확정한다. 지금 추상 WorldModel Base를 추가하지 않는다.

### 11.2 Planning 함수와 검색제어

일반적인 후속 조합 형태는 `sample_batch()`와 `learn_batch(batch)`를 budget만큼 호출하는 작은 함수면 충분하다. 현재 두 plan을 즉시 이 함수로 합치지는 않는다. 표 업데이트 1회와 Neural gradient 1회의 의미가 다르므로 budget 단위를 드러낸다.

| 전략 | query 선택 | World 책임 | 학습/feedback 책임 |
|---|---|---|---|
| Uniform | 관측 쌍 또는 Replay state 선택; 독립 RNG | 지정 쌍 outcome 예측 | 같은 Q Learner/Rule |
| Prioritized | ID·priority queue/sampling | 모델 예측; 필요 시 predecessor 정보 | detached TD error→ID feedback; 가중치/정규화 외부 |
| All-action | state 선택 후 가능한 action 열거 | B×A outcome 질의 | Batch로 합칠지 순차 갱신할지 명시 |
| Confidence 기반 | 후보와 fidelity/uncertainty 정책 | 원본 uncertainty·모델 version 제공 | σ/가중치/허용 horizon 결정과 기록 |

Prioritized Dyna sampling과 predecessor를 역전파하는 Prioritized Sweeping은 동일하지 않다. 후자를 원하면 predecessor 관리·queue 갱신 규칙이 추가된다. All-action을 한 Batch optimizer step으로 처리하는 것과 A번 step하는 것도 Adam 상태·평균 Loss·target snapshot이 달라 동일하지 않다.

### 11.3 연구 요구에 대한 현재 충족도

| 요구 | 현재 근거 | 추가 결정 |
|---|---|---|
| Real/Plan 독립 horizon | X의 real n=1/3/5 + planning 1 | Phase 1 optimize로 Loss/Optimizer 반복 제거 완료 |
| 공유 Online/Adam/Target | 동일 DQNLearner 인스턴스 | 별도 Learner·Optimizer를 source별로 만들지 않음 |
| Real 지연 중 Planning | X.run_steps가 real readiness와 별개로 observe/plan | 유지; real이 아직 0회여도 planning 가능 |
| 독립 Planning budget / real 빈도 | X의 planning_steps, train_frequency, gradient_steps | budget 의미는 gradient step, synthetic 수는 B배임을 유지 |
| 실제/가상 분리 | raw Replay는 실제만, synthetic 즉시 Batch | 혼합·source 비율과 provenance는 외부 |
| Offline/online 모델 적응 | Tabular.observe로 사전/온라인 채우기 가능 | Neural/GPR fitting·drift·uncertainty 미구현 |
| Idle planning | W/X.plan은 환경 없이 호출 가능 | 시간 제한·중단 정책은 호출자; 병렬 optimizer 쓰기 안 함 |
| Target sync | X는 실제 env step 기준 Real/Plan 뒤 hard sync | idle 중 env step이 없으면 자동 sync도 없음. gradient 기준으로 바꾸려면 명시적 다른 실험 |
| Priority feedback | signed detached TD error 이미 반환 | sample ID·priority 규칙·stale ID 처리 추가 필요 |
| All-action query | 현재 없음 | 지정 쌍 predict, 후보 action, 미관측 쌍, Batch/순차 의미 필요 |
| 신뢰도 기반 Q(σ) | σ provider와 sampling-time σ 보존은 있음 | synthetic 순서 경로·π/b·모델 version/fidelity/uncertainty·snapshot·경계 정의 필요 |

모델 신뢰도 기반 Q(σ)는 단일 synthetic TransitionBatch만으로 완성되지 않는다. 예측 신뢰도와 σ의 계산 시점을 고정하고 행동 시점에 저장해야 한다. 나중 모델로 과거 σ/log b를 재계산하지 않는다. 본문 경계 조건·control variate·비율 범위도 유지한다. 현재 callback closure로 모델을 조회할 수 있다는 사실은 이 전체 연구 계약이 검증됐다는 뜻이 아니다.

## 12. Runner와 실행 스케줄

비교할 기존 Runner 클래스는 없다. 실제 기준은 `gridworld_sarsa.train_and_evaluate`, n-step 각 `run_episode`, W/X의 `run_steps`, `cartpole_dqn.train_and_evaluate`다.

| 질문 | 현재에 근거한 답 |
|---|---|
| 기존 Runner로 Dyna를 표현 가능한가? | Runner 없음. 예제의 명시적 함수가 이미 실제/가상 학습과 sync를 조합함 |
| 책임은 어디까지인가? | 이후 추출한다면 reset/step/final observation·경계·카운터와 명시적 호출 순서. Target 수학·Loss·모델 fitting 수식은 제외 |
| Planning schedule 소유권 | 환경 step/clock을 아는 외부 실행 코드. plan 자체는 env clock이나 sync를 몰라야 함 |
| Callback이면 충분한가? | 현재 규모에서는 직접 함수 호출 또는 좁은 콜백으로 충분. 공용 Runner callback API는 현재 없음 |
| 순서 변경 | 지금 함수 순서를 직접 바꿔 실험 가능. 순서 변경은 알고리즘·RNG 변화이므로 설정 옵션 뒤에 숨기지 않음 |
| Flush 책임 | pending을 소유하는 수집/실행 함수. Tabular tail update와 Neural metadata 기록을 구분 |
| Idle planning | 환경 loop 밖에서도 같은 plan/학습 상태를 호출. 예산·취소는 호출자; 새 Event Bus/Execution Graph 불필요 |

현재 X 순서는 실제 step/저장·eligible 생성→조건부 Real gradient→World.observe→Planning→경계 metadata flush→조건부 Target sync→reset이다. Tabular W는 planning 뒤 flush가 실제 Q 갱신이라는 차이가 있다. action 선택 시점도 SARSA와 Tree/Q(σ)가 다르다. 향후 Runner 추출의 완료 조건은 코드 길이 감소가 아니라 이 순서를 명시적으로 표현하고 재현하는 것이다.

## 13. 단계별 마이그레이션과 예상 파일

승인받아 구현한 범위는 **1단계만**이다. 뒤 단계는 자동 후속 작업이 아니며 필요한 연구 기능과 추가 승인에 따라 선택한다. 회귀 검증은 마지막에 몰아 하지 않고 각 단계의 완료 조건으로 둔다.

### 1단계: DQN 최적화 경계 — 구현 완료

| 작은 작업 단위 | 변경 예상 파일·정의 | API/영향 | 테스트·완료 기준·위험 |
|---|---|---|---|
| 1a. additive API | `modular_rl/learner/dqn.py`: optimize 추가, learn 마지막 부분 위임; 신규 `tests/test_dqn_optimization.py`, 기존 `tests/test_dqn_learner.py` 회귀 | 기존 생성자/learn/sync/exports 유지. 새 메서드 1개 | 같은 forward→generate→Loss→optimizer trace·device/dtype·detach·Target 불변; graph target 거절과 foreign prediction 책임 문서화. 위험: 순서/zero_grad 변경 |
| 1b. 실제 호출부 재사용 | `examples/gridworld_dyna_dqn.py:learn_real_n_step`; `tests/test_dyna_dqn_composition.py`, `tests/test_gridworld_dyna_dqn.py` | Metadata·Target식·public 함수명 유지; 5문장→optimize | n=1/3/5 target·같은 Adam state·Planning=0 bit exact·flush/readiness/sync/재개 일치. 위험: hidden extra forward/step |
| 1c. 사용 계약 반영 | `docs/ARCHITECTURE.md`, 필요 시 `README.md` | 새 경계와 semi-gradient 명시; 예제 코드와 일치 | 기존 Notebook 출력 보존. 이번 설계 문서 작성 단계에서는 해당 파일 수정 안 함 |

Phase 1은 승인된 `refactor(dqn): extract shared optimization step` Commit 단위로 검증한다. 신규 검증은 `tests/test_dqn_optimization.py`에 두고 기존 DQN/Dyna 테스트는 변경 없이 실행한다. Notebook 소스·이전 Cell·Metadata는 보존한다. 변경 범위는 기존 Learner·예제 각각 한 파일과 관련 테스트·문서뿐이다.

### 2단계: 경험 재개·공유 데이터 검사 — 조건부

| 작업 | 변경 예상 파일 | 완료 기준 / 영향 / 위험 |
|---|---|---|
| Replay 공개 checkpoint | `component/buffer/replay.py`, `tests/test_replay_buffer.py`, `tests/test_gridworld_dyna_dqn.py`, ARCHITECTURE | 제안 state_dict/load_state_dict가 owned records·ring 위치·RNG·schema를 보존하고 검증 실패 전 상태 유지. add/sample/clear는 호환. 장기 재개 수요 있을 때 수행 |
| 데이터 전용 Window 검사 | **신규 후보** `modular_rl/data/sequence.py`; 각 n-step 예제와 composition 테스트 | Model에 의존하지 않는 연속 B=1 검사만 옮김. 기존 예제 함수/Notebook loader 경로 보존. 행동 선택·flush 순서 공통화 안 함 |
| n-step 저장소 공용화 | **신규 후보** `data/n_step.py`, `component/buffer/n_step.py`, 전용 테스트 | 두 번째 실제 Neural 소비자가 필요할 때 별도 설계. Transition 필드 변경 없이 m/gamma·snapshot·sample/checkpoint 확정. 현재 단계에서 만들지 않음 |

중립 공유 위치를 examples 내부로 잡으면 설치·Notebook cwd 문제가 남는다. 설치 패키지로 옮길 함수는 데이터 계약으로 설명 가능한 것만 선정한다. 예제의 전체 Return/Runner를 이동하지 않는다. 작은 private 검사 중복을 줄이기 위해 위 파일 모두를 만들 필요는 없다.

### 3단계: World prediction·검색제어 — 첫 교체 모델이 있을 때

후보 파일은 **신규** `modular_rl/component/world_model/tabular.py`와 initializer, 해당 테스트, Dyna 두 예제다. 현재 예제의 `TabularWorldModel` import 경로를 alias로 보존하고 sample/observe/checkpoint 결과를 유지해야 한다. Neural/GPR 공통 Base부터 만들지 않는다.

첫 목표는 관측 쌍 선택과 지정 쌍 예측을 분리하는 것뿐이다. Offline fitting 함수·새 Neural 모델·priority queue·all-action은 각각 별도 연구 작업이다. 완료 기준은 두 번째 모델/검색 방식이 Q Model/Rule 변경 없이 adapter로 연결되고 장치·uncertainty 의미가 명확한 것이다. 모델/Replay/환경을 섞어 하나의 Model 클래스로 만드는 안은 제외한다.

### 4단계: Runner 통합 — 현재 제외

현재 예제와 함수 호출로 요구를 표현한다. 여러 실제 실행 경로에서 같은 수집 책임이 반복되어 추출 효과가 입증되기 전에는 Runner/Agent/Callback 계층을 추가하지 않는다. 향후 필요 시 수집·일정만 좁게 다루며 Window/Target을 Runner가 소유하도록 강제하지 않는다.

### 5단계: 연구 회귀 — 독립 구현 단계로 두지 않음

Advanced Dyna/Neural World/PER/Q(σ)-Planning을 이번 리팩터링의 완료 조건으로 확대하지 않는다. 현재 구현된 혼합 horizon·idle planning·공유 optimizer·확률/RNG 계약을 각 단계마다 검증한다. 아직 없는 연구 기능은 해당 기능을 구현할 때 별도의 수학·데이터·성능 검증을 추가한다.

## 14. 회귀 검증 계획과 이번 실제 실행

### 승인 후 변경에 대한 회귀 계획

| 보존할 계약 | 실제 기존 테스트 근거·추가 확인 |
|---|---|
| 1. Planning=0 DQN 동등성 | `test_gridworld_dyna_dqn.py:test_planning_zero_bit_exact_independent_dqn_reference` 유지; 중간 parameter/optimizer/RNG trace 일치 |
| 2. σ=0 Tree Backup | `test_q_sigma_composition.py:test_sigma_zero_is_bit_exact_tree_backup_with_same_rng` |
| 3. n=1 Expected SARSA | `test_tree_backup_composition.py:test_dynamic_one_step_matches_expected_sarsa_actions_rewards_targets_errors_table_rng`; Q(σ) n1=4.6 유지 |
| 4. Real n-step Target | `test_dyna_dqn_composition.py:test_n_step_reward_sum_horizon_current_target_and_gradient`; 저장 후 Target 변경해도 현재 Target 사용 |
| 5. Planning 1-step Target | 같은 파일 `test_target_from_target_network_shape_dtype_device`; 같은 one-step Learner 호출 확인 |
| 6. Target sync | `test_dqn_learner.py:test_sync_copies_persistent_and_nonpersistent_buffers_and_learn_preserves_target_buffers`; env-step 기준 횟수 유지 |
| 7. 공유 Optimizer state | `test_dyna_dqn_composition.py:test_real_then_planning_same_sgd_momentum_parameter_and_target`, `test_n_step_one_matches_existing_dqn_and_uses_same_adam_state` |
| 8. Truncation Bootstrap | Q/SARSA/TD0/DQN/composition의 terminal/truncated 조합, 양쪽 True, 실제 final observation. 단일 done으로 축약 금지 |
| 9. Flush | `test_n_step_sarsa_composition.py:test_truncation_tails_re_evaluate_current_q_after_each_write_without_resampling`; Tree/Q(σ) no endpoint sampling; Dyna metadata-only flush |
| 10. RNG | probabilities 무소비, 실행 action 재사용, log b/σ 선택 시 보존, 독립 planning RNG; stochastic module 회귀도 신규 API에서 추가 고려 |
| 11. Checkpoint | `test_gridworld_dyna_dqn.py:test_checkpoint_restores_models_optimizer_replay_rng_pending_metadata_and_counters`; n-step/Q(σ) 고정 stream 재개 유지; 실환경 일반 복원으로 확대 해석 안 함 |
| 12. Notebook | `test_dlfs4_notebooks.py:test_run_all_twice_reproduces_text_and_figures_in_same_kernel`: 04–08 각각 같은 kernel Run All 두 번, text/numeric/PNG 비교 |

기존 Q-learning/SARSA/Expected/Double/n-step/Off-policy/Tree/Q(σ)/Dyna-Q/Dyna-DQN 관련 수학 테스트와 Model/Estimator/Buffer/TD0/Rule 전체 pytest를 함께 유지한다. Q(σ) 필수 n=3 target `3.303325`, 업데이트 Q `2.6516625`, n=1 target `4.6`, σ=1 control-variate 차이를 고정한다. float32/64, CPU 및 가능한 CUDA를 검증한다. 확률적 학습 성능 우위·단일 seed RMSE 감소를 통과 조건으로 만들지 않는다.

승인 후 구현 검증에는 quickstart, FrozenLake, CartPole 짧은 smoke, 기존 DP/MC/TD/Control 및 n-step/Dyna 예제 smoke를 포함한다. 수식 검증과 장기 학습 성능 실험을 구분한다.

### 이번 감사에서 새로 실행한 테스트

실행 환경: `modularrl-py312`, Python 3.12.15, CUDA 미지원. 아래 **선택 테스트만** 이번 감사에서 새로 실행했다.

```bash
python -m pytest -q \
  tests/test_dqn_learner.py \
  tests/test_dyna_dqn_composition.py tests/test_gridworld_dyna_dqn.py \
  tests/test_update_rules.py tests/test_n_step_sarsa_composition.py \
  tests/test_off_policy_n_step_sarsa_composition.py \
  tests/test_tree_backup_composition.py tests/test_q_sigma_composition.py \
  tests/test_expected_sarsa_composition.py tests/test_replay_buffer.py
```

결과: **729 passed / 71 skipped / 26.18초**. skipped는 CUDA 미지원이다. 이것은 설계안의 미구현 API가 아니라 현재 코드의 수학·호출 경계에 대한 확인이다.

기준 Commit의 기존 전체 검증 기록은 **1,974 passed / 111 skipped**, 다섯 Notebook의 Run All 두 번 및 기존 예제 smoke 통과다. 이는 ARCHITECTURE.md의 기준 기록을 인용한 것이며 **이번 감사에서 전체 pytest나 Notebook/학습 예제를 다시 실행했다고 보고하지 않는다**. 코드가 변경되지 않은 문서 감사라 전체 재실행은 생략했다.

### 승인 후 Phase 1의 신규 실행 기록

최초 감사와 구분하여, Phase 1에서는 신규 최적화 테스트 **45 passed / 24 skipped**, 기존 DQN/Dyna
세 파일 **140 passed / 45 skipped**를 실제로 실행했다. skipped는 CUDA 미지원이다. 독립 초기 상태의
기준 Optimizer 절차와 비교하며 dtype·Parameter/Buffer·Adam/SGD Momentum·RNG까지 정확히 일치했다.

Real n=1/3/5 × float32/64 × Adam/SGD Momentum의 독립 전체 루프 비교 12개 설정도 통과했다.
각 80 실제 Step의 결과·공유 학습 상태·Flush·후속 Replay 표본과 모든 RNG가 기존 절차와 bit-exact였다.
18개 예제 CLI Smoke Test를 통과했다. 이번 전체 pytest의 실제 결과는
**2,019 passed / 135 skipped(CUDA 미지원), 2,840.62초**다. 기준 기록 1,974/111과 구분한다.
04–08 각 Notebook의 같은 Kernel Run All 두 번을 통과했고, text/numeric/PNG 40개가 두 실행 및
기존 저장본과 일치했다. Cell source/ID/metadata와 Notebook metadata도 같으며 원본 Notebook의
byte hash는 변경되지 않았다. 세부 내용은 [ARCHITECTURE.md](ARCHITECTURE.md)의 Phase 1 검증 기록에 남겼다.

## 15. 호환성과 예상 Breaking Changes

1단계는 additive API다. Model/Estimator/ActionSelector/Transition/TransitionBatch/TD0/Rule/기존 Learner 생성자·learn·sync_target·공개 import·checkpoint 필드를 바꾸지 않는다. 기존 `learn` 결과의 detach·signed error·행 순서·Loss 의미도 유지한다.

새 optimize의 graph target 거절은 신규 API 계약이다. 기존 TDLossUpdateRule의 target gradient를 제한하지 않는다. 기존 learn 호출 경로의 validation 오류·optimizer 호출 순서도 유지한다.

후속 단계에서 자료형/Buffer 저장 형식/공유 함수 위치를 변경하면 backward loading·alias·삽입 순서·RNG state version을 따로 설계한다. 예제 import 경로 변경, 필드 통합, runtime 재개 형식의 암묵적 교체는 호환 변경으로 취급하지 않는다. 지원 불가능한 경우에는 사전에 범위를 설명하고 승인받는다.

당장 해결하지 않을 반복은 Q/SARSA의 짧은 Batch 검증·표 준비, greedy 평가 루프, seed/CLI/통계 출력, DP와 MC/TD의 별도 비교 설정이다. 표본별 순차 갱신, Neural Batch 평균 Loss, MC 누적 평균, DP 기대값의 차이도 공통 Base 뒤로 숨기지 않는다.

## 16. 승인된 결정과 남은 질문

| 결정 | 권장안 | 미결·후속 범위 |
|---|---|---|
| 첫 구현 범위 | 1단계 DQNLearner.optimize 추가 + n-step 호출부 교체 + 회귀/문서 | Window/Replay/World/Runner 확대는 포함하지 않음 |
| 신규 API | `optimize(prediction, target)` | weights keyword는 PER 실제 작업 때 검토 |
| Gradient | DQN 신규 경계는 고정 target 요구·graph target 거절, 반환 detached | target 미분 연구는 기존 TDLossUpdateRule 사용 |
| Tabular | 기존 Rule.update 그대로 사용 | 동일 이름을 위해 Learner API 추가 안 함 |
| 데이터 | Transition과 NStepExperience 분리, 실제 m+생성 gamma 보존 | 공용 n-step Batch/Replay는 두 번째 실제 수요 때 설계 |
| 예제 의존 | 당장 유지하되 private/runpy chain을 더 늘리지 않음 | 데이터 전용 공유 검사 추출 위치·호환 wrapper는 조건부 2단계 |
| 장기 재개 | 공개 Replay checkpoint는 별도 작은 후속 후보 | 버전 형식·실환경 mid-episode 복원 범위 미결 |
| World 교체 | Q Model과 분리, query selection과 prediction 경계 | Neural/GPR joint output·uncertainty·sampling RNG는 실제 모델로 검증 후 확정 |
| Runner | 현재 추가 안 함 | idle scheduling·동시성/취소 API는 새 요구가 있을 때 검토 |

사용자가 위 1단계 구현 범위와 optimize의 semi-gradient 계약을 승인했으며 해당 범위만 구현했다. 이 문서의 후속 후보를 일괄 구현 승인으로 간주하지 않는다.

### 최초 감사와 Phase 1의 변경 범위

최초 감사는 이 문서만 추가했고 추적 파일 143개와 HEAD를 보존했으며 Commit/Push하지 않았다.
Phase 1은 위에서 승인된 DQN 최적화 경계·호출부·관련 테스트·문서에 한정한다.
설계 문서도 보존하여 검증 후 함께 Commit하고, Push하지 않는다.
