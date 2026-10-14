# 밑바닥부터 시작하는 딥러닝 4 — ModularRL 실습

프로젝트 가상환경에 `pip install -e ".[notebooks]"`로 설치하고 이 폴더에서 세 Notebook을
위에서부터 실행한다. Kernel 등록·선택 명령은 [프로젝트 README](../../README.md#jupyter-notebooks--deep-learning-from-scratch-4)에 있다.
`plotting.py`는 좌표에 값·정책 화살표를 표시하는 작은 함수만 제공한다. 환경 생성·샘플링·RMSE·학습은 포함하지 않는다.

| Notebook | 실행 지원 |
|---|---|
| [04_dp_modularrl.ipynb](04_dp_modularrl.ipynb) | 두 상태 평가, GridWorld 표시, Policy Evaluation/Improvement/Iteration, Value Iteration |
| [05_mc_modularrl.ipynb](05_mc_modularrl.ipynb) | 주사위 평균, MC Return, On/Off-policy Prediction/Control, Ordinary/Weighted IS·DP 비교 |
| [06_td_modularrl.ipynb](06_td_modularrl.ipynb) | TD(0) Prediction, 동일 Episode MC 비교, Q-learning/SARSA, 실제 Next Action·Greedy 비교 |

## 확인한 원본

원본은 [WegraLee/deep-learning-from-scratch-4](https://github.com/WegraLee/deep-learning-from-scratch-4)의
커밋 `b82cd6432b4e63ce6a4ab2b925fc74a1227fb06a`다. 세 `.ipynb`와 ch04/ch05/ch06의 모든
Python 파일, common/gridworld.py·gridworld_render.py·utils.py를 직접 읽었다.
원본 README는 소스의 MIT 라이선스를 명시한다. 원본 코드를 실행 시 내려받거나 import하지 않는다.

원본 Notebook은 셀에서 `dezerogym`을 설치하고 해당 GridWorld를 import한다. 같은 저장소의
장별 Python 예제는 `common.gridworld.GridWorld`를 사용한다. 본 환경 계약은 버전이 고정되지 않은
설치 패키지 대신 확인한 common/gridworld.py를 기준으로 삼았다.

| 원본 Notebook | 셀의 실습 순서 | 설정·출력 |
|---|---|---|
| [04_dynamic_programming.ipynb](https://github.com/WegraLee/deep-learning-from-scratch-4/blob/b82cd6432b4e63ce6a4ab2b925fc74a1227fb06a/notebooks/04_dynamic_programming.ipynb) | dp → dp_inplace → GridWorld 표시 → policy_eval → policy_iter → value_iter | γ=.9; 두 상태 허용오차 .0001, GridWorld .001; dict V·반복 수·V/정책 그림 |
| [05_montecarlo.ipynb](https://github.com/WegraLee/deep-learning-from-scratch-4/blob/b82cd6432b4e63ce6a4ab2b925fc74a1227fb06a/notebooks/05_montecarlo.ipynb) | dice → mc_eval → mc_control → importance_sampling → mc_control_offpolicy | 주사위 1,000표본; MC Prediction 1,000 Episode; Control 10,000; IS 100표본; V/Q 그림·평균/분산 |
| [06_temporal_difference.ipynb](https://github.com/WegraLee/deep-learning-from-scratch-4/blob/b82cd6432b4e63ce6a4ab2b925fc74a1227fb06a/notebooks/06_temporal_difference.ipynb) | 공통 행동 함수 → td_eval → sarsa → sarsa_off_policy → q_learning → q_learning_simple | γ=.9; TD α=.01/1,000 Episode; Control α=.8, ε=.1/10,000; Simple Q 1,000; V/Q 그림 |

common의 주요 구성은 `GridWorld`의 next_state/reward/reset/step과 `Renderer`의 render_v/render_q,
utils의 argmax/greedy_probs/plot_total_reward다. 원본 Agent들은 행동 선택·기억·추정·갱신을
한 클래스에 넣지만 새 Notebook은 Model/Learner와 외부 수집 루프를 구분한다.

## 환경 계약

3×4, 시작 (2,0), 목표 (0,3), 벽 (1,1), 행동 순서 up/down/left/right다.
다음 칸의 reward를 받으며 목표 진입은 +1·종료, (1,3) 진입은 −1·계속, 다른 칸은 0이다.
경계·벽 이동은 제자리이고 그 결과 칸의 보상을 받는다. 따라서 (1,3)에서 오른쪽 경계로
움직여 머무를 때도 −1이다. 목표 이후 추가 step은 생성하지 않는다.

`dlfs4_gridworld()`는 벽을 제외한 11개 State의 TabularMDP다. State ID는 `DLFS4_STATES`의
위에서 아래·왼쪽에서 오른쪽 순서이며 시작 ID는 7, 목표 ID는 3이다.
원본 states()는 벽 좌표도 열거하지만 정상 에피소드에서는 도달할 수 없다. 여기서는 벽에
가치·Count를 만들지 않고 표시에서 회색 칸으로 남긴다. RMSE는 목표도 제외한 10개 상태다.
Sutton 4×4 환경과 별도의 정의이며 기존 `sutton_gridworld()`는 바꾸지 않았다.

DP는 알려진 `dlfs4_gridworld()` 모델의 기대값을 직접 계산한다. MC/TD/Q-learning은
`make_dlfs4_gridworld_env()`의 Gymnasium reset/step을 사용하며 같은 kernel을 공유한다.
정수 observation/action은 Discrete space에 속한다. 기본 시작은 7이며 reset options의
start_state로 이번 Episode만 바꿀 수 있다. render_mode는 None/ansi/human이다.

환경은 첫 reset의 seed=7과 이후 reset(seed=None)으로 전이 RNG stream을 이어간다.
MC/TD의 외부 행동 Generator는 seed=8, Q-learning 행동 선택은 PyTorch seed=7이다.
TabularEpisodeSampler는 기존 API만 유지하며 이 Notebook에서는 사용하지 않는다.
난수 사용 순서가 달라졌으므로 이전 Sampler의 동일 seed 결과와 같은 Episode를 요구하지 않는다.

MC/TD는 작은 `collect_episode(env, select_action, ...)`로 동일한 Episode를 수집한다.
MC/TD 비교의 TimeLimit은 2,000 Step, Q-learning은 200 Step, Greedy 평가는 20 Step이다.
terminated 또는 truncated이면 reset하며 실제 마지막 관측을 보존한다. MC는 truncated-only
Episode를 학습하지 않고 비교를 중단한다. TD/Q는 terminated에서만 Bootstrap을 제거한다.
제한 시점의 자연 종료는 두 flag가 True일 수 있다. 환경은 finally에서 닫는다.

원본 환경의 40개 비종료 상태·행동 전이와 대조해 next state/reward/termination이 일치했다.
γ=.9, 허용오차 1e−12에서 원본 Policy Evaluation과 V의 최대 오차는 2.763e−12,
원본 Value Iteration과 최적 V의 오차는 0이었다. 시작 최적 V는 .9⁴=.6561이다.

## 원본 실습 대응표

장별 경로는 위에 연결한 동일 커밋의 소스다. 미구현 항목을 포함하므로 Chapter 4–6 전체
알고리즘이 구현됐다고 간주하지 않는다.

| 원본 파일·실습·주요 클래스/함수 | 새 Notebook Section | 사용 API | 상태 | 의미 있는 차이·남은 기능 |
|---|---|---|---|---|
| ch04/dp.py: 두 상태 동기식 평가 | 04 §2 | TabularMDP, TransitionOutcome, uniform_policy, policy_evaluation | 실행 완료 | dict 대신 배열, θ=1e−10 |
| ch04/dp_inplace.py: 제자리 평가 | 04 §2 설명 | policy_evaluation | 설명만 | 현재 동기식만 지원; 제자리 모드를 재구현하지 않음 |
| ch04/gridworld_play.py: 무작위 V 표시 (Notebook 표제는 gridwolrd.py) | 04 §3 | dlfs4_gridworld; 작은 plot_grid | 실행 완료 | Seed=7, 벽 State 제외, Matplotlib 표시 |
| ch04/policy_eval.py: eval_onestep/policy_eval | 04 §4 | policy_evaluation | 실행 완료 | 원본 제자리 sweep → 동기식; θ=.001 → 1e−10 |
| ch04/policy_iter.py: greedy_policy/policy_iter | 04 §5–6 | action_values, policy_improvement, policy_iteration | 실행 완료 | 동률 마지막 행동 → 균등 확률; 중간 애니메이션 없이 개선·최종 결과 표시 |
| ch04/value_iter.py: value_iter_onestep/value_iter | 04 §7–8 | value_iteration | 실행 완료 | 동기식; PI와 같은 최적 V인지 확인 |
| ch05/dice.py: sample, 표본 평균 | 05 §2 | NumPy 표본·누적 평균 | 실행 완료 | RL 갱신이 아닌 통계 실습; 중간 print 대신 그래프 |
| ch05/mc_eval.py: RandomAgent, Every-visit V | 05 §3–7 | make_dlfs4_gridworld_env, collect_episode, Episode, MonteCarloReturnGenerator, MCPredictionLearner | 실행 완료 | Every는 유지; First·Count·DP RMSE 추가; 원본 역순 평균 갱신 → 시간순 평균 |
| ch05/mc_control.py: McAgent, Q와 ε-greedy 개선 | 05 §8 | Model.act, EpsilonGreedyActionSelector, TabularMCControlLearner, collect_episode | 실행 완료·확장 | 원본 고정 α=.1 → First-visit Sample Average; 10,000 → 1,000 Episode; γ=.9, ε=.1 유지 |
| ch05/importance_sampling.py: 확률변수 평균/분산 | 05 §9 | NumPy 표본·ρ=π/b | 실행 완료 | Seed 명시; 이 통계 실습을 Off-policy RL 구현으로 부르지 않음 |
| Off-policy Prediction 확장: 단일 확률변수 IS에서 Episode V 추정으로 확장 | 05 §10 | ActionOutput, collect_episode, OffPolicyMCPredictionLearner, policy_evaluation | 실행 완료·추가 | γ=.9, 고정 π, 기록 log b; Ordinary/Weighted·First/Every 지원, Control과 구분 |
| ch05/mc_control_offpolicy.py: McOffPolicyAgent | 05 §11 | OffPolicyMCControlLearner, Model.act, Episode, collect_episode | 실행 완료·표준 알고리즘 적용 | 원본 고정 α=.2 → C 기반 Weighted IS; 10,000 → 1,000 Episode; γ=.9, ε=.1 유지 |
| ch06/td_eval.py: TdAgent | 06 §2–5 | collect_episode, TabularTD0PredictionLearner, TD0TargetGenerator, TabularVUpdateRule, Transition.to_batch | 실행 완료 | 원본 전이별 TD 유지; Gymnasium Episode는 먼저 수집 후 같은 순서 처리 |
| ch06/sarsa.py: SarsaAgent, deque(maxlen=2) | 06 §6·11 | TabularSARSALearner, Model.act, Transition.to_batch, TD0TargetGenerator, TabularQUpdateRule | 실행 완료 | 실제 a′를 학습 전에 선택·다음 step에 재사용; deque 대신 명시적 next_action; 10,000 → 1,000 Episode |
| ch06/sarsa_off_policy.py: SarsaOffPolicyAgent | 06 §7 | 해당 Learner 없음 | 미구현 | 다음 행동·행동 확률의 수집 시점과 IS Target 보정 계약 필요 |
| ch06/q_learning.py: QLearningAgent, π/b 분리 | 06 §8–9 | env.reset/step, Model.act, EpsilonGreedyActionSelector, TabularQLearner, TD0TargetGenerator, TabularQUpdateRule | 실행 완료 | 10,000 → 1,000 Episode; π/b dict 대신 Model/Selector; Q 네 행동 표시 |
| ch06/q_learning_simple.py: 간소화 QLearningAgent | 06 §8–10 | 같은 Model/Learner | 실행 완료·통합 | 중복 루프 없이 두 예제의 공통 갱신 사용; stochastic 동률은 균등, deterministic은 최소 인덱스 |
| Expected SARSA 조합 검증 확장 | 06 §12 | Model.forward, 동일 ActionSelector.probabilities, TD0TargetGenerator, TabularQUpdateRule | 실행 완료·예제 함수 | 새로운 Learner 없이 ΣπQ; 기존 코드 셀·출력 유지, 본체 API 변경 없음 |
| Double Q-learning 조합 검증 확장 | 06 §13 | 두 Model.forward, EstimatorOutput, 기존 Selector.select, TD0TargetGenerator, TabularQUpdateRule | 실행 완료·예제 함수 | 독립 table·Q 합 Behavior·교차 Bootstrap; 원본 Gaussian Bias 실험은 현재 환경 계약으로 정확히 표현 불가 |
| Online n-step SARSA 확장 | 06 §14 | Model.act/forward, Transition.to_batch, TD0TargetGenerator 역순 합성, TabularQUpdateRule | 실행 완료·예제 함수 | deque의 B=1 시간순 window; actual action 유지·boundary flush, TDN/SequenceReplay는 현재 없음 |
| Off-policy n-step SARSA 확장 | 06 §15 | 기존 Model/Selector, Transition.log_prob, TD0 역순 합성, Q UpdateRule | 실행 완료·고정 정책 평가 예제 | 현재 행동 제외·sampled 끝점 포함 IS, π=b 동등성, π≠b의 동일 π DP Q 비교 |
| Online n-step Tree Backup 확장 | 06 §16 | Model의 Q Snapshot, Selector.probabilities, TD0 역순 합성, Q UpdateRule | 실행 완료·예제 함수 | 중간 실제 행동 분기·π/Q 재귀, Expected Bootstrap, IS/끝점 행동 선택 없음 |
| common/gridworld.py: GridWorld | 세 Notebook 환경 구성 | dlfs4_gridworld, make_dlfs4_gridworld_env, TabularEnv, DLFS4_STATES/START_STATE | 실행 완료 | DP의 알려진 kernel과 Gymnasium 실행 환경 분리; compact State ID |
| common/gridworld_render.py: Renderer | 세 Notebook 결과 표시 | plot_grid, Matplotlib | 실행 완료·간소화 | Q 삼각형 대신 행동별 heatmap; 목표 V=0과 진입 +1 구분 |
| common/utils.py: argmax/greedy_probs/plot_total_reward | 06 §8–10 | ActionSelector.select/probabilities, plt.plot | 실행 완료·대체 | 원본 무작위 단일 argmax와 달리 동률 집합의 정확한 확률 사용 |
| Notebook SETUP: 셀 내 pip install | 각 §1·프로젝트 README | notebooks extra, 공개 import | 실행 완료·대체 | Notebook에 설치·sys.path 탐색 코드 없음 |

## 데이터 흐름과 실행 검증

TD/Q의 전이별 호출은 다음 한 줄로 끝난다. 변환은 dtype/device를 보존한 복사이며 graph를 제거한다.

```python
learner.learn(transition.to_batch())
```

MC는 `learner.learn(episode)`다. MC/TD 비교는 동일한 완료 Episode를 공유하고 MC는 전체,
TD는 순서대로 B=1을 처리한다. TD의 일반적 Time-limit Bootstrap과 MC의 미완성 Episode 거절은
각 Notebook에서 구분한다. 출력과 테스트는 RMSE의 단조 감소나 MC/TD 성능 순위를 요구하지 않는다.

MC Control은 `Model(IdentityRepresentation(), TabularQEstimator(...), EpsilonGreedyActionSelector(...))`와
`TabularMCControlLearner(model, MonteCarloReturnGenerator(.9))`를 조합한다. Model.act를 callback으로
collect_episode에 전달하고 자연 종료 후 learn한다. 수집 중에는 Q를 고정하고 다음 Episode부터
새 Q를 사용한다. visit_counts `(S,A)`는 Model과 별도로 저장·복원한다. 평가에는 독립 환경과
deterministic=True를 사용하며 Training Reward와 Greedy 평가의 길이·목표 도달을 구분한다.

Off-policy Prediction은 `collect_episode(env, behavior_model.act)`로 ActionOutput과 sampling-time
log b를 기록한다. 기존 int callback은 유지하며 log_prob=None이다. 수집 후 Q/epsilon 변경이나
Tensor 재사용에도 기록은 유지된다. Target `(S,A)`는 고정하며 현재 행동 확률로 과거 b를 다시 계산하지 않는다.
Ordinary는 ΣrhoG/N, Weighted는 ΣrhoG/Σrho다. 전체 suffix 비율에 현재 행동을 포함한다.
Count·Weight 합·Weighted Return 합은 V-table과 함께 별도로 복원한다. 미관측 행동의 Coverage는
기록만으로 확인할 수 없으므로 이 실험은 모든 행동의 b=1/4로 보장한다.
직접 reset/step으로 Transition을 수집한 뒤 Episode로 묶는 경로도 지원한다.
log_prob은 step 전에 detach·CPU clone하며, 전체 코드는
[ARCHITECTURE.md](../../docs/ARCHITECTURE.md#sampling-시점의-behavior-확률)에 있다.

Off-policy Control은 기존 TabularQEstimator Model과 OffPolicyMCControlLearner를 조합한다.
Behavior는 ε-greedy로 탐색하고 Target은 갱신 후 Q의 최소 ID argmax다. 역순 W=1에서 Q/C를
갱신한 뒤 기록 행동이 Target과 다르면 중단한다. 현재 행동의 비율을 먼저 곱하지 않는다.
Notebook은 첫 Episode를 reset/step으로 직접 수집하고 나머지는 collect_episode를 사용한다.
반환값은 전체 시간순 Return이며 Q-table과 별도 float64 weight_sums(C)를 함께 저장·복원한다.
원본 고정 α와 달리 표준 Weighted IS를 적용하며 Ordinary/First/Every 옵션은 추가하지 않는다.

```bash
python -m examples.gridworld_mc_control --episodes 1000 --seed 7 --epsilon 0.1
python -m examples.gridworld_off_policy_mc --episodes 1000 --seeds 7 11 19
python -m examples.gridworld_off_policy_mc --episodes 100 --seeds 7 --target-uniform
python -m examples.gridworld_off_policy_mc_control --episodes 1000 --seeds 7 11 19
python -m examples.gridworld_sarsa --episodes 1000 --seeds 7 11 19 --alpha .8 --epsilon .1
python -m examples.gridworld_expected_sarsa --episodes 1000 --seeds 7 11 19 --alpha .8 --epsilon .1
python -m examples.gridworld_double_q_learning --episodes 1000 --seeds 7 11 19 --compare-all
python -m examples.gridworld_n_step_sarsa --episodes 1000 --seeds 7 11 19 --ns 1 3 5
```

SARSA는 설치된 ModularRL의 공개 API만 사용하며 환경 루프를 Notebook에서 직접 보여준다.
Next Action은 학습 전에 선택해 다음 step에 유지하고, terminated에서는 None이며 truncated-only는
final observation의 선택 행동으로 Bootstrap한다. B>1의 Frozen Target과 B=1 Online 차이는 별도로 검증한다.

Expected SARSA §12는 프로젝트의 `examples/gridworld_expected_sarsa.py` 함수를 그대로 로드한다.
examples는 설치 패키지가 아니므로 프로젝트 파일을 찾되 sys.path를 변경하지 않는다.
핵심 함수의 소스를 표시하고, 기존 컴포넌트로 다음 상태 Q와 동일 Selector의 확률을 읽어 ΣπQ를
TD0TargetGenerator/TabularQUpdateRule에 전달한다. 직접 reset/step B=1 수집과 공통 실험 함수 호출을
보여주며 Notebook에서 Target/Update 계산을 다시 정의하지 않는다. 확률 조회에는 RNG 소비가 없다.
종료 mask와 frozen Batch/중복 순서 계약은 유지한다. 새 Learner와 공통 Bootstrap 인터페이스는 추가하지 않는다.

Double Q §13는 두 독립 TabularQEstimator Model을 사용하고 합산 Q를 단일 EstimatorOutput으로
기존 Selector에 전달한다. Model.act 대신 공개 Selector.select를 직접 호출하며 별도 행동용 Q-table은 없다.
예제의 동일 교차 Bootstrap 함수는 기존 TargetGenerator/UpdateRule로 하나의 table만 갱신한다.
환경 reset/step·Behavior·갱신 동전의 RNG를 분리하고 Q-learning/SARSA/Expected 결과는 §12에서 재사용한다.
DP와의 가치 비교에는 max((Q1+Q2)/2)를 쓰고 행동에는 Q1+Q2를 쓴다.
교재 Gaussian 보상과 상태별 행동 수는 현재 TabularMDP로 정확히 표현할 수 없어 Bias 실험은 제외한다.

n-step SARSA §14는 현재 존재하는 TD0TargetGenerator를 역순 합성한다. TDNTargetGenerator,
SequenceReplayBuffer, n-step DQN은 현재 저장소에 없다. deque는 owned B=1 snapshots를 시간순 보관하며
n개 보상이 모이는 즉시 가장 오래된 행을 학습하고 boundary에서 추가 step/act 없이 잔여 head를 모두 처리한다.
실제 다음 행동은 학습 전에 선택해 다음 step까지 유지한다. Truncation은 마지막 선택 행동으로 실제
남은 보상 수의 discount를 적용하고, 각 residual call의 현재 Q를 읽는다. 기존 49개 셀은 모두 그대로 보존한다.

Off-policy n-step §15는 별도의 고정 확률표 π를 평가한다. 실행한 행동의 log b는 Transition,
아직 실행하지 않은 끝점의 행동/log b는 별도로 보관한다. 중요도 비율은 head 행동을 제외하고 sampled
Bootstrap 행동까지 포함하며 Truncation flush에서 다시 선택하지 않는다. `Yeff=Q+rho*(G-Q)`로
기존 Q Rule을 재사용하고 rho=1은 원래 G를 직접 전달한다. 이전 59개 셀·출력·메타데이터를 보존한다.
새 섹션은 100 Episode, gamma=.9, alpha=.05, Seeds7/11/19, n=1/3/5를 실행하며 CLI는 기본 200 Episode다.
Uniform Behavior와 Uniform/다른 고정 Target을 비교하고 각각의 DP Qpi 기준 RMSE·ρ 통계·ESS를 출력한다.
이는 고정 정책 평가이며 최적 제어 성공으로 해석하지 않는다.

`tests/test_dlfs4_notebooks.py`는 nbformat 검증 후 각 Notebook을 같은 Kernel에서 두 번
실행하고 텍스트·숫자·PNG 출력이 같은지 확인한다. 별도의 작은 테스트 kernelspec으로 현재
pytest Python을 사용하며 사용자의 Kernel 등록 상태를 바꾸지 않는다. 기본 library 설치에서는
Notebook 관련 선택 의존성이 없으면 이 실행 테스트만 skip한다.

```bash
python -m pytest -q tests/test_dlfs4_notebooks.py
```

## 실행 결과

선택적 Notebook 의존성을 설치한 Python 3.12에서 세 Notebook의 Run All을 같은 Kernel에서
각각 두 번 실행했다. 모든 텍스트·숫자·PNG 출력이 일치했고, 기존 Prediction·Control·통계 실습을
보존하면서 Off-policy Prediction의 그림 2개와 Control의 Q/Target·DP/C 비교 및 성능 그림 2개를 추가했다.
실행 출력은 `.ipynb`에 저장되어 있으며 다음 값은 재현성 검증 결과다.

| 실습 | 조건 | 결과 |
|---|---|---|
| DP | γ=.9 | Policy/Value Iteration의 V 차이 0, 최적 V(start)=.6561 |
| MC | 환경 seed=7, 행동 seed=8, 1,000 Episode, 45,188 Step, γ=.9 | First RMSE=.0306, Every RMSE=.0317 |
| MC/TD 동일 경험 비교 | 같은 Episode/Step, TD α=.01 | Every MC RMSE=.0317, TD RMSE=.0239 |
| On-policy MC Control | 환경·PyTorch seed=7, 1,000 Episode, 5,672 Step, γ=.9, ε=.1, First-visit | 초기 Greedy는 20 Step cutoff; 학습 후 5 Step·+1로 목표 도달, 마지막 100 Training 평균 보상 1.0 |
| Off-policy Prediction | 각 Seed 1,000 Episode, b=균등, π=(.3,.2,.2,.3), γ=.9 | Ordinary RMSE(seed7/11/19)=.055652/.047766/.046639; Weighted=.062241/.051419/.040393 |
| Off-policy Control | γ=.9, ε=.1, 1,000 Episode, Seed7, 5,701 Step | 초기 20 Step cutoff → 학습 후 Greedy 5 Step/+1; DP 최적 경로와 일치; max Q RMSE=.250643 |
| Q-learning | seed=7, 1,000 Episode, γ=.9, α=.8, ε=.1 | Truncation 0, 마지막 100 Episode 평균 보상 1.0; Greedy 5번 이동으로 목표 도달 |
| SARSA | γ=.9, α=.8, ε=.1, Seed7, 1,000 Episode/8,604 Step | 실제 a′ 선택·재사용; Truncation 0, 마지막 100 Reward 평균 1.0, Greedy 5 Step/+1 |
| SARSA/Q-learning 비교 | 같은 설정·독립 실행, Seed7/11/19 | SARSA Step=8,604/6,659/6,627, Greedy=5/5/20(cutoff); Q-learning Step=5,738/5,630/5,611, Greedy=5/5/5 |
| Expected SARSA 조합 | γ=.9, α=.8, ε=.1, 1,000 Episode, Seed7/11/19 | Step=53,664/5,592/5,621; Greedy 모두 Goal 5 Step/+1; max Q의 DP RMSE=.260824/.050405/.305310 |
| Double Q-learning 조합 | 동일 설정, 1,000 Episode, Seed7/11/19 | Step=6,017/5,790/5,796; Greedy 모두 Goal 5 Step/+1; max 평균 Q의 DP RMSE=2.449e-9/.148032/1.394e-8 |
| Online n-step SARSA | 동일 설정·각 1,000 Episode, Seed7/11/19 | n=1 Step=8,604/6,659/6,627; n=3=7,137/6,818/7,089; n=5=9,920/7,308/8,134; Update=Step; n=3/5 Greedy 모두 Goal 5 |

이 수치는 해당 설정의 관측 결과다. 방문이 적은 상태·행동의 Q는 초기값으로 남을 수 있으며
시작 경로의 성공을 모든 Q의 수렴으로 해석하지 않는다. Gymnasium 경로는 기존 Sampler의
동일 seed 결과를 재현하는 대신, 자체 재실행 결정성과 정책·전이 분포를 검증한다.
계산은 반복 상태-행동 쌍·Sample Average·Count·Checkpoint·입력 경계의 결정적 테스트로,
환경 실행은 짧은 재현성/종료 테스트로 검증한다. 단일 Seed의 학습 성공률로 합격을 판정하지 않는다.
Behavior=Target 균등 예제(seed7, 100 Episode)는 모든 Weight가 1이고 두 방법의 값이 같다.
단일 종료 전이 Reward=4, pi=.5, b=.25는 Ordinary V=8, Weighted V=4이며 여러 단계의
비율 곱·반복 방문·Zero Weight·수치 오류·Checkpoint도 결정적 테스트로 검증한다.
Control 단일 표본 reward=4/b=.25는 Q=4/C=1, 두 단계 rewards=[1,3]은 Q=[4,3]/C=[4,1]이다.
갱신 후 Greedy·Early Break·동률·반복 쌍·수치 오류의 부분 갱신 방지·Checkpoint를 결정적으로 검증한다.
SARSA 수치 예제의 a′=1은 Error=-1.2/Q=3.4, a′=0은 Error=6/Q=7, terminal은 Error=-3/Q=2.5다.
직접 루프와 비교 루프의 Seed7 Q는 일치한다. 종료·Truncation·선택 횟수/실행 재사용·Batch/Online
차이·dtype/device·Model 복원을 결정적으로 검증한다. Seed19의 Greedy cutoff도 그대로 보고한다.
전체 pytest는 1,693 passed, 53 skipped(CUDA 미지원)다. 세 Notebook Run All 두 번의
텍스트·수치·32개 PNG는 일치했고 기존 06 Notebook의 69개 셀 전체(코드 34개)는 이전 commit과 같다.
quickstart·DP/MC/TD/On/Off-policy MC·FrozenLake·CartPole Smoke Test도 통과했다.

Expected SARSA 추가 실험은 학습 중 Reward/Length와 누적 Step을 기록한다. Seed7의 학습 cutoff 23회와
큰 Step 비용, SARSA Seed19의 Greedy cutoff를 그대로 보여준다. DP 최적 경로는 5 Step이다.
기존 Learner의 max/gather Bootstrap은 고정돼 있지만 컴포넌트의 출력 계약만으로 기대값 계산을
조합할 수 있었다. 중복은 Batch 검증/준비와 Target/Update 호출 및 환경 루프이며, 향후 최소 공통화
후보는 외부에서 계산한 next_value `(B,)`를 받는 작은 조합 함수다. 이번에는 그 API를 구현하지 않는다.

Double Q의 수치 예제는 Q1 갱신 Error=-2.1/Q1=2.95, Q2 갱신 Error=-3.2/Q2=4.4이며 반대 table은 유지한다.
Q1/Q2 동전은 Seed7/11/19에서 3,044·2,973 / 2,872·2,918 / 2,868·2,928회를 선택했다.
초기 Greedy는 20 Step cutoff, 학습 후 DP와 같은 5 Step이고 training cutoff는 모두 0이었다.
합산 정책 확률·실제 Sampling·Storage 독립성·Frozen Target·중복 순서·Checkpoint를 결정적으로 검증한다.
원본 Bias 분포와 표현 한계 및 향후 결합 함수 후보는 [ARCHITECTURE.md](../../docs/ARCHITECTURE.md#double-q-learning-컴포넌트-조합-검증)에 기록한다.

n=1은 기존 SARSA와 세 Seed의 Q/Reward/Length가 정확히 같고 Seed19의 Greedy cutoff도 유지한다.
n=3의 수치 Target은 8.146, 짧은 terminal의 Targets=[2.8,2], truncated-only는 [6.04,5.6]이다.
반복 self-transition의 잔여 Bootstrap은 Q=4에서 첫 update 후 5.02를 다시 읽으며 Target=6.518이 된다.
n=3, T=5는 3번째 환경 Step부터 갱신하고 마지막 두 행을 종료 후 처리한다. 모든 Transition을 한 번씩 갱신한다.
관련 책임·checkpoint pending/선택 행동·향후 endpoint 조합 후보는
[ARCHITECTURE.md](../../docs/ARCHITECTURE.md#online-n-step-sarsa-컴포넌트-조합-검증)에 기록한다.

Off-policy n-step 수치 예제 A/B/C/D의 (G,ρ,Qnew)는 (2.8,2,2.8), (8.146,3,11.219),
(2.8,2,2.8), (6.04,6,14.12)다. n=1/3/5의 Matched π=b는 기존 On-policy의 행동·Reward·Target·
Error·Q·Step·RNG와 정확히 일치한다. 기록 확률 유지, 수치 실패 전 쓰기 차단, 고정 stream의
Model/pending/끝점/정책/RNG 복원을 결정적으로 검증한다. §15를 포함한 06은 69개 셀, 코드 34개, PNG 15개다.
Notebook 100 Episode의 Different Qpi RMSE는 n=1/3/5에서 세 Seed 평균±표준편차
.209765±.014594 / .111875±.011099 / .135703±.015378이다. CLI 200 Episode 결과는
[ARCHITECTURE.md](../../docs/ARCHITECTURE.md#off-policy-n-step-sarsa-컴포넌트-조합-검증)에 기록한다.

Tree Backup §16은 같은 Snapshot의 Q/π에서 중간 실제 행동 분기를 재귀하고 마지막은 ΣπQ로
Bootstrap한다. n=1은 기존 Expected SARSA의 Action/Target/Error/Q/RNG와 정확히 같으며 n=3의
Target=3.8206, Error=1.8206, Qnew=2.9103을 확인한다. 자연 종료의 짧은 Targets=[2.8,2],
Time-limit은 [3.61,5.6]이고 Bootstrap용 행동은 추가로 선택하지 않는다. π=0은 다른 행동의
기대값을 남기며 π=1은 G를 그대로 전달한다. Behaviour log_prob이나 IS Weight를 사용하지 않는다.

200 Episode·Seeds7/11/19·n=1/3/5의 45개 비교 run은 384,075 Step=Update였고 모두 자연 종료했다.
Different Target의 Tree RMSE 평균±표준편차는 n=1/3/5에서
.100217±.014008 / .073076±.012630 / .072202±.012607이다. n=1은 SARSA보다 평균이 약간 높고,
n=3/5는 낮은 관측값이지만 성능 보장은 아니다. On-policy 기준·SARSA·Matched 값과 Seed별 수치는
[ARCHITECTURE.md](../../docs/ARCHITECTURE.md#online-n-step-tree-backup-컴포넌트-조합-검증)에 기록한다.
현재 06 Notebook은 87개 셀, 코드 47개, PNG 17개이며 이전 69개 셀·출력·메타데이터를 모두 보존한다.

## 최소 확장 제안

- MC Control 확장: 현재 Sample Average와 별도로 원본 고정 α 갱신·GLIE 등의 의미를 설계한다. 이번 구현에 이미 포함됐다고 간주하지 않는다.
- SARSA 확장: Expected SARSA는 §12, Online n-step은 §14, 고정 π Off-policy n-step은 §15 예제 조합으로 검증한다. Q(σ)의 혼합·확률 계약은 별도로 설계하며 Transition 필드를 임의로 추가하지 않는다.
- Tree Backup 확장: §16에서 고정 π의 중간 분기·Expected Bootstrap을 기존 컴포넌트로 조합한다. Q(σ)의 sampled/expected 혼합 및 Off-policy 보정 계약은 별도로 설계한다.
- Off-policy 확장: 현재 V Prediction·Weighted Q Control·n-step 고정 정책 평가와 구분해 Per-decision IS와 n-step Control의 계약을 별도로 설계한다.
- 제자리 DP: 필요할 때 별도 모드의 수치 의미를 설계한다. Notebook 단순화를 위해 지금 추가하지 않는다.
