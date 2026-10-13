# 밑바닥부터 시작하는 딥러닝 4 — ModularRL 실습

프로젝트 가상환경에 `pip install -e ".[notebooks]"`로 설치하고 이 폴더에서 세 Notebook을
위에서부터 실행한다. Kernel 등록·선택 명령은 [프로젝트 README](../../README.md#jupyter-notebooks--deep-learning-from-scratch-4)에 있다.
`plotting.py`는 좌표에 값·정책 화살표를 표시하는 작은 함수만 제공한다. 환경 생성·샘플링·RMSE·학습은 포함하지 않는다.

| Notebook | 실행 지원 |
|---|---|
| [04_dp_modularrl.ipynb](04_dp_modularrl.ipynb) | 두 상태 평가, GridWorld 표시, Policy Evaluation/Improvement/Iteration, Value Iteration |
| [05_mc_modularrl.ipynb](05_mc_modularrl.ipynb) | 주사위 평균, MC Return, First/Every Prediction, On-policy MC Control·Q/정책·DP 비교, 확률변수 IS |
| [06_td_modularrl.ipynb](06_td_modularrl.ipynb) | TD(0) Prediction, 동일 Episode MC 비교, Q-learning, Greedy 실행 |

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
| ch05/mc_control_offpolicy.py: McOffPolicyAgent | 05 §10 | 해당 Learner 없음 | 미구현 | γ=.9, α=.2, ε=.1; Return 보정·target/behavior 계약 필요 |
| ch06/td_eval.py: TdAgent | 06 §2–5 | collect_episode, TabularTD0PredictionLearner, TD0TargetGenerator, TabularVUpdateRule, Transition.to_batch | 실행 완료 | 원본 전이별 TD 유지; Gymnasium Episode는 먼저 수집 후 같은 순서 처리 |
| ch06/sarsa.py: SarsaAgent, deque(maxlen=2) | 06 §6 | 해당 Learner 없음 | 미구현 | 실제 next_action 전달 계약 필요; max Q로 대신하지 않음 |
| ch06/sarsa_off_policy.py: SarsaOffPolicyAgent | 06 §7 | 해당 Learner 없음 | 미구현 | 다음 행동·행동 확률의 수집 시점과 IS Target 보정 계약 필요 |
| ch06/q_learning.py: QLearningAgent, π/b 분리 | 06 §8–9 | env.reset/step, Model.act, EpsilonGreedyActionSelector, TabularQLearner, TD0TargetGenerator, TabularQUpdateRule | 실행 완료 | 10,000 → 1,000 Episode; π/b dict 대신 Model/Selector; Q 네 행동 표시 |
| ch06/q_learning_simple.py: 간소화 QLearningAgent | 06 §8–10 | 같은 Model/Learner | 실행 완료·통합 | 중복 루프 없이 두 예제의 공통 갱신 사용; stochastic 동률은 균등, deterministic은 최소 인덱스 |
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

```bash
python -m examples.gridworld_mc_control --episodes 1000 --seed 7 --epsilon 0.1
```

`tests/test_dlfs4_notebooks.py`는 nbformat 검증 후 각 Notebook을 같은 Kernel에서 두 번
실행하고 텍스트·숫자·PNG 출력이 같은지 확인한다. 별도의 작은 테스트 kernelspec으로 현재
pytest Python을 사용하며 사용자의 Kernel 등록 상태를 바꾸지 않는다. 기본 library 설치에서는
Notebook 관련 선택 의존성이 없으면 이 실행 테스트만 skip한다.

```bash
python -m pytest -q tests/test_dlfs4_notebooks.py
```

## 실행 결과

선택적 Notebook 의존성을 설치한 Python 3.12에서 세 Notebook의 Run All을 같은 Kernel에서
각각 두 번 실행했다. 모든 텍스트·숫자·PNG 출력이 일치했고, 기존 13개 그림을 보존해
MC Control의 3개 그림을 추가했다.
실행 출력은 `.ipynb`에 저장되어 있으며 다음 값은 재현성 검증 결과다.

| 실습 | 조건 | 결과 |
|---|---|---|
| DP | γ=.9 | Policy/Value Iteration의 V 차이 0, 최적 V(start)=.6561 |
| MC | 환경 seed=7, 행동 seed=8, 1,000 Episode, 45,188 Step, γ=.9 | First RMSE=.0306, Every RMSE=.0317 |
| MC/TD 동일 경험 비교 | 같은 Episode/Step, TD α=.01 | Every MC RMSE=.0317, TD RMSE=.0239 |
| On-policy MC Control | 환경·PyTorch seed=7, 1,000 Episode, 5,672 Step, γ=.9, ε=.1, First-visit | 초기 Greedy는 20 Step cutoff; 학습 후 5 Step·+1로 목표 도달, 마지막 100 Training 평균 보상 1.0 |
| Q-learning | seed=7, 1,000 Episode, γ=.9, α=.8, ε=.1 | Truncation 0, 마지막 100 Episode 평균 보상 1.0; Greedy 5번 이동으로 목표 도달 |

이 수치는 해당 설정의 관측 결과다. 방문이 적은 상태·행동의 Q는 초기값으로 남을 수 있으며
시작 경로의 성공을 모든 Q의 수렴으로 해석하지 않는다. Gymnasium 경로는 기존 Sampler의
동일 seed 결과를 재현하는 대신, 자체 재실행 결정성과 정책·전이 분포를 검증한다.
계산은 반복 상태-행동 쌍·Sample Average·Count·Checkpoint·입력 경계의 결정적 테스트로,
환경 실행은 짧은 재현성/종료 테스트로 검증한다. 단일 Seed의 학습 성공률로 합격을 판정하지 않는다.
전체 pytest는 1,136 passed, 20 skipped(CUDA 미지원)였고 세 Notebook Run All 두 번,
quickstart·DP/MC/TD/FrozenLake·CartPole Smoke Test도 통과했다. 기존 05 Notebook의
코드 셀 10개 출력은 이전 commit과 정확히 일치했다.

## 최소 확장 제안

- MC Control 확장: 현재 Sample Average와 별도로 원본 고정 α 갱신·GLIE 등의 의미를 설계한다. 이번 구현에 이미 포함됐다고 간주하지 않는다.
- SARSA: `learn(batch, next_actions)`처럼 실제 수집한 다음 행동을 전달하고 기존 TD0TargetGenerator/TabularQUpdateRule을 재사용한다. 현재 Transition 필드를 임의로 추가하지 않는다.
- Off-policy: 어떤 시점의 π/b를 쓰는지, Ordinary/Weighted/Per-decision 보정과 gradient/저장 경계를 먼저 결정한다. 기존 log_prob만 있다고 완성되는 것은 아니다.
- 제자리 DP: 필요할 때 별도 모드의 수치 의미를 설계한다. Notebook 단순화를 위해 지금 추가하지 않는다.
