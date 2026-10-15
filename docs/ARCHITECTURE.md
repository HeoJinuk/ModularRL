# ModularRL 아키텍처와 데이터 계약

> 현재 공개 API는 [API.md](API.md)를 따른다. 아래 개발 단계별 조합 검증에서
> “examples 전용/새 Learner 없음”은 당시 범위다. Expected SARSA, Double Q,
> n-step SARSA, Off-policy n-step SARSA, Tree Backup, Q(σ)의 학습 계산은 현재 패키지로
> 승격됐으며 이전 함수는 호환 위임 경로다. 수학과 수집·업데이트 순서는 유지한다.


ModularRL은 연구·교육용 강화학습 라이브러리를 위한 Model·경험 데이터·기본
ReplayBuffer/SequenceReplayBuffer·TD(0)/MC/n-step TargetGenerator·Tabular/Neural UpdateRule·TabularQLearner·TabularSARSALearner·DQNLearner와
Tabular DP, On/Off-policy Monte Carlo Prediction/Control 및 TD(0) Prediction 참조 구현이다.
Tabular Q-learning은 FrozenLake, 기본 DQN은 CartPole, DP/MC/TD Prediction 비교는
Sutton Gridworld, On/Off-policy MC Control은 교재 3×4 Gridworld에서 실행할 수 있다. Runner, Agent,
범용 Off-policy SARSA Learner, 다른 Neural Learner, 라이브러리 수준 WorldModel/Planner는 아직 포함되지 않는다.
Expected SARSA·Double Q·Online multi-step·Dyna는 예제 조합 지원이며, 신규 Gymnasium Notebook과
Sequence Replay/DQN n-step 경로의 현재 계약은 문서 마지막 절과 GYMNASIUM_TUTORIALS.md에 정리한다.
목표 구성은 `Runner = Environment + Agent`, `Agent = Model + Learner`이며,
현재 Model은 다음 흐름을 구현한다.

```text
Observation -> InputAdapter -> Representation -> Estimator -> EstimatorOutput
                                                               |
                                                        ActionSelector
                                                               |
                                                          ActionOutput
```

## 컴포넌트의 책임

- InputAdapter: NumPy, Tensor, scalar를 Model의 device와 dtype에 맞는 Tensor로 변환한다. Tabular 정수 state는 `torch.long`으로 유지한다.
- Representation: Tensor 관측을 Tensor features로 변환한다. 추정 대상이나 행동 선택은 담당하지 않는다.
- Estimator: Policy, V, Q, Advantage 중 추정 대상과 `EstimatorOutput`의 의미를 정의한다.
- Network: Estimator에 속한 `nn.Module`로 실제 추정 함수를 계산한다. 공유 Representation의 출력 features를 입력으로 받는다.
- ActionSelector: EstimatorOutput을 사용해 행동을 선택하고 `ActionOutput(action, log_prob)`을 만든다.

Estimator에는 NumPy 변환, 강제 device 이동, Loss, Optimizer, 행동 Sampling을
포함하지 않는다. `Model.forward()`는 gradient를 유지하며 `Model.act()`는
`no_grad`로 동작한다. train/eval 모드는 호출자가 설정한다.

## Model 통합 인터페이스

Model은 컴포넌트를 연결하는 `nn.Module`이며 학습 알고리즘을 알지 않는다.
Loss, Optimizer, ReplayBuffer, TargetGenerator와 학습 절차는 Model에 포함하지 않는다.
현재 세 추론 경로와 기존 `encode()` 보조 메서드를 유지한다.

| 메서드 | 계산 경로 | 반환값과 gradient |
|---|---|---|
| `forward(observation)` | InputAdapter -> Representation -> Estimator.forward | EstimatorOutput, 호출자의 autograd context 유지 |
| `act(observation, *, deterministic=False)` | forward -> ActionSelector.select | ActionOutput, 전체 경로 no_grad |
| `evaluate_q(observation, actions)` | 관측 encode, 행동 InputAdapter 변환 -> Estimator.evaluate | EstimatorOutput, 호출자의 autograd context 유지 |
| `encode(observation)` | InputAdapter -> Representation | Features Tensor |

`forward()`와 `act()`는 관측 변환, Representation, Estimator를 각각 한 번 호출한다.
`evaluate_q()`는 관측과 행동을 각각 한 번 변환하고 Representation은 관측에만
한 번 적용한다. 행동은 Representation을 통과하지 않는다. Estimator.forward를
거치지 않고 선택적인 `evaluate(features, actions)`를 직접 호출한다.
그 메서드가 없는 Estimator는 명확한 TypeError로 거절하고 기존 forward 계약은 유지한다.

ActionSelector가 없는 Model에서 `act()`는 명확한 RuntimeError를 발생시킨다.
Continuous Policy Model은 `forward()`, Continuous Critic Model은 `evaluate_q()`를
사용한다. 현재 없는 Continuous ActionSelector를 대신하는 임시 Selector는 두지 않는다.

`act()`는 train/eval 모드를 변경하지 않는다. `deterministic=True`는 Selector의
행동 선택 규칙만 바꾼다. Dropout과 BatchNorm 등의 모드는 호출자가 설정해야 한다.
`no_grad`는 계산 그래프를 만들지 않도록 하며 모듈을 자동으로 eval 모드로 전환하지 않는다.
학습에서 Policy gradient가 필요하면 forward로 다시 추정하고 저장된 행동의
log probability를 계산한다. Target 평가의 gradient 여부도 호출자가 관리한다.

### 검증한 조합과 shape

`...`는 모든 앞쪽 batch 차원이며 Model이 임의로 batch 축을 추가하거나 제거하지 않는다.

| 조합 | 추정 출력 | 행동 선택 또는 평가 |
|---|---|---|
| Identity + TabularQ + EpsilonGreedy | 정수 state `(...)` -> q `(..., A)` | action와 log_prob `(...)` |
| MLP + DiscreteQ + Greedy | 관측 `(..., O)` -> q `(..., A)` | action와 log_prob `(...)` |
| MLP + ActorCritic + CategoricalActionSelector | policy.probs `(..., A)`, v `(...)` | action와 log_prob `(...)` |
| MLP + ContinuousQ | 관측 `(..., O)`, 행동 `(..., A)` -> q `(...)` | evaluate_q, Selector 없음 |

Single 입력은 batch 축이 없는 입력이다. 이산 행동은 scalar Tensor이며,
batch 입력에서는 앞쪽 batch 차원이 그대로 남는다. `deterministic=True`의
log_prob는 None이다. 환경에 필요한 Python/NumPy 변환은 향후 Runner의 책임이다.

### Device, dtype과 checkpoint

컴포넌트를 조합한 뒤 `Model.to(device, dtype)`로 Model 전체의 Parameter와 buffer를
이동한다. 기본 InputAdapter는 floating 관측·행동을 Model의 dtype/device로 옮기고,
정수는 `torch.long`으로 유지한다. Tensor의 floating dtype 변환도 autograd를 유지한다.
파라미터가 없는 Tabular Model도 같은 이동 규칙을 따른다.

명시적인 `InputAdapter(dtype=...)`는 Model의 기본 floating dtype보다 우선한다.
정수로 저장된 연속 관측을 float로 바꿀 때 유용하지만 Adapter의 명시적 dtype과
Network dtype을 일치시켜야 한다. 예를 들어 float64로 설정한 Adapter를 사용할
때는 Model도 double로 구성한다. 이 우선순위를 암묵적으로 변경하지 않는다.

`state_dict()`는 Representation, Estimator의 Network, Tabular table, Selector의
epsilon 등 등록된 상태를 저장한다. 내부 device/dtype 추적용 `_input_reference`는
non-persistent buffer이므로 checkpoint 키에 포함되지 않는다. 복원 시 같은 구성의
Model을 만들고 device/dtype을 설정한 뒤 state_dict를 로드한다. Network 구조,
InputAdapter 설정, train/eval 모드, RNG와 Optimizer 상태는 별도로 관리한다.

## Experience Data Contract와 기본 ReplayBuffer

`data/experience.py`의 Transition은 한 번의 실제 또는 가상 환경 전이다.
필드는 observation, action, reward, next_observation, terminated, truncated,
선택적인 log_prob의 일곱 개다. done으로 합쳐 저장하지 않으며 Gymnasium info,
value prediction, advantage, discount, 학습 알고리즘별 target을 기본 필드에 넣지 않는다.

Transition은 Python numeric scalar, NumPy scalar/array, Tensor를 그대로 받는
frozen dataclass다. 내부 배열·Tensor까지 불변인 것은 아니다. 실제 저장 경계인
ReplayBuffer.add에서 shape·dtype 계약을 검증하고 독립적인 CPU Tensor snapshot을
만든다. Tensor는 detach한 후 복사하므로 observation, action, reward와 log_prob 등의
graph를 유지하지 않는다. NumPy 재사용, view와 negative stride도 저장값을 바꾸지 않는다.

### Transition과 Replay Batch의 shape·dtype

| 필드 | 단일 Transition | TransitionBatch |
|---|---|---|
| observation / next_observation | 임의의 동일한 관측 shape | `(B, *observation_shape)` |
| discrete action | integer scalar | `(B,)`, integer dtype 보존 |
| continuous action | floating vector `(A,)` | `(B, A)`, floating dtype 보존 |
| reward | real numeric scalar | `(B,)`, 입력의 numeric dtype 보존 |
| terminated / truncated | bool scalar, 서로 독립 | 각각 `(B,)`, torch.bool |
| log_prob | floating scalar 또는 None | `(B,)` floating Tensor 또는 None |

관측의 feature 수를 고정하거나 Flatten하지 않는다. Tabular scalar state는
Replay에서 `(B,)` 정수 Tensor가 되고 Neural 관측은 `(B, ...)` floating Tensor로
유지된다. 이미지의 uint8 등 정수 관측도 임의로 float로 바꾸지 않는다. NumPy와
Tensor의 dtype은 보존하며 Python scalar는 PyTorch의 기본 dtype 추론을 사용한다.
현재 행동 계약은 scalar 이산 행동과 1차원 연속 행동 벡터다. MultiDiscrete나
구조화된 dict 관측 등의 별도 지원은 이번 범위에 포함하지 않는다.

한 Buffer 안에서는 각 필드의 shape·dtype이 일치해야 한다. observation과
next_observation도 같은 shape·dtype이어야 한다. dtype이 섞인 데이터의 자동 승격을
막고 원본 계약을 유지하기 위한 기준이다. log_prob은 Buffer 전체에서 제공하거나
전체에서 생략한다. 두 경우를 섞으면 add가 명확히 실패한다. None을 0이나 NaN으로
채우거나 알려진 log_prob을 버리지 않는다. 서로 다른 계약은 별도 Buffer로 관리하거나
호출자가 명시적으로 정규화한다. clear 이후에는 새로운 계약으로 다시 저장할 수 있다.

log_prob은 행동을 생성한 Behavior Policy의 log probability 또는 continuous
log density다. 연속 density의 log_prob은 양수일 수도 있다. 행동 event 차원은
미리 합산한 scalar를 전달한다. PPO 등의 업데이트에서는 저장된 log_prob을 Behavior
값으로 사용하고 현재 Policy의 미분 가능한 값은 Model.forward로 다시 계산한다.

### Device와 dtype의 책임

Replay 저장·샘플 결과는 detached CPU Tensor다. CUDA Tensor 입력도 CPU로 복사한다.
sample은 새로 stack한 Tensor를 반환하므로 Batch를 수정해도 저장된 전이는 바뀌지 않는다.
Buffer가 observation의 정수/실수 의미를 추측해 dtype을 바꾸지 않는다.

학습 device·floating dtype의 선택과 이동은 호출자, 향후 Learner의 책임이다.
`TransitionBatch.to(device, dtype=...)`는 새 Batch를 반환하고 floating 필드만
dtype을 바꾼다. 정수 관측·행동·reward와 bool 플래그는 dtype을 유지한다. 따라서
정수 reward를 floating target 계산에 사용할 때는 호출자가
`reward.to(dtype=next_value.dtype)` 등으로 floating dtype을 명시한다.
TD0TargetGenerator는 입력 dtype을 바꾸지 않는다. 정수로 저장된 연속 관측은 기존 InputAdapter의 명시적인
dtype 설정을 사용할 수 있다. `.to()`에서 이동이 필요 없는 Tensor는 일반 Tensor.to처럼
원래 Batch와 storage를 공유할 수 있지만 Replay 저장소와는 독립적이다.

### ReplayBuffer API와 샘플링

`component/buffer/base.py`의 BaseBuffer는 add, sample, clear, len만 정의한다.
ReplayBuffer는 capacity가 양의 정수인 원형 저장소다.

| API | 동작 |
|---|---|
| `ReplayBuffer(capacity, *, seed=None)` | 고정 capacity와 전용 NumPy Generator 구성 |
| `add(transition)` | 입력 검증·CPU snapshot 후 저장, 가득 차면 가장 오래된 전이 교체 |
| `sample(batch_size)` | 균등 비복원 샘플링, `1 <= batch_size <= len(buffer)` |
| `clear()` | 데이터와 원형 위치를 초기화하고 shape/dtype 계약을 해제 |
| `len(buffer)` | 현재 저장된 전이 수 |

같은 seed·입력·호출 순서는 같은 샘플을 재현한다. 전역 NumPy/PyTorch RNG에는
영향을 주지 않는다. clear는 전용 RNG를 초기 seed로 되감지 않는다. 샘플 순서는
환경의 시간 순서를 의미하지 않는다. Buffer checkpoint나 RNG 상태 복원 API,
Prioritized, n-step, Sequence Replay는 아직 구현하지 않는다.

```python
import torch
from modular_rl.data import Transition, TransitionBatch
from modular_rl.component.buffer import ReplayBuffer

replay = ReplayBuffer(capacity=3, seed=123)
for state in range(3):
    replay.add(Transition(
        observation=state, action=state % 2, reward=1.0,
        next_observation=state + 1, terminated=state == 2, truncated=state == 1,
    ))
batch = replay.sample(2)
assert isinstance(batch, TransitionBatch)
assert batch.observation.shape == batch.action.shape == batch.reward.shape == (2,)
assert batch.terminated.dtype == batch.truncated.dtype == torch.bool
assert batch.log_prob is None
training_batch = batch.to("cpu", dtype=torch.float64)
assert training_batch.reward.dtype == torch.float64
assert training_batch.observation.dtype == training_batch.action.dtype == torch.long
```

### Bootstrap, 에피소드 경계와 향후 Trajectory

terminated는 MDP의 종료, truncated는 time limit 등 외부 중단을 별도로 표현한다.
둘 다 True인 전이도 그대로 보존한다. 일반적인 Time-limit Truncation에서는
마지막 상태로 Bootstrap할 수 있으므로 one-step bootstrap mask는 보통
`~terminated`다. 반면 에피소드 경계는 `terminated | truncated`다. 이것은 이번
Buffer에서 target이나 mask를 계산한다는 의미가 아니며 그 규칙은 TargetGenerator의 책임이다.

향후 on-policy Trajectory는 `(T, B, ...)`를 사용한다. T는 시간, B는 환경/stream
축이다. GAE와 n-step은 종료 값의 Bootstrap 여부와 다음 에피소드로 trace가 이어지는
문제를 구분해야 한다. Truncation에서 마지막 상태의 value는 사용할 수 있어도 다음
에피소드의 advantage나 reward trace를 연결해서는 안 된다. 현재 무작위 Replay Batch는
시간 순서·에피소드 식별을 제공하지 않으며 PPO용 ordered trajectory를 대체하지 않는다.

자동 Reset 환경의 next_observation은 Reset 후 첫 관측이 아니라 해당 전이의
실제 마지막 관측이어야 한다. 이 값을 환경 반환 정보에서 올바르게 선택하는 책임은
향후 Runner에 있다. Buffer는 전달된 값만 복사하며 환경 reset이나 Gymnasium info를
해석하지 않는다. Dyna의 가상 전이도 동일한 기본 필드를 사용한다.

## TD(0) TargetGenerator

`component/target_generator/base.py`의 BaseTargetGenerator는 `generate()` 하나만
정의한다. nn.Module이 아니며 Model, Buffer, Optimizer를 소유하지 않는다.
`td0.py`의 TD0TargetGenerator는 외부에서 계산한 Tensor를 사용해 다음 수식만 계산한다.

```text
target = reward + gamma * (1 - terminated) * next_value
```

| API / 입력 | 계약 |
|---|---|
| `TD0TargetGenerator(gamma)` | 실수 discount, `0 <= gamma <= 1` |
| `generate(reward, next_value, terminated)` | 세 Tensor -> 하나의 target Tensor |
| reward / next_value | 동일 floating dtype, 동일 shape·device |
| terminated | bool Tensor, reward와 동일 shape·device |
| target | 입력 shape·floating dtype·device 유지 |

Scalar `()`, Replay `(B,)`, 추가 앞쪽 차원의 `(T, B)` 등 동일한 shape를 허용한다.
`(B,)`와 `(B, 1)`처럼 Broadcast 가능한 다른 shape도 명확히 실패한다.
NumPy 변환, Device 이동, 암묵적인 dtype 승격을 하지 않는다. 필요하면 호출자가
Replay Batch를 이동하고 reward를 next_value와 같은 floating dtype으로 변환한다.

종료된 항목의 next_value는 0으로 마스킹한 뒤 discount를 곱한다. 해당 항목의
사용하지 않는 NaN/Inf Bootstrap Value도 target에 섞이지 않는다. terminated=True이면
target은 reward이고, False이면 Bootstrap을 유지한다. truncated는 이 API의 입력이
아니다. 일반적인 Time-limit에서 terminated=False이면 실제 마지막 관측으로
계산한 next_value를 사용할 수 있다. Episode Boundary는 `terminated | truncated`이며
Bootstrap Mask와 다르다. 이 컴포넌트는 다음 에피소드와의 연결을 추론하지 않는다.

next_value 계산은 호출자(Learner)의 책임이다. TD Prediction은 V(s'),
SARSA는 외부에서 선택한 다음 행동의 Q(s', a'), Q-learning은 max_a Q(s', a)를
전달할 수 있다. TargetGenerator는 이런 선택 규칙이나 EstimatorOutput을 알 필요가 없다.

Gradient도 호출자 책임이다. generate는 입력 graph를 유지하고 reward에 대한 미분은 1,
next_value에 대한 미분은 `gamma * (1 - terminated)`다. 표준 semi-gradient TD에서는
호출자가 Bootstrap Model 평가를 `torch.no_grad()`로 감싸거나 next_value를 detach한다.
이렇게 계산한 next_value와 graph 없는 reward를 전달하면 target에도 graph가 없다.
TargetGenerator 내부에서 no_grad/detach를 적용하거나 Model을 호출하지 않는다.

```python
import torch
from modular_rl.component.target_generator import TD0TargetGenerator

reward = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
next_value = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64, requires_grad=True)
terminated = torch.tensor([False, True, False])
target = TD0TargetGenerator(gamma=0.9).generate(reward, next_value.detach(), terminated)
torch.testing.assert_close(target, torch.tensor([10.0, 2.0, 30.0], dtype=torch.float64))
assert not target.requires_grad
```

TD(n), GAE, lambda-return은 아직 구현하지 않는다. 순서가 없는 Replay 샘플로
Trajectory 계산을 대신하지 않으며 `(T, B, ...)`의 경계·순서 계약은 별도로 설계한다.

### 향후 Target 확장에서 결정할 사항

- 알고리즘별 입력·출력: Expected SARSA/Importance Sampling의 Policy 확률과 Behavior log_prob 사용 규칙, Distributional target 표현.
- 종료 정책: Time-limit 외 truncation의 의미, Bootstrap mask와 trace 경계의 분리, Runner의 실제 마지막 관측 제공 계약.
- Trajectory 확장: `(T, B, ...)`의 정렬·에피소드 경계, GAE/n-step 계산에 필요한 순서 정보와 On-policy Buffer 계약.
- Target Model 및 복원: 현재/Target Model 선택·갱신 책임, Buffer 내용·원형 위치·전용 RNG의 향후 checkpoint 범위.

## Tabular / Neural UpdateRule

공통 BaseUpdateRule을 두지 않는다. Tabular 갱신과 Neural Loss 계산은 서로 다른
메서드와 결과를 사용하며, Model, Estimator, Buffer, Optimizer 객체를 전달받지 않는다.
공통 TD error 정의는 `target - current_value`다. Neural에서는 current_value가
외부에서 전달된 prediction이고, Tabular에서는 각 행 갱신 직전의 테이블 값이다.

### Tabular 직접 갱신

`component/update_rule/tabular.py`의 두 Rule은 다음 계약을 따른다.

| API / 입력 | 계약 |
|---|---|
| `TabularVUpdateRule(alpha).update(v_table, states, target)` | V[s] 직접 갱신, signed TD error Tensor 반환 |
| `TabularQUpdateRule(alpha).update(q_table, states, actions, target)` | Q[s,a] 직접 갱신, signed TD error Tensor 반환 |
| alpha | 양의 유한 실수, 첫 구현은 `(0, 1]`; NumPy real scalar 허용, bool/NaN/Inf 거절 |
| v_table / q_table | floating Tensor, `(N,)` / `(N, A)` |
| states / actions | torch.long, Scalar `()` 또는 Batch `(B,)`, 유효한 인덱스 |
| target | 인덱스와 동일 shape, 테이블과 동일 floating dtype·device |
| 반환 TD error | target과 동일 shape·dtype·device, graph 없음 |

입력 계약을 검증한 뒤 `torch.no_grad()`에서 원본 테이블을 in-place로 갱신한다.
테이블을 교체하거나 Estimator를 복사하지 않으므로 Model의 등록된 buffer와
state_dict에 갱신 값이 그대로 반영된다. 원래 Tensor의 requires_grad 속성은 바꾸지 않는다.

```text
각 입력 행 i:
    delta_i = target_i - table의 현재 값
    table의 값 += alpha * delta_i
```

중복 상태 또는 상태-행동 쌍도 입력 순서대로 처리한다. 초기 값 0, alpha=0.5,
target=[1,3]이면 값은 `0 -> 0.5 -> 1.75`, 반환 error는 `[1,2.5]`다.
target 순서를 [3,1]로 바꾸면 최종 값은 1.25이고 error는 [3,-0.5]다.
중복 인덱스에 대한 일괄 Advanced Indexing 연산, 평균 또는 합산 갱신으로 대체하지 않는다.

target은 갱신 시작 시 한 번 복사해 고정한다. target이 테이블 storage를 참조해도
내부 snapshot은 중간 갱신으로 바뀌지 않는다. 원본 target이 table view인 경우 그 view의
값은 테이블과 함께 변하지만 갱신 계산은 복사한 값을 사용한다.
Bootstrap을 각 행 사이에 다시 계산하지 않는다. 온라인 TD에서 매 전이마다 변경된
테이블로 target을 만들려면 Learner가 전이별로 target 생성과 update를 호출한다.
Replay의 무작위 행 순서는 환경의 시간 순서가 아니다.

```python
import torch
from modular_rl.component.update_rule import TabularQUpdateRule

q_table = torch.zeros(1, 1)
td_error = TabularQUpdateRule(alpha=0.5).update(
    q_table, torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([1.0, 3.0]),
)
torch.testing.assert_close(td_error, torch.tensor([1.0, 2.5]))
assert q_table.item() == 1.75 and not td_error.requires_grad
```

### Neural TD Loss

`component/update_rule/td_loss.py`의
`TDLossUpdateRule(loss_fn=None).compute(prediction, target, *, weights=None)`는
`data/updates.py`의 frozen TDLossOutput을 반환한다. 필드는 signed td_error와 scalar loss다.

```text
td_error = target - prediction
per_sample_loss = loss_fn(prediction, target)
loss = mean(per_sample_loss)                  # weights 없음
loss = mean(weights * per_sample_loss)        # weights 제공
```

prediction/target은 동일 shape·floating dtype·device의 비어 있지 않은 Tensor다.
Scalar `()`, Replay `(B,)`, 추가 앞쪽 차원도 허용하며 자동 Broadcasting은 금지한다.
weights도 제공하면 동일 shape·dtype·device여야 한다. 기본 Loss는
`nn.MSELoss(reduction="none")`이며 `nn.SmoothL1Loss(reduction="none")` 또는
Callable을 주입할 수 있다. custom loss는 입력과 동일 shape의 floating Loss Tensor를
반환해야 한다. Batch의 reduction된 scalar 등 잘못된 출력은 명확히 실패한다.
Rule은 dtype/device를 변환하지 않으며 loss dtype은 loss_fn의 결과를 따른다.

가중치는 전달된 값 그대로 사용하며 합으로 나누거나 보정·정규화하지 않는다.
이는 향후 Importance Sampling/Prioritized Replay의 Sampler 또는 Learner 책임이다.
td_error는 prediction의 shape·dtype·device를, loss는 scalar shape `()`를 유지한다.

입력과 prediction/target의 graph를 유지한다. Optimizer, backward, 자동 detach는 없다.
표준 semi-gradient TD에서는 Learner가 Bootstrap Model과 TargetGenerator 호출을
no_grad로 감싸거나 target을 detach한다. 로그·Priority용 td_error의 detach도 Learner가
수행한다. custom loss의 미분 방식은 해당 callable이 정의한다.

```python
import torch
from modular_rl.component.update_rule import TDLossUpdateRule

prediction = torch.tensor([1.0, 3.0], requires_grad=True)
target = torch.tensor([3.0, 1.0], requires_grad=True)
result = TDLossUpdateRule().compute(prediction, target.detach())
torch.testing.assert_close(result.td_error, torch.tensor([2.0, -2.0]))
assert result.loss.item() == 4.0 and prediction.grad is None
result.loss.backward()  # Learner owns backward/step; this is outside compute().
assert prediction.grad is not None and target.grad is None
priority_error = result.td_error.detach().abs()
assert not priority_error.requires_grad
```

### Learner 경계와 알고리즘 재사용

학습 호출자가 Buffer를 샘플링하고 Learner는 Batch device/dtype을 관리하며 Model에서
prediction과 Bootstrap 후보를 계산한다. Buffer는 정수 관측·행동 dtype을 보존하므로 Tabular update에
전달하는 indices는 호출자가 long으로 정규화한다. 정수 reward도 호출자가 target 계산용
floating dtype으로 변환한다. UpdateRule과 TargetGenerator는 이런 변환을 하지 않는다.

| 알고리즘 | Learner가 선택하는 Bootstrap | 재사용 Rule |
|---|---|---|
| TD(0) Prediction | V(s') | TabularVUpdateRule 또는 Neural TDLossUpdateRule |
| SARSA | 외부에서 선택한 다음 행동의 Q(s',a') | TabularQUpdateRule |
| Q-learning | 현재 테이블의 max Q(s',·) | TabularQUpdateRule |
| DQN | Target Model의 max Q(s',·) | TDLossUpdateRule |
| Double DQN | Online Model의 argmax 행동을 Target Model에서 평가 | 같은 TDLossUpdateRule |

TD0TargetGenerator는 선택된 next_value와 reward/terminated로 target만 만든다.
DQN/Double DQN용 별도 Loss Rule은 필요하지 않다. 복수 Critic 중 어떤 Q를 쓸지도
Learner가 명시적으로 선택하며 Rule은 EstimatorOutput을 해석하지 않는다.

Optimizer의 zero_grad/backward/step, gradient clipping·누적, Target Model 갱신은
Neural Learner 책임이다. 현재 TabularQLearner의 진입점은 `learn(batch)`이며
Bootstrap -> target -> Tabular update를 조합한다. DQNLearner는
prediction -> target -> Neural compute -> optimizer step을 조합한다. Runner는 아직 구현하지 않는다.

SARSA의 실제 next_action은 현재 Transition에 없다. 수집 경계에서 선택한 다음
행동을 TabularSARSALearner.learn의 next_action 인자로 별도로 전달한다. 무작위 Replay의 이웃 행이나 현재
action을 다음 행동으로 간주하지 않는다. 일반적인 Time-limit은 Bootstrap할 수 있지만
Episode Boundary와 Bootstrap Mask는 여전히 별개의 개념이다.

Prioritized Replay는 Learner가 샘플 ID와 `abs(td_error.detach())`를 연결해 Buffer로
전달한다. 현재 TransitionBatch에는 ID/importance weight가 없으므로 향후 별도 sampling
metadata 계약을 설계한다. 반환 error는 Neural step 전, Tabular 각 행의 갱신 직전 값이다.
갱신 후 잔차가 필요하면 Learner가 재평가한다. Dyna는 실제·가상 전이에 같은 Rule을
사용할 수 있으며 WorldModel, planning queue와 predecessor 관계는 별도 책임이다.

## TabularQLearner와 FrozenLake 검증

`learner/tabular_q.py`의 TabularQLearner는 기존 컴포넌트를 조합하는 일반 Python 객체다.
BaseLearner를 추가하지 않고 Optimizer, Target Network 또는 Buffer를 소유하지 않는다.

| API / 입력 | 계약 |
|---|---|
| `TabularQLearner(model, target_generator, update_rule)` | TabularQEstimator Model + TD0TargetGenerator + TabularQUpdateRule |
| `learn(batch: TransitionBatch) -> Tensor` | 외부 Batch 학습, 행 순서를 유지하는 signed TD error `(B,)` |
| observation / next_observation / action | 정수 indices `(B,)`; 입력의 정수 dtype을 long으로 정규화 |
| reward | floating Tensor `(B,)` |
| terminated / truncated | 서로 독립적인 bool Tensor `(B,)` |
| 반환 TD error | 현재 테이블의 dtype·device, graph 없음 |

한 전이는 `B=1`로 전달한다. 모든 필드의 shape는 동일해야 하며 비어 있는 Batch와
floating/bool state-action indices는 거절한다. 연속 관측이나 연속 행동을 지원하는
Learner가 아니다. Model은 이미 table index인 관측을 Representation에서 그대로 보존해야
하며 기본 구성은 IdentityRepresentation이다. 다른 Estimator를 연결하면 생성 시 실패한다.
ActionSelector는 선택적으로 존재할 수 있고 learn에서는 호출하지 않는다.

```text
호출자: Environment -> Model.act -> Transition -> ReplayBuffer.add
호출자: ReplayBuffer.sample -> TransitionBatch -> learner.learn
Learner: Batch device/dtype 준비, state/action long 변환
         -> no_grad에서 Model.forward(next_state)
         -> action 축 max Q 선택
         -> TD0TargetGenerator.generate(reward, next_value, terminated)
         -> TabularQUpdateRule.update(q_table, states, actions, target)
```

Learner는 매 호출에서 현재 Model의 q_table을 참조한다. 따라서 생성 후 Model.to나
double로 이동·변환한 테이블에도 대응하며, 직접 갱신한 값은 Model.state_dict에 저장된다.
Batch.to는 floating 필드만 테이블 dtype으로 바꾸므로 입력 reward는 먼저 floating이어야
한다. Buffer에 integer reward를 저장했다면 호출자가 reward를 floating으로 준비한다.
Model, Estimator, InputAdapter, Buffer, TargetGenerator와 UpdateRule의 공개 API는 유지한다.

Batch의 Bootstrap 평가와 target 생성은 어떤 테이블 갱신보다 먼저 완료한다.
중복 상태-행동은 기존 UpdateRule에 의해 입력 순서대로 갱신된다. 중간 행에서 Bootstrap을
다시 계산하지 않는다. 결과와 target에는 gradient가 없으며 Optimizer가 필요하지 않다.
truncated=True, terminated=False인 일반적인 Time-limit은 Bootstrap을 유지한다.
자동 reset의 실제 final observation 선택이나 에피소드 경계 추론은 호출자 책임이다.

```python
import torch
from modular_rl.component.buffer import ReplayBuffer
from modular_rl.component.estimator import TabularQEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TabularQUpdateRule
from modular_rl.data import Transition
from modular_rl.learner import TabularQLearner
from modular_rl.model import Model

model = Model(IdentityRepresentation(), TabularQEstimator(3, 2))  # Selector is optional.
model.estimator.q_table[1] = torch.tensor([10.0, 20.0])
learner = TabularQLearner(model, TD0TargetGenerator(0.9), TabularQUpdateRule(0.5))
replay = ReplayBuffer(1, seed=7)
replay.add(Transition(0, 0, 1.0, 1, False, True))
error = learner.learn(replay.sample(1))
torch.testing.assert_close(error, torch.tensor([19.0]))
assert model.estimator.q_table[0, 0].item() == 9.5
```

### FrozenLake 예제 실행

Gymnasium은 실행 환경의 표준 API이므로 필수 의존성이다. 기존 `examples` extra 설치 명령은
호환성을 위해 계속 허용한다. 렌더링하지 않으므로 pygame 등의 GUI 의존성은 필요하지 않다.

```bash
python -m pip install -e ".[examples]"
python -m examples.frozenlake_tabular_q --episodes 1000 --eval-episodes 100 --seed 7
```

`examples/frozenlake_tabular_q.py`는 명시적인 환경 루프로 모델, Learner와 외부 ReplayBuffer를
조합한다. FrozenLake-v1의 4x4 map과 is_slippery=False를 사용한다.
환경 설정과 종료 조건은 [Gymnasium FrozenLake 공식 문서](https://gymnasium.farama.org/main/environments/toy_text/frozen_lake/)를 참고한다.
PyTorch 행동 RNG, Replay 전용 RNG, episode reset seed를 설정하고 epsilon을 1.0에서
최소 0.05까지 감소시킨다. gamma=0.99, alpha=0.5, capacity=2000이며 매 step에서
`sample(min(32, len(replay)))`를 외부에서 호출한다. 환경 reward는 float으로 변환해
Transition에 저장한다. log_prob은 학습 행동의 값을 저장하되 Q-learning에서는 사용하지 않는다.

종료 또는 truncation이면 episode를 마치며 step의 실제 next_observation을 저장한다.
Autoreset wrapper는 사용하지 않는다. 평가에서는 deterministic=True로 Greedy 행동을 선택하고
Buffer를 저장하거나 학습하지 않는다. 학습 전후 동일한 평가 seed를 사용하며 환경은 finally에서 닫는다.
출력은 episode별 진행 정보, 평가 성공률·평균 길이·종료 플래그 수, Q-table 변경량과 Greedy 행동이다.

검증 실행(Gymnasium 1.4.0, seed=7, 1,000 학습 episode, 100 평가 episode)에서는
학습 전 Greedy 성공률 0%, 평균 길이 100 step에서 학습 후 성공률 100%, 평균 길이
6 step으로 변했다. Q-table 64개 항목 중 35개가 변경됐고 최대 절대 변화량은 1.0이었다.
이 성공률은 보고용 실험 결과이며 단위 테스트 합격 기준이 아니다. 수학적 정확성은
결정적인 Bellman/종료/중복 target 테스트로, 실제 환경 루프는 짧은 실행·seed 재현·TimeLimit
처리·환경 close 테스트로 검증한다. 기본 설치에도 Gymnasium이 포함된다.

### DQNLearner와 실제로 겹치는 코드

두 Learner는 Batch.to로 floating dtype·device를 준비하고 이산 action을 long으로 변환한다.
Batch의 reward/종료 shape·dtype 검사와 no_grad에서 action별 max Q -> TD0TargetGenerator를
호출하는 흐름도 겹친다. 반면 Tabular는 정수 state indices와 테이블의 입력 순서 갱신을,
DQN은 feature 관측과 미분 가능한 gather prediction·Optimizer·독립 Target Model을 사용한다.
이 작은 반복을 숨기기 위해 공통 BaseLearner나 Validation Framework를 추가하지 않는다.

## Tabular SARSA와 실제 Next Action

[Sutton & Barto Chapter 6](https://people.cs.umass.edu/~barto/courses/cs687/Chapter%206.pdf)의
On-policy TD Control을 Model, TabularQEstimator, TD0TargetGenerator, TabularQUpdateRule로 구성한다.
Q-learning의 max Q 선택을 실제 next_action의 Q gather로 바꾼 독립적인 작은 Learner다.
기존 컴포넌트/API를 유지하고 BaseLearner나 공통 Validation Framework를 추가하지 않는다.

| 공개 API / 입력 | 계약 |
|---|---|
| `TabularSARSALearner(model, target_generator, update_rule)` | 같은 Tabular Q Model, TD0 target, Q update rule |
| `learn(batch, *, next_action=None) -> Tensor` | signed TD Error `(B,)`, live table floating dtype/device, graph 없음 |
| TransitionBatch fields | 기존 tabular `(B,)`; 정수 State/Action, floating reward, bool 종료 flags |
| next_action | `(B,)` 정수 Tensor; B=1은 정수 scalar도 허용; table device의 long으로 준비 |
| 모든 행 terminated=True | None 허용; 다음 상태 Model 평가도 생략 |
| 혼합 Batch | terminal 행도 유효 ID(예: 0)를 넣되 해당 Q는 TargetGenerator가 무시 |
| truncated-only | 실제 마지막 관측의 Next Action으로 Bootstrap; 종료 환경에서 실행하지 않음 |

```text
호출자: Environment.step(action) -> nonterminal이면 Model.act(next observation)를 한 번 호출
        -> Transition.to_batch(), 선택한 next_action을 함께 전달
Learner: Batch를 Q dtype/device에 맞추고 인덱스 long 준비
         -> no_grad Model.forward(next_state) -> Q(next_state, next_action) gather
         -> 기존 TD0TargetGenerator.generate(reward, next_value, terminated)
         -> 기존 TabularQUpdateRule.update(table, states, actions, target)
호출자: 경계가 아니면 이미 선택한 next_action을 다음 step에서 실행
```

Learner는 행동 선택·Buffer 샘플링·환경·Optimizer를 소유하지 않는다. On-policy 수집과 정책 일관성은
호출자 책임이며 argmax를 실제 선택 행동 대신 전달하지 않는다. 기존 Transition 필드를 바꾸지 않고
무작위 Batch의 다음 행이나 현재 action을 a′로 추측하지 않는다. B=1 scalar를 B>1로 Broadcasting하지
않는다. terminal placeholder도 유효 정수 범위를 검사한다. 두 종료 flag가 True이면 Bootstrap하지 않는다.
같은 Model을 갱신하므로 이후 선택에는 새 Q가 반영되지만 이미 선택한 a′는 다시 뽑지 않는다.

### Frozen Batch와 Online SARSA

모든 Bootstrap Q와 Target은 table 갱신 전에 계산한다. B>1은 Frozen Target Batch Update이며
중복 쌍은 기존 Rule의 입력 순서 계약을 따른다. 진정한 Online SARSA는 B=1을 시간순으로 호출해
다음 전이의 Bootstrap에서 새 table을 읽는다.
초기 Q(1,0)=4, Q(2,:)=[10,2], reward=1, gamma=.9, alpha=.5에서 a′=1은
Target=2.8/Error=-1.2/Q(1,0)=3.4다. a′=0은 Target=10/Error=6/Q=7이다.
terminated는 Target=1/Error=-3/Q=2.5이며 truncated-only는 Bootstrap을 유지한다.

반복 쌍 s=1,a=0,s′=1,a′=0, 초기 Q=4, gamma=1, alpha=.5, reward=[1,3]이면 Frozen Batch의
Target=[5,7], Error=[1,2.5], Q=5.75다. B=1 Online 호출은 두 번째 Target=7.5로 Q=6이 된다.
terminal 갱신이 뒤 행의 Bootstrap 상태와 겹치는 별도 테스트도 포함한다.
Model 저장·복원은 기존 state_dict이며 추가 영구 Learner 통계는 없다. 재개 시 gamma/alpha와
환경·CPU/CUDA PyTorch RNG 상태도 함께 보존한다.

### Gymnasium Step-by-step 비교

```bash
python -m examples.gridworld_sarsa --episodes 1000 --seeds 7 11 19 --gamma .9 --alpha .8 --epsilon .1
```

예제는 a′를 학습 전에 선택해 다음 step까지 유지한다. terminated에서는 a′를 선택하지 않고,
truncated-only에서는 final observation의 a′를 Bootstrap에만 사용한다. 실제 TimeLimit 경계에서
종료된 환경을 다시 step하지 않는지도 검증한다. Q-learning은 max Q Target을 만든 뒤 다음 실행
행동을 선택한다. 두 독립 실행은 동일 예산·설정이지만 같은 Seed도 동일 Episode를 뜻하지 않는다.
Episode별 Reward/Length, 누적 Step, Q, 별도 Greedy 평가와 DP 최적 V를 보고한다.

| Seed | SARSA Step / 평가 | Q-learning Step / 평가 | max Q의 DP RMSE: SARSA / Q-learning |
|---|---|---|---|
| 7 | 8,604 / Goal 5 Step | 5,738 / Goal 5 Step | .375808 / .230544 |
| 11 | 6,659 / Goal 5 Step | 5,630 / Goal 5 Step | .300900 / .069670 |
| 19 | 6,627 / 20 Step cutoff | 5,611 / Goal 5 Step | .255833 / .243963 |

각 1,000 Episode의 training cutoff는 0, 평가 cap은 20이며 초기 Greedy는 모두 cutoff였다.
DP 최적 경로는 5 Step/V*(start)=.6561이다. RMSE는 비종료 10개 상태의 max Q와 V* 차이다.
SARSA는 탐색을 포함한 행동 정책을 학습하므로 DP Q*를 직접 추정한다고 간주하지 않는다.
고정 alpha=.8과 유한 표본의 Greedy 실패를 숨기거나 방법 간 순위를 테스트 조건으로 고정하지 않는다.
성능 평가와 수학적 정확성·Next Action 선택 횟수/실행 재사용 테스트는 분리한다.

06 Notebook은 기존 TD(0)/Q-learning을 보존하고 마지막에 SARSA 수치 예제·B=1 Step-by-step·
독립 다중 Seed 비교와 Greedy/학습 성능 그림을 추가한다. 환경 실행만 Notebook에 두고
Target/Q 갱신은 Learner의 공개 API에 맡긴다. 직접 SARSA 루프와 비교 루프의 Seed 7 Q도 일치한다.
Expected SARSA, Online n-step SARSA와 고정 정책 평가용 Off-policy n-step SARSA는 아래의 예제 조합으로
검증하며 라이브러리 Learner는 추가하지 않는다. Off-policy n-step Control은 아직 구현하지 않는다.

## Expected SARSA 컴포넌트 조합 검증

`modular_rl/` 본체와 공개 API는 유지한다. `examples/gridworld_expected_sarsa.py`의 작은 함수가
기존 컴포넌트를 직접 호출하며 새로운 Learner·TargetGenerator·Base Class를 정의하지 않는다.

```python
learn_expected_sarsa(model, batch, target_generator, update_rule, *, policy_selector) -> Tensor
train_and_evaluate(algorithm="expected_sarsa", episodes=1000, seed=7,
                   gamma=.9, alpha=.8, epsilon=.1, max_steps=2000, log_every=100)
```

첫 함수는 예제 API이며 signed TD Error `(B,)`를 반환한다. 기존 tabular batch 계약을 검사하고
정수 state/action은 long, floating reward는 live Q-table의 dtype/device로 준비한다.
`policy_selector is model.action_selector`를 요구해 실제 행동 선택과 기대값 계산의 정책을 일치시킨다.
`train_and_evaluate`는 독립 Model과 실험 결과를 반환한다. 계산용 Learner 상태는 따로 만들지 않는다.

```text
TransitionBatch (B,) -> table dtype/device 준비
no_grad Model.forward(next_state) -> EstimatorOutput, 단일 Q (B,A)
동일 ActionSelector.probabilities(estimate) -> pi (B,A), RNG 소비 없음
next_value = (pi * Q).sum(-1) -> (B,)
TD0TargetGenerator.generate(reward, next_value, terminated) -> frozen target
TabularQUpdateRule.update(table, states, actions, target) -> signed TD Error (B,)
```

`probabilities()`가 단일 Scalar Discrete Q를 검증한 뒤 같은 estimate의 Q를 사용한다.
Q/확률 Shape·dtype·device가 같아야 하며 Broadcasting하지 않는다. 현재 조합은 Tabular Q와 이산
확률 조회를 지원하는 Selector를 대상으로 한다. 모든 Selector에 probabilities를 강제하거나 연속
정책을 이산 확률로 바꾸지 않는다. no_grad·table 직접 갱신에는 Optimizer가 없다.

행동 수 A와 최대 Q 집합 M에서 기존 ε-greedy 확률은
`pi(a|s) = epsilon/A + (1-epsilon) I(a in M)/|M|`다.
따라서 ε=0은 max Q, ε=1은 Q 평균이다. `[0,3,3,1]`, ε=.2의 기대값은 2.75로
동률 Greedy 질량과 탐색 질량을 모두 포함한다. `[10,2]`, ε=.2는 pi=[.9,.1], next_value=9.2다.
현재 Q=4, r=1, gamma=.9, alpha=.5이면 Target=9.28/Error=5.28/새 Q=6.64다.
terminated이면 Target=1/Error=-3/새 Q=2.5, truncated-only이면 Bootstrap을 유지한다.

Batch의 모든 Q·pi·Target은 첫 갱신 전에 계산한다. terminal update가 다른 행의 bootstrap 상태를
바꿔도 Target은 고정된다. 중복 쌍은 기존 Rule이 입력 순서로 갱신한다. 초기 Q=4, gamma=1,
epsilon=0, alpha=.5, rewards=[1,3]의 중복 self-transition은 Error=[1,2.5], 최종 Q=5.75다.
B=1의 순차 online 호출은 다음 전이에서 갱신된 Q/정책을 다시 사용한다.

환경은 Gymnasium reset/step으로 즉시 학습하며 ReplayBuffer를 소유·샘플링하지 않는다.
실제 다음 실행 행동은 Expected/Max-Q의 업데이트 후 샘플링한다. SARSA 비교는 실제 next_action을
업데이트 전에 선택해 다음 step까지 유지한다. terminated/truncated에서 환경을 더 step하지 않고
reset하며 실제 마지막 관측을 전달한다. 확률 조회·기대값 계산은 행동 RNG를 소비하지 않는다.

### 동일 설정의 DLFS4 Gridworld 실험

각 방법은 1,000 Episode, gamma=.9, alpha=.8, epsilon=.1, 시작 (2,0), 학습 cap=2,000 Step,
평가 cap=20으로 독립 실행한다. 같은 Seed도 동일 Episode를 의미하지 않는다.
DP 최적 경로는 5 Step, V*(start)=.6561이다. RMSE는 비종료 10개 상태의 max Q와 V* 차이다.
Expected/SARSA의 탐색 정책 Q가 Q*를 추정한다고 가정하지 않는다.

| Seed | Expected Step / Greedy / RMSE | Q-learning Step / Greedy / RMSE | SARSA Step / Greedy / RMSE |
|---|---|---|---|
| 7 | 53,664 / Goal 5 / .260824 | 5,738 / Goal 5 / .230544 | 8,604 / Goal 5 / .375808 |
| 11 | 5,592 / Goal 5 / .050405 | 5,630 / Goal 5 / .069670 | 6,659 / Goal 5 / .300900 |
| 19 | 5,621 / Goal 5 / .305310 | 5,611 / Goal 5 / .243963 | 6,627 / cutoff 20 / .255833 |

Expected Seed7의 학습 cutoff는 23, 나머지 실행은 0이다. 마지막 100 Episode의 평균 Reward는
Expected .99/.95/1.0, Q-learning 1.0/.98/1.0, SARSA 1.0/1.0/1.0이다.
유한 표본·고정 alpha의 편차와 평가 실패를 그대로 기록하고 방법 간 성능 순위를 테스트하지 않는다.
Q-learning/SARSA 비교의 Q와 Episode Reward/Length는 이전 예제와 동일함을 검증한다.

### 실제 중복과 공통화 후보

Q-learning/SARSA는 동일한 Batch 검증 본문 20줄, table dtype/device 준비, no_grad, TD0 target 호출,
Q UpdateRule 호출을 공유한다. 차이는 next_value의 max/gather/확률 가중합이며 SARSA에는 실제
next_action 준비와 terminal-only 처리도 있다. 예제 함수는 같은 입력 계약을 독립적으로 검사한다.
기존 private 검증 메서드를 호출하거나 본체 리팩터링으로 새 공개 API를 만들지 않는다.
실험 루프에도 Seed/reset/step/Transition/경계/평가/통계가 겹치지만 이번에는 재작성하지 않는다.

현재 Learner는 next_value 선택이 고정돼 있어 기존 learn에 기대값 계산만 주입할 수 없다.
TD0TargetGenerator는 이미 줄어든 `(B,)` next_value만 받으므로 Q/정책을 가져오는 새 TargetGenerator는
필요하지 않다. 출력 계약 수준에서는 모든 계산을 재사용할 수 있고, 상위 조합 코드만 필요하다.
향후 공통화한다면 **외부에서 갱신 전에 계산한 `next_value: Tensor (B,)`와 Batch를 받아 기존
TargetGenerator/UpdateRule을 호출하는 작은 함수**가 최소 후보다. 기존 Learner를 유지한 채
내부에서 재사용할 수 있지만, next_action 선택 시점·검증·dtype/device 준비·no_grad 소유권을 먼저
결정해야 한다. 이번에는 그 인터페이스, BaseLearner 또는 별도 Bootstrap 계층을 추가하지 않는다.

06 Notebook §12는 위 함수를 예제 파일에서 로드해 표시하고 수학적 계산을 다시 정의하지 않는다.
직접 B=1 환경 수집 셀, 세 Seed 비교, Q/Greedy/DP와 학습 Reward/Length/누적 Step 그림을 추가한다.
기존 코드 셀 13개의 소스·출력은 보존한다. 예제는 별도 학습 통계를 갖지 않으며 Q/epsilon은
Model.state_dict에 포함된다. 학습 재개에는 gamma/alpha와 환경·PyTorch RNG도 보존해야 한다.

## Double Q-learning 컴포넌트 조합 검증

`examples/gridworld_double_q_learning.py`는 두 독립적인 TabularQEstimator Model을 조합한다.
`modular_rl/` 내부 구현·공개 API를 변경하지 않고 새 Learner/Estimator/Runner 클래스를 만들지 않는다.
두 테이블은 동일 shape·floating dtype·device를 사용하며 별도 storage를 요구한다.
storage를 공유하는 두 view는 data_ptr가 달라도 거절한다. Representation은 정수 State ID를 유지한다.

| 예제 API | 책임 |
|---|---|
| `combined_estimate(model1, model2, observation)` | 두 공개 forward 결과의 Q 합을 단일 Q payload로 반환; shape (...,A) |
| `act_double_q(model1, model2, observation, selector, *, deterministic=False)` | 기존 Selector.select로 합산 Q의 ActionOutput 생성 |
| `choose_update_first(rng)` | 명시적인 NumPy Generator로 50% 동전 한 번 선택 |
| `learn_double_q(model1, model2, batch, target_generator, update_rule, *, update_first)` | 교차 Bootstrap과 기존 Target/Rule 조합; signed error (B,), graph 없음 |
| `train_and_evaluate(...)` | Gymnasium B=1 online 학습·독립 Greedy 평가·DP 비교; 두 Model/Selector와 실험 결과 반환 |

Behavior Policy는 반드시 epsilon-greedy(Q1+Q2)다. 기존 Model은 하나의 Estimator를 소유하므로
두 Model.forward를 호출하고 `EstimatorOutput(values=ValueOutput(q=(q1+q2,)))`로 명시적으로 결합해
Selector의 공개 select/probabilities를 사용한다. 두 Critic tuple을 Selector에 그대로 전달하지 않는다.
합산 Tensor는 현재 출력일 뿐 별도의 영구 Q-table 복사본이나 매 Step 동기화 대상이 아니다.
정상 입력의 단일/Batch 행동·log_prob, stochastic 균등 동률과 deterministic 최소 ID는 기존 Selector를 따른다.

```text
Q1 갱신: Q1.forward(next_state) -> argmax -> Q2.forward(next_state)의 해당 행동 gather
Q2 갱신: Q2.forward(next_state) -> argmax -> Q1.forward(next_state)의 해당 행동 gather
no_grad 교차 next_value (B,) -> 기존 TD0TargetGenerator.generate(reward, next_value, terminated)
선택된 원본 q_table만 기존 TabularQUpdateRule.update(table, states, actions, target)로 갱신
```

현재 Q1(s,a)=4, Q2(s,a)=6, 다음 Q1=[10,2], Q2=[1,7], r=1, gamma=.9, alpha=.5이면
Q1 갱신은 Target=1.9/Error=-2.1/새 Q1=2.95, Q2 갱신은 Target=2.8/Error=-3.2/새 Q2=4.4다.
각 방향에서 반대 테이블·storage는 유지한다. Bootstrap의 argmax 동률은 최소 행동 ID이며
합산 Behavior의 Greedy 행동 또는 반대 테이블의 max로 대체하지 않는다. Argmax는 RNG를 소비하지 않는다.

입력은 기존 tabular TransitionBatch의 (B,) 정수 State/Action, floating reward, bool flags다.
Batch는 두 테이블의 dtype/device에 맞추고 인덱스를 long으로 정규화한다.
update_first는 호출 전체에 적용되는 단일 bool이며 행별 동전을 묵시적으로 허용하지 않는다.
Q 선택·평가와 모든 Target을 갱신 전에 완료한 뒤 중복 쌍을 기존 Rule이 입력 순서로 갱신한다.
예를 들어 Q1(s,:)=[4,0], Q2(s,:)=[6,0], gamma=1, alpha=.5, rewards=[1,3]의 Q1 중복 self-transition은
Target=[7,9], Error=[3,3.5], 최종 Q1=7.25다. B=1 순차 호출은 매 전이의 새로운 Argmax를 사용한다.
terminated에서 Bootstrap을 제거하고 truncated-only에서 실제 final observation으로 유지한다.

### RNG와 저장·복원 경계

환경 RNG는 첫 reset(seed=seed)의 Gymnasium np_random, Behavior는 torch.manual_seed(seed)의
PyTorch RNG, 갱신 대상은 독립 default_rng(seed+2)다. 매 환경 Step마다 갱신 동전을 한 번 뽑는다.
확률 조회, argmax, learn 함수와 deterministic Greedy 평가는 두 학습 RNG를 소비하지 않는다.
환경의 전이 샘플링은 평가 환경의 독립 RNG로 수행한다. 동일 Seed 재실행은 Q1/Q2·행동·통계가 일치한다.

별도 Learner 통계는 없지만 두 Model.state_dict와 외부 Selector.state_dict(epsilon), gamma/alpha,
Torch CPU/CUDA RNG, 갱신 Generator.bit_generator.state를 함께 복원한다.
수집자가 live 환경을 재개할 때는 환경 RNG와 Episode 위치도 관리한다. 현재 환경은 범용
checkpoint API를 제공하지 않는다. 고정 경험 stream의 저장·복원 후 행동/동전/Error/테이블 재현성을 검증한다.

### DLFS4 Gridworld 결과

각 1,000 Episode, gamma=.9, alpha=.8, epsilon=.1, 시작 (2,0), 학습 cap=2,000 Step,
Greedy 평가 cap=20이다. Q-learning 비교는 기존 Expected SARSA 예제의 comparator 함수를 그대로 재사용한다.
`--compare-all`은 같은 함수로 SARSA/Expected SARSA도 실행한다. 각 알고리즘은 독립 환경·Model 실행이다.
Notebook은 §12의 기존 comparator 결과를 재사용하며 다시 학습시키지 않는다.

| Seed | Double Q Step / Q1·Q2 갱신 수 | Double Greedy | max 평균 Q와 V* RMSE | Q-learning Step / Greedy / RMSE |
|---|---|---|---|---|
| 7 | 6,017 / 3,044·2,973 | Goal 5 Step/+1 | 2.449e-9 | 5,738 / Goal 5 / .230544 |
| 11 | 5,790 / 2,872·2,918 | Goal 5 Step/+1 | .148032 | 5,630 / Goal 5 / .069670 |
| 19 | 5,796 / 2,868·2,928 | Goal 5 Step/+1 | 1.394e-8 | 5,611 / Goal 5 / .243963 |

Double Q의 마지막 100 Episode 평균 Reward는 .99/.99/.97, 평균 Length는 5.62/5.58/5.44다.
training cutoff는 세 Seed 모두 0이다. 초기 Greedy는 모두 평가 cutoff이며
DP 최적 경로는 5 Step, V*(start)=.6561이다. Episode별 Reward/Length, 누적 Step, 두 Q-table,
합산 Q Greedy 경로·DP 경로를 기록한다. 가치 비교는 비종료 10개 상태의 **max((Q1+Q2)/2)**를
DP V*와 비교한다. 합산 Q의 두 배 스케일을 상태가치로 해석하지 않는다. 특정 Seed의 성능 순위나
모든 Q값의 수렴을 테스트 조건으로 삼지 않으며 Gridworld 결과만으로 일반적인 Bias 감소를 주장하지 않는다.

### Maximization Bias 원본 실험의 한계

[Sutton & Barto §6.7, Example 6.7, pp.134–136](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf#page=156)을
확인했다. A의 right는 보상 0으로 종료, left는 보상 0으로 B에 간다. B의 여러 행동은
평균 -.1·분산 1의 정규분포 보상으로 종료한다. A는 두 행동이며 B는 더 많은 행동을 사용한다.
현재 TabularMDP의 유한 Outcome은 고정 실수 reward만 저장하고 모든 상태가 같은 n_actions를
사용한다. TabularEnv에는 연속 Reward Sampler나 상태별 유효 행동 mask가 없다.
원본 보상 분포와 정책을 정확히 재현하려면 핵심 환경 계약의 확장이 필요하므로 이번 실험은 제외한다.
유한 보상 근사, 중복 행동 또는 새 환경 클래스로 원본 재현을 우회하지 않는다.

### 범용성 평가와 최소 공통화 후보

기존 Model/Estimator/출력/Selector/TargetGenerator/UpdateRule만으로 Double Q-learning을 실행할 수 있었다.
새 클래스는 필요 없고 예제 함수만 추가했다. Model.act 자체가 두 Estimator를 자동 결합하지는 않아
현재 Q 출력의 명시적 합산과 Selector 직접 호출이 필요했다. 불필요한 Q-table 복사·동기화는 없었다.
TargetGenerator는 교차 next_value를 그대로 받고 Rule은 선택된 table만 갱신하므로 재사용에 문제가 없다.

단일 Model facade와 기존 알고리즘별 Learner의 고정 Bootstrap은 복수 Model의 역할·결합·갱신 선택을
자동 표현하지 않는다. 출력 tuple은 복수 Critic을 담을 수 있지만 결합 수학과 table 소유권은 상위 조합의 책임이다.
공통화 후보는 **Q 출력의 명시적 결합 함수**와 **외부 next_value (B,)를 받아 기존 Target/Update를
호출하는 작은 함수**다. 갱신 대상과 RNG 소유권은 호출자가 지정해야 한다.
이번에는 BaseLearner/Bootstrap 계층이나 공개 API를 추가하지 않으며 함수 조합 결과만 기록한다.
06 Notebook §13는 같은 예제 함수를 로드해 표시하고 수치·직접 Online 수집·두 테이블·Greedy/DP·
다중 Seed 비교를 추가한다. 이전 코드 셀 18개와 출력은 보존한다.

## Online n-step SARSA 컴포넌트 조합 검증

현재 소스에는 TDNTargetGenerator, SequenceReplayBuffer, n-step DQN이 없다. 기존 TD0TargetGenerator는
외부 reward/continuation/terminated의 한 번의 backup을 제공하므로 이를 역순으로 합성해 n-step Return을
계산한다. 기존 함수의 수식을 복제하거나 TargetGenerator/Learner/Buffer 클래스를 추가하지 않는다.
`examples/gridworld_n_step_sarsa.py`만 시간 관리와 계산 조합을 담당하며 `modular_rl/`는 변경하지 않는다.

| 예제 API | 계약 |
|---|---|
| `learn_n_step_sarsa(model, window, target_generator, update_rule, *, next_action=None)` | 순서 있는 B=1 batch window의 가장 오래된 행 하나 갱신; signed error (1,), table dtype/device, graph 없음 |
| `run_episode(env, model, target_generator, update_rule, *, n=3, seed=None)` | reset/step, 실제 다음 행동 선택·유지, Online head update, boundary flush; Reward/Step/Update/flags 반환 |
| `train_and_evaluate(n=3, episodes=1000, seed=7, ...)` | 독립 Gridworld 실행, Q/종료/갱신 수/Greedy/DP/Reward/Length 결과 기록 |

Window는 `deque`의 TransitionBatch snapshots다. 각 필드는 (1,), State/Action은 integer,
reward는 floating, flags는 bool이며 table dtype/device로 준비하고 인덱스를 long으로 정규화한다.
Transition.to_batch의 기존 owned·detached 계약으로 배열/Tensor 재사용과 Autograd alias를 피한다.
연속 관측의 일치와 중간 episode boundary 부재를 검사하되 timestamp를 추론하지 않는다.
순서와 실제 행동의 기록은 수집자 책임이다. Episode는 전체 완료 기록용으로 유지하고 훈련 전에
완성 Episode를 수집하지 않는다. ReplayBuffer.sample은 무작위 순서여서 이 Online window에 사용하지 않는다.

### 기존 TargetGenerator의 역순 합성

```text
no_grad: 마지막 next observation에서 실제 endpoint action의 Q -> continuation (1,)
         natural terminal이면 Q 평가 없이 zeros_like(reward)
         reversed(window): continuation = TD0.generate(reward, continuation, terminated)
TabularQUpdateRule.update(table, oldest_state, oldest_action, continuation)
```

각 TD0 호출의 shape/dtype/device·termination mask·gradient 계약은 그대로다. 첫 continuation은
Bootstrap Q이고 이후에는 계산한 suffix Return이다. m번의 기존 backup 합성으로
`sum(gamma**k * reward[k]) + gamma**m * Q(endpoint, actual_action)`가 된다.
이름이 TD0라고 해서 TDN API가 구현된 것으로 주장하지 않는다. n과 window/horizon의 책임은 예제에 있다.
Target/Bootstrap은 각 호출에서 table 갱신 전에 계산하며, 실제 endpoint action을 max나 재샘플링으로 대체하지 않는다.

n=3, gamma=.9, rewards=[1,2,3], endpoint Q=4는 Target=8.146이다. 역순 TD0 호출의
보상 순서는 [3,2,1]이다. 짧은 T=2 natural terminal은 Targets=[2.8,2], truncated-only는
[6.04,5.6]이며 실제 보상 수 m=2/1의 discount를 사용한다. terminated와 truncated가 함께 True이면
natural terminal 규칙이 우선한다. 반환 Error는 `target-current Q`이며 원본 table에 기존 Rule을 적용한다.

### Online 시점과 잔여 갱신

[Sutton & Barto §7.2, p.147](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf#page=169)의
순서를 따른다. 저장한 A_t를 step에서 실행하고 자연 종료가 아니면 A_(t+1)을 한 번 선택한다.
그 다음 window 길이가 n이면 tau=t-n+1의 가장 오래된 행을 학습하고 pop한다.
다음 step은 이미 선택한 행동을 그대로 실행한다. 환경 RNG는 reset(seed), 행동 RNG는
PyTorch manual_seed(seed)며 Target 계산과 잔여 flush는 RNG를 소비하지 않는다.

n=3, T=5의 업데이트는 (실행 완료 Step 수, tau)=(3,0),(4,1),(5,2),(5,3),(5,4)다.
마지막 두 갱신은 step/act 없이 수행한다. n>T의 자연 종료는 관측한 보상으로 모든 잔여 head를 갱신한다.
교재의 종료 후 가상 시간 증가를 실질적 변화가 없는 대기 없이 같은 순서의 flush로 표현한다.
각 Transition은 정확히 한 번 갱신하며 pending 최대 길이는 n이다. 다음 Episode에 window를 이어 붙이지 않는다.

Truncation은 마지막 관측 S_T의 A_T를 선택만 하고 실행하지 않는다. 각 residual call은 같은 A_T로
**그 호출 직전의 현재 Q**를 읽는다. 전체 flush에 하나의 Q를 캐시하거나 Target을 한 번에 동결하지 않는다.
반복 self-transition의 초기 Q=4, rewards=[1,2], gamma=.9, alpha=.5이면 첫 Target=6.04 후 Q=5.02,
다음 Target=2+.9*5.02=6.518, 최종 Q=5.769다. Bootstrap Q=[4,5.02]를 실제로 읽는지 검사한다.
자연 종료에서는 모든 residual call에 Bootstrap이 없다. 단일 head당 B=1이므로 기존 Rule의 in-place
계약을 유지하며 반복 쌍은 각 호출 시점의 current Q를 사용한다. 무리하게 B>1로 일반화하지 않는다.

n=1은 기존 TabularSARSALearner와 같은 forward/gather/TD0/Rule을 사용한다.
float32/float64, natural terminal/truncation/동시 flag에서 실제 행동·환경 전이·Target·Error·table·Torch RNG를
정확히 비교했다. Gridworld 세 Seed의 1,000 Episode에서도 Q/Reward/Length가 기존 SARSA와 일치한다.
길이 <n/=n/>n, n=1/3/5, 반복 state/action, actual next-action 재사용과 누락 없는 갱신을 검증한다.

### Gridworld 동일 예산 결과

각 1,000 Episode, gamma=.9, alpha=.8, epsilon=.1, Seed7/11/19, 학습 cap=2,000, Greedy cap=20이다.
각 n은 독립 환경·Model을 사용하며 같은 Seed도 동일 경험을 뜻하지 않는다.
기존 SARSA/Q-learning comparator는 수정하지 않는다. Notebook은 기존 결과를 재사용한다.

| Seed | n=1 Step / Greedy / RMSE | n=3 Step / Greedy / RMSE | n=5 Step / Greedy / RMSE | Q-learning Step / Greedy / RMSE |
|---|---|---|---|---|
| 7 | 8,604 / Goal 5 / .375808 | 7,137 / Goal 5 / .274961 | 9,920 / Goal 5 / .250066 | 5,738 / Goal 5 / .230544 |
| 11 | 6,659 / Goal 5 / .300900 | 6,818 / Goal 5 / .372150 | 7,308 / Goal 5 / .298566 | 5,630 / Goal 5 / .069670 |
| 19 | 6,627 / cutoff 20 / .255833 | 7,089 / Goal 5 / .204578 | 8,134 / Goal 5 / .246335 | 5,611 / Goal 5 / .243963 |

모든 수집 Step에 대해 업데이트 수가 일치한다. 각 n의 자연 종료 Episode는 1,000, 학습 Truncation은 0이었다.
DP 최적 경로는 5 Step/V*(start)=.6561이며 보상·길이·Q·누적 Step·Greedy/DP 경로를 기록한다.
RMSE는 비종료 10개 상태의 max Q와 최적 V 차이이며 탐색 정책 Q가 Q*와 같다고 가정하지 않는다.
고정 alpha의 유한 표본 결과에서 n-step의 우위나 특정 Seed의 성공을 요구하지 않는다.

### Checkpoint와 구조적 평가

Episode 경계에는 Model.state_dict(epsilon 포함), n/gamma/alpha, 환경·Torch CPU/CUDA RNG를 저장한다.
도중에는 owned pending batches, 현재 관측, 이미 선택한 action/log_prob, Episode 누적 통계,
환경 위치/RNG와 TimeLimit 잔여 예산도 필요하다. 고정 경험 stream 도중의 pending/Model/행동/RNG를
복원한 뒤 후속 Error/Q가 정확히 같은지 테스트한다. 현재 환경에는 범용 mid-episode checkpoint API가 없다.
예제는 그런 상태를 Model이나 Learner에 숨기지 않고 수집자 소유권을 명시한다.

재사용 컴포넌트는 Model/InputAdapter/Representation/TabularQEstimator/Selector, Transition/Batch,
TD0TargetGenerator와 TabularQUpdateRule이다. 추가된 것은 ordered window 관리와 delayed update/flush의
조합 함수다. TDN/SequenceReplay/n-step DQN이 없다는 사실을 소스와 문서에서 확인했으며 새로 만들지 않는다.
기존 Learner는 독립 Batch를 즉시 학습하고 temporal window·actual endpoint action·flush 상태를 소유하지 않는다.
이 한계를 예제 수준의 명시적 deque/함수로 해결할 수 있었으며 핵심 컴포넌트 변경은 필요 없었다.

최소 공통화 후보는 ordered B=1 window, 실제 horizon, endpoint bootstrap value와 boundary를 외부에서
전달하는 작은 함수다. reward backup과 update 시점 관리는 n-step Q-learning/Expected SARSA에 재사용할 수 있지만
끝점 값 선택은 max/Policy 기대값/실제 행동으로 다르다. Q(sigma)는 중간 분기·확률·sigma 혼합이 필요하므로
endpoint 교체만으로 구현된다고 보지 않는다. 이번에는 공통 인터페이스나 BaseLearner/Buffer를 추가하지 않는다.
06 Notebook §14는 같은 예제 함수와 기존 comparator를 사용하고 이전 49개 셀 전체를 그대로 보존한다.

## Off-policy n-step SARSA 컴포넌트 조합 검증

`examples/gridworld_off_policy_n_step_sarsa.py`는 고정 Target Policy의 Qπ를 평가한다.
`modular_rl/`의 61개 추적 파일과 공개 API는 변경하지 않는다. 새 Learner, TargetGenerator,
UpdateRule, Policy/Runner 클래스나 Framework를 만들지 않는다.

| 예제 함수 | 책임 |
|---|---|
| `learn_off_policy_n_step_sarsa(model, window, target_generator, update_rule, *, target_policy, bootstrap_action=None, bootstrap_log_prob=None)` | ordered B=1 window의 head 하나 갱신; 반환 dict의 `target=G`, `effective_target`, `raw_td_error=G-Q`, `td_error=Yeff-Q`, `importance_ratio` |
| `run_episode(env, model, target_generator, update_rule, *, target_policy, n=3, seed=None)` | Online 수집, sampling-time 확률 저장, 지연 head update/flush, 실제 Step당 한 번 갱신 |
| `train_and_evaluate(mode="different", n=3, episodes=200, ...)` | 고정 정책 평가, 같은 π의 DP Qπ 참조, Episode/Step/RMSE/ρ/ESS 기록 |
| `run_comparison(seed=7, episodes=200, ns=(1,3,5), ...)` | 기존 On-policy, π=b, π≠b 독립 실행과 정확한 동등성 검증 |

Window/Index 검증과 On-policy 비교 실행은 원래 n-step 예제의 함수를 그대로 재사용한다.
예제 패키지는 설치 대상이 아니므로 Notebook cwd에서도 동작하는 표준 `runpy.run_path`로 그 파일을 읽는다.
기존 함수의 소스와 Notebook inspect 출력은 유지한다. 원래 수집 함수는 최종 행동만 전달하며
log_prob이나 학습 callback 확장 경로가 없다. 따라서 새 예제에 약 30줄의 같은 deque 수집·head/pop·flush
흐름이 남고, 짧은 forward/gather와 TD0 역순 합성도 명시적으로 배치한다. 계산 컴포넌트는 충분하지만
수집 조합 인터페이스는 아직 공유되지 않았다는 한계다. 이를 감추기 위한 monkeypatch나 Wrapper는 만들지 않는다.

### Return, Importance Ratio와 Effective Target

[Sutton & Barto §7.3, 식 (7.11), p.149](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf#page=171)을 따른다.
실제 reward 수 m에 대해 기존 TD0를 역순 호출하여
`G = sum(gamma**k * reward[k]) + gamma**m * Q(endpoint, sampled_action)`을 얻는다.
자연 종료는 Bootstrap을 제거한다. 갱신 대상 Aτ는 비율에서 제외한다.
Bootstrap이 있으면 Aτ+1부터 **sampled endpoint Aτ+m까지** π/b를 곱한다.
자연 종료면 마지막 실제 행동 A(T−1)까지만 포함하고 빈 곱은 1이다.
Expected SARSA는 끝점 기대값을 사용하므로 끝점 비율이 빠진다는 차이가 있다.

Time-limit truncated-only는 실제 final observation에서 행동과 log b를 한 번 선택해 보관한다.
그 행동을 실행하거나 flush 중 다시 선택하지 않는다. 같은 끝점 행동으로 각 head 갱신 직전의 Q를
다시 평가한다. 각 호출 내부 Target/ρ는 write 전에 확정하지만 전체 flush의 Q를 한 번에 동결하지 않는다.
terminated=True가 우선하므로 두 flag가 True인 경우에도 Bootstrap하지 않는다.
이는 자연 종료 교재 수식을 외부 Time-limit Bootstrap에 확장한 규칙이다.

TD0와 Q Rule의 API는 그대로다. `Yeff = current_Q + rho * (G-current_Q)`를 기존 Rule에 전달하여
`Q += alpha * (Yeff-current_Q)`를 수행한다. ρ=1은 G를 그대로 전달하고 ρ=0은 current_Q를 전달한다.
출력 (1,) Tensor는 table dtype/device이며 graph가 없다. ρ만 float64다.
`td_error`는 실제 Rule의 가중 Error이고 `raw_td_error`는 원래 signed Error다.
유한 정밀도에서 Effective Target을 만드는 덧셈/뺄셈 때문에 일반 ρ의 반환 Error는
수학적인 `rho * raw_td_error`와 작은 반올림 차이가 있을 수 있다. 이 방식은 B=1 Tabular 갱신의
조합이며 Neural Loss 가중치나 중복 B>1 Effective Target에 그대로 일반화하지 않는다.

실행한 행동은 기존 Transition.log_prob에 저장한다. 아직 실행하지 않은 Bootstrap 행동은
별도 action/log_prob으로 전달한다. Model.act의 실제 확률은 Q/ε 갱신 전 선택한 값이며 현재 Selector를
다시 조회하지 않는다. Batch.to가 floating field를 cast하더라도 IS에는 원본 log_prob을 사용한다.
Target 확률의 log는 그 확률표 dtype에서 계산하고 기록 log와 함께 float64로 누적·exp한다.
float32 기록은 원래 Sampling의 정밀도 한계를 유지하며 복구할 수 없는 정밀도를 만들어내지 않는다.

π=0이면 ρ=0, b=0에 해당하는 -inf log나 NaN/Inf/양수 log는 거절한다.
선택 행동의 로그만으로 미관측 행동의 Coverage `pi>0 ⇒ b>0`를 검사할 수 없으며 수집자가 보장한다.
양의 비율의 overflow/underflow, table dtype으로 표현 불가능한 비율·Target·Error·갱신을 검사하고
**현재 head의 write 이전에** 실패한다. Clipping은 없다. 앞선 성공한 Online 갱신을 rollback하는
Episode 단위 트랜잭션은 아니다. 실패한 head는 호출자가 pop하지 않는다.

수치 검증: n=1은 G=2.8, ρ=2, raw δ=-1.2, Qnew=2.8이다.
n=3은 G=8.146, ρ=3, Qnew=11.219다. m=2 자연 종료는 G=2.8/ρ=2/Qnew=2.8,
Truncation은 G=6.04/ρ=6/Qnew=14.12다. ρ=0은 변화 없이 후속 head를 처리한다.
π=b Uniform에서 n=1/3/5, float32/64, 길이 <n/=n/>n, 두 종료 flag 조합의 행동·Reward·Target·
Error·Q·Step·RNG가 기존 On-policy 코드와 정확히 일치한다.

### 고정 정책 실험과 재개 상태

Behavior는 기존 EpsilonGreedyActionSelector(1)의 실제 Uniform 확률 .25다. Matched π=.25,
Different π=[.3,.2,.2,.3]이며 모두 caller-owned 확률표 (S,A)로 보관하고 Q에 따라 개선하지 않는다.
gamma=.9, alpha=.05, n=1/3/5, Seeds7/11/19, CLI 200 Episode, Notebook 100 Episode를 사용한다.
`--episodes 1000`으로 예산을 늘릴 수 있다. 동일 Uniform Behavior와 Seed는 세 모드의 실제 경험도 일치시킨다.
비종료 10개 상태의 모든 행동에 대해 기존 policy_evaluation/action_values의 **동일 π Qπ**와 RMSE를 비교한다.
Episode reward/length, 누적 Step/Update, ρ 평균/최대/0 비율과 ESS=(sum ρ)²/sum ρ²를 기록한다.
ESS는 Weight 집중도 진단이며 중첩 window의 독립 표본 수라고 해석하지 않는다.
고정 alpha의 유한 표본 결과를 최적 제어 성공이나 알고리즘 우위로 해석하지 않는다.

CLI 실제 실행 결과(각 200 Episode):

| Seed | 각 run Step=Update | n=1 Matched / Different RMSE | n=3 Matched / Different RMSE | n=5 Matched / Different RMSE |
|---|---|---|---|---|
| 7 | 8,419 | .074577 / .071736 | .113803 / .166660 | .144084 / .213516 |
| 11 | 8,412 | .104793 / .098586 | .053557 / .073633 | .076829 / .110648 |
| 19 | 8,774 | .127772 / .124573 | .088591 / .102298 | .106447 / .131470 |

On-policy와 Matched의 모든 Q/Reward/Length는 동일했다. Different의 ρ 평균은 .996924~1.012976,
최댓값은 n=1/3/5에서 1.2/1.728/2.48832이고 0 비율은 모두 0이었다.
ESS는 n=1에서 8,097~8,443, n=3에서 7,520~7,822, n=5에서 7,005~7,284였다.
자연 종료·Time-limit 경계 처리는 별도 결정적 테스트로 검증한다.
세 Seed의 RMSE 평균±표준편차는 실행 결과에서 계산할 수 있고, Notebook은 100 Episode 예산으로 이를 출력한다.

재개에는 Model.state_dict(ε 포함), 고정 π, n/gamma/alpha, owned pending batches,
Bootstrap action/log_prob, 현재 관측·선택 행동, Torch CPU/CUDA RNG가 필요하다.
실제 수집 재개는 환경 위치/RNG/TimeLimit 잔여 예산과 누적 통계도 보존해야 한다.
테스트는 고정 경험 stream에서 Model/pending/끝점/정책/RNG 복원 후 Error/Q/다음 행동을 비교한다.
범용 환경 mid-episode checkpoint API나 checkpoint 클래스는 만들지 않는다.

### 구조적 평가와 향후 최소 공통화 후보

- Model/InputAdapter/Identity/TabularQ/Selector, Transition/Batch, TD0, Q Rule을 그대로 조합할 수 있었다.
  시간 Window 검증·On-policy comparator는 재사용하지만 수집 scheduling의 함수 추출은 하지 않았다.
- Target Policy는 고정 확률 Tensor면 충분하다. Transition.log_prob은 실행된 행동에 충분하며,
  실행 전 끝점 ActionOutput의 확률은 별도로 보관해야 한다.
- Effective Target으로 Q Rule을 바꾸지 않고 IS를 적용할 수 있었다. 원래/가중 Error의 차이와
  rounding, B=1 적용 범위가 한계다. Neural Loss에 같은 기법을 강제하지 않는다.
- Off-policy MC와 공유 가능한 것은 sampling-time log, 고정 확률표, log-space 비율·Coverage 검증이다.
  MC Prediction은 현재 행동부터 전체 Episode, 여기서는 현재 행동 이후부터 짧은 sampled 끝점이라는 범위 차이가 있다.
  Expected SARSA와는 Q 평가·정책 확률표·TD0·Q Rule을 공유할 수 있지만 Return/IS 범위는 다르다.
- 최소 후보는 ordered B=1 window, actual horizon/boundary, 끝점 action/log_prob 또는 명시적
  continuation을 받는 함수다. n-step Expected SARSA는 끝점 기대값과 중간 비율을 사용한다.
  Tree Backup은 각 중간 상태의 Q/π 분기 기대값, Q(σ)는 이들 및 시점별 σ·sampled log를 추가로 필요로 한다.
  단일 Bootstrap callback만으로 모두 해결된다고 보지 않으며 이번에는 공통 Framework를 구현하지 않는다.
- Control 확장은 현재 Q를 사용한 별도 Policy Improvement와 Target Policy를 평가하는 시점의 계약이 필요하다.
  과거 b는 항상 보존한다. 이번 고정 평가에 Control을 섞지 않는다.

06 Notebook §15는 같은 예제 함수를 호출하며 이전 59개 셀·출력·메타데이터 전체를 보존한다.

## Online n-step Tree Backup 컴포넌트 조합 검증

`examples/gridworld_tree_backup.py`는 기존 컴포넌트로 고정 정책의 Qπ를 평가한다.
`modular_rl/`의 61개 파일과 공개 API는 그대로다. 새로운 Learner, TargetGenerator,
UpdateRule, Policy/Base/Runner 클래스는 추가하지 않는다.

| 예제 함수 | 책임 |
|---|---|
| `learn_tree_backup(model, window, target_generator, update_rule, *, target_policy)` | contiguous B=1 ordered window의 head만 갱신; signed `G-current_Q` (1,), table dtype/device, graph 없음 |
| `run_episode(env, model, target_generator, update_rule, *, target_policy, n=3, seed=None)` | step → owned Transition snapshot → eligible head update → 다음 행동 선택; boundary flush |
| `train_and_evaluate(target_mode="different", n=3, episodes=200, ...)` | Uniform Behavior와 고정 π, 같은 π의 DP Qπ 참조·RMSE·Reward/Length/Step/Update 기록 |
| `run_comparison(seed=7, episodes=200, ns=(1,3,5), ...)` | 기존 On-policy 기준과 Off-policy SARSA matched/different, Tree matched/different 비교 |

기존 Off-policy 예제의 고정 정책 검증과 SARSA 실험 함수, n-step 예제의 Window/차원·경계 검증을
재사용한다. 설치하지 않는 examples 파일은 Notebook에서도 읽을 수 있도록 기존 방식인
`runpy.run_path`로 로드한다. 원래 예제 함수 및 Notebook inspect 출력은 수정하지 않는다.

### 같은 Snapshot에서의 재귀 계산

[Sutton & Barto §7.5, 식 (7.15–16), p.153](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf#page=175)의
Tree Backup은 실제 다음 행동의 분기를 재귀 Return으로 확장하고 다른 행동은 현재 Q로 평가한다.

```text
no_grad: 필요한 next states의 Model estimate를 한 번 계산
         Q rows와 고정 π rows 또는 Selector.probabilities(같은 estimate) 보관
         Vπ = sum(pi * Q)
         G = terminal ? 0 : Vπ(endpoint)
         reversed(window):
             last row: C = G
             earlier row: C = Vπ(next) + pi(actual_next_action) * (G - Q(next,actual_next_action))
             G = TD0.generate(reward, C, terminated)
         TabularQUpdateRule.update(table, head_state, head_action, G)
```

π=0은 sampled 분기만 끊고 Vπ를 그대로 사용한다. π=1은 G를 직접 전달하여
불필요한 상쇄·반올림을 피한다. 일반 π에서는 위 식을 직접 사용하며 IS 비율·Weight·Clipping·
Effective Target은 계산하지 않는다. Behavior log_prob은 Transition 계약대로 저장할 수 있지만
Tree Target에서 조회하거나 유효성을 요구하지 않는다. 미관측 상태-행동 학습을 보장하려면
Behavior의 충분한 방문은 여전히 필요하며 IS가 없다는 이유로 Coverage가 자동 확보되지는 않는다.

Model의 한 batched forward가 필요한 모든 다음 상태 Q rows의 owned lookup 결과를 반환한다.
같은 호출에서 table을 갱신하는 것은 마지막 Rule 호출뿐이며 동일 Snapshot의 Q/π로 재귀한다.
자연 종료의 마지막 next observation은 평가 목록에서 제외한다. 중간 실제 행동은 다음 window row의
action이므로 별도 endpoint action/log_prob이 필요하지 않다. invalid shape/action/policy는 쓰기 전에
거절하고 비유한 Target/Error/예상 갱신도 head write 전에 실패한다. 이전 성공 호출을 rollback하지 않는다.

고정 π는 caller-owned floating Tensor (S,A), 각 행의 유한 확률합 1, Q와 같은 device다.
선택 rows를 Q dtype으로 명시적으로 cast한다. Q 갱신으로 π를 개선하지 않는다.
동적 π는 기존 Selector 객체를 전달하며 `probabilities(estimate)`를 사용한다. 반환 확률은 Q와
shape/dtype/device가 같아야 한다. Model.forward나 행동 선택을 다시 호출하지 않으므로 같은
Snapshot 및 RNG 불변 계약을 유지한다. 이 경로는 동적 Target의 계산 지원이며 새로운 Control
실험이나 Policy Framework를 구현한 것이 아니다.

### Online 시점, 경계와 수치 검증

매 Step 이후 Transition을 pending에 추가하고 len==n이면 oldest row를 갱신한 뒤 pop한다.
종료가 아니면 그 **갱신 이후** 다음 Behavior 행동을 선택한다. 이는 n=1 Expected SARSA와
같은 타이밍이다. SARSA 예제의 next-action-before-update 수집 함수를 그대로 호출하지 않는다.
n=3, T=5의 갱신 시점은 Step 3/4/5/5/5이고 각 Transition을 한 번씩 처리한다.

terminated=True는 Bootstrap을 제거하고 두 flag가 True여도 이 규칙이 우선한다.
truncated-only는 actual final observation의 ΣπQ로 Bootstrap하며 마지막 행동을 샘플링하지 않는다.
flush에는 step/act가 없고 각 학습 호출은 앞선 갱신이 반영된 최신 Q를 다시 평가한다.
하나의 호출 내부 Snapshot 고정과 전체 flush의 Q 동결은 다르다.

| 검증 | Return/Target | signed Error | 갱신 후 Q (초기 2, alpha=.5) |
|---|---|---|---|
| n=1, Qnext=[10,2], π=[.25,.75], reward=1 | 4.6 | 2.6 | 3.3 |
| n=3, rewards=[1,2,3], 실제 A1=0/A2=1 | 역순 6.6 → 6.536 → 3.8206 | 1.8206 | 2.9103 |
| m=2 자연 종료 | head 2.8, tail 2 | .8 (head) | 2.4 |
| m=2 Time-limit | head 3.61, tail 5.6 | 1.61 (head) | 2.805 |

n=1 동적 Selector는 기존 Expected SARSA와 float32/64, 종료/Truncation/동시 flag에서
Action/Reward/Target/Error/Q/Step/RNG가 정확히 일치한다. π=0의 중간 분기에도 다른 행동의
기대값 항과 갱신은 남는다. n=1/3/5의 결정적 π를 실제 경로가 따르면 Sampled SARSA와
Return이 같지만 일반 확률적 π의 n>1에서는 일치하지 않는다.
반복 self-transition의 Q=[4,10], rewards=[1,2], π=.5, gamma=.9/alpha=.5는
첫 Snapshot [[4,10],[4,10]]에서 Target=9.235/Qnew=6.6175, 다음 flush에서
Target=9.477875/Qnew=8.0476875다. 매 호출 한 Snapshot과 최신 flush Q를 검증한다.

### 고정 정책 비교와 Checkpoint

Seeds7/11/19, n=1/3/5, 200 Episode, gamma=.9, alpha=.05, Behavior epsilon=1이다.
Matched π=.25, Different π=[.3,.2,.2,.3]을 고정한다. On-policy Uniform 기준은 기존 SARSA
함수를 재사용하고, Off-policy SARSA의 matched/different 및 Tree 두 정책을 각각 실행한다.
`policy_evaluation`과 `action_values`의 동일 π Qπ를 비종료 10개 상태·모든 행동에 대해 비교한다.
Reward/Length, Step/실제 Update, Q/π, RMSE 곡선과 Seed 평균·표준편차를 보존한다.
Tree에는 Weight/ESS를 계산하지 않는다. 고정 정책 평가를 최적 제어 성공으로 해석하지 않는다.

Uniform Behavior의 자연 종료 경험은 방법 간 Action/Reward/Length/Step이 같다.
단, SARSA는 Time-limit 마지막 행동을 추가로 선택하고 Tree는 선택하지 않아, Truncation 뒤 RNG·
경험까지 동일하다고 가정하지 않는다. 그 경우 각 방법의 독립적인 seed 재실행으로 재현성을 검증한다.
Tree의 단일 Seed 우위나 RMSE 단조 감소를 테스트 조건으로 두지 않는다.

실제 200 Episode 비교 결과(각 행의 SARSA는 Off-policy 버전, Matched는 On-policy 기준과 정확히 같다):

| Seed | n | Step=Update (각 run) | Matched SARSA / Tree RMSE | Different SARSA / Tree RMSE |
|---|---|---|---|---|
| 7 | 1 | 8,419 | .074577 / .097615 | .071736 / .088023 |
| 7 | 3 | 8,419 | .113803 / .071858 | .166660 / .061638 |
| 7 | 5 | 8,419 | .144084 / .070951 | .213516 / .060784 |
| 11 | 1 | 8,412 | .104793 / .101074 | .098586 / .092793 |
| 11 | 3 | 8,412 | .053557 / .075692 | .073633 / .066914 |
| 11 | 5 | 8,412 | .076829 / .074770 | .110648 / .066053 |
| 19 | 1 | 8,774 | .127772 / .124823 | .124573 / .119835 |
| 19 | 3 | 8,774 | .088591 / .097037 | .102298 / .090675 |
| 19 | 5 | 8,774 | .106447 / .096101 | .131470 / .089769 |

Different RMSE 평균±표준편차(SARSA / Tree)는 n=1에서 .098298±.021572 / .100217±.014008,
n=3에서 .114197±.038899 / .073076±.012630, n=5에서 .151878±.044406 / .072202±.012607이다.
Tree Matched는 n=1/3/5에서 .107838±.012094 / .081529±.011077 / .080607±.011066이다.
45개 run은 총 9,000 Episode, 384,075 Step/Update이며 모두 자연 종료했다. 동일 π별 DP 기준을 사용한다.
n=1 평균은 Tree가 약간 높고, n=3/5 평균은 낮지만 고정 alpha와 세 Seed의 유한 표본 관측이다.
Reward/Length는 Uniform Behavior가 같아서 일치하며 이 곡선을 Target Policy의 제어 성능으로 해석하지 않는다.

Checkpoint는 Model.state_dict, 고정 π 또는 동적 Selector의 구조/ε, owned pending batches,
현재 관측·이미 선택한 Behavior action/log_prob, n/gamma/alpha와 Torch CPU/CUDA RNG를 보존한다.
실제 환경 도중 재개는 환경 위치/RNG/TimeLimit 잔여 예산·누적 통계도 별도 보존해야 한다.
테스트는 고정 stream에서 pending/선택 행동을 복원하여 후속 모든 Error/Q 및 다음 행동이
일치하는지 확인한다. 범용 환경 Checkpoint 클래스/API는 추가하지 않는다.

### 구조적 평가와 Q(σ) 요구사항

- 기존 Model/Estimator는 그대로 사용하고 Selector.probabilities로 동적 π의 기대값·분기 확률을
  계산할 수 있었다. TD0는 외부 continuation을 받아 Tree의 모든 재귀 Bellman Backup에 재사용된다.
- Ordered B=1 window와 검증·comparator는 재사용한다. deque 수집/head/pop/flush 및 실험 집계는
  다른 예제와 겹치지만 행동 선택 타이밍이 달라 기존 수집 함수와 억지로 공유하지 않는다.
- Tree의 추가 입력은 중간 실제 행동과 같은 Snapshot의 Q/π rows다. 실행된 행동은 기존
  Transition/Batch로 충분하고 sampled endpoint나 Behavior probability는 필요하지 않다.
- 기존 Q/SARSA Learner는 Batch의 max/gather Bootstrap을 고정하며 recursive return/temporal window
  조합 경로를 제공하지 않는다. 계산 컴포넌트는 충분하여 예제 함수만으로 구현할 수 있었다.
  신규 Learner나 기존 Learner 리팩터링은 하지 않는다.
- 최소 공통화 후보는 ordered window, actual horizon/boundary, 같은 Snapshot의 Q/π 및 실제
  중간 행동을 받는 Return 조합 함수와, 행동 선택 시점을 명시한 학습 callback이다.
  endpoint bootstrap 교체만으로 Tree Backup의 모든 중간 분기를 표현할 수 없다.
- Q(σ)는 시점별 σ, sampled/expected 혼합, 중간 Q/π, 실제 행동 및 필요한 Off-policy 보정의
  선택·기록·평가 시점이 추가로 필요하다. Off-policy sampling 성분에는 과거 b의 기록이 필요할 수 있다.
  현재 Transition.log_prob을 활용할 수 있지만 σ의 소유권과 mixing/ratio 범위는 먼저 설계해야 한다.
  Q(σ), 공통 Framework, Target Policy Control은 이번에 구현하지 않는다.

06 Notebook §16은 예제 함수로 같은 실험을 실행하고 이전 69개 셀·출력·메타데이터를 그대로 보존한다.
전체 pytest는 1,693 passed, 53 skipped(CUDA 미지원)이며 신규 테스트는 92 passed/8 skipped다.
기존 예제 14개의 Smoke Test와 세 Notebook의 같은 Kernel Run All 두 번을 통과했다.
기존 모든 출력 payload도 이전 commit과 일치하며 새 그래프 두 개를 포함한 PNG는 총 32개다.
현재 06은 87개 셀/코드 47개/PNG 17개이고 라이브러리 61개 파일의 byte hash는 변경되지 않았다.

## DQNLearner와 CartPole 검증

`learner/dqn.py`의 DQNLearner는 외부 Online Model, Target Model, Optimizer,
TD0TargetGenerator, TDLossUpdateRule을 조합하는 일반 Python 객체다. Buffer와
ActionSelector는 학습에서 사용하지 않는다. 수집·샘플링·평가는 외부 호출자의 책임이다.

| API / 입력 | 계약 |
|---|---|
| `DQNLearner(online_model, target_model, optimizer, target_generator, update_rule)` | 독립된 동일 구조 Model, Online 전용 Optimizer, 기존 Target/UpdateRule |
| `learn(batch: TransitionBatch) -> TDLossOutput` | 한 번의 backward/step, 반환값은 detach된 업데이트 직전 통계 |
| `optimize(prediction: Tensor, target: Tensor) -> TDLossOutput` | 외부 Online prediction과 고정 target으로 같은 Loss/Optimizer 절차 실행 |
| `sync_target() -> None` | 명시적인 hard copy; 초기 복사와 이후 주기는 호출자가 결정 |
| observation / next_observation | 동일 shape `(B, *observation_shape)`; Model의 Representation 입력 계약 준수 |
| action / reward / terminated / truncated | `(B,)`; 정수 index / floating / bool / bool |
| Online Q / Target Q | 하나의 floating Scalar Discrete Tensor `(B, A)`; action 수 일치 |
| prediction / target / 반환 td_error | `(B,)`; Online Model floating dtype·device |
| 반환 loss | scalar `()`; graph 없음 |

단일 경험도 `B=1`로 전달한다. 추가 Time 축 `(T, B, ...)`은 이 Learner의 Replay
입력이 아니다. Observation feature 수는 Model이 결정한다. Batch.to는 floating 필드만
변환하므로 정수로 저장된 연속 관측에는 명시적인 InputAdapter dtype 설정을 사용한다.
floating/bool action index를 묵시적으로 반올림하지 않는다. 빈 Batch, 잘못된 행 shape,
행동 범위, 누락된 Q·복수 Critic·Distributional Q는 명확하게 실패한다.

```text
호출자: Model.act -> Environment.step -> Transition -> ReplayBuffer.add/sample
Learner: Batch를 Online device/dtype으로 준비, action long 변환
         -> Online Model.forward(observation) -> gather Q(s,a)
         -> no_grad: Target.eval -> forward(next_observation) -> max Q(s',·)
         -> no_grad: TD0TargetGenerator.generate(reward, next_value, terminated)
         -> optimize(prediction, target): TDLossUpdateRule.compute
         -> optimizer.zero_grad -> loss.backward -> optimizer.step
         -> TDLossOutput(td_error.detach(), loss.detach())
호출자: 지정 시점에 learner.sync_target()
```

TD Error는 `target - prediction`이며 반환 행 순서는 Batch와 같다. 기본 Loss는 기존
UpdateRule의 MSE지만 SmoothL1Loss(reduction="none") 등도 그대로 주입할 수 있다.
TargetGenerator와 TDLossUpdateRule 자체의 gradient 보존 계약은 바꾸지 않는다.
DQNLearner가 no_grad로 Target 평가·계산을 감싸고 반환 통계만 detach한다. Online의
Representation과 Estimator에는 gradient가 전달되고 Target에는 전달되지 않는다.
truncated=True, terminated=False인 Time-limit에서도 Bootstrap한다. 실제 final observation과
다음 에피소드 reset 구분은 호출자가 관리하며, Learner는 이를 추론하지 않는다.

### 공유 최적화 경계와 Semi-gradient

`DQNLearner.optimize(prediction, target)`는 기본 DQN과 외부 n-step Target 조합이 같은
Loss/Optimizer/반환 절차를 사용하도록 한다. 두 Tensor는 비어 있지 않은 `(B,)`이며 동일한
floating dtype·device를 사용한다. Prediction은 Online Model의 dtype/device와 gradient 경로를
유지해야 한다. 호출자가 해당 Online Model에서 만들었는지 확인하는 graph 탐색은 하지 않는다.

Target은 호출자가 no_grad에서 계산한 고정 Tensor다. `target.requires_grad=True`는 Optimizer 호출
전에 명확하게 거절하며 조용히 detach하지 않는다. Shape/dtype/device 일치 등 공통 검증은 기존
TDLossUpdateRule이 수행한다. 모든 검증은 zero_grad/backward/step 전에 끝난다.
TargetGenerator와 TDLossUpdateRule의 gradient 보존 계약은 유지한다. Target 미분을 연구하는
호출자는 기존 Rule과 명시적인 별도 Optimizer 절차를 사용할 수 있다.

optimize는 Model Forward, Target Forward/생성/Sync, Replay sampling, 환경 상호작용과 ε 변경을
수행하지 않는다. `compute -> zero_grad -> backward -> step` 뒤 signed TD error와 scalar loss를
detach하여 반환한다. 결과는 업데이트 직전 값이며 Prediction을 미리 detach하지 않는다.

기존 learn(batch)는 Online Forward/gather, no_grad Target eval/Forward, TD0 Target 계산을
현재 순서대로 수행한 뒤 optimize에 위임한다. 생성자·learn·sync_target의 시그니처와 반환 계약은
그대로다. Optimizer는 같은 인스턴스를 계속 사용하고 Target 복사는 명시적 sync에서만 수행한다.

### Target Model과 Optimizer 소유권

`deepcopy(online_model)` 또는 같은 구조를 별도로 구성해 Target을 전달한다. 생성 시
모듈 경로·종류, Parameter/Buffer 이름·shape·device·dtype 및 저장 공간 독립성을 확인한다.
동일 Parameter 객체뿐 아니라 다른 객체가 같은 Tensor storage를 참조하는 경우도 거절한다.
Target의 기존 grad를 비우고 requires_grad=False, eval로 설정하며 각 learn에서도 eval을 유지한다.
BatchNorm 등 평가 시 고정되어야 하는 Buffer가 학습 중 바뀌지 않도록 한다.
Online train/eval은 호출자가 관리하고 Learner가 임의로 바꾸지 않는다.

생성자는 자동 동기화하지 않는다. `sync_target()`은 호환성을 먼저 확인한 후 strict
state_dict 복사와 Buffer 복사를 수행하며 nonpersistent Buffer도 포함한다. Target의
저장 공간은 유지되고 Online과 alias되지 않는다. 모든 Model 상태를 복사하므로
ActionSelector도 있다면 양쪽에 호환되는 구성을 사용한다. 초기 복사는 학습 시작 전에
명시적으로 호출한다. 이후 환경 step 기준 주기와 시점은 예제/향후 Runner가 결정한다.

Optimizer에는 Online Parameter만 넣는다. 두 Model의 device/dtype 이동을 마친 뒤
Optimizer를 구성해야 Parameter와 Optimizer state가 맞는다. 기본 구현은 한 floating
dtype·device를 사용하며 AMP, Gradient Clipping/Accumulation은 포함하지 않는다.
Optimizer state는 Model.state_dict에 포함되지 않는다. 향후 Learner checkpoint는 Online
state_dict, Target state_dict와 optimizer.state_dict를 따로 저장·복원해야 하며,
외부 Loop의 step·epsilon schedule·동기화 시점, Replay/RNG 상태도 별도로 결정해야 한다.
이번 작업에서는 checkpoint wrapper나 새 공개 메서드를 추가하지 않는다.

```python
from copy import deepcopy
import torch
from modular_rl.component.estimator import DiscreteQEstimator
from modular_rl.component.representation import MLPRepresentation
from modular_rl.component.target_generator import TD0TargetGenerator
from modular_rl.component.update_rule import TDLossUpdateRule
from modular_rl.data import TransitionBatch
from modular_rl.learner import DQNLearner
from modular_rl.model import Model

representation = MLPRepresentation(4, hidden_dims=(128, 128))
online = Model(representation, DiscreteQEstimator(128, 2))  # Selector is optional.
target = deepcopy(online)
optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
learner = DQNLearner(online, target, optimizer, TD0TargetGenerator(.99), TDLossUpdateRule())
learner.sync_target()
batch = TransitionBatch(torch.zeros(2, 4), torch.tensor([0, 1]), torch.ones(2),
                        torch.ones(2, 4), torch.tensor([False, True]), torch.tensor([True, False]))
result = learner.learn(batch)
assert result.td_error.shape == (2,) and result.loss.shape == ()
assert not result.td_error.requires_grad and not result.loss.requires_grad
```

### CartPole 예제와 향후 Double DQN

```bash
python -m pip install -e ".[examples]"
python -m examples.cartpole_dqn
python -m examples.cartpole_dqn --episodes 4 --eval-episodes 2 --max-steps 5 --batch-size 2 --warmup-steps 2
```

`examples/cartpole_dqn.py`는 일반 Gymnasium CartPole-v1 환경을 사용하며 렌더링과
Autoreset을 사용하지 않는다. 종료 조건과 기본 보상은
[Gymnasium CartPole 공식 문서](https://gymnasium.farama.org/environments/classic_control/cart_pole/)를 따른다.
명시적인 환경 루프에서 탐색·Warm-up·Replay Sampling·Target 동기화를 처리한다.
기본 구성은 MLP (128, 128), capacity=10,000, batch_size=64, Adam lr=1e-4,
gamma=.99와 SmoothL1Loss(reduction="none")다. 1,000 전이를 모은 뒤 환경 step마다
한 번 학습한다. 250 환경 step마다 hard sync하고 epsilon은 10,000 step에
걸쳐 1.0에서 .05로 감소한다. 기본 학습은 400 episode, 평가는 20 episode, seed=7이다.
Network와 하이퍼파라미터는 예제 함수와 CLI에 직접 기록한다. PyTorch RNG, Replay 전용
RNG와 episode reset seed를 설정한다. 학습 전후 동일한 별도 평가 seed로 Greedy Policy를
평가하며 평가 중에는 학습하거나 Replay에 저장하지 않는다. 환경과 Torch thread 설정은
finally에서 정리한다. reward, epsilon, loss, signed/absolute TD Error 통계와 평가 평균 보상을 출력한다.

검증 실행(Gymnasium 1.4.0, CPU, seed=7)에서 Greedy 평균 보상은 학습 전 9.40에서
학습 후 180.95로 증가했다. 학습은 43,998 환경 step, 42,999 Optimizer step,
175번의 주기적 Target 복사(초기 복사 제외)였고 최종 Loss=.006164,
평균 절대 TD Error=.074323이었다. 같은 설정의 seed=42에서는 14.10 -> 73.20이었다.
학습률 5e-4/1e-3 등 초기 구성에서 보상 정체와 변동을 관찰해 기본값을 1e-4로 정했다.
이는 실제 학습 결과이며 500 보상으로의 수렴이나 여러 Seed에서 같은 성능을 보장하지 않는다.

전체 학습 보상은 실험 결과로 보고하고 매 pytest의 성공 임계값으로 사용하지 않는다.
결정적 테스트는 Bellman/gather/종료 mask, Online gradient·Optimizer 호출, Target 불변성,
모든 Buffer 동기화, 반환 detach와 Model/Optimizer state 복원을 확인한다. 짧은 실제 환경
Smoke Test는 Warm-up·동기화 주기·실제 final observation·seed 재현성·환경 종료를 확인한다.
기본 설치에도 Gymnasium이 포함되며, CUDA가 없으면 CUDA 검사만 skip한다.

Double DQN의 변경 지점은 no_grad Bootstrap 선택 경로다. Online next Q의 argmax 행동을
Target next Q에서 gather해 next_value로 전달하면 된다. 기존 TargetGenerator,
TDLossUpdateRule, 반환 계약과 Optimizer 절차는 재사용할 수 있다. 이번 구현에 조건문이나
Double DQN 옵션을 미리 추가하지 않는다.

## Tabular Dynamic Programming Phase 1

DP는 알려진 유한 MDP의 모든 전이 결과를 합산해 기대값을 계산한다.
Sutton & Barto의 [Reinforcement Learning: An Introduction, Chapter 4](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf)를
참고하며 Policy Evaluation, Policy Improvement/Iteration, Value Iteration을 구현한다.
경험 샘플을 업데이트하는 기존 Learner 경로와 달리 Buffer, Optimizer 또는 Network가 필요하지 않다.
DP의 알려진 환경 모델은 관측으로 Policy/Value를 추정하는 `model.Model`과 별개의 데이터다.
학습된 WorldModel과 Dyna Planner는 이번 범위에 포함하지 않는다.

### 알려진 Tabular MDP와 Policy

`data/mdp.py`의 두 frozen dataclass가 kernel을 표현한다.

| API / 데이터 | 계약 |
|---|---|
| `TransitionOutcome(probability, next_state, reward, terminated=False)` | P(s', r, terminated \| s, a)의 한 확률 사건 |
| `TabularMDP(n_states, n_actions, transitions, terminal_states=())` | 0부터 시작하는 상태·행동 indices, 모든 상태에 공통 행동 수 |
| `mdp.transitions[s][a]` | 하나 이상의 TransitionOutcome tuple; 결과별 확률합 1 |
| `mdp.terminal_states` | 선택적인 terminal state index tuple |
| Policy `policy[s, a]` | pi(a\|s); shape `(S, A)`, 비음수·유한, 각 행의 합 1 |
| V / Q | shape `(S,)` / `(S, A)` |

입력 nested list는 생성 시 tuple로 복사한다. 확률·보상은 유한 real scalar,
next_state는 유효한 정수 index, terminated는 bool이어야 한다. 각 (s,a) 확률합은
absolute tolerance 1e-12로 검사하며 임의로 정규화하지 않는다. 같은 다음 상태를 갖더라도
reward나 terminated가 다른 결과를 합쳐 버리지 않는다. Gymnasium의 내부 P dict 형식에 의존하지 않는다.
모든 상태·행동 행을 제공하며 terminal 행도 정상 확률을 갖는 placeholder를 제공한다.

DP 알고리즘은 CPU NumPy float64 배열을 사용하며 입력 V·Policy를 복사한다. Policy 입력도
float64로 준비하고 동일한 1e-12 확률합 검사를 통과해야 한다. 자동 확률 정규화와 Device 관리는
포함하지 않는다. 결과 dataclass는 frozen이지만 배열 원소는 수정할 수 있다. 반환 배열 수정이
입력 Policy/V나 kernel을 변경하지 않는다. 기존 Tensor 기반 Model/Estimator의 API는 유지하며
DP 값을 옮기는 경우 호출자가 `torch.from_numpy(result.values)` 등을 명시적으로 사용한다.

Terminal outcome의 보상은 포함하고 Bootstrap만 중단한다. 선언된 terminal state로 들어가는
경우도 Bootstrap하지 않는다. 현재 상태가 terminal이면 V와 Q는 0이고 이후 보상을 누적하지 않는다.
terminated=True인 결과가 있다는 이유로 그 출발 상태나 다음 상태 index 전체를 terminal로 추론하지 않는다.
따라서 같은 next_state에 종료·비종료 결과가 공존하는 joint distribution도 표현할 수 있다.
Time-limit truncation은 알려진 kernel의 자연 종료와 별개이며 Phase 1은 sampling loop를 구현하지 않는다.

### Bellman Backup과 알고리즘 API

공유하는 계산은 `action_values()`의 한 단계 기대 행동가치와 균등 동률 Greedy Policy다.

```text
Q_V(s,a) = sum_outcomes p(o|s,a) * [r(o) + gamma * bootstrap(o)]
bootstrap(o) = 0                       if terminated or next_state is terminal
               V(next_state)           otherwise
Policy Evaluation: V_new(s) = sum_a pi(a|s) * Q_V_old(s,a)
Value Iteration:   V_new(s) = max_a Q_V_old(s,a)
Policy Improvement: pi_new(a|s) = I(a in argmax Q_V(s,.)) / number_of_maxima
```

| 공개 함수 | 역할 |
|---|---|
| `uniform_policy(mdp)` | 균등 행동 확률 `(S, A)` 생성 |
| `action_values(mdp, values, gamma=1.0)` | 모든 joint outcome의 full expectation으로 Q `(S, A)` 계산 |
| `policy_evaluation(mdp, policy, gamma=1.0, theta=1e-8, max_iterations=10000, *, initial_values=None)` | 고정 Policy의 V를 동기식으로 반복 평가 |
| `policy_improvement(mdp, values, gamma=1.0)` | 정확히 동률인 최대 Q 행동에 균등 확률을 배분 |
| `policy_iteration(mdp, gamma=1.0, theta=1e-8, max_iterations=100, *, max_evaluation_iterations=10000, initial_policy=None)` | 초기 균등 Policy 또는 외부 Policy의 평가·개선을 반복 |
| `value_iteration(mdp, gamma=1.0, theta=1e-8, max_iterations=10000, *, initial_values=None)` | Bellman Optimality Backup 반복 후 최종 V에서 Greedy Policy 추출 |

기본 방식은 동기식이다. 한 sweep에서는 이전 전체 V로 Q를 만든 뒤 V 전체를 교체한다.
제자리 방식은 동일 sweep에서 앞서 갱신한 상태의 V를 뒤 상태가 사용할 수 있어 중간 값과
반복 횟수가 달라진다. Phase 1에는 동기식 한 방식만 구현하며 별도 mode나 공통 Base Class는 두지 않는다.

Policy Iteration은 Policy Evaluation의 계산 루프를 재사용하고 이전 V로 평가를 warm-start한다.
개선 Policy의 상태별 확률 행렬이 현재 행렬과 같으면 stable로 판단한다. 정확한 최대 Q 동률에는
언제나 같은 균등 확률을 만들어 Random tie-breaking으로 정책이 바뀌지 않는다. 거의 같은 Q를
임의의 tolerance로 동률 처리하지 않는다. Terminal의 행동 확률은 placeholder이며 개선 시 균등이다.

### 수렴 결과와 실패 계약

평가·제어 함수는 같은 `DPResult`를 반환한다.

| 필드 | 의미 |
|---|---|
| `values` / `policy` | owned float64 배열 `(S,)` / `(S, A)` |
| `iterations` | 평가·Value Iteration은 sweep 수, Policy Iteration은 평가·개선 cycle 수 |
| `evaluation_iterations` | Policy Evaluation에서 수행한 총 sweep 수; Value Iteration은 0 |
| `delta` | 마지막 value sweep의 max abs(V_new - V_old); Policy Iteration은 마지막 평가 sweep |
| `converged` | 평가·Value Iteration은 delta < theta, Policy Iteration은 평가 수렴과 Policy 안정성 |

gamma는 [0,1], theta는 양의 유한 real, 두 반복 한도는 양의 정수다. 반복 한도 도달은
`converged=False`와 마지막 계산값으로 보고한다. Policy Iteration은 평가가 수렴하지 않으면
개선을 진행하지 않는다. 외부 cycle 한도에서도 아직 평가하지 않은 개선 Policy를 반환하지 않고,
마지막으로 평가한 Policy와 그 V를 함께 반환한다. 비수렴 Value Iteration의 Policy는 마지막
근사 V에 대한 Greedy Policy이며 최적이라고 주장하지 않는다. 수치 범위를 넘겨 Backup이
비유한 값이 되면 FloatingPointError로 실패한다.

gamma=1인 Episodic MDP를 지원하지만 Proper Policy를 자동 탐지하지 않는다. 종료되지 않는
reward=1 self-loop는 반복마다 V가 커져 한도에 도달한다. 반대로 delta 수렴 자체가 모든
정책의 종료를 증명하는 것은 아니다. delta는 update 변화량으로, 참조 V와의 절대 오차가 아니다.

### Sutton Gridworld 예제와 MC/TD 재사용

`environment/gridworld.py`의 `sutton_gridworld()`는 16 states, terminals=(0,15),
actions=(up,down,left,right)의 알려진 kernel을 반환한다. 경계 밖 이동은 제자리이며 모든
nonterminal 행동의 reward는 -1이고 terminal에 들어가는 이동도 -1이다. Terminal 행의
zero-reward self-loop는 추가 환경 step이 아닌 placeholder다. 교재의 두 terminal 구석을
요청에 맞춰 서로 다른 0/15 index로 표현한다.

```python
import numpy as np
from modular_rl.dp import policy_evaluation, policy_iteration, uniform_policy, value_iteration
from modular_rl.environment import sutton_gridworld

mdp = sutton_gridworld()
policy = uniform_policy(mdp)
evaluation = policy_evaluation(mdp, policy, gamma=1, theta=1e-10)
optimal_pi = policy_iteration(mdp, gamma=1, theta=1e-10)
optimal_vi = value_iteration(mdp, gamma=1, theta=1e-10)
assert evaluation.converged and optimal_pi.converged and optimal_vi.converged
np.testing.assert_array_equal(optimal_pi.values, optimal_vi.values)
print(evaluation.values.reshape(4, 4))
```

```bash
python -m examples.gridworld_dp
```

예제는 Uniform Random Policy의 V, 두 최적 V, 선택 가능한 모든 최적 행동, 수렴 여부·반복 수와
Sutton 참조값 최대 오차를 출력한다. theta=1e-10, gamma=1에서 평가 426 sweep,
참조 오차 1.735e-9였다. Policy Iteration은 3 cycle·431 평가 sweep,
Value Iteration은 4 sweep이며 두 최적 V의 참조 오차와 상호 차이는 모두 0이었다.
테스트는 최적 Policy 배열 하나를 정답으로 고정하지 않고 선택 행동이 최대 Q를 갖는지 확인한다.

MC/TD Prediction은 같은 `mdp.transitions[s][a]`와 `policy[s]`로 실제 전이를
샘플링하되 수집 루프·RNG·에피소드 종료를 Learner와 분리한다. DP의 `policy_evaluation`을
고정 Policy의 V_pi 참조값으로 사용하고 같은 kernel, reward, gamma, 초기 상태 분포와
terminal 규칙을 맞춰 오차를 비교한다. TD Prediction에는 기존 TabularVEstimator,
TD0TargetGenerator, TabularVUpdateRule을 재사용할 수 있다. 현재 Q-learning Learner의
max Q Bootstrap은 고정 Policy의 V_pi 평가와 목적이 다르므로 이를 대신하지 않는다.
Phase 1의 DP는 sampling과 독립적이다. Phase 2의 Episode Sampler/MC Learner와
Phase 3의 TD Prediction Learner는 아래에 기술하며 Dyna용 인터페이스는 아직 구현하지 않는다.

## Gymnasium 실행 환경

`TabularMDP`는 알려진 유한 MDP의 수학적 데이터, `TabularEnv(gymnasium.Env)`는 그 데이터로
실제 전이를 샘플링하는 실행 환경이다. DP는 모든 Outcome의 확률 가중합을 계산하며
환경의 step을 반복하지 않는다. MC/TD/Q-learning은 Gymnasium API에서 관측한 경험만
Learner에 전달한다. FrozenLake·CartPole도 같은 reset/step 계약을 사용한다.
추론용 `model.Model`과 알려진 MDP를 혼동하지 않으며 기존 Model/Estimator/Learner 계약은 유지한다.

| 공개 API | 책임 |
|---|---|
| `TabularEnv(mdp, *, start_state=None, render_mode=None)` | Discrete 관측·행동 공간, MDP 확률에 따른 Outcome 샘플링 |
| `reset(*, seed=None, options=None) -> (int, dict)` | 시작 상태와 환경 RNG 초기화, options의 start_state는 이번 reset만 적용 |
| `step(action) -> (int, float, bool, bool, dict)` | next state·reward·자연 종료 반환; 기본 Env의 truncated는 False |
| `render()` / `close()` | ansi 상태 ID 문자열 / human 출력, 멱등 close; reset으로 재사용 가능 |
| `make_sutton_gridworld_env(*, render_mode=None, max_episode_steps=10000)` | 비종료 상태 균등 시작 + 공식 TimeLimit |
| `make_dlfs4_gridworld_env(*, render_mode=None, max_episode_steps=2000)` | 교재 고정 시작 ID 7 + 공식 TimeLimit |
| `collect_episode(env, select_action, *, seed=None, options=None) -> Episode` | 외부 행동 callback으로 시간순 scalar State/Action 경험 수집 |

두 생성 함수는 단순한 환경 구성 함수이며 별도 Factory/Registry 계층이 아니다.
max_episode_steps=None이면 기본 Env를 반환한다. `env.unwrapped.mdp`로 알려진 kernel을
명시적으로 참조할 수 있다. 기존 sutton_gridworld()/dlfs4_gridworld()는 계속 TabularMDP를 반환한다.

```python
from modular_rl.data import Transition
from modular_rl.environment import make_dlfs4_gridworld_env

env = make_dlfs4_gridworld_env(max_episode_steps=200)
try:
    observation, info = env.reset(seed=42, options={"start_state": 7})
    while True:
        action = int(model.act(observation).action.item())
        next_observation, reward, terminated, truncated, info = env.step(action)
        transition = Transition(observation, action, reward, next_observation, terminated, truncated)
        learner.learn(transition.to_batch())
        observation = next_observation
        if terminated or truncated:
            break
finally:
    env.close()
```

환경은 `super().reset(seed=seed)`와 `self.np_random`으로 시작 상태 및 joint Outcome만 선택한다.
첫 reset에 seed를 전달하고 이후 None으로 같은 stream을 이어간다. 행동 선택은 외부 NumPy
Generator나 Model의 PyTorch RNG를 사용한다. action_space.sample()을 쓰면 별도로
action_space.seed(...)를 설정한다. 환경이 정책·Model·Learner·추가 RNG를 소유하지 않으며
global NumPy RNG를 변경하지 않는다. 실험 재개에는 환경/행동 RNG 상태도 함께 필요하다.

자연 종료는 Outcome의 terminated 또는 선언된 terminal 상태 진입이다. 알려진 terminal에서
reset 시작을 거절하며 step 전 reset·자연 종료 후 reset을 요구한다(ResetNeeded).
기본 Env는 자동 reset이나 자체 step cap이 없다. 외부 TimeLimit의 truncation은 기본 Env가
알 수 없으므로 호출자는 어느 flag든 True이면 reset해야 한다. 제한에 도달한 자연 종료에서는
공식 TimeLimit이 두 flag를 모두 True로 반환할 수 있다.
최종 관측은 reset 첫 관측으로 바꾸지 않는다. TD/Q는 terminated에서만 Bootstrap을 중단하고
MC는 truncated-only Episode를 거절한다. info 전체를 경험에 저장하지 않는다.

collect_episode는 이산 scalar 관측·행동을 사용하는 TabularEnv/FrozenLake 같은 환경에 한정한다.
환경의 reset/step만 호출하며 정책·학습·추가 RNG·중복 길이 제한을 포함하지 않는다.
반환 Episode가 기존 owned snapshot 계약을 적용한다. 범용 Trajectory/Runner는 추가하지 않는다.
기존 int callback은 log_prob=None을 유지한다. 추가된 ActionOutput callback은 scalar integer
action Tensor를 환경 경계에서 Python action으로 바꾸고, log_prob을 선택 직후·step 전에
detach/CPU/clone한다. Callback의 Tensor 재사용이나 수집 후 Q/epsilon 변경이 과거 확률을
바꾸지 않는다. `collect_episode(env, behavior_model.act)`로 기존 Model 출력을 바로 기록한다.
deterministic=True의 log_prob=None은 평가 경험으로 수집할 수 있지만 Off-policy IS에는 충분하지 않다.

Gymnasium은 필수 의존성(`gymnasium>=1.0`)이며 `examples=[]`는 이전 설치 명령과의 호환성만
남긴다. Matplotlib/Jupyter 등은 선택적 notebooks extra다. 공식
[환경 API](https://gymnasium.farama.org/api/env/)와
[TimeLimit](https://gymnasium.farama.org/api/wrappers/misc_wrappers/) 의미를 따른다.
검증은 unwrapped TabularEnv의 공식 check_env와 wrapper의 별도 boundary 테스트를 구분한다.
Sutton/DLFS4의 모든 비종료 전이, 시작·행동·Outcome 분포와 재실행 결정성을 검사한다.
기존 Sampler와 RNG 사용 순서가 다르므로 동일 seed의 Episode 자체가 같다고 가정하지 않는다.

## Tabular Monte Carlo Prediction Phase 2

Sutton & Barto의 [Chapter 5, Monte Carlo Prediction](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf)을
참고한다. DP는 알려진 확률 모델의 모든 결과를 합산하고, MC는 고정 정책에서 수집한 실제
완료 Episode의 Return을 평균한다. 환경에서 경험을 수집하는 코드와 그 Episode만 보는
Learner를 분리한다. 이 Prediction API는 V 추정만 담당하며 On-policy Q Control은 아래 별도
Learner로 구현한다. Off-policy IS Prediction도 아래 별도 Learner로 기술한다. Function Approximation은 아직 포함하지 않는다.

### Episode와 Replay Batch

`data/episode.py`의 `Episode(transitions)`는 기존 Transition의 비어 있지 않은 순서열이다.
observation, action, reward, next_observation, terminated, truncated, optional log_prob를
그대로 보존하되 모든 present field를 owned CPU Tensor로 복사하고 graph를 제거한다.
NumPy/Tensor dtype을 유지하므로 환경 배열 재사용이나 원본 Tensor 변경이 저장값을 바꾸지 않는다.
Python numeric scalar는 torch.as_tensor의 기본 dtype 규칙을 사용한다. 필드는 frozen이지만
Tensor 원소까지 immutable인 것은 아니므로 직접 수정할 때 계약을 지키는 것은 호출자 책임이다.

Transition 간 next_observation과 다음 observation의 연속성을 검사한다. observation/next_observation은
같은 shape·dtype이고 reward는 finite real scalar, 종료 flag는 bool scalar다. 중간에 종료나
truncation이 있으면 뒤 전이를 붙일 수 없다. `len(episode)`, `episode.transitions`,
`episode.terminated`, `episode.truncated`로 데이터와 최종 경계를 읽는다.
열린 마지막 전이나 truncated-only 조각도 데이터로 표현할 수 있으나 MC 학습 입력으로는 거절한다.
Episode는 `(T,B,...)` 범용 Trajectory가 아니며 ReplayBuffer의 무작위 `(B,...)`와 역할이 다르다.

### Return Generator와 방문 규칙

`MonteCarloReturnGenerator(gamma).generate(rewards: Tensor) -> Tensor`는 finite floating
rewards `(T,)`에 대해 같은 dtype/device의 Return `(T,)`를 반환한다. gamma는 [0,1]이며
빈 reward, 추가 batch 축, integer dtype과 비유한 입력을 거절한다.

```text
G = 0
for t = T-1, ..., 0:
    G = rewards[t] + gamma * G
    returns[t] = G
```

Bootstrap 값은 없으며 rewards만으로 Episode 완료 여부를 판단할 수 없으므로 그 책임은
호출자에게 있다. Generator는 gradient를 유지하고 Learner가 no_grad를 적용한다.
Return 수치가 overflow하면 table/count 갱신 전에 FloatingPointError로 실패한다.
기존 TD0TargetGenerator와 BaseTargetGenerator의 bootstrap 인터페이스는 변경하거나 상속하지 않는다.

First-visit는 전체 Return을 계산한 후 원래 시간 순서로 상태를 확인해 첫 방문만 사용한다.
역순 Return 계산 중 처음 발견한 상태를 첫 방문으로 취급하지 않는다. Every-visit는 각 시점의
Return을 모두 평균에 포함한다. 방문 선택 이후의 계산은 다음 세 줄이다.

```text
count[state] += 1
value[state] += (return_at_time - value[state]) / count[state]
```

고정 alpha TabularVUpdateRule과 달리 학습된 표본 수에 따른 Sample Average다.
수식이 짧으므로 별도 Rule/Base Class를 추가하지 않고 MC Learner에서 직접 계산한다.
초기 V를 pseudo sample로 세지 않는다. 동일 Episode에서 반복 방문한 상태도 Every-visit이면
시간 순서대로 Count와 평균을 갱신한다. 상태 [1,2,1], reward [1,2,3], gamma=1의 Return은
[6,5,3]이며 상태 1은 First-visit=6(count=1), Every-visit=4.5(count=2)다.

### Learner API와 저장 상태

| API / 데이터 | 계약 |
|---|---|
| `MCPredictionLearner(model, return_generator, visit_mode="first")` | TabularVEstimator Model, 독립 MC Return Generator, mode="first"/"every" |
| `learn(episode: Episode) -> Tensor` | V-table in-place sample-average 갱신, 전체 시간 순서 Returns `(T,)` 반환 |
| 입력 observation | scalar integer table index, table device의 long으로 정규화 |
| 준비된 reward / 반환 Return | V-table의 floating dtype·device, 반환 graph 없음 |
| `visit_counts` | Learner 소유 long Tensor `(N,)`, learn 시 현재 table device로 이동 |

Representation은 Model 추론에서 State Index를 보존해야 하며 기본은 IdentityRepresentation이다.
ActionSelector는 선택적이고 학습에 사용하지 않는다. Learner는 정책·MDP·ReplayBuffer를 받거나
Model에서 Bootstrap을 계산하지 않는다. 최종 next_observation은 마지막 결과일 뿐 추가 방문으로
세거나 갱신하지 않는다. Terminal placeholder V는 수집된 현재 상태로 나오지 않으므로 그대로 남는다.

V-table은 기존 Model.state_dict에 저장되고 Count는 포함되지 않는다. 재개하려면 아래처럼 둘을
함께 저장·복원하며 Model dtype/device와 gamma/visit_mode를 유지한다. Count만 0으로 초기화한
채 학습된 V만 복원하면 첫 새 Return이 이전 평균을 덮어쓰므로 같은 학습의 재개가 아니다.

```python
import io
import torch
from modular_rl.component.estimator import TabularVEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import MonteCarloReturnGenerator
from modular_rl.data import Episode, Transition
from modular_rl.learner import MCPredictionLearner
from modular_rl.model import Model

model = Model(IdentityRepresentation(), TabularVEstimator(3)).double()
learner = MCPredictionLearner(model, MonteCarloReturnGenerator(1), visit_mode="every")
episode = Episode((Transition(1, 0, 1., 2, False, False),
                   Transition(2, 0, 2., 1, False, False), Transition(1, 0, 3., 0, True, False)))
torch.testing.assert_close(learner.learn(episode), torch.tensor([6., 5., 3.], dtype=torch.float64))
assert model.estimator.v_table[1].item() == 4.5
checkpoint = io.BytesIO()
torch.save({"model": model.state_dict(), "visit_counts": learner.visit_counts.clone(),
            "gamma": learner.return_generator.gamma, "visit_mode": learner.visit_mode}, checkpoint)
checkpoint.seek(0)
saved = torch.load(checkpoint, weights_only=True)
restored_model = Model(IdentityRepresentation(), TabularVEstimator(3)).double()
restored_model.load_state_dict(saved["model"], strict=True)
restored = MCPredictionLearner(restored_model, MonteCarloReturnGenerator(saved["gamma"]), saved["visit_mode"])
restored.visit_counts.copy_(saved["visit_counts"])
torch.testing.assert_close(restored.visit_counts, learner.visit_counts)
```

고정된 정책도 같은 것으로 복원해야 한다. 완전한 실험 재현에는 현재 Episode 번호,
`env.unwrapped.np_random.bit_generator.state`와 외부 행동 RNG를 별도로 저장한다.
호환 Sampler 경로에서는 기존 시작 RNG와 `sampler.rng.bit_generator.state`가 해당한다. Policy 또는 gamma를 바꾸면
이전 Return 평균의 추정 대상이 달라지므로 기존 학습과 섞지 않는다. Checkpoint Framework는 추가하지 않는다.

### 외부 Episode Sampling과 Truncation 정책

현재 예제·Notebook은 `collect_episode(env, select_action, seed=...)`를 사용한다. 정책 확률에
따른 행동 callback과 환경 RNG를 분리하며 공식 TimeLimit이 Episode 경계를 보고한다.
아래 TabularEpisodeSampler의 공개 API와 이전 경계 동작은 호환성을 위해 수정하지 않는다.

`TabularEpisodeSampler(mdp, policy, *, seed=None).sample(start_state, *, max_steps=10000)`는
owned Policy float64 `(S,A)`와 독립 NumPy Generator를 사용한다. Policy shape·확률합을
생성 시 검증한다. 매 step에서 정책으로 action을, 전이확률로 joint outcome을 선택하며
reward와 next_state를 기록한다. DP와 같은 종료 규칙(outcome.terminated 또는 선언된 terminal
진입)을 사용한다. 알려진 terminal에서 시작하면 RNG를 사용하거나 추가 전이를 만들지 않고
ValueError를 발생시킨다. 특정 Gymnasium 환경의 P 포맷에 의존하지 않는다.

마지막 자연 종료이면 max_steps와 같은 시점이어도 terminated=True, truncated=False다.
끝나지 않은 채 max_steps에 도달하면 마지막 Transition만 truncated=True로 표시한다.
MC Learner는 terminated=True인 Episode만 허용한다. 두 flag가 모두 True라면 실제 종료가
확인되므로 학습할 수 있지만 truncated-only/open Episode는 Full Return이 없어 거절한다.
TD의 Time-limit Bootstrap 규칙과 달리 순수 MC는 truncated Return을 Bootstrap으로 보완하지 않는다.

기본 실험은 잘린 Episode가 나오면 두 Learner의 갱신 전에 명시적으로 중단한다.
완료된 Episode만 계속 골라 수집하면 긴 Episode를 제외하는 selection bias가 생길 수 있으므로
조용히 조각을 버리고 새 Episode를 수집하지 않는다. 종료가 어려운 정책은 max_steps 증가나
정책의 Proper 여부를 별도로 검토해야 한다.

### Gridworld 검증과 다음 TD(0) Prediction

```bash
python -m examples.gridworld_mc
python -m examples.gridworld_mc --episodes 100 --seed 7
```

`examples/gridworld_mc.py`는 make_sutton_gridworld_env(), collect_episode(), uniform_policy(),
policy_evaluation()을 재사용한다. gamma=1이며 환경 RNG(seed=7)가 14개 비종료 상태 중
균등 시작과 Outcome을 선택하고 외부 행동 RNG는 seed=8이다. 동일 Episode를 First/Every Learner에 전달한다. 예제 출력은
Episode 수, 전체 환경 step, 시작 횟수, 상태별 update count, V, DP 대비 RMSE다.
RMSE는 terminal 0/15를 제외한다. 전체 계산은 float64이며 default max_steps=10000이다.

이전 Sampler 경로의 검증 기록: 20,000 완료 Episode, 365,007 step에서 초기 RMSE=18.516402였고 First-visit=.199544,
Every-visit=.283310이었다. 모든 비종료 상태에 sample이 있었고 terminal count/V는 0이었다.
5,000/10,000/15,000 Episode의 RMSE는 First=.621222/.284128/.231667,
Every=.619912/.295867/.240866이었다. 마지막 Every RMSE는 이전 구간보다 커졌지만 DP값에
통계적으로 접근했다. 이 수치를 새로운 Gymnasium RNG 경로의 결과로 간주하지 않는다.
특정 Seed의 단조 감소나 전체 실험 RMSE 임계값을 pytest 합격 조건으로 삼지 않는다.
결정적 테스트는 반복 상태·Return·평균·Count·경계·저장 복원을, 통계 테스트는 고정 Seed의
행동/Outcome 빈도와 이론 확률을, 짧은 실험 테스트는 실행·RMSE 정의·재현성을 확인한다.

다음 TD(0) Prediction은 같은 Transition, Episode의 시간 순서/종료 flags, 외부 Sampler,
고정 Policy와 TabularVEstimator Model을 재사용할 수 있다. TD는 완성된 Episode를 기다리는 대신
각 전이의 next V로 TD0TargetGenerator와 TabularVUpdateRule을 적용한다. MC와 달리 일반적인
truncated 전이에서 Bootstrap이 가능하다. 고정 정책 V_pi 비교에는 기존 DP 참조값을 그대로 사용한다.
Phase 2의 Episode 계약을 아래 Phase 3에서 그대로 재사용한다. 범용 Trajectory Framework는 추가하지 않는다.

## Tabular On-policy Monte Carlo Control

Sutton & Barto의 [Chapter 5 강의 자료](https://people.cs.umass.edu/~barto/courses/cs687/Chapter%205.pdf)의
Episode 단위 평가·정책 개선 흐름을 따른다. Prediction이 고정 정책의 V(s)를 추정한다면
Control은 Q(s,a)를 추정해 다음 Episode의 행동을 개선한다. 기존 MonteCarloReturnGenerator,
Episode, TabularQEstimator, Model, EpsilonGreedyActionSelector와 collect_episode를 재사용한다.
새 BaseLearner, UpdateRule, 정책 행렬 관리자, Sampler는 추가하지 않는다.

| 공개 API / 데이터 | 계약 |
|---|---|
| `TabularMCControlLearner(model, return_generator, visit_mode="first")` | TabularQEstimator Model과 기존 MC Return Generator, first/every 지원 |
| `learn(episode: Episode) -> Tensor` | 같은 Model의 table을 no_grad에서 in-place 갱신, 전체 Return `(T,)` 반환 |
| observation / action | scalar integer 인덱스, table 범위 확인 후 long 정규화 |
| reward / 반환 Return | live Q-table의 floating dtype·device, 반환 graph 없음 |
| `visit_counts` | Learner 소유 long Tensor `(S,A)`, learn 시 table device를 따름 |

학습의 두 단계는 기존 Generator로 전체 Return을 계산한 뒤, 원래 시간 순서에서 상태-행동
방문을 고르고 아래 Sample Average를 갱신하는 것이다.

```text
count[s,a] += 1
Q[s,a] += (G_t - Q[s,a]) / count[s,a]
```

First-visit는 상태가 아닌 (s,a)의 가장 이른 방문이다. 같은 상태의 다른 행동은 각각 첫 방문이
될 수 있다. Every-visit는 반복 쌍의 Return도 시간순으로 모두 포함한다. 마지막 next_observation은
방문이 아니다. states=[1,2,1], actions=[0,1,0], rewards=[1,2,3], gamma=1이면 Return=[6,5,3],
First Q(1,0)=6/count=1, Every Q(1,0)=4.5/count=2이고 Q(2,1)=5다.
초기 Q를 가상 sample로 세지 않는다. 입력 Episode는 수정하지 않고 Return overflow 및 잘못된
인덱스는 table/count 갱신 전에 실패한다. 열린 Episode·truncated-only는 거절한다.
자연 종료와 TimeLimit이 동시에 True면 완료된 Return으로 학습할 수 있다.

### 정책과 Episode 경계

```python
model = Model(IdentityRepresentation(), TabularQEstimator(11, 4),
              EpsilonGreedyActionSelector(epsilon=0.1))
learner = TabularMCControlLearner(model, MonteCarloReturnGenerator(gamma=0.9))
episode = collect_episode(env, lambda state: model.act(state).action.item(), seed=7)
returns = learner.learn(episode)
```

collect_episode의 기존 int callback으로 Model.act를 바로 사용할 수 있다. 수집 중 Q 및 epsilon은
고정하고 자연 종료 후 learn을 호출한다. 다음 Episode부터 같은 Model과 Selector가 새 Q를
참조하므로 별도 Policy 행렬 생성이나 동기화가 없다. Representation은 Identity처럼 인덱스를 보존한다.
Learner 자체는 Selector를 요구하거나 호출하지 않는다. On-policy 경험 수집은 호출자 책임이며
과거의 다른 행동 정책 경험을 재생하거나 IS 보정을 자동으로 적용하지 않는다.

기존 행동 확률은 `epsilon/A + (1-epsilon)*I(a in maxima)/len(maxima)`다. 탐색을 포함한
stochastic 동률은 균등, deterministic 평가에서는 epsilon을 무시하고 최소 인덱스 최대 행동을
고른다(log_prob=None). epsilon 속성 setter로 외부 Episode 경계에서 schedule을 설정할 수 있다.
환경은 첫 reset seed와 이후 None의 RNG stream, 행동 선택은 PyTorch RNG를 사용한다.
평가는 별도 환경의 deterministic=True로 수행하며 학습하지 않는다.

### Checkpoint와 수렴 해석

Q-table과 epsilon은 기존 Model.state_dict에 포함되지만 visit_counts는 포함되지 않는다.
재개 시 Model과 같은 형태의 long Count, gamma, visit_mode를 함께 저장·복원한다.
예를 들어 `torch.save({"model": model.state_dict(), "visit_counts": learner.visit_counts.clone(),
"gamma": learner.return_generator.gamma, "visit_mode": learner.visit_mode}, path)`로 저장하고,
새 Model/Learner 구성 후 `model.load_state_dict(saved["model"])`,
`learner.visit_counts.copy_(saved["visit_counts"])`로 복원한다. Model dtype/device도 유지한다.
완전한 실험 재개에는 Episode 번호와 환경 RNG, CPU/CUDA PyTorch 행동 RNG 상태도 필요하다.
Count만 초기화하면 이전 Return 평균의 정상적인 재개가 아니다.

고정 epsilon은 탐색을 유지한다. 변화하는 행동 정책에서 관측한 과거 Return의 평균을 사용하므로
유한 표본의 max Q가 DP의 최적 V와 같다고 주장하지 않는다. GLIE schedule, Exploring Starts,
Off-policy IS 또는 다른 수렴 기능을 구현한 것으로 간주하지 않는다.

### 교재 Gridworld 검증

```bash
python -m examples.gridworld_mc_control --episodes 1000 --seed 7 --epsilon .1
```

`examples/gridworld_mc_control.py`는 Gymnasium 환경과 Model.act로 Episode를 수집한 뒤 Learner를
호출한다. gamma=.9, epsilon=.1, First-visit, fixed start (2,0), float64다. 학습 cap=2000,
Greedy 평가 cap=20이며 두 환경은 finally에서 닫는다. truncated-only 학습 표본은 조용히
버리지 않고 갱신 전에 실험을 중단한다. Episode/누적 Step별 Training Reward와 Greedy 성공·길이·보상,
Q, Count, 정책 및 DP 최적 경로를 보고한다.

seed=7, 1,000 Episode 실행은 5,672 Step, cutoff 0이었다. 초기 Greedy는 목표에 도달하지 못하고
20 Step에서 잘렸으며, 학습 후에는 5 Step·보상 +1로 목표에 도달했다. 마지막 100 Training Episode의
평균 보상은 1.0이었다. DP 최적 경로도 5 Step이며 V*(start)=.6561이다. 전체 상태·행동의
Count가 충분하다는 뜻은 아니며 단일 Seed의 성공률·길이를 pytest 합격 조건으로 삼지 않는다.

`05_mc_modularrl.ipynb`의 기존 Prediction 뒤에 On-policy MC Control 구성·학습·Q와 정책 표시,
DP 비교를 추가한다. 원본 고정 alpha=.1/10,000 Episode와 달리 Sample Average/1,000 Episode다.
Off-policy Prediction은 아래의 기록 확률·IS 통계로 구현한다. On-policy Control은 기존 int callback과
log_prob=None을 계속 사용할 수 있다. Off-policy Control은 아래의 별도 Weighted IS Learner를 사용한다.
SARSA는 아래의 별도 Learner에 실제 next_action을 전달하며 MC Learner에 TD 수식을 혼합하지 않는다.

## Off-policy Monte Carlo Prediction

Behavior b로 생성한 경험에서 고정 Target π의 상태가치 Vπ를 추정한다.
[Sutton & Barto Chapter 5 자료](https://people.cs.umass.edu/~barto/courses/cs687/Chapter%205.pdf)의
Importance Sampling을 전체 Episode Return에 적용한다. Q Control·SARSA·Per-decision IS는 포함하지 않는다.
Model/Estimator/ActionSelector/Data Class와 기존 MC Generator API는 변경하지 않는다.

| 공개 API / 상태 | 계약 |
|---|---|
| `OffPolicyMCPredictionLearner(model, return_generator, target_policy, method="weighted", visit_mode="first")` | TabularVEstimator Model, 기존 MC Return Generator, 고정 `(S,A)` 확률 정책 |
| `learn(episode: Episode) -> Tensor` | 전체 raw Return `(T,)`, no_grad, V-table dtype/device 유지 |
| `target_policy` | 입력에서 복사한 owned float64 Tensor; shape·finite·확률 범위·행 합은 생성 시 검사 |
| `visit_counts` | long `(S,)`, First/Every로 선택한 방문 수, rho=0도 포함 |
| `weight_sums` | float64 `(S,)`, 선택 방문의 누적 Σrho |
| `weighted_return_sums` | float64 `(S,)`, 선택 방문의 누적 Σ(rho*G) |

Target 정책은 변경하지 않는다. Observation/action은 scalar 정수 범위 인덱스이며 long으로
정규화한다. Reward/Return은 table dtype/device를 따른다. Learner는 Behavior Model·환경·Sampler·
Optimizer를 받거나 현재 정책을 조회하지 않고, Episode의 모든 sampling-time log_prob을 요구한다.
마지막 next_observation은 방문으로 추가하지 않는다. Input Episode도 수정하지 않는다.

### Sampling 시점의 Behavior 확률

기존 EpsilonGreedy는 `epsilon/A + (1-epsilon)*I(a in maxima)/len(maxima)`에서 실제로 샘플링하고
그 선택의 Categorical log_prob을 반환한다. 동률에 대한 확률도 포함된다. Model.act의 no_grad는
이 의미를 바꾸지 않는다. collector가 해당 log_prob을 즉시 복사하므로 이후 Q/epsilon 변화나
Callback Tensor 재사용 후에도 기록된 log b(a|s)는 유지된다.

```python
episode = collect_episode(env, behavior_model.act, seed=seed)
learner.learn(episode)
```

같은 데이터 계약을 직접 Gymnasium의 reset/step으로 구성할 수도 있다.
아래는 Tabular State/Action용이며 Episode는 기존 Transition을 시간순으로 묶는다.
log_prob은 env.step 전에 복사해 Callback이 같은 Tensor를 재사용해도 선택 시점의 값을 보존한다.

```python
from modular_rl.data import Episode, Transition

transitions = []
try:
    observation, _ = env.reset(seed=seed)
    while True:
        output = behavior_model.act(observation)
        action = output.action.item()
        log_prob = output.log_prob.detach().cpu().clone()
        next_observation, reward, terminated, truncated, _ = env.step(action)
        transitions.append(Transition(observation, action, reward, next_observation,
                                      terminated, truncated, log_prob))
        if terminated or truncated:
            break
        observation = next_observation
    episode = Episode(tuple(transitions))
    learner.learn(episode)  # truncated-only Episode는 Learner가 거절한다.
finally:
    env.close()
```

두 수집 경로는 같은 환경·행동 RNG 상태에서 같은 기록과 IS 학습 결과를 만든다.
정수 Callback의 기존 collect_episode 경로도 유지되지만 log_prob=None이므로 IS 학습에는 사용할 수 없다.

각 log_prob은 scalar floating Tensor, finite, <=0이어야 한다. b=0을 뜻하는 -inf, NaN/Inf,
log b>0, None은 갱신 전에 거절한다. log 공간의 유한 값은 양의 확률을 뜻하며, 불필요하게
exp(log b)를 먼저 계산해 작은 양의 b를 0으로 만들지 않는다.
Coverage는 π(a|s)>0 ⇒ b(a|s)>0다. 기록된 선택 행동만으로 미관측 행동의 support를 검증할 수
없으므로 전체 Coverage는 호출자가 보장한다. 실험의 균등 b=1/4는 모든 Target 행동을 지원한다.
int callback이나 평가용 deterministic ActionOutput의 None을 확률 1로 자동 추정하지 않는다.
전체 Return을 사용하는 이 MC 계약은 Behavior와 Target이 거의 확실하게 종료하는 Episodic 정책을
전제로 한다. 관측된 Episode의 종료만으로 Target 정책의 종료나 전체 Coverage를 증명할 수는 없다.

### Return·Suffix Weight·방문 규칙

```text
G_t = reward[t] + gamma * G_(t+1)        # existing MonteCarloReturnGenerator
rho_t = product(pi(a_k|s_k) / b(a_k|s_k), k=t,...,T-1)
N[s] += 1
C[s] += rho_t
S[s] += rho_t * G_t
ordinary: V[s] = S[s] / N[s]
weighted: V[s] = S[s] / C[s], when C[s] > 0
```

V Prediction의 rho에는 현재 행동도 포함된다. log(pi)-log_b의 reverse cumulative sum 뒤 exp로
전체 suffix 비율을 계산한다. Target 확률 0은 log=-inf이므로 해당 시점과 앞선 suffix Weight가
정확히 0이 된다. First-visit는 Return/Weight 역순 계산과 독립적인 시간상 첫 **상태** 방문이며,
Every-visit는 같은 상태의 모든 Return을 포함한다. On-policy Q Control의 (state,action) 방문과 다르다.

Ordinary는 rho=0 표본도 N에 포함하므로 첫 zero sample에서 V=0이다. Weighted는 C=0이면
기존 V를 유지하며 이후 양의 Weight가 처음 생기면 그 유효 표본으로 평균을 시작한다.
초기 V를 가상 Return로 세지 않는다. Reward=4, gamma=1, b=.25, pi=.5인 종료 전이는 rho=2,
Ordinary V=8, Weighted V=4다. 반복 상태 [1,2,1], reward [1,2,3], 매 행동 pi/b=2이면
Return=[6,5,3], rho=[8,4,2]다. 상태1의 First 값은 Ordinary=48/Weighted=6, Every는 27/5.4다.

### 수치·종료·저장 경계

통계와 Weight는 float64로 계산해 float32 V의 범위와 구분한다. V는 원래 dtype을 유지한다.
정확한 zero-target Weight와 양의 Weight가 underflow로 0이 되는 상황을 구분해 후자는 실패한다.
Weight exp, 누적 Weight/Weighted Return, V dtype 변환에서 overflow·NaN·Inf가 발생하면
FloatingPointError를 발생시킨다. 이번 Episode의 통계는 작업 복사본에 갱신한 뒤 유한성을 확인하고
원본에 copy하므로 실패한 Episode가 일부 상태만 갱신하지 않는다. 유효한 기존 V와 통계도 보존한다.

자연 종료된 Episode만 사용한다. open·truncated-only는 거절하고 자연 종료와 TimeLimit이 동시에
True면 완료된 Return을 사용할 수 있다. Bootstrap이나 final reset 관측 추론은 추가하지 않는다.

Model.state_dict는 V만 포함한다. 재개하려면 `visit_counts`, `weight_sums`, `weighted_return_sums`,
고정 target_policy, gamma, method, visit_mode를 모두 함께 저장한다. 복원은 같은 구성의 Model을
load한 뒤 각 통계 Tensor에 copy_한다. V만 복원해 통계를 0으로 두는 것은 정상적인 재개가 아니다.
통계는 learn에서 live table device로 옮기며 Count의 long·합의 float64는 Model.float/double과 독립적이다.
전체 실험의 재현에는 Episode 번호와 환경 RNG·CPU/CUDA PyTorch 행동 RNG도 필요하다.
Target/gamma/method/visit_mode 변경은 추정 대상이나 누적 통계의 의미가 달라지므로 기존 재개와 구분한다.

### 교재 Gridworld와 DP 검증

```bash
python -m examples.gridworld_off_policy_mc --episodes 1000 --seeds 7 11 19
python -m examples.gridworld_off_policy_mc --episodes 100 --seeds 7 --target-uniform
```

고정 시작 (2,0), gamma=.9, First-visit다. b는 TabularQEstimator의 초기 Q와 EpsilonGreedy(1)의
균등 Model이고 학습하지 않는다. π=(up,down,left,right)=(.3,.2,.2,.3)는 고정 확률 행렬로 별도 지정한다.
두 독립적인 V Model에 동일한 기록 Episode를 전달한다. DP는 같은 π의 policy_evaluation을 사용하며
RMSE는 Goal을 제외한 비종료 10개 상태다. cap=2000에서 truncated-only이면 두 Learner 갱신 전에
실험을 중단한다. 환경은 finally에서 닫는다.

| Seed | 1,000 Episode의 Step | Ordinary RMSE | Weighted RMSE |
|---|---:|---:|---:|
| 7 | 45,541 | .055652 | .062241 |
| 11 | 45,887 | .047766 | .051419 |
| 19 | 46,242 | .046639 | .040393 |
| 평균 ± std (ddof=0) | 총 137,670 | .050019 ± .004010 | .051351 ± .008920 |

Cutoff는 없었다. Ordinary IS의 높은 분산 가능성과 Weighted의 유한 표본 bias를 고려해 특정 Seed의
성능 임계값·단조 감소·방법 순위를 합격 조건으로 두지 않는다. b=π 균등 실험(seed7, 100 Episode,
4,203 Step)은 두 방법의 V/RMSE가 같고 max|Σrho-N|=0이었다. 결정적 테스트에서는 b=(.25,.75)의
정확한 빈도 데이터와 Target=(.5,.5)에서 두 IS 방법이 DP 기대 보상 2를 복원하는 것도 검증한다.

05 Notebook은 기존 On-policy Prediction/Control·확률변수 IS 뒤에 이 Prediction을 추가한다.
Notebook에서 비율이나 V 갱신을 재구현하지 않고 고정 Target, Behavior Model, collect_episode,
Learner, DP 비교만 사용한다. Off-policy MC Control은 다음 섹션의 별도 Learner를 사용하며 SARSA는 범위 밖이다.

## Off-policy Monte Carlo Control

표준 Weighted IS Control을 기존 TabularQEstimator와 Model 위에 추가한다.
Behavior는 같은 Model의 epsilon-greedy ActionSelector, Target은 갱신된 Q의 결정적 Greedy다.
정책 행렬이나 Target Model을 Learner에 두지 않고, 환경·수집·RNG는 외부에 유지한다.
Prediction의 고정 V 정책 및 Ordinary/First/Every 옵션을 이 Control에 혼합하지 않는다.

| 공개 API / 상태 | 계약 |
|---|---|
| `OffPolicyMCControlLearner(model, return_generator)` | 기존 TabularQEstimator Model과 MC Return Generator |
| `learn(episode: Episode) -> Tensor` | 전체 raw Return `(T,)`, 시간순, no_grad, Q dtype/device |
| `weight_sums` | 누적 중요도 Weight C(s,a), float64 `(S,A)`, table device |
| Target 행동 | 갱신된 Q 행의 argmax, 동률은 최소 Action ID |

기존 Return Generator로 전체 G를 역순 계산한 뒤 아래 절차를 수행한다.
Q는 현재 행동을 조건으로 하므로 시점 t의 W는 **t 이후** 행동의 비율만 포함한다.
V Prediction의 rho_t는 현재 행동도 포함한다는 점과 다르다.

```text
W = 1
for t = T-1, ..., 0:
    G = returns[t]                       # existing Return Generator
    C[s,a] += W
    Q[s,a] += (W / C[s,a]) * (G - Q[s,a])
    greedy = argmax(Q[s,:])              # after the update; first maximum
    if a != greedy: break
    W /= b(a|s)                          # recorded log b; only if an earlier t remains
```

Greedy 검사 전에 현재 Q를 갱신하므로 이 행동이 새 Greedy가 되면 이전 시점으로 계속한다.
갱신 후 Greedy가 아니면 그 시점까지는 갱신되지만 앞부분은 변경하지 않는다.
반복 상태-행동 쌍은 방문 필터 없이 역순으로 갱신한다. Return 반환에는 Early Break로
건너뛴 앞부분도 포함한다. 행동 선택이나 Bootstrap을 Learner에서 호출하지 않는다.

단일 reward=4, gamma=1, b=.25 표본은 Q=4/C=1이다. 두 단계 states=[0,1], actions=[0,1],
rewards=[1,3], b=.25에서는 Q(1,1)=3/C=1, Q(0,0)=4/C=4다. 같은 쌍에 다른 Weight를 갖는
Episode를 추가하면 C 분모를 공유한 Weighted Average가 된다. 수치 테스트는 갱신 전/후 Greedy
변경과 동률 Early Break를 별도로 검증한다.

### 수집·수치·Checkpoint

직접 reset/step → Transition → Episode 또는 `collect_episode(env, model.act)`를 사용한다.
직접 수집 코드는 위의 [Behavior 확률 기록](#sampling-시점의-behavior-확률) 예제와 같으며
학습기를 OffPolicyMCControlLearner로 구성하면 된다. step 전에 log_prob을 detach·CPU clone하고
수집 중 Q/epsilon을 고정한다. 종료 후 learn하면 다음 Episode의 같은 Model이 새 Q를 사용한다.
기존 정수 callback도 유지하지만 log_prob=None 경험은 IS Control에 사용할 수 없다.

모든 전이의 scalar integer 인덱스 및 scalar floating log b(finite, <=0)를 갱신 전에 확인한다.
열린 Episode·truncated-only는 거절하고 자연 종료와 TimeLimit이 동시에 True면 허용한다.
Coverage는 호출자 책임이며 기록된 선택 행동만으로 미관측 행동의 support를 검증할 수 없다.
예제는 epsilon>0으로 모든 행동을 지원한다. 전체 Return의 Episodic 종료 전제를 유지한다.

Weight와 C는 float64로 계산하며 Q는 원래 floating dtype/device를 유지한다.
Return·W·C·Q 연산의 Overflow/NaN/Inf는 오류로 보고한다. Q/C 작업 복사본에서 역순 갱신하며
성공한 경우만 기존 table storage와 C에 복사하므로 실패 시 기존 학습 상태가 보존된다.
Early Break 뒤나 t=0 뒤에는 사용하지 않을 W를 계산하지 않는다.
이는 별도 영구 Q-table이나 Estimator를 생성하는 구조가 아니다.

```python
torch.save({"model": model.state_dict(), "weight_sums": learner.weight_sums.clone(),
            "gamma": learner.return_generator.gamma}, path)
# Reconstruct the same dtype/device model and learner, then:
model.load_state_dict(saved["model"], strict=True)
learner.weight_sums.copy_(saved["weight_sums"])
```

C는 Model.state_dict에 포함되지 않는다. C 없이 이전 Q만 복원하면 같은 Weighted Average를
재개하는 것이 아니다. Episode 번호와 환경/CPU·CUDA PyTorch RNG도 실험 재개 시 복원한다.

### DLFS4 Gridworld와 Notebook

`python -m examples.gridworld_off_policy_mc_control --seeds 7 11 19`는 gamma=.9, epsilon=.1,
1,000 Episode, 고정 시작 7, float64, 학습 cap=2000·평가 cap=20을 사용한다.
Episode Reward·누적 Step·Q·C·결정적 Target 행동·Greedy 평가·DP 최적 경로를 보고한다.
기존 On-policy 예제의 별도 환경 Greedy 평가 함수를 재사용한다. cutoff-only 학습은 중단하며
환경은 finally에서 닫는다. DP 및 기존 Model/수집/Data Class API는 변경하지 않는다.

| Seed | Step | 학습 후 Greedy Step / Reward | max Q의 비종료 10개 상태 DP RMSE |
|---|---:|---|---:|
| 7 | 5,701 | 5 / +1 | .250643 |
| 11 | 5,625 | 5 / +1 | .060083 |
| 19 | 5,574 | 5 / +1 | .000000 |

초기 Greedy는 20 Step cutoff, 학습 후는 DP와 같은 5 Step 경로였다.
모든 Seed에서 training cutoff=0, 마지막 100 Episode 평균 보상=1.0이었다.
일부 상태-행동의 C가 0인 점과 Seed별 오차를 함께 보고하며 전체 Q* 수렴이나 단일 Seed 성공을
테스트 조건으로 삼지 않는다. 05 Notebook의 마지막 섹션은 첫 Episode를 직접 reset/step으로,
나머지를 collect_episode로 수집하고 Learner만 호출한다. 기존 실습의 계산 코드는 유지한다.
원본 고정 alpha=.2와 달리 이번 구현은 Sutton & Barto의 C 기반 표준 Weighted IS다.

## Tabular TD(0) Prediction Phase 3

Sutton & Barto의 [Chapter 6, TD Prediction](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf)을
따른다. MC는 완료된 Episode의 Return을 사용하지만 TD(0)는 관측한 reward와 현재 추정한
next V로 Bootstrap한다. `learner/tabular_td0.py`는 기존 Model, TD0TargetGenerator,
TabularVUpdateRule을 조합하며 새로운 TargetGenerator/UpdateRule/BaseLearner를 만들지 않는다.

### 공개 API와 계산 경계

| API / 데이터 | 계약 |
|---|---|
| `TabularTD0PredictionLearner(model, target_generator, update_rule)` | TabularVEstimator Model + TD0TargetGenerator + TabularVUpdateRule |
| `learn(batch: TransitionBatch) -> Tensor` | 원본 V-table 갱신, 행 순서 signed TD error `(B,)` 반환 |
| observation / next_observation | 비어 있지 않은 integer `(B,)`, 현재 table device의 long으로 정규화 |
| reward / terminated / truncated | 같은 `(B,)`, floating reward와 bool flags |
| 준비된 reward / 반환 error | V-table의 floating dtype·device, 반환 graph 없음 |
| action / optional log_prob | 기존 TransitionBatch 필드 보존, V Prediction 계산에 사용하지 않음 |

단일 전이도 `B=1`로 전달한다. Representation은 State Index를 보존해야 하며 기본 구성은
IdentityRepresentation이다. 생성 후 Model.to/double을 호출해도 learn은 현재 v_table을 참조한다.
입력 검증은 `_validate()`에 두고 기존 컴포넌트가 shape, table bounds와 target 계약을 검사한다.
정책과 MDP, EpisodeSampler, ReplayBuffer, 행동 선택은 외부 책임이며 Optimizer가 없다.

```text
Batch.to(table.device, dtype=table.dtype), states -> long
no_grad:
    next_value = Model(next_states).values.v
    target = TD0TargetGenerator.generate(reward, next_value, terminated)
signed_error = TabularVUpdateRule.update(v_table, states, target)
```

수식은 `y = r + gamma * (1-terminated) * V(s')`, `delta = y - V(s)`,
`V(s) += alpha * delta`다. TargetGenerator 자체의 gradient 보존 계약은 유지하고 Learner가
Bootstrap과 Target 계산을 no_grad로 감싼다. Table 갱신/반환 graph 제거는 기존 Rule 책임이다.
V(1)=4, V(2)=10, r=1, gamma=.9, alpha=.5이면 y=10, delta=6, 새 V(1)=7이다.
같은 전이가 terminated=True면 y=1, delta=-3, 새 V(1)=2.5다.

terminated만 Bootstrap을 제거한다. truncated=True/terminated=False인 일반적인 외부
Time-limit은 Bootstrap하며 다음 Episode reset 대신 실제 마지막 관측을 전달하는 것은 수집자
책임이다. Learner에는 terminal 상태 집합이나 Episode 연결을 추론하는 코드가 없다.
Terminal placeholder V는 Sampler가 추가 전이를 만들지 않아 갱신하지 않으며 초기값 0을 유지한다.

### Frozen Target Batch와 전이별 Online 갱신

`B>1`에서는 모든 next V와 Target을 갱신 시작 전에 확정한다. 이후 같은 상태가 반복되면
기존 UpdateRule이 이전 행에서 바뀐 현재 V를 사용해 순서대로 error/update를 계산한다.
Bootstrap을 행 사이에 다시 평가하지 않는다. `B=1`을 순서대로 호출하면 다음 호출의 Target은
이미 갱신된 테이블에서 계산된다. 이 차이는 의미 있는 알고리즘 계약이다.

초기 V(1)=4, V(2)=10, gamma=.9, alpha=.5에서 전이 `(2,0,1)`, `(1,1,2)`의
튜플은 각각 `(state,reward,next_state)`다. 한 Batch로 전달하면 delta=[-6.4,6],
최종 V(1)=7, V(2)=6.8이다. 전이별 B=1 호출은 delta=[-6.4,3.12],
최종 V(1)=5.56, V(2)=6.8이다. 두 방식 모두 Rule의 중복 상태 순차 갱신을 유지한다.

V-table은 Model.state_dict에 포함된다. TD Learner에는 별도 Count나 Optimizer 상태가 없으며
재개할 때 gamma/alpha, 고정 정책, Model dtype/device와 외부 수집 RNG를 함께 보존한다.
MC의 visit_counts와 TD table은 서로 독립적이다.

### 동일 경험을 사용하는 DP/MC/TD 비교

```bash
python -m examples.gridworld_td
python -m examples.gridworld_td --episodes 5000 --seeds 7 11 19 --alphas .01 .05 .1
python -m examples.gridworld_td --episodes 20 --log-every 5
```

`examples/gridworld_td.py`는 MC와 같은 make_sutton_gridworld_env(), collect_episode(),
uniform_policy(), policy_evaluation()을 재사용한다. gamma=1, 시작 상태는 14개 비종료
상태에서 균등 추출한다. 환경 RNG는 seed, 행동 RNG는 seed+1이며 현재 MC 예제와 동일하다.
각 Seed에서 두 MC 및 세 alpha의 TD Model을 독립적으로 0에서 시작한다.

MC는 같은 완료 Episode를 전체 입력으로 받고 TD는 그 Episode의 Transition을 시간 순서대로
B=1 Batch로 만들어 학습한다. Episode 생성 후 처리하므로 실제 환경 종료 전에 학습하는
Streaming Runner는 아니다. Cutoff이면 모든 Learner가 해당 Episode를 처리하기 전에 실패한다.
이를 조용히 버려 긴 Episode를 제외하거나 MC에 미완성 Return을 전달하지 않는다.
TD의 Truncation Bootstrap은 별도의 결정적인 단위 테스트로 검증한다.

RMSE는 DP의 고정 정책 V_pi를 기준으로 terminal 0/15를 제외해 계산한다. 출력과 Python
`train_and_compare()`의 `history`에는 초기값과 checkpoint별 Episode 수, 누적 환경 step,
각 방법의 RMSE를 기록한다. 최종 V와 Seed/alpha/gamma도 보고하며 여러 Seed의 마지막
RMSE 평균·표준편차는 ddof=0이다. 같은 경험은 방법 수와 관계없이 한 번의 환경 step으로 센다.
Constant alpha TD는 표본 변동이 남을 수 있다. 특정 alpha가 MC보다 우수하거나 한 Seed의
RMSE가 매 checkpoint 감소해야 한다는 테스트는 두지 않는다.

이전 Sampler 경로의 검증 실행은 Seed마다 5,000 완료 Episode, 초기 V=0, float64, gamma=1이었다.
초기 RMSE는 18.516402이며 아래는 각 Seed의 최종 RMSE다. 세 Seed의 실제 환경 step 합은
274,415이고 MC/TD 모두 각 Seed 안에서 동일 Episode를 사용했다. Cutoff는 없었다.

| Seed | 환경 step | First MC | Every MC | TD alpha=.01 | TD alpha=.05 | TD alpha=.1 |
|---|---:|---:|---:|---:|---:|---:|
| 7 | 89,019 | .621222 | .619912 | 1.567009 | 1.753061 | 2.018980 |
| 11 | 92,728 | .351567 | .469013 | .680536 | 1.120250 | 1.701403 |
| 19 | 92,668 | .217053 | .269155 | .564713 | 1.189102 | 2.026688 |
| 평균 ± std | — | .396614 ± .168048 | .452694 ± .143660 | .937419 ± .447691 | 1.354138 ± .283479 | 1.915691 ± .151557 |

예를 들어 seed=7, alpha=.05의 1,000/2,000/3,000/4,000/5,000 Episode RMSE는
1.237541/1.155240/.856019/2.237295/1.753061이었다. 해당 checkpoint의 step은
18,398/37,122/54,592/71,943/89,019다. 이 실행의 순위와 오르내림은 관측 결과이며
MC 또는 TD의 일반적인 성능 우위로 해석하지 않는다. Gymnasium 경로의 표본과 수치는 달라질 수 있다.

현재 Gymnasium 경로의 검증(seed=7, gamma=1, 1,000 완료 Episode)은 17,609 step,
cutoff 0이었다. First MC RMSE=.758770, Every MC=.600755이고 TD alpha=.01/.05/.1은
각각 9.454563/1.738354/1.741590이었다. MC 단독 예제와 비교 예제의 MC 값·Count·Step은
같았다. 새 경로 자체의 seed 재실행, 분포 및 결정적인 갱신 수식으로 정확성을 검증한다.

### 공통 데이터와 실제 중복

| 공통 요소 | 사용 위치 |
|---|---|
| TabularMDP / TransitionOutcome, Policy `(S,A)` | DP 기대값과 Gymnasium 환경·외부 행동 callback이 공유 |
| State index, reward, 종료 flag와 실제 next state | 경험 Transition/Episode/TransitionBatch가 보존 |
| 같은 reward/gamma/terminal 의미, 고정 정책 V_pi | 세 방식의 비교 기준 |
| TabularVEstimator / Model | MC와 TD의 값 저장·추론·checkpoint; DP는 NumPy V 유지 |

Episode 수집은 공통 collect_episode로 처리한다. 비교 예제와 MC 예제의 행동 RNG 설정,
비종료 상태 목록, Model 초기화와 RMSE 계산은
작게 반복된다. TD0 Learner와 TabularQLearner는 Batch 준비 및 reward/flag/index 검증이
겹치지만 Bootstrap 선택과 Rule 호출이 다르다. 이 정도의 반복을 공통 Base나 통합 Framework로
숨기지 않는다. DP Bellman 합산, MC 전체 Return 평균, TD 한 단계 Bootstrap은 별도 계산으로
유지한다. 추가 통합 리팩터링은 현재 필요하지 않다.

SARSA는 수집 시 실제 선택한 next_action을 별도 인자로 받아 기존 TD0TargetGenerator와 TabularQUpdateRule을 재사용한다. On-policy MC Control은
위의 별도 Q Learner로 구현했으며 기존 V Prediction을 변경하지 않는다. Off-policy V Prediction은 위의 별도 IS Learner로 구현한다.

## 교재 Notebook과 최소 사용성 API

`notebooks/dlfs4/`는 『밑바닥부터 시작하는 딥러닝 4』 Chapter 4–6의 확인한 원본 실습을
공개 API 호출로 표현한다. 알고리즘은 Notebook에서 재구현하지 않는다. DP의 제자리 모드,
Off-policy SARSA는 미구현으로 명시한다. SARSA는 실제 Next Action의 B=1 학습으로,
Off-policy Control은 표준 Weighted IS로 구성하며
[원본 실습 대응표](../notebooks/dlfs4/README.md)에 각 파일·설정·지원 상태·차이를 기록한다.

`environment/dlfs4_gridworld.py`는 3×4 교재 TabularMDP와 그 Gymnasium 실행 환경 생성 함수를 제공한다.
벽 (1,1)을 제외한 11개 State ID와 좌표 tuple을 사용하며 시작은 (2,0), 목표는 (0,3)이다.
목표 진입 +1·종료, (1,3) 진입 -1·계속, 다른 진입 0이며 벽·경계는 제자리다.
기존 Sutton 환경과 다르다. MC/TD Sampling은 collect_episode의 reset/step,
Q-learning은 명시적인 Gymnasium 환경 루프를 사용한다. 표시는 별도의 작은
notebooks/dlfs4/plotting.py가 담당한다. DP의 NumPy 계산과 교재 전이 kernel은 변경하지 않는다.

`Transition.to_batch() -> TransitionBatch`는 모든 present field를 dtype 유지·graph 없는
복사로 만들고 앞에 B=1을 추가한다. Tensor는 원래 device를 보존하고 Python/NumPy는 CPU다.
이산 action은 `(1,)`, 연속 action `(1,A)`, observation은 `(1,*shape)`다. log_prob=None을
유지한다. 알고리즘 입력 검증은 Learner/Buffer의 기존 책임이며 float 강제 변환을 하지 않는다.
기존 생성자·필드·Batch.to·Learner API를 유지하는 추가 메서드다. 기존 TD 비교 예제의
전이별 private helper도 이 공개 변환으로 대체했다.

Notebook의 기본 MC/TD는 γ=.9, 고정 시작·Uniform Policy·1,000 Episode와 동일 표본을
사용한다. MC는 완료 Episode 전체, TD는 수집된 Episode의 시간순 B=1을 처리하며 Streaming
Runner 구현으로 해석하지 않는다. Q-learning은 α=.8, ε=.1, 1,000 Episode와 명시적
환경 루프를 사용한다. Kernel 실행·Matplotlib·nbformat/nbclient는 선택적 notebooks extra다.
Run All 두 번의 실제 출력 비교는 같은 Kernel에서 수행하며 core API의 부재를 우회하지 않는다.

Gymnasium 경로의 Notebook Run All 검증에서 MC/TD는 동일한 1,000 Episode, 45,188 step을
사용했다. 비종료 상태 DP 대비 RMSE는 First MC=.0306, Every MC=.0317, TD=.0239였고
Q-learning은 cutoff 0, 마지막 100 Episode 평균 보상 1.0, Greedy 5번 이동으로 목표에 도달했다.
MC Control 섹션은 seed=7의 1,000 Episode·5,672 step 이후 Greedy 5 Step·보상 +1을 보였다.
기존 13개 그림을 보존하고 Control의 Q/정책·DP 비교·성능 그림 3개를 추가했으며,
Off-policy Prediction은 고정 Target의 DP 값과 Ordinary/Weighted V를 비교하는 그림 2개를 추가했다.
seed7, 1,000 Episode·45,541 step에서 RMSE는 .055652/.062241이었고 b=pi의 Weight=1도 확인했다.
Off-policy Control은 첫 Episode를 reset/step으로 직접 수집하고 나머지를 collect_episode로 수집한다.
seed7, 1,000 Episode·5,701 Step 이후 Greedy는 5 Step·보상 +1이었다. Q/Target·DP/C 비교와
학습/평가 그래프 2개를 추가했다. 세 Notebook의 텍스트·수치·20개 PNG 출력이 같은 Kernel의
두 실행에서 일치했고 기존 05의 코드 셀 21개 소스·출력은 이전 commit과 일치했다.
SARSA는 γ=.9, α=.8, ε=.1, Seed7의 1,000 Episode/8,604 Step 이후 Greedy 5 Step/+1을 보였다.
직접 Next Action을 재사용하는 루프와 다중 Seed 비교 루프의 Q가 일치하며 비교 결과는 위에 기록한다.
Greedy 정책·학습 Reward/Length/Step 그림 2개를 추가했다. Expected SARSA §12, Double Q-learning §13,
Online n-step SARSA §14, Off-policy n-step 고정 정책 평가 §15, Tree Backup §16은 예제 함수 조합으로 검증하며 그림 2개씩 추가해 총 32개 PNG와 텍스트·수치가
두 Run All에서 일치했다. 기존 06의 49개 셀 전체(코드 23개)의 소스·출력·실행 번호·메타데이터도 이전 commit과 같다.
최종 전체 pytest는 1,479 passed, 37 skipped(CUDA 미지원)였으며 본체 추적 파일 61개는 변경하지 않았다.

## 향후 Learner 구현 전에 남은 결정 사항

- On-policy Trajectory 계약: 확정한 Replay `(B, ...)`와 별도로 `(T, B, ...)` 정렬·에피소드 경계·GAE/n-step에 필요한 정보를 결정한다.
- Gradient 경계: 수집에는 act, 업데이트에는 forward/evaluate_q를 사용하는 범위와 Target 평가의 no_grad 정책.
- Parameter 소유권: 공유 Representation과 Policy/Critic의 Optimizer 구성, 동결·Target 복사 책임, 중복 업데이트 방지.
- Device/dtype와 checkpoint: Buffer의 CPU 저장 이후 학습 Batch 전송·floating dtype, Adapter 설정, Buffer/Optimizer/Target/RNG 상태의 저장·복원 범위.
- 연속 행동 지원: 실제 Continuous ActionSelector, action bound 변환과 log probability의 일관성. 현재 Policy/Critic 분리 구성은 유지한다.

## Base Class와 출력

`BaseEstimator`는 `forward(features) -> EstimatorOutput`만 정의한다.
`BasePolicyEstimator`, `BaseValueEstimator`는 기존 `PolicyEstimator`,
`ValueEstimator`의 alias다. Network 주입을 위해 Base Class에 새 메서드나
Factory, Registry, Builder를 추가하지 않는다.

출력 필드는 기존 계약을 유지한다.

- `EstimatorOutput.policy`: Distribution 또는 deterministic 행동 Tensor.
- `EstimatorOutput.values`: `ValueOutput(v, q, advantage)`.
- `ValueOutput.q`: 항상 tuple. 단일 critic은 `(q,)`, 복수 critic은 `(q1, q2)`로 표현한다.

Distributional payload도 원본을 담을 수 있지만 아래 Neural Estimator는
scalar value와 기본 Policy를 계산한다. Distributional 알고리즘의 구현을
의미하지 않는다.

## Network 주입과 shape 계약

대상 7개 Estimator는 기존 생성자 인자 뒤에 keyword 전용
`network: nn.Module | None = None`을 받는다. `None`이면 기존 기본 Network를
만들고, 전달된 모듈은 복사하거나 감싸지 않고 `self.network`에 등록한다.
TabularVEstimator와 TabularQEstimator는 기존 lookup buffer를 사용한다.

7개 Neural Estimator의 `feature_dim`, `n_actions` 또는 `action_dim`은 기본·custom
Network 모두 양의 정수여야 한다. `numbers.Integral`에 해당하는 NumPy 정수도
허용하지만 bool, 실수, 0과 음수는 명확한 ValueError로 거절한다. 생성 시 한 번
검사하며 매 계산마다 반복하지 않는다. Continuous Q의 `hidden_dim`은 기본
Network일 때만 같은 기준으로 검사한다.

`...`는 모든 앞쪽 batch 차원, `F`는 feature 차원이다. 이산 Estimator의
`A`는 행동 수, 연속 Estimator의 `A`는 행동 벡터 차원이다. 모든 Neural
Network 입력은 Float Tensor이며, 반환값은 batch 차원을 유지하는 Tensor다.

| Estimator | Network 입력 | Network 출력 | Estimator 출력 | 기본 Network |
|---|---|---|---|---|
| CategoricalPolicyEstimator | features `(..., F)` | logits `(..., A)` | `policy = Categorical(logits)` | `Linear(F, A)` |
| GaussianPolicyEstimator | features `(..., F)` | mean `(..., A)` | `policy = Independent(Normal, 1)` | `Linear(F, A)` |
| DeterministicPolicyEstimator | features `(..., F)` | action `(..., A)` | `policy` Tensor `(..., A)` | `Linear(F, A)` |
| StateValueEstimator | features `(..., F)` | value `(..., 1)` | `values.v (...)` | `Linear(F, 1)` |
| DiscreteQEstimator | features `(..., F)` | Q `(..., A)` | `values.q = (q (..., A),)` | `Linear(F, A)` |
| DiscreteAdvantageEstimator | features `(..., F)` | advantage `(..., A)` | `values.advantage (..., A)` | `Linear(F, A)` |
| ContinuousQEstimator | concat(features, actions) `(..., F+A)` | Q `(..., 1)` | `values.q = (q (...),)` | `Linear(F+A, H) -> ReLU -> Linear(H, 1)` |

V와 Continuous Q의 scalar head는 마지막 singleton 차원을 포함해 반환한다.
Estimator의 `squeeze(-1)`가 그 차원만 제거한다. 예를 들어 single state의
Network 출력은 `(1,)`, batch 하나의 출력은 `(1, 1)`이며, 최종 V/Q는 각각
scalar와 `(1,)`가 된다. 이미 squeeze한 `(...)` 출력을 반환하지 않는다.

Continuous Q는 features와 actions의 batch 차원이 같아야 한다. Network는
features 다음에 actions를 연결한 하나의 Tensor를 받는다. `hidden_dim=128`은
기본 Network에만 사용하며 custom Network에서는 사용하지 않는다.
Q(s,a)는 행동 입력이 필요하므로 `evaluate(features, actions)` 또는
`Model.evaluate_q(observation, actions)`로 계산한다. `forward(features)`는
사용할 수 없으며 모든 Estimator에 `evaluate()`를 강제하지 않는다.

## 기본 Network와 Custom MLP

```python
import numpy as np
import torch
from torch import nn
from modular_rl.component.estimator import DiscreteQEstimator
from modular_rl.component.representation import MLPRepresentation
from modular_rl.component.action_selector import GreedyActionSelector
from modular_rl.model import Model

default_q = DiscreteQEstimator(feature_dim=64, n_actions=4)
my_network = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 4))
custom_q = DiscreteQEstimator(feature_dim=64, n_actions=4, network=my_network)

model = Model(
    representation=MLPRepresentation(4, hidden_dims=(64,)),
    estimator=custom_q,
    action_selector=GreedyActionSelector(),
)
assert model.forward(np.zeros((8, 4), dtype=np.float32)).values.q[0].shape == (8, 4)
assert model.act(np.zeros(4, dtype=np.float32)).action.shape == ()
```

같은 입력·출력 계약을 지키는 임의의 `nn.Module`을 사용할 수 있다.
custom Network의 내부 hidden layer나 activation은 Estimator가 해석하지 않는다.
Representation은 공유 features를 만들고, 주입한 head는 추정 대상에 맞는
함수를 계산한다.

## Gaussian Mean Network

```python
from modular_rl.component.estimator import GaussianPolicyEstimator

mean_network = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 2))
gaussian = GaussianPolicyEstimator(64, 2, log_std_init=-0.5, network=mean_network)
policy = gaussian(torch.zeros(8, 64)).policy
assert policy.mean.shape == (8, 2)
assert policy.log_prob(torch.zeros(8, 2)).shape == (8,)
```

교체하는 Network는 mean만 반환한다. 상태와 독립적인 행동 차원별 `log_std`
Parameter는 Estimator가 유지한다. `Independent(Normal, 1)`가 행동 차원의
log probability를 합산한다. 상태 의존적 표준편차, action bound와 SAC용
tanh-squashed Distribution은 별도 구현 범위다. Estimator가 행동을 Sampling하지
않으며, 현재 CategoricalActionSelector는 이 Gaussian을 처리하지 않는다.

`log_std_init`은 유한한 값이어야 한다. 학습이나 checkpoint 복원으로 변경된
`log_std`도 실제 Distribution을 만들기 전에 행동 차원별 scale만 점검한다.
`variance = exp(log_std)**2`는 사용 중인 dtype에서 다음 범위여야 한다.

```text
torch.finfo(dtype).tiny <= variance <= torch.finfo(dtype).max / 2
```

하한은 subnormal variance의 역수에서 gradient가 overflow하는 것을 피하고,
상한은 PyTorch Normal.log_prob의 `2 * variance`가 overflow하는 것을 피한다.
범위를 벗어나거나 NaN/Inf인 값은 dtype와 조정 방법을 담은 ValueError로
실패시킨다. 임의 clamp, Parameter 변경 또는 자동 dtype 변환은 수행하지 않는다.
검사는 상태와 독립적인 작은 행동 벡터에만 적용하고 batch 전체를 순회하지 않는다.
같은 `log_std`라도 float32와 float64의 표현 범위가 다르므로 극단값의 허용 여부는
현재 Model dtype에 따라 달라진다. 이 검사는 큰 mean-action 차이 등 모든 가능한
입력에서 유한한 log probability를 보장하는 것은 아니다.

향후 확장 사항은 상태 의존적 `log_std`, action-bound 처리, SAC용 Squashed
Gaussian과 log probability 보정, 호환되는 연속 ActionSelector다. 기본 Gaussian의
현재 상태 독립적 `log_std`와 Mean Network 주입 계약은 유지한다.

## Continuous Q Network

```python
from modular_rl.component.estimator import ContinuousQEstimator
from modular_rl.component.representation import IdentityRepresentation

critic_network = nn.Sequential(nn.Linear(66, 64), nn.ReLU(), nn.Linear(64, 1))
critic = ContinuousQEstimator(64, 2, network=critic_network)
critic_model = Model(IdentityRepresentation(), critic)
q = critic_model.evaluate_q(
    np.zeros((8, 64), dtype=np.float32),
    np.zeros((8, 2), dtype=np.float32),
).values.q[0]
assert q.shape == (8,)
```

## Actor-Critic 조합

```python
from modular_rl.component.estimator import ActorCriticEstimator, CategoricalPolicyEstimator, StateValueEstimator
from modular_rl.component.action_selector import CategoricalActionSelector

actor_network = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 3))
value_network = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
actor_critic = ActorCriticEstimator(
    CategoricalPolicyEstimator(64, 3, network=actor_network),
    StateValueEstimator(64, network=value_network),
)
actor_critic_model = Model(
    MLPRepresentation(4, hidden_dims=(64,)), actor_critic, CategoricalActionSelector()
)
estimate = actor_critic_model.forward(np.zeros((8, 4), dtype=np.float32))
assert estimate.values.v.shape == (8,)
assert estimate.policy.probs.shape == (8, 3)
```

Wrapper는 두 Estimator에 같은 features를 전달하고 Policy와 전체 ValueOutput을
그대로 결합한다. Representation과 학습 절차를 중복 구현하지 않는다.
행동 조건부 critic은 Wrapper의 `forward(features)`와 별도로 평가한다.

현재 ActorCriticEstimator에 ContinuousQEstimator를 value child로 직접 연결하면
`forward(features)`에 행동이 없어서 NotImplementedError가 발생한다. Wrapper에는
`evaluate(features, actions)`도 없으므로 이 Wrapper를 감싼 Model.evaluate_q()가
그 critic으로 자동 전달하지 않는다. Q(s,a)를 Q(s,·)처럼 계산하거나 forward
계약에 행동을 추가하지 않는다.

현재 해결 방법은 Policy Model과 Continuous Q Model을 별도로 구성하고 Policy는
`forward()`, Critic은 `evaluate_q()`로 호출하는 것이다. 필요하면 같은 Representation
인스턴스를 공유할 수 있다. 공유 Parameter는 호출자가 Optimizer 등에 중복 등록하지
않도록 관리한다. 아래처럼 호출자가 준비한 행동 Tensor를 Critic에 전달한다.

```python
policy_model = Model(
    IdentityRepresentation(),
    GaussianPolicyEstimator(64, 2, network=mean_network),
)
q_model = Model(IdentityRepresentation(), ContinuousQEstimator(64, 2, network=critic_network))
observations = np.zeros((8, 64), dtype=np.float32)
policy_output = policy_model.forward(observations)
provided_actions = torch.zeros(8, 2)
q_output = q_model.evaluate_q(observations, provided_actions)
assert policy_output.policy.event_shape == (2,)
assert q_output.values.q[0].shape == (8,)
```

행동 조건부 Actor-Critic 조합은 향후 별도 확장 사항이다. 현재 Wrapper, Model,
Base Class의 공개 인터페이스는 유지한다.

## 등록, 저장과 검증

`self.network`에 직접 할당한 모듈은 PyTorch의 자식 모듈로 등록된다.
custom Network의 Parameter와 buffer는 Model의 `parameters()`, `state_dict()`,
`to(device)`, `double()` 등에 포함된다. Gaussian의 `log_std`도 함께 저장·이동한다.
Estimator가 모듈을 임의로 다른 device나 dtype으로 옮기지 않는다.

기존 생성자의 위치 인자와 기본 Network checkpoint 키는 유지한다.
Gaussian의 세 번째 인자는 `log_std_init`, Continuous Q의 세 번째 인자는
`hidden_dim`이며 `network`는 keyword로 전달한다. custom checkpoint를 복원할
때는 같은 구조의 Network를 만들어 `load_state_dict()`에 전달한다.
`state_dict()`는 Network 구조 자체를 저장하지 않는다.

생성 시 `network`가 `nn.Module`인지 한 번 검사하며, dummy 입력으로 Network를
실행하지 않는다. 실제 계산 뒤 `_validate_output()`에서 Tensor와 정확한 출력
shape만 확인해 잘못된 행동 차원이나 batch 축의 소실을 명확히 실패시킨다.
V/Q의 마지막 singleton axis도 확인한다. Gaussian의 작은 scale 벡터 외에 관측과
Network 출력 전체의 finite 검사, dtype/layout 열거 검사, 입력 변환, 중복 Network
실행은 수행하지 않는다. Network 입력의
feature 크기 오류는 일반적으로 Network 연산에서 보고한다. Continuous Q의
기존 state-action 입력 검증은 `_validate()`에서 유지한다.

검증 명령은 프로젝트 루트에서 `python -m pytest -q`와
`python -m examples.quickstart`다. CPU에서 dtype, gradient, checkpoint와
Model/ActionSelector 연결을 검증하며, CUDA는 사용 가능한 환경에서 실행한다.

## n-step Q(σ) 컴포넌트 조합 검증

`examples/gridworld_q_sigma.py`는 기존 컴포넌트로 고정 정책의 Qπ를 추정하는 예제다.
`modular_rl/`와 기존 API, Transition 필드, Learner 클래스는 변경하지 않는다.
기준은 Sutton & Barto 2판 §7.6 본문 식 (7.17), 인쇄 쪽수 155–156이다.
[기준 PDF](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf#page=177).

### 기준 수식과 본문/의사코드의 차이

```text
Vbar_k = sum_a pi(a|S_k) Q(S_k,a)
rho_k  = pi(A_k|S_k) / b(A_k|S_k)
c_k    = sigma_k * rho_k + (1-sigma_k) * pi(A_k|S_k)
G_j:h  = R_(j+1) + gamma * [Vbar_(j+1) + c_(j+1) * (G_(j+1):h - Q(S_(j+1),A_(j+1)))]
Q(S_t,A_t) += alpha * (G_t:h - Q(S_t,A_t))
```

Q와 π는 한 호출 안에서 업데이트 전 동일 Snapshot이다. σ·Behavior log b는 실제 행동
선택 시 결정한 값이다. 현재 head의 σ·ρ를 적용하지 않고 실제 중간 행동만 보정한다.
Control Variate는 중간 `Vbar-Q(sampled)` 항을 보존한다. σ=1을 일반적인 sampled SARSA와
표본별로 같다고 가정하거나, σ=0/1 최종 Target을 단순 선형보간하지 않는다.

비종료 Horizon의 본문 경계는 `G_h:h = Q(S_h,A_h)`다. 따라서 마지막 residual은 0이며
`G_(h-1):h = R_h + gamma*Vbar_h`로 시작한다. 끝점 행동·log b·σ_h는 필요 없다.
자연 종료는 `G_(T-1):T = R_T`다. 일반적 Time-limit은 실제 final observation의 기대값으로
Bootstrap한다. 두 flag가 True면 terminated가 우선한다. Boundary flush는 step/act를 호출하지 않는다.

같은 페이지 의사코드는 `G=0`으로 시작하여 본문의 비종료 경계와 다른 결과를 낸다.
이번 구현은 사용자와 확정한 **본문 정의**를 선택한다. 공식 정오표에서 확인하지 못했으므로
공식적으로 인정된 오류라고 단정하지 않는다. 사용자 n=3 예제에서 본문 방식의
Target/Qnew는 3.303325/2.6516625지만, 의사코드 초기화를 문자 그대로 적용하면
1.437540625/1.7187703125다. σ=0의 Tree Backup 동등성도 후자는 보존하지 않는다.

본문은 Return 뒤에 일반 Off-policy 업데이트 (7.11)을 참조한다. (7.11)의 전체 비율을
이미 `c_k`로 보정한 (7.17)에 다시 곱하면 선택한 정의에서 중복 보정이 된다.
상자 의사코드의 직접 `alpha*(G-Q)` 및 이번에 승인한 계약대로 별도 전체 IS 비율을 곱하지 않는다.
이 업데이트 해석과 Horizon 초기화 선택을 명시하며, 본문·상자를 완전히 동일하다고 기술하지 않는다.

### Control Variate를 쓰지 않는 On-policy 표현과의 구분

[De Asis 외, AAAI 2018의 식 (13–14)](https://cdn.aaai.org/ojs/11631/11631-13-15159-1-2-20201228.pdf#page=4)는
`delta_sigma = R + gamma*[sigma*Q(sampled) + (1-sigma)*Vbar] - Q(current)`를 사용한다.
동일 Snapshot의 On-policy 재귀로 표현하면 `c_on=sigma+(1-sigma)*pi(sampled)`와
`C_alt=sigma*Q(sampled)+(1-sigma)*Vbar+c_on*(G-Q(sampled))`다.
이번에 선택한 본문 정의의 `C_cv=Vbar+c_on*(G-Q(sampled))`와는
`sigma*(Vbar-Q(sampled))`만큼 다르다. 대체 정의는 끝점에도 sampled/expected 혼합을
사용하여 sigma=1에서 일반 Sampled SARSA를 얻지만, 이번 본문 경계는 끝점이 항상 기대값이다.
이를 별도의 구현 모드로 추가하지 않는다. 해당 논문의 Off-policy 보정 방식을 이번 식 (7.17)에
옮겨 섞지도 않는다. 여러 Q(σ) 정의의 이름이 같다는 이유로 표본별 동등성을 주장하지 않는다.

### 공개 예제 함수와 데이터 계약

- `learn_q_sigma(model, window, target_generator, update_rule, *, target_policy, sigma)`:
  연속된 owned B=1 TransitionBatch Window에서 head 한 행만 갱신한다.
  반환 dict의 `target`, signed `td_error`는 graph 없는 `(1,)`의 table dtype/device Tensor다.
  `importance_ratios`, `coefficients`는 실제 중간 시점의 순서대로 float64 `(m-1,)`다.
  빈 중간 구간(n=1)에는 빈 통계 Tensor를 반환한다. Sampling은 수행하지 않는다.
- `sigma`는 고정 Scalar 또는 `[σ_(t+1),…,σ_(t+m)]`의 m개 frozen 값이다.
  마지막 값은 유효 범위를 검사하는 placeholder로 Return에 영향을 주지 않는다.
  Callback은 learn에 전달하지 않는다. 고정 σ=0은 Tree의 π=0/1 shortcut과 같은 계산을 사용한다.
- `target_policy`는 고정 floating `(S,A)` Tensor 또는 기존 discrete Selector다.
  Fixed Policy는 table device여야 하며 Tree와 동일하게 필요한 행을 Q dtype으로 변환한다.
  Dynamic `probabilities(estimate)`는 Return의 Q와 같은 단일 Model estimate를 읽으며 RNG를 소비하지 않는다.
- `run_episode(env, model, target_generator, update_rule, *, target_policy, sigma=.5, n=3, seed=None)`:
  reset/step 기반 Online 수집·n-step scheduling·boundary flush를 수행한다.
  σ는 Scalar, 반복 시간 패턴 또는 `callback(state, action, timestep)`이다.
  패턴의 timestep은 Episode마다 0부터 시작한다. 다음 행동은 eligible head 갱신 뒤 선택한다.
- `train_and_evaluate(...)`는 Uniform Behavior와 matched/different 고정 Target으로 DP Qπ RMSE,
  Reward/Length, 누적 Step=Update, 종료 flag, frozen σ 및 c·ρ 통계를 반환한다.
  `run_comparison(..., reference_results=None)`은 기존 SARSA/Tree 실험 함수를 재사용한다.
  Notebook에서는 이미 실행한 동일 설정 결과를 전달해 비교 실험을 중복 실행하지 않는다.
  전달한 결과는 seed/budget/n/gamma/alpha를 검사하며 환경 TimeLimit 설정도 호출자가 일치시킨다.

### Snapshot, Sampling-time 메타데이터와 수치 안전성

Window/인덱스/경계 검증은 원래 n-step SARSA 함수, 정책 및 log b 검증은 기존 Off-policy 예제,
확률 조회·Q 갱신 사전 검사는 Tree Backup의 작은 함수를 재사용한다. 새 예제는 일반 Python
import를 사용한다. 이전 Tree/Off-policy 파일 내부의 runpy 로딩은 변경하지 않으며 새 runpy
체인을 추가하지 않는다. Notebook은 예제 디렉터리를 표준 namespace package로 한 번 등록한다.
설치 패키지를 늘리거나 sys.path를 수정하지 않는다.

각 필요한 next state의 Q와 π를 한 번에 평가한다. Tabular lookup의 advanced-index 결과는
테이블과 독립인 Tensor이며, 동적 Selector도 같은 estimate에서 확률을 구한다.
Return을 모두 계산하고 수치를 검사한 뒤 원본 table의 head를 기존 Rule로 한 번 갱신한다.
반복 상태·행동이 들어와도 현재 호출은 한 Snapshot이고, 후속 flush 호출은 앞선 갱신 후 Q를 읽는다.

기존 TransitionBatch의 log_prob에 실제 sampling-time `log b`를 저장한다. 현재 epsilon/Q로
과거 확률을 재계산하지 않는다. `.to(dtype=table.dtype)`가 log b도 변환할 수 있으므로 IS에는
원본 record의 precision을 유지한 값을 float64로 옮겨 사용한다. π=0이면 rho=0으로 처리해
`0*inf`를 만들지 않고, 기대값은 남겨 해당 sampled residual 분기만 끊는다.
Coverage `π>0 ⇒ b>0`는 호출자 책임이며, 관측한 행동의 log b만으로 미관측 행동의 support는
확인할 수 없다. Uniform Behavior 실험은 모든 행동에 b=1/4를 보장한다.

σ의 finite [0,1], 로그의 finite ≤0, 정책 확률/정규화와 Shape·인덱스를 검사한다.
양수 ratio overflow/underflow, table dtype의 coefficient 표현 불가, correction/update 곱의
underflow, 비유한 Snapshot/Return/TD error/제안 Q는 **쓰기 전에** 실패한다. Clipping하지 않는다.
실패한 호출의 head/window는 pop하지 않으며 이전 성공한 Online 호출까지 되돌리지는 않는다.
본 실험은 학습계산의 graph를 보존하지 않는다. core TargetGenerator의 gradient 계약은 그대로다.

### 결정적인 수치 검증

| 조건 | Target | Qnew (현재 Q=2, α=.5) |
|---|---:|---:|
| n=1, 다음 Q=[10,2], π=[.25,.75], γ=.9, r=1, 모든 σ | 4.6 | 3.3 |
| n=3, σ=[.5,.25,.75], 아래의 중간 행동·확률 | 3.303325 | 2.6516625 |
| 같은 n=3, 고정 σ=0 (Tree Backup) | 3.8206 | 2.9103 |
| 같은 n=3, 고정 σ=.5 | 3.17575 | 2.587875 |
| 같은 n=3, 고정 σ=1 (Control Variate) | 2.3608 | 2.1804 |
| 같은 n=3, 일반 Sampled SARSA, A3=0 | 9.604 | 5.802 |

n=3은 rewards=[1,2,3], S1 Q=[10,2]/π=[.25,.75]/A1=0/b=.5,
S2 Q=[4,8]/π=[.6,.4]/A2=1/b=.25, S3 Q=[6,2]/π=[.5,.5]다.
시점별 σ의 rho=[.5,1.6], c=[.375,.7], 역순 G=[6.6,6.158,3.303325]를 독립 검증한다.
σ_h를 0/.25/.75/1로 바꿔도 마지막 residual=0이므로 결과는 같다.
고정 σ=.5 Target은 σ=0/1 Target 평균 3.0907과 다르다.

σ=1에서도 π=b의 중간 `Vbar-Q(sampled)`가 일반적으로 0은 아니다. Deterministic π=b가
실제 경로를 따르는 경우나 관련 Q가 행동에 무관한 경우 등에는 Sampled Return과 일치한다.
일반 확률적 정책의 표본별 동등성은 요구하지 않는다. n=1의 Sampled Target은 선택한 마지막
행동 Q를 쓰지만 본문 정의는 기대값이므로 두 Target은 첫 단계부터 다를 수 있다.

### Checkpoint와 범용성 평가

1. Model/TabularQEstimator/기존 EpsilonGreedySelector는 그대로 재사용한다.
2. Transition/TransitionBatch의 필드와 공개 API를 변경하지 않는다.
3. TD0TargetGenerator는 각 `reward + gamma*continuation`을, Q Rule은 직접 table 갱신을 담당한다.
4. 시점별 σ는 예제의 `pending_sigmas` deque에 현재 observation/action과 정렬해 저장한다.
   learn에는 tail σ와 무효 영향의 Horizon placeholder를 전달한다. provider는 선택 시 한 번만 호출한다.
5. Q·π는 단일 Model estimate에서 같은 시점의 Snapshot으로 계산한다.
6. sampling-time 행동 log b는 기존 record, 현재 아직 실행하지 않은 action/log b/σ는 별도 수집 상태다.
7. Online scheduling/flush의 핵심은 On-policy 약 30줄, Off-policy 약 40줄, Tree 약 30줄과
   Q(σ)의 약 45줄에서 반복된다. Q(σ)는 aligned σ와 branch 통계가 추가된다.
   서로 다른 next-action 선택 시점과 endpoint 계약을 숨기지 않아 수집 루프를 강제로 공유하지 않았다.
8. 기존 Learner의 고정 max/gather 경로에는 중간 Q·π·σ를 쓰는 재귀를 외부에서 주입할 수 없다.
   기존 컴포넌트 출력은 이 계산을 표현할 수 있지만 시간 순서·Snapshot·σ 책임은 예제에 남는다.
9. TargetGenerator의 reward 수식과 UpdateRule의 table 갱신은 잘 분리돼 있다.
   여러 시점의 continuation/IS/CV 계산을 받을 공통 Return 인터페이스는 현재 없다.
10. 최소 공통화 후보는 owned ordered window, 실제 horizon/boundary, Online eligible-head/flush,
    sampling-time action metadata, target probability provider, frozen Q snapshot, Return→Rule 연결이다.
    endpoint callback 하나만으로 일반 재귀를 표현할 수 없다. 알고리즘별 재귀/importance 범위를
    거대한 조건문에 합칠 필요도 없다. 이번 Commit에서는 해당 공통화와 기존 예제 리팩터링을 하지 않는다.
11. 재개에는 Model.state_dict, 고정 π 또는 Selector 구성/ε, pending records, aligned σ,
    현재 선택 action/log b/σ, provider 설정, n/γ/α, Torch RNG 및 환경 위치/RNG/TimeLimit 잔여 예산을
    함께 저장한다. 고정 stream의 pending·σ·선택 행동 복원 후 후속 TD error/Q/RNG 일치를 테스트한다.
    `run_episode` 편의 함수에는 중간 Checkpoint를 내보내거나 복원하는 API가 없다. 중간 재개가
    필요하면 호출자가 직접 reset/step과 learn을 조합해 이 상태를 소유해야 한다. 테스트는 그
    명시적 Window 경로의 재개를 검증하며, 임의 Gymnasium 환경의 자동 복원을 보장하지 않는다.
12. 향후 WorldModel 신뢰도로 σ를 조절하려면 행동 선택 시점의 상태·행동·시간과 함께 모델 예측의
    불확실성/신뢰도 및 provider 설정을 명시적으로 제공해야 한다. Callback의 closure로 참조할 수
    있지만 선택 시 계산한 σ를 저장한다. 학습 시 갱신된 WorldModel로 과거 σ를 다시 정하지 않는다.

```bash
python -m examples.gridworld_q_sigma --episodes 200 --seeds 7 11 19 --ns 1 3 5
```

CLI는 γ=.9/α=.05와 고정 σ=0/.25/.5/.75/1 및 시간 패턴[0,1]을 비교한다.
Notebook §17은 n=3의 같은 6개 설정과 3개 Seed를 실행하며 §16의 비교 결과를 재사용한다.
각 Target 자체의 DP Qπ를 기준으로 비종료 상태·모든 행동 RMSE를 기록한다. 통계는 update별
중간 branch에 대한 것이므로 같은 실제 행동이 서로 다른 Window에서 반복 집계될 수 있다.
Uniform Behavior에서 자연 종료 경험은 모두 같고 Q(σ)/Tree는 Truncation에도 같은 수집 순서다.
SARSA는 Truncation에서 별도 endpoint action을 선택하므로 그 이후의 RNG/경험 일치는 강제하지 않는다.
성능 순위나 단일 Seed의 RMSE 감소를 테스트 합격 조건으로 삼지 않는다.

### Gridworld 실행 결과 (γ=.9, α=.05, 각 200 Episode)

153개 run(기존 comparator 45개 + Q(σ) 108개), 총 30,600 Episode와 1,305,855 Step=Update를 실행했다. 모든 Episode가 자연 종료했고 Truncation은 0이다. Seed7/11/19의 각 run Step은 8,419/8,412/8,774다. σ=0의 Q/Reward/Length/Step은 Tree와 정확히 같으며, n=1에서는 모든 σ의 Q가 같다.

아래는 π≠b의 DP Qπ RMSE다. [0,1]은 실제 선택 시각의 반복 σ 패턴이며 마지막 Horizon σ는 해당 Return에서 사용하지 않는다.

| n | σ | Seed7 | Seed11 | Seed19 | 평균±표준편차 |
|---:|---|---:|---:|---:|---:|
| 1 | 0.0 | 0.088023 | 0.092793 | 0.119835 | 0.100217±0.014008 |
| 1 | 0.25 | 0.088023 | 0.092793 | 0.119835 | 0.100217±0.014008 |
| 1 | 0.5 | 0.088023 | 0.092793 | 0.119835 | 0.100217±0.014008 |
| 1 | 0.75 | 0.088023 | 0.092793 | 0.119835 | 0.100217±0.014008 |
| 1 | 1.0 | 0.088023 | 0.092793 | 0.119835 | 0.100217±0.014008 |
| 1 | [0,1] | 0.088023 | 0.092793 | 0.119835 | 0.100217±0.014008 |
| 3 | 0.0 | 0.061638 | 0.066914 | 0.090675 | 0.073076±0.012630 |
| 3 | 0.25 | 0.044088 | 0.049285 | 0.069645 | 0.054340±0.011029 |
| 3 | 0.5 | 0.029966 | 0.034738 | 0.051111 | 0.038605±0.009055 |
| 3 | 0.75 | 0.019562 | 0.023749 | 0.035969 | 0.026427±0.006961 |
| 3 | 1.0 | 0.012501 | 0.016183 | 0.024526 | 0.017737±0.005030 |
| 3 | [0,1] | 0.032931 | 0.040662 | 0.056554 | 0.043382±0.009834 |
| 5 | 0.0 | 0.060784 | 0.066053 | 0.089769 | 0.072202±0.012607 |
| 5 | 0.25 | 0.040596 | 0.045683 | 0.065585 | 0.050621±0.010783 |
| 5 | 0.5 | 0.023090 | 0.027462 | 0.042134 | 0.030895±0.008145 |
| 5 | 0.75 | 0.010723 | 0.014219 | 0.022732 | 0.015891±0.005043 |
| 5 | 1.0 | 0.004412 | 0.007531 | 0.010218 | 0.007387±0.002372 |
| 5 | [0,1] | 0.028358 | 0.035869 | 0.050506 | 0.038244±0.009196 |

같은 정책의 비교 결과는 다음과 같다. SARSA/Tree는 기존 함수를 다시 실행했으며 Q(σ)는 중간 Control Variate와 마지막 기대값 때문에 σ=1에도 Target 정의가 다르다.

| n | Target | Off-policy sampled SARSA | Tree Backup / σ=0 | Q(σ=.5) | Q(σ=1) |
|---:|---|---:|---:|---:|---:|
| 1 | matched | 0.102381±0.021784 | 0.107838±0.012094 | 0.107838±0.012094 | 0.107838±0.012094 |
| 1 | different | 0.098298±0.021572 | 0.100217±0.014008 | 0.100217±0.014008 | 0.100217±0.014008 |
| 3 | matched | 0.085317±0.024704 | 0.081529±0.011077 | 0.045665±0.008253 | 0.021788±0.004861 |
| 3 | different | 0.114197±0.038899 | 0.073076±0.012630 | 0.038605±0.009055 | 0.017737±0.005030 |
| 5 | matched | 0.109120±0.027522 | 0.080607±0.011066 | 0.036794±0.007596 | 0.008664±0.002633 |
| 5 | different | 0.151878±0.044406 | 0.072202±0.012607 | 0.030895±0.008145 | 0.007387±0.002372 |

이 설정에서는 n=3/5에서 σ=1의 평균 RMSE가 작았지만 성능 보장은 아니다. n=1은 모든 σ에서 같고 Sampled SARSA와의 성능 순위도 한 방향이 아니다.
Branch 통계의 matched rho는 1, different rho는 .8/1.2이며 실제 방문 빈도에 따라 평균이 달라진다. c는 σ=0에서 π, σ=1에서 rho다. n=1에는 중간 Branch가 없으므로 통계 Tensor는 비어 있다. 고정/시점별/Callback σ, Dynamic Selector, 수치 실패, Checkpoint와 CPU/float32·64는 결정적인 테스트로 별도 검증한다.

### 최종 회귀 검증

전체 pytest는 **1,823 passed, 55 skipped(CUDA 미지원)**이며 실행 시간은 43분 57초다.
신규 Q(σ) 테스트는 130 passed/2 skipped이고 기존 예제 15개 Smoke Test를 통과했다.
세 Notebook을 각각 같은 Kernel에서 Run All 두 번 실행해 텍스트·수치·PNG가 일치했다.
기존 출력 payload도 이전 commit과 같고, 새 그림 2개를 포함한 PNG는 총 34개다.
06 Notebook은 106개 셀/코드 60개/PNG 19개이며 기존 87개 셀·출력·메타데이터와
루트 metadata를 모두 그대로 보존했다. 기존 라이브러리 61개 Python 파일의 byte hash가 같다.

초기 전체 검증에서는 새 Notebook의 Sampled SARSA 비교 참조값 한 곳이 실패했다.
이번 Window의 끝점 Q=6인 Target은 9.604인데 다른 예제의 8.146을 옮긴 것이 원인이었다.
참조값·문서를 정정하고 결정적인 숫자 비교 테스트를 추가한 뒤 전체 검증을 다시 통과했다.
학습 알고리즘을 참조값에 맞춰 바꾸지 않았다.

## Tabular Dyna-Q 컴포넌트 조합 검증

이번 구현은 `examples/gridworld_dyna_q.py`의 작은 World Model과 함수 조합이다.
`modular_rl/`의 공개 API·구현·Transition/Batch를 바꾸지 않고 DynaLearner/Agent/Runner를
추가하지 않았다. 기존 설명의 WorldModel/Planner 미포함은 라이브러리 본체 기준이다.
참고는 [Sutton & Barto 2판 Chapter 8, §8.2, pp.161–164](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)다.

### 두 종류 Model과 공개 함수

| 예제 API | 책임 |
|---|---|
| `TabularWorldModel().observe(transition)` | 관측한 (S,A)의 최신 R/S'/terminated 예측을 대체 |
| `world.sample(rng) -> Transition` | 관측된 쌍 균등 추출 후 owned 가상 전이 생성 |
| `world.reset()`, `len(world)` | 예측 초기화와 관측된 고유 쌍 수 |
| `world.state_dict()`, `world.load_state_dict(state)` | 삽입 순서와 owned 예측 저장·복원 |
| `plan(world, learner, *, steps, rng) -> Tensor` | 환경 없이 1-step Planning; 실제 갱신 수 길이의 signed TD Error |
| `learn_real_n_step(learner, window) -> Tensor` | 실제 연속 B=1 Window의 최오래된 행 1개 갱신; error (1,) |
| `run_steps(env, real_learner, *, steps, world_model=None, planning_learner=None, planning_steps=0, rng=None, n=1, seed=None)` | 고정 실제 Step 예산의 명시적 online 루프·통계 |
| `train_and_evaluate(...)`, `comparison(steps=2000, seeds=(7,11,19))` | 별도 Greedy 환경·DP 기준과 예산/Seed 비교 |

Q 추정 `Model`은 V/Q/정책 출력의 기존 façade다. World Model은 환경 동역학의 예측이다.
알려진 `TabularMDP`를 읽어 World Model을 채우지 않으며 실제 관측만 `observe`한다.
DP kernel은 실험 후 관측된 쌍의 예측 정확성 검증과 V* 참조 계산에만 사용한다.

### World Model과 ReplayBuffer의 차이

이 예제의 World Model은 `(state, action) -> 최신 Transition 결과` dictionary다.
상태·행동은 정수 scalar, reward는 원래 floating dtype의 graph 없는 owned CPU scalar Tensor,
다음 상태는 정수 scalar, terminated는 bool이다. 보상은 저장과 반환 시 복사한다.
외부 `truncated`는 학습된 환경 종료가 아니므로 예측은 항상 `truncated=False`이고 log_prob은 None이다.
두 종료 flag가 True인 실제 전이는 terminated=True를 보존한다. NaN/Inf reward와 잘못된
scalar/index/flag 계약은 저장 전에 거절하며 잘못된 checkpoint는 기존 모델을 부분 대체하지 않는다.

별도 NumPy Generator로 **관측된 (S,A) 쌍**을 균등 추출한다. 교재 p.164 의사코드의
상태 균등 → 그 상태에서 관측한 행동 균등과는 상태별 관측 행동 수가 다르면 분포가 다르다.
이번 요청의 쌍 균등 규칙을 따른다. 미관측 쌍의 결과는 만들지 않는다.
빈 모델의 직접 sample은 ValueError이고 plan은 0개 error를 반환하며 RNG도 사용하지 않는다.

ReplayBuffer는 개별 경험의 ring/history라서 같은 쌍의 과거 결과가 여러 개 남고,
관측 빈도에 비례한 샘플 분포가 생긴다. Dataset/모델 학습 데이터 저장에는 재사용 가능하지만
쌍별 최신 예측과 생성 모델 자체를 대신하지 않는다. 결정적 최신 결과 모델에서는 가상 전이의
값이 마지막 실경험과 같은 것이 정상이다. 계획은 이 모델의 예측을 샘플링하며 과거 history를
ReplayBuffer에서 재생하는 루프가 아니다. 확률적 환경에는 Outcome 빈도/확률 또는 학습 분포가
추가로 필요하다. 이번 모델이 확률적 전이의 정확한 분포를 추정한다고 주장하지 않는다.

### Real / Planning 학습 흐름과 갱신 순서

1. Model.act → env.step → 실제 Transition/owned B=1 Batch.
2. 기본 n=1은 기존 TabularQLearner.learn으로 직접 Q 갱신.
3. World Model.observe로 실제 결과 반영.
4. `plan`은 model.sample → 기존 learner.learn을 예산만큼 실행.
5. 다음 행동은 모든 갱신 후 선택. 자연 종료/Truncation에서 reset.

두 경로 모두 기존 TargetGenerator와 UpdateRule을 통해 계산하며 Q 수식을 복제하지 않는다.
Planning은 환경·행동 Selector·실제 Window를 참조하지 않고 observe도 호출하지 않는다.
정책 RNG는 기존 PyTorch RNG, 환경 RNG는 Gymnasium np_random, Planning RNG는 별도
NumPy Generator다. plan의 max Q 평가와 학습은 행동 RNG를 소비하지 않는다.
Planning을 늘리면 Q가 달라져 다음 행동/환경 궤적이 바뀔 수 있으며 RNG 독립성 위반이 아니다.

동일 Q Model을 가진 별도 TabularQLearner를 전달하면 TargetGenerator와 UpdateRule 설정도
독립적으로 줄 수 있다. 동일 Model 공유를 run_steps에서 확인한다. 예제 기본은 한 Learner를
두 경로에서 공유한다. 장기적 독립 Target 전략이 필요하더라도 Model/Buffer API 변경은 필요 없다.

Planning=0의 순서는 기존 `gridworld_sarsa.py`의 Q-learning 분기와 같다. 기존 예제는
Episode 예산, 이번 루프는 정확한 실제 Step 예산이라 중단 조건만 다르다. 독립적인 기준
루프로 같은 초기 Q/Seed/γ/α/ε·환경 제한에서 행동·reward·length·Target·TD Error·Q·Torch RNG를
bit-exact 비교한다. World Model 관측만으로 행동 또는 Planning RNG는 소비하지 않는다.

### Real n-step, Planning 1-step과 경계

Real Window에는 실제 연속 Transition.to_batch만 넣는다. n=1은 기존 QLearner에 위임한다.
m>1은 마지막 관측의 max Q에서 시작해 기존 TD0.generate를 보상 역순으로 합성하여
`sum(gamma**k * reward[k]) + gamma**m * max Q(last_next_state)`를 만들고 기존 Q Rule로
head만 갱신한다. 자연 종료는 TD0의 terminated mask로 Bootstrap을 제거한다.
SARSA의 실제 endpoint 행동이나 Q(σ)의 Target 정책을 섞지 않는다.
기존 n-step SARSA 예제의 Window 형식·연속성 및 n 검증 함수만 재사용한다.

매 Step에 eligible head 1개를 처리한 뒤 모델을 관측하고 Planning을 실행한다.
n=3의 처음 두 Step에도 observe/Planning은 수행하지만 아직 Real 업데이트는 없다.
Episode 경계에서는 **그 Step의 Planning 후** 잔여 Real Window를 순서대로 Flush한다.
각 flush는 당시 Q로 Target을 고정하고 Q를 갱신하며 다음 flush가 그 변경을 볼 수 있다.
Flush는 env.step/act/observe/plan을 추가 호출하지 않는다. 따라서 실제 Step 100개는 Real 갱신
100개, 모델 관측 100개, Planning=5이면 가상 갱신 500개다. n은 이 개수를 바꾸지 않는다.

전역 실제 Step 예산이 중간 Episode에서 끝나면 마지막 전이에 외부 cutoff를 표시해
최종 관측의 max Q로 Bootstrap하고 Window를 전부 Flush한다. env가 보고한 원래 flag와
budget_cutoff를 trace에서 구별하고 부분 Episode의 reward/length는 완료 Episode 배열과
분리한다. 자동 reset의 실제 final observation을 추론하지 않는다.
모든 Target은 해당 head의 쓰기 전에 계산하며 여러 head/Planning 호출은 각각 현재 Q를 읽는다.

### 수치·독립 실행·Checkpoint 검증

Q(0,1)=2, 다음 Q=[4,6], r=1, γ=.9, α=.5에서 Real Target=6.4,
error=4.4, 새 Q=4.2다. 모델이 같은 전이를 예측하면 Planning error=2.2, 새 Q=5.3이다.
종료 r=3이면 큰 다음 Q에도 Target=3, 새 Q=2.5다. truncated-only는 Target=6.4를 유지한다.
Planning 예산 0/1/5/10, 중복 쌍 순서, dtype/device와 in-place/no-graph 계약을 검증한다.

Offline Transition을 world.observe로 사전학습해도 Q는 바뀌지 않는다. 이후 같은 쌍의 새 결과로
온라인 대체가 가능하다. Idle Planning은 `plan(..., steps=k, rng=rng)`를 반복 호출하면 된다.
실제 환경 state/RNG/elapsed_steps·World Model·행동 RNG를 보존하면서 Q와 Planning RNG만
변할 수 있다. 중단/재개는 호출 사이에서 수행한다. wall-clock/async/sleep Scheduler는 없다.

재개 시 Q Model.state_dict, World Model.state_dict, Planning bit_generator.state,
Torch RNG(사용 시 CUDA RNG), γ/α/ε/n/Planning 예산과 Pending real Window를 함께 저장한다.
World Model 삽입 순서는 같은 RNG 추출의 재현성에 필요하다. 테스트는 실제 파일 저장·복원 후
고정 후속 경험과 행동 선택의 error/Q/RNG 일치를 검증한다. `run_steps`는 pending을 노출하는
중간 checkpoint API가 아니다. 중간 재개에는 caller가 explicit window/learn/plan 조합을 소유한다.
임의 환경의 Episode 중간 재개는 환경 내부 위치·RNG·TimeLimit 경과 Step도 복원해야 하며
Model/World Model만으로 보장할 수 없다.

### 구조 평가와 향후 확장

- 기존 Q Model은 두 학습 경로에 충분하다. World Model은 Q Model과 독립적인 객체가 적절하다.
  입력 실제 Transition, 출력 예측 Transition이 현재 표 형식에서 연결 경계가 된다. Neural/GPR
  World Model은 자체 입력·예측·불확실성 계약을 정의하고 Planning 호출자가 기존 Transition/Batch로
  명시적으로 변환할 수 있다. 학습용 Model 출력에 환경 예측을 억지로 추가하지 않는다.
- 기존 QLearner는 1-step max bootstrap을 고정한다. n-step Return을 주입하는 메서드는 없으므로
  Real n-step은 Model/TD0/Rule을 직접 조합하는 함수가 필요하다. 기존 두 경로의 1-step은 같은
  QLearner를 재사용하며 서로 다른 horizon을 위해 핵심 API를 바꿀 필요는 없었다.
- Window와 schedule은 외부 deque/실제 Step 루프, model update는 observe, Planning은 독립 plan으로
  분리했다. World Model 관측이 Real n-step 업데이트 지연을 기다리지 않는다.
- Uniform 검색제어는 sample 한 곳에 있다. Prioritized에는 쌍 ID·TD Error/priority·필요 시 predecessor,
  All-action에는 후보 상태/가능 행동 목록, 신뢰도 기반에는 예측 불확실성·confidence가 필요하다.
  단순 sample API만으로 이들 전략의 모든 feedback을 표현한다고 보장하지 않는다.
  외부 Sampler를 모델과 완전히 독립적으로 교체하려면 관측 쌍 조회와 `predict(state, action)` 같은
  지정 쌍 예측 경계를 분리하는 후속 설계가 필요하다. 이번 예제는 sample 한 곳의 쌍 균등 추출만 구현한다.
- 모델 신뢰도 기반 Q(σ)-Planning에는 다단계 synthetic 경로·종료/시간 경계·정책 확률·sampling-time
  σ/신뢰도·Return 재귀가 추가로 필요하다. 단일 전이 생성만으로 완성되지 않으며 이번에 구현하지 않았다.
- Dyna-DQN은 기존 DQNLearner가 TransitionBatch를 받으므로 실제/가상 Batch source를 바꿀 여지가
  있다. Q network/Target sync/Optimizer 상태·주기·두 source의 비율·분포 변화/모델 오차를 명시적으로
  관리해야 한다. 학습된 Neural/GPR 환경 모델과 continuous action search는 별도 설계다.
  현재 plan 함수는 TabularQLearner.model/q_table과 Tensor TD Error를 사용하므로 DQNLearner를
  그대로 전달할 수는 없다. DQNLearner는 online_model과 TDLossOutput 계약을 가지므로 향후 작은
  Neural Planning 함수에서 같은 sample→to_batch→learn 흐름을 연결하고 td_error/loss를 수집해야 한다.
  이를 위해 공통 BaseLearner나 기존 Learner API 변경이 반드시 필요한 것은 아니다.
  후속 Neural 조합 검증은 아래 `Dyna-DQN 컴포넌트 조합 검증` 절에서 이 최소 경로를 구현한다.
  기존 ReplayBuffer는 log_prob 유무를 한 Buffer 안에서 혼합하지 못한다. 실제 전이의 log b와
  가상 전이의 None을 유지하려면 source별 Buffer를 분리하고 Batch 조합/메타데이터 정책을
  외부에서 정해야 한다. 가상 log probability를 0으로 만들어 넣지 않는다.
  신경 환경모델의 확률적 예측이나 Dropout이 Torch RNG를 사용한다면 행동 Selector의 전역
  Torch RNG와도 분리해야 한다. 현재 검증의 Planning RNG 독립성은 결정적 Tabular Q 경로다.
- 관수·온도 제어의 연속 상태·지연·시간 간격은 이번 정수 최신 결과 테이블의 범위를 벗어난다.
  Offline pretraining, online adaptation, idle planning의 실행 분리는 가능하지만 확률적 모델 학습,
  drift 대응, 안정성/신뢰도 제어가 구현된 것으로 간주하지 않는다.
- 실제 중복은 ordered window 검증/eligible-head/flush와 Target→Rule 연결이며 앞 n-step 예제들과
  반복된다. Frozen Q snapshot·sampling-time metadata는 Return별 요구가 다르다. 공통화 후보는
  owned Window, 명시적 horizon/boundary, snapshot, real/planning schedule, 예측과 검색제어 경계다.
  새로운 BaseLearner/Factory/Registry나 알고리즘 조건문 Framework로 이번에 통합하지 않았다.

```bash
python -m examples.gridworld_dyna_q --steps 2000 --seeds 7 11 19
```

신규 `07_dyna_modularrl.ipynb`는 예제 함수만 호출해 1,000 Step×3 Seed 비교·그림과 Idle Planning을
보여준다. 기존 04–06 Notebook의 셀·출력을 보존한다. 실험의 학습 성능은 결과로 보고하고
단일 Seed의 최적 정책/RMSE 우위를 pytest 통과 조건으로 삼지 않는다.

### Gridworld 실행 결과 (실제 2,000 Step/run)

γ=.9, α=.8, ε=.1, Seed7/11/19, 6구성×3 Seed=18 run을 실행했다.
총 실제 Step=Real 갱신은 36,000개, 가상 Planning 갱신은 150,000개다.
모든 run의 별도 Greedy 평가가 목표까지 DP 최적 5 Step에 도달했다.
아래 RMSE는 비종료 10개 상태의 max Q와 DP 최적 V 비교다(6자리 반올림).

| 구성 | Seed7 | Seed11 | Seed19 | 평균±표준편차 |
|---|---:|---:|---:|---:|
| Q-learning / Dyna Planning=0 | .240595 | .069672 | .279962 | .196743±.091279 |
| Dyna Planning=5 | <.000001 | .098377 | <.000001 | .032792±.046375 |
| Dyna Planning=10 | <.000001 | <.000001 | .117240 | .039080±.055267 |
| Real n=3 / Planning=5 | .166549 | .081283 | .119837 | .122556±.034863 |
| Real n=5 / Planning=5 | .003355 | .000062 | .470413 | .157943±.220954 |

Planning=0과 Q-learning은 궤적·Target·error·Q가 같았다. Planning=5/10은 각각
10,000/20,000번의 가상 갱신을 추가했다. n=3/5의 Real 지연 중에도 매 Step 관측·Planning을
수행했고 각각 총 2,000 Real 갱신을 마쳤다. Seed별 flush 수는 n=3: 568/690/698,
n=5: 1278/1304/1320이다. n=1은 flush 0이다.
World Model은 run별 관측한 25–39개 쌍에서 kernel과 정확히 같은 결과를 예측했다.
불완전한 상태·행동 탐색과 제한된 갱신 때문에 Greedy 시작 경로가 최적이어도 모든 상태 max Q의
RMSE가 0인 것은 아니다. Planning을 늘릴 때 탐색 경로가 바뀌어 관측 쌍도 달라지므로 예산별 RMSE가
단조 개선되지 않는다. 고정 탐색 정책의 다단계 실제 보상을 쓰는 Real n-step max backup을
샘플별 1-step Q-learning 또는 중요도 보정 off-policy return과 동일하다고 주장하지 않는다.

### 최종 회귀 검증

전체 `python -m pytest -q`는 **1,896 passed, 69 skipped(CUDA 미지원)**로 통과했다.
실행 시간은 3,674.71초(1시간 1분 14초)이며 기존 대규모 Notebook 실험을 포함한다.
신규 Dyna 단위·환경 테스트는 72 passed/14 skipped다. 기존 예제 16개 Smoke Test도 통과했다.

04/05/06/07 네 Notebook을 각각 같은 Kernel에서 Run All 두 번 실행해 텍스트·수치·PNG가
일치했다. 실행 payload는 저장된 Notebook 출력과도 같다. PNG는 총 37개(4/11/19/3개)다.
07은 20개 셀/코드 10개/그림 3개이며 새 그림을 시각적으로 확인했다.
기존 04–06 세 파일은 셀·출력·metadata를 포함한 전체 byte hash가 작업 전과 같고,
`modular_rl/`의 기존 Python 61개 파일도 byte hash와 파일 집합이 같다.

## Dyna-DQN 컴포넌트 조합 검증

### 실제 저장소 분석과 최소 조합

기준 `2c3263b`의 `DQNLearner`는 외부 `TransitionBatch`를 직접 학습한다. Buffer를 소유하거나
샘플링하지 않는다. 실제 구현은 `DiscreteQEstimator`, `TD0TargetGenerator`, `TDLossUpdateRule`이며
DQNUpdateRule·DQN/DDQN 전용 TargetGenerator·Agent·Runner는 없다. 이름만 보고 존재를 가정하지 않는다.
TDLoss 기본은 MSE이고 기존 CartPole 예제는 `SmoothL1Loss(reduction="none")`를 주입한다.
이번 Neural 예제도 이 Huber 구성을 재사용한다.

| 기존 DQN 검토 질문 | 확인 결과 |
|---|---|
| 외부 Batch 직접 학습? | `learn(batch)`가 가능하다. |
| Replay Sampling이 내부에 고정? | 없다. 호출자가 임의 source의 Batch를 전달한다. |
| Real/Planning 구분? | 서로 다른 호출 경로와 카운터로 구분하며 source 필드를 Transition에 추가하지 않는다. |
| 같은 Online Network 공유? | 같은 DQNLearner 객체를 두 경로에 전달한다. |
| 같은 Target 공유? | 같은 `target_model`을 no_grad/eval로 사용한다. |
| Optimizer 상태 공유? | 동일 Adam 한 개가 두 source의 Gradient Step과 Moment를 누적한다. |
| Target 동기화 시점? | Learner는 자동 주기를 갖지 않는다. 기존 CartPole처럼 예제의 실제 환경 Step 주기를 유지한다. |
| Target/Loss/Optimizer 분리? | TD0/TDLoss는 독립 컴포넌트다. 기본 learn은 세 단계를 묶고 optimize(prediction,target)가 외부 Target 경로를 제공한다. |
| n-step 지원? | 기존 예제는 보상합/Horizon Metadata를 조합한다. 신규 라이브러리 경로는 SequenceBatch·NStepTargetGenerator·DQNLearner.learn이며 아래 회귀 경로와 구분한다. |
| Replay source 구분? | 공개 source 필드가 없다. Real 전이만 한 Replay에 넣어 출처를 유지한다. |

`modular_rl/`과 기존 Tabular Dyna/Q-learning 예제는 변경하지 않는다. 새로운 DynaDQNLearner,
BaseLearner, Runner나 Registry는 없다. `examples/gridworld_dyna_dqn.py`의 작은 함수가 조합을 담당한다.

```text
Environment.step ── actual Transition ── Real Replay ── Real Batch ──┐
                    └── world.observe                               │
TabularWorldModel.sample ── Synthetic Batch ──────────────────────────┤
                                                       same DQNLearner
                                                       same Online Q
                                                       same Optimizer/Moments
                                                       same Target Q
```

### 공개 예제 함수와 데이터 흐름

- `create_learner(..., gamma=.9, learning_rate=1e-4, dtype=float32, device="cpu")`는
  기존 Model·MLPRepresentation·DiscreteQEstimator·EpsilonGreedy·TD0·TDLoss·Adam을 조합하고 초기 Target을 복사한다.
- `synthetic_batch(world_model, batch_size, rng)`는 예측 전이를 묶은 기존 TransitionBatch를 반환한다.
- `plan(world_model, learner, steps, batch_size, rng, record_batches=False)`는 환경과 독립적으로
  같은 `learner.learn(batch)`를 반복한다. 반환 dict는 requested_updates, 실제 updates,
  detached TDLossOutput results와 선택적인 QA용 batches를 구분한다.
- `run_steps(env, learner, replay, ...)`는 고정 실제 Step 예산의 명시적인 수집·학습 루프다.
  Real 전이 저장 → 설정된 Real 업데이트 → World 관측 → Planning → 경계 Metadata Flush → 필요한 Sync
  순서를 따른다. 편의 실험 함수이며 범용 Runner나 live resume API가 아니다.
- `make_n_step_experience(window, target_generator)`는 실제 시간순 B=1 Window의 보상합과 실제 Horizon을 만든다.
- `learn_real_n_step(learner, experiences)`는 같은 Learner의 공개 학습 컴포넌트를 조합한다.
- `train_and_evaluate(...)`, `comparison(steps=3000, seeds=(7,11,19))`는 독립 환경의 Greedy 평가와
  DP 참조값·Loss·카운터·실제 실행시간을 보고한다.

신경망 입력에는 기존 구현에 없던 One-hot 변환이 필요하다. 프레임워크 클래스를 추가하지 않고
`nn.Sequential(nn.Embedding.from_pretrained(torch.eye(S), freeze=True), MLPRepresentation(...))`를
기존 Representation 자리에 전달한다. 단위행렬은 학습되지 않고 Model.to와 state_dict를 따른다.
InputAdapter는 정수 State ID를 long으로 유지하며, 이 Tensor→Tensor 경계에서만
`(...) -> (..., S)` floating One-hot과 MLP Features로 변환한다. Q 출력은 `(..., A)`다.
Replay에는 raw 정수 ID, World에는 Python 정수 ID를 유지한다. float32/64와 CPU/CUDA 계약은 테스트한다.
현재 기본 Network는 Dropout 없는 128×128 ReLU MLP와 Linear Q Head다.
고정 단위행렬 Lookup은 O(S²) 저장 공간을 쓰는 작은 Gridworld 예제의 선택이다.
큰 이산 상태공간이나 연속 관측에는 다른 Representation이 필요하며 이 예제로 일반화 성능을 주장하지 않는다.

### Real Replay와 모델 생성 Minibatch

ReplayBuffer에는 실제 `env.step`의 Transition만 넣는다. 실제 log_prob은 수집 당시 값을 유지하며
Synthetic의 None과 같은 Replay에 섞지 않는다. Raw ReplayBuffer.sample은 비복원 추출한다.
Planning은 Replay를 읽지 않고 기존 TabularWorldModel의 최신 쌍별 예측에서 직접 Batch를 만든다.
World는 관측된 `(s,a)` 쌍을 균등 추출하고 B번 독립 복원 추출하므로 한 쌍만 있어도 B=32 학습이 가능하다.
빈 모델의 Planning은 0 업데이트이며 RNG를 소비하지 않는다. 가상 전이를 다시 observe하지 않는다.

World는 자연 종료를 보존하며 외부 Time-limit을 환경 동역학으로 학습하지 않아 예측 truncated=False,
log_prob=None이다. 결정적 Gridworld에서는 예측과 과거 경험이 같을 수 있지만, source는 분리돼 있다.
이는 환경모델의 일반화·확률적 분포 추정 검증이 아니라 모델 생성 경험의 Neural 학습 경로 검증이다.
같은 Pair를 새로 observe하면 최신 결과로 대체되는 기존 World의 API·테스트를 그대로 재사용한다.

기본 설정은 Batch=32, Real Warm-up=64, Replay capacity=10,000, train_frequency=1,
gradient_steps=1이다. Warm-up은 Real 경로에만 적용한다. 매 실제 Step의 World 관측 직후
설정한 Planning Budget(0/1/5)만큼 Gradient Step을 추가한다. Requested/실제 Planning Step을 별도로 센다.
Real min-size/Batch 조건이 안 되면 Real 업데이트만 건너뛰고 관측·Planning은 계속한다.

### 공유 학습 상태, 수학 검증과 동기화

Real과 Planning 모두 기존 DQN 식을 사용한다.

```text
prediction = Q_online(states).gather(actions)
next_value = max_a Q_target(next_states, a)       # no_grad, eval
target = TD0.generate(reward, next_value, terminated)
result = TDLoss.compute(prediction, target)       # SmoothL1 unreduced -> mean
optimizer.zero_grad(); result.loss.backward(); optimizer.step()
```

위 Loss/Optimizer 세부 절차는 이제 `learner.optimize(prediction, target)` 한 곳에 있다.
일반 1-step learn과 예제의 real n-step 함수가 이 경계를 공유한다.

DQNLearner와 TDLoss의 기존 Gradient 계약을 유지한다. 반환 TD Error/Loss는 업데이트 직전 계산값을
detach한 Tensor다. Target은 독립 저장 공간·동결·eval이며 Optimizer에 포함하지 않는다.
Target 파라미터가 변하는 시점은 초기 복사와 명시적인 hard sync뿐이다. 온라인 train/eval은 호출자가 관리한다.
별도 평가 환경에서 deterministic 행동을 선택해 탐색 RNG와 학습을 간섭하지 않는다.

수치 검증은 Target Next Q=[4,6], Reward=1, gamma=.9에서 Y=6.4다. Online Next Q=[44,66]으로
다르게 두어 Target Network를 쓰는지 확인한다. 자연 종료는 Y=3이며, truncated-only는 Y=6.4다.
두 flag가 True이면 자연 종료가 우선한다. B=1/4/32, 중복/혼합 Pair, dtype/device를 검증한다.
Prediction=2일 때 signed error=4.4, Huber=3.9다. SGD(lr=.1,momentum=.9)로 Real 업데이트 후
Q=2.1, Momentum=-1이고, 같은 Optimizer의 Planning 후 Q=2.29, Momentum=-1.9, Loss=3.8이다.
Tabular의 Q=4.2/5.3 갱신을 신경망에 적용하지 않는다. 실제 실험은 Adam을 사용한다.

초기 `sync_target()` 1회는 periodic sync count와 구분한다. 기존 CartPole 예제처럼 실제 환경 Step의
배수(기본 250)에서 Real·Planning 뒤 동기화한다. Planning=5라고 Sync가 5배 빨라지지 않는다.
환경 Step 없는 Idle Planning에는 자동 Sync가 없다. 필요하면 외부에서 명시적으로 sync_target을 호출한다.
환경·Real 전이·World 관측·Real/Planning/전체 Gradient Step·Periodic Sync 카운터를 각각 기록한다.

Planning=0 회귀 테스트의 Reference Loop는 Dyna 실행 함수를 호출하지 않는 별도 DQN 루프다.
Seed 7/11/19, float32/64에서 초기화·실제 행동/전이·Replay 내용/후속 샘플·모든 Target·TD Error/Loss·
Online/Target 파라미터·Optimizer 전체 상태·Sync Count·환경/행동/Planning RNG를 정확히 비교한다.
최종 Reward 유사성만 비교하지 않는다. Planning RNG는 독립 NumPy Generator이며 deterministic MLP의
Planning은 Torch 행동 RNG를 소비하지 않는다. 확률적 Neural World/Dropout을 도입한다면 이 RNG 분리는
다시 설계해야 한다. Planning으로 Q가 바뀌어 이후 행동이 바뀌는 현상은 RNG 독립성 위반이 아니다.

### Real n-step과 Planning 1-step의 경계

이 절은 보존된 Dyna-DQN 예제의 Metadata 경로를 설명한다. 신규 SequenceReplayBuffer 경로는
후행 `Ordered replay and reward-only n-step targets` 절을 참고한다. 두 경로의 수학적 동등성을 검증한다.

`NStepExperience`는 예제에만 있는 frozen dataclass다. owned CPU scalar인 시작 observation/action,
discounted reward_sum, 마지막 next_observation/terminated/truncated와 실제 horizon m, gamma를 보존한다.
보상합의 의미가 바뀌므로 다른 gamma로 기존 Metadata를 재사용하지 않는다. Bootstrap Q는 저장하지 않는다.
원래 Transition과 TransitionBatch 공개 필드는 변경하지 않는다.

기존 ReplayBuffer는 horizon·임의 Metadata·sample index 공개 API가 없다. Raw Replay는 실제 1-step 전이를
유지하고 n>1에만 별도 `deque(maxlen=replay.capacity)`에 n-step 기록을 저장한다.
이는 작은 예제 Metadata 저장소이며 범용 n-step Buffer 구현이 아니다. 별도 명시적인 NumPy RNG로
기록을 비복원 추출한다. Pending Window에는 실제 경험만 넣고 첫 n-1 Step 중에도 World 관측과 Planning은 진행한다.

```text
reward_sum = R1 + gamma*R2 + ... + gamma**(m-1)*Rm
next_value = max_a Q_current_target(last_next_state, a)
continuation = gamma**(m-1) * next_value
target = TD0.generate(reward_sum, continuation, terminated)
```

보상합은 기존 TD0를 역순 합성해 만들고, 학습 시점의 Target Network에서 Bootstrap을 읽는다.
TD0가 한 gamma를 곱하므로 continuation에 gamma^(m-1)을 적용하면 요구한 gamma^m이 된다.
gamma=0,m=1도 지원한다. 자연 종료는 Bootstrap을 제거하고 Time-limit은 실제 마지막 관측으로 Bootstrap한다.

DQNLearner.learn은 계속 1-step TransitionBatch를 받는다. 작은 `learn_real_n_step`은 기존처럼
같은 Online prediction/gather와 no_grad Target 조회·실제 m의 Target 계산을 직접 수행한 뒤,
`learner.optimize(prediction, target)`를 호출한다. 기존에 반복하던 Rule.compute, zero_grad,
backward, step, detached TDLossOutput 반환의 다섯 문장을 공유하며 Horizon 수식은 숨기지 않는다.
Private 메서드나 Optimizer 내부 상태 조작으로 우회하지 않는다. n=1 Metadata 경로와 기존 learn의
파라미터·Adam 상태·결과가 정확히 일치함을 테스트한다. 일반 1-step Real과 모든 Planning은 기존 learn 그대로다.

경계에서 Planning 후 남은 Window를 실제 m으로 Flush한다. Flush는 Replay 기록 생성이며 Gradient Step,
환경 Step, 행동 선택, observe, Planning이나 RNG 소비를 추가하지 않는다. 학습은 설정된 train_frequency와
Gradient Steps를 따른다. 실행 마지막에 생성된 Flush 기록은 후속 Replay 업데이트가 없으면 해당 run에서
학습되지 않을 수 있다. 원래 Tabular Dyna-Q의 head 직접 갱신과 Neural Replay 기록의 책임을 구분한다.
고정 실제 Step 예산의 마지막 열린 Episode는 외부 truncated cutoff로 표시하고 Bootstrap한다.
이 부분 Episode는 완료 Episode의 Reward/Length 평균에서 제외해 별도 partial_episode로 보고한다.

추가 검증은 n=1/3/5, gamma=0/.9/1, float32/64, CPU/CUDA, 짧은 종료·Truncation·잔여 Horizon·
현재 Target 변경 후 재샘플 학습·중복 Minibatch·Real 지연 중 Planning과 실제 Sync 주기를 다룬다.
Real 다단계 보상은 탐색 정책 아래 수집한 uncorrected n-step max backup이며 Importance Sampling이나
Retrace, Q(σ) 보정을 구현한 것으로 간주하지 않는다. Planning Horizon은 항상 1로 독립적이다.

### Checkpoint와 독립 Planning

`plan`은 환경이나 Replay를 받지 않으며 step/observe/행동 선택/ε 변경을 하지 않는다.
기존 World를 observe만으로 사전 구성한 뒤 호출하거나 온라인 관측 사이에 여러 번 호출할 수 있다.
Idle 테스트는 환경 위치·TimeLimit 경과·환경/행동 RNG·World·Target을 유지하고 Online 파라미터와
같은 Optimizer 상태만 변경되는지 확인한다. Timer, sleep 기반 Scheduler와 비동기 Worker는 구현하지 않는다.

Checkpoint 테스트는 Online/Target/Optimizer의 공개 state_dict, World의 공개 state_dict,
실제 Replay·Replay RNG·Planning/Behavior/n-step RNG·ε·카운터·Sync 설정·Pending Window·n-step Metadata를
저장한다. 복원 후 같은 후속 실제 전이와 다음 샘플에서 Loss·파라미터·Optimizer 상태·행동/log_prob·RNG가
정확히 일치한다. 샘플링 상태를 소실하지 않도록 실제 Replay 객체를 torch.save/load(weights_only=False)한다.
이는 **현재 버전의 신뢰할 수 있는 자체 생성 객체**에 한정된 회귀 검증이다. 기존 ReplayBuffer에 공개
checkpoint API가 없어 장기 호환 형식이나 보편적인 Learner checkpoint API가 검증된 것은 아니다.
Buffer private RNG/저장소를 직접 읽거나 수정하지 않으며 새 프레임워크 API를 만들지 않는다.

중간 Episode의 환경 위치·환경 RNG·TimeLimit 경과 Step까지 일반적으로 저장한 것은 아니다.
실험 편의 run_steps 자체의 중단/재개 API도 제공하지 않는다. 고정 후속 전이를 입력한 학습 상태의 재개를
검증하며 실제 환경 전체 Checkpoint 지원으로 과장하지 않는다. CUDA 재개에는 별도 CUDA RNG도 필요하다.

### 구조적 평가와 향후 최소 확장 후보

| 질문 | 검증 결과와 남은 범위 |
|---|---|
| 1. 기존 DQN 경로 공유? | 1-step Real/Planning은 동일 learn을 호출한다. n-step은 같은 공개 컴포넌트를 직접 조합한다. |
| 2. UpdateRule 재사용? | 기존 TDLossUpdateRule에 Huber를 주입하며 수정하지 않는다. |
| 3. Optimizer 하나? | 하나의 Adam과 Moment가 두 source를 누적한다. 중복 Optimizer를 만들지 않는다. |
| 4. Target 하나? | 동일 Target을 읽고 외부 실제 Step 스케줄로 hard sync한다. |
| 5. Replay 책임? | 실제 경험 저장·무작위 Batch 구성만 담당한다. 모델 예측이나 스케줄을 소유하지 않는다. |
| 6. Tabular World + Neural Q? | raw 정수 ID 모델과 입력 경계 One-hot을 조합했다. World는 Neural일 필요가 없다. |
| 7. 모델에서 Batch 직접 생성? | sample을 B회 호출해 Synthetic Batch를 만든다. Planning Buffer가 필요 없다. |
| 8. Sampler 교체? | 예제의 synthetic_batch/plan 호출부를 교체할 수 있다. 공통 Strategy API는 아직 없다. |
| 9. Environment/Planning 분리? | plan은 환경을 받지 않으며 Idle 반복 호출을 지원한다. |
| 10. 사전학습/온라인 적응? | 기존 observe를 사전·실행 중 호출할 수 있다. Neural 모델 학습/Drift 대응은 없다. |
| 11. 독립 Horizon? | Real n=1/3/5와 Planning 1을 같은 Network/Optimizer로 검증했다. |
| 12. n-step Metadata 위치? | examples의 NStepExperience와 bounded deque다. 기존 경험 필드를 확장하지 않는다. |
| 13. Window/Replay 관계? | Window는 시간순 pending 실제 경험, raw Replay는 실제 1-step, Metadata Replay는 eligible reward sum/m이다. |
| 14. Real 지연 중 Planning? | 처음 n-1 Step과 Warm-up 중에도 관측·Planning한다. 경계 Flush는 별도 기록 작업이다. |
| 15. Target 동기화 일관성? | Real/Planning이 같은 Target을 읽으며 실제 Step 기준 250마다 한 번 복사한다. |
| 16. Idle-time Planning? | 독립 함수 호출로 연결 가능하다. 실행시간 Scheduler는 추가하지 않았다. |
| 17. 연속 상태 World? | raw continuous 관측·모델 예측 dtype/device·Representation·상태/행동 후보 추출을 정의해야 한다. One-hot/Tabular Pair 저장은 교체 대상이다. |
| 18. Priority Feedback? | 반환 signed td_error를 재사용할 수 있다. 선택 Pair ID와 연결·우선순위 갱신/추출은 추가 책임이다. |
| 19. All-action? | World에 predict(state, action)/enumerate actions 등 명시적 질의와 미관측 Pair 처리 규칙이 필요하다. 현재 sample-only 결과만 있다. |
| 20. 불확실성 기반 Q(σ)? | 모델 version/fidelity·uncertainty와 수집 시 σ·π/b·행동/경로·Horizon·경계·snapshot 의미가 필요하다. 단일 1-step Batch만으로 완성되지 않는다. |

최소 공통화의 첫 단계로 `DQNLearner.optimize(prediction, target) -> TDLossOutput`를 추가했다.
Optimizer 소유권을 유지하면서 반복되던 Loss/zero_grad/backward/step/반환을 공유한다.
Model 평가·Horizon·no_grad Target 정책은 호출자에 남긴다. Real과 Planning은 같은 Optimizer moment와
같은 Target Network를 계속 사용하며 raw/Metadata Replay, sample RNG, readiness, 경계 Flush와
실제 환경 Step 기준 Sync 일정은 변경하지 않았다. World Model·Planning·Runner API도 확장하지 않았다.
raw 경험과 Metadata의 안정적 저장·샘플 ID·공개 checkpoint, 예제 private/runpy 의존과
명시적인 source별 sampling/scheduling은 남은 후속 과제다.
현재 검증만으로 거대한 BaseLearner나 모든 알고리즘 Bootstrap 추상화를 추가할 근거는 없다.
학습 기반 WorldModel, 불확실성/우선순위/All-action Planning과 Q(σ)-Planning은 향후 연구 범위다.

```bash
python -m examples.gridworld_dyna_dqn --steps 3000 --seeds 7 11 19
python -m pytest -q tests/test_dyna_dqn_composition.py tests/test_gridworld_dyna_dqn.py
```

`08_dyna_dqn_modularrl.ipynb`는 별도 Neural 실습으로 600 실제 Step×3 Seed의 빠른 비교와 공유 상태,
One-hot, Minibatch, n-step Metadata, Idle Planning을 예제 함수로 실행한다. 학습 수식을 중복 구현하지 않는다.
실제 실행시간은 CLI에 보고하며 Notebook의 재현성 비교 출력에서는 제외한다. 기존 04–07의 셀·출력은 보존한다.

### Gridworld 실험 결과 (실제 3,000 Step/run)

γ=.9, Adam1e-4, Huber, MLP128×128, B32, Warm-up64, Replay10,000, train_frequency=1, gradient_steps=1,
ε1→.05/2,000 실제 Step, Target250 실제 Step 설정으로 6구성×Seed7/11/19=18 run을 실행했다.
모든 run은 실제 전이·환경 Step 3,000개, Real Gradient Step 2,937개, Periodic Sync 12회다.
총 실제 Step=54,000, Real 업데이트=52,866, Planning 업데이트=72,000이며 총 Optimizer Step=124,866이다.
모든 run의 별도 Greedy 평가가 DP 최적 경로와 같은 5 Step으로 목표에 도달했다. Dyna는 모두 40쌍을 관측했다.

다음 max Q 대 DP V* RMSE는 비종료 10개 상태에서 비교한다. CLI 6자리 반올림 값을 집계했다.
표준편차는 세 Seed의 모집단 표준편차(ddof=0)다.

| 구성 | Seed7 | Seed11 | Seed19 | 평균±표준편차 | Planning/run | 시간 평균(초) |
|---|---:|---:|---:|---:|---:|---:|
| DQN | 0.000632 | 0.000219 | 0.000314 | 0.000388±0.000177 | 0 | 7.36 |
| Dyna p=0 | 0.000632 | 0.000219 | 0.000314 | 0.000388±0.000177 | 0 | 7.23 |
| Dyna p=1 | 0.000001 | 0.000000 | 0.000000 | 0.000000±0.000000 | 3,000 | 18.58 |
| Dyna p=5 | 0.004351 | 0.000437 | 0.002491 | 0.002426±0.001599 | 15,000 | 69.49 |
| Real n=3 / p=1 | 0.072063 | 0.093504 | 0.089485 | 0.085017±0.009306 | 3,000 | 21.47 |
| Real n=5 / p=1 | 0.065167 | 0.091467 | 0.106628 | 0.087754±0.017129 | 3,000 | 21.44 |

각 run에서 완료 Episode의 평균 Reward/Length를 계산한 뒤 Seed별 결과를 평균했다.
실제 Step 예산 끝의 열린 Episode는 truncated cutoff로 학습하되 완료 Episode 평균에서 제외했다.

| 구성 | Episode Reward | Episode Length | Real 평균 Loss | Planning 평균 Loss |
|---|---:|---:|---:|---:|
| DQN | 0.8971 | 8.13 | 0.003682 | — |
| Dyna p=0 | 0.8971 | 8.13 | 0.003682 | — |
| Dyna p=1 | 0.8854 | 7.75 | 0.001601 | 0.003499 |
| Dyna p=5 | 0.8973 | 7.86 | 0.000462 | 0.001178 |
| Real n=3 / p=1 | 0.9244 | 7.73 | 0.023168 | 0.005408 |
| Real n=5 / p=1 | 0.9312 | 7.84 | 0.044330 | 0.007189 |

실제 측정시간 합계는 436.743초다. 다른 검증을 병행한 동일 머신의 관측치이며 성능 벤치마크가 아니다.
Planning=0은 DQN과 수치·동작이 같았고, 추가 observe/계측·초기 실행 비용으로 시간은 달라질 수 있다.
Planning=1의 6자리 RMSE는 .000001/.000000/.000000으로, 0으로 출력된 값도 수학적 exact zero를 뜻하지 않는다.
Planning=5는 더 많은 계산에도 Planning=1보다 RMSE가 컸다. 고정된 실제 Step Sync 사이에 더 많은 업데이트가
들어가며 탐색 경로도 바뀌므로 단조 성능 향상을 요구하지 않는다. Real n=3/5 역시 최적 시작 경로를 얻었지만
전체 상태 max Q의 오차는 1-step보다 컸다. 같은 시작 경로의 최적성과 모든 상태가치의 정확성은 구분한다.
각 n=3/5의 Seed별 Flush 기록 수는 778/761/790 및 1544/1505/1544다. 실제 Horizon을 보존했으며
Flush가 추가 Optimizer Step을 만들지 않아 Real 업데이트 수는 같은 2,937개다.

### 최종 회귀 검증

전체 `python -m pytest -q`는 **1,974 passed, 111 skipped(CUDA 미지원)**로 통과했다.
실행 시간은 3,121.96초(52분 1초)다. 신규 Dyna-DQN 단위·환경/Checkpoint 테스트는
77 passed/42 skipped이며 종료 Target=[100,200]에서도 Bootstrap을 제거하는 보강을 포함한다.
기존 예제 17개 Smoke Test, 3,000 실제 Step×18 run의 새 CLI 비교와 Warm-up 이전 1-Step CLI도 통과했다.
경험 부족 시 실제 업데이트가 0인 것을 정상 출력하며 빈 평균의 NaN/RuntimeWarning을 만들지 않는다.

04/05/06/07/08 다섯 Notebook을 각각 같은 Kernel에서 Run All 두 번 실행했다.
텍스트·수치·PNG는 두 실행 사이에 같았으며 저장된 Notebook 출력과도 같다.
PNG는 총 40개(4/11/19/3/3개)다. 새 08은 23개 셀/코드 12개/PNG 3개이며 그림을 시각적으로 확인했다.
목표 State의 Neural Q 출력은 직접 학습하지 않아 임의 값일 수 있음을 설명했고,
종료 Mask와 비종료 상태 RMSE를 구분했다. 실행 후 추가한 설명은 계산 코드나 출력을 바꾸지 않는다.

기존 `modular_rl/` Python 61개 파일은 byte hash와 파일 집합이 모두 작업 전과 같다.
기존 04–07 네 Notebook은 셀·출력·metadata를 포함한 전체 byte hash가 같고,
Tabular Dyna-Q·SARSA·n-step SARSA·CartPole 보호 대상 예제도 그대로다.
모든 구현 변경은 새 예제에 한정하며 기존 공개 API나 프레임워크 추상 계층은 바꾸지 않았다.

## Phase 1: DQN 공유 최적화 경계 검증

위 Dyna-DQN 검증 기록 이후 `DQNLearner.optimize(prediction, target)`를 추가하고 기본 learn과
real n-step 함수의 동일한 Loss/Optimizer/반환 다섯 문장을 공유했다. 기존 1-step/다단계 Target 계산,
데이터·Replay·World Model·Planning·Runner 계약과 기존 공개 시그니처는 유지했다.

신규 `tests/test_dqn_optimization.py`는 45 passed/24 skipped(CUDA 미지원)다.
B=1/4/32, float32/64, gradient와 detached 반환, Parameter storage 유지, Target 불변,
추가 Forward/Sync/RNG 소비 부재 및 잘못된 입력에서 기존 Parameter/Gradient/Adam 상태 보존을 확인한다.
독립적으로 복제한 초기 상태에서 기준 Commit의 Optimizer 절차와 현재 learn/optimize를 비교하며,
Adam/SGD Momentum·Dropout RNG·BatchNorm buffer까지 bit-exact인지 검증한다.
기존 DQN/Dyna-DQN 세 테스트 파일도 변경 없이 140 passed/45 skipped로 통과했다.

별도의 전체 루프 참조 실험은 real n=1/3/5 × float32/64 × Adam/SGD Momentum의 12개 설정을
비교했다. 80 실제 Step마다 Real 74회·Planning 80회·Target Sync 11회를 수행하며 모든 trace,
반환 통계·Online/Target·Optimizer 상태·Flush metadata·후속 Replay 표본·행동/환경/Planning/n-step RNG가
기존 최적화 절차와 정확히 일치했다. 이는 이 리팩터링의 동등성 검증이며 새 알고리즘 성능 실험이 아니다.
quickstart, FrozenLake, CartPole와 기존/신규 Gridworld를 포함한 18개 CLI Smoke Test도 통과했다.

이번 Phase 1의 전체 `python -m pytest -q`는 **2,019 passed / 135 skipped(CUDA 미지원)**로
통과했다. 실행 시간은 2,840.62초(47분 20초)다. 기준 기록 1,974/111을 재인용한 결과가 아니라
이번 소스와 신규 테스트를 포함해 실제 실행한 결과다.

04–08 다섯 Notebook을 각각 같은 Kernel에서 Run All 두 번 실행했으며 text/numeric/PNG가
두 실행 사이에 일치했다. 실행 결과를 기존 저장본과도 대조하여 PNG 40개(4/11/19/3/3개)를 포함한
출력이 같고 Cell source/ID/metadata와 Notebook metadata도 유지되는지 확인했다.
다섯 원본 Notebook 파일은 byte hash도 동일하다.

원본 핵심 Python 61개 중 dqn.py만, 예제 18개 중 gridworld_dyna_dqn.py만 변경했다.
기존 테스트 52개 파일은 그대로이며 신규 최적화 테스트만 추가했다. 두 학습 함수의 public signature와
최적화 이전 계산은 AST 비교에서도 동일하다. 후속 Window/Replay/World Model/Runner 리팩터링은 수행하지 않았다.

## Ordered replay and reward-only n-step targets

### Replay input compatibility

`ReplayBuffer.add(transition)` and `add(transition=transition)` remain supported.
The same operation also accepts the six Transition fields, an optional seventh
`log_prob`, or their positional/keyword combination. Python's Transition argument
binding rejects missing/duplicate fields. Mixing a Transition object with extra
fields is an error. All forms enter the same schema validation and owned,
detached CPU snapshot path before changing storage. Sampling remains uniform,
without replacement, using the existing private NumPy Generator. `clear()` does
not rewind that RNG. Transition remains useful for Episode and B=1 online updates.

### SequenceReplayBuffer and SequenceBatch

`SequenceReplayBuffer(capacity, seed=...).add(...)` uses the same add forms and
field schema as ReplayBuffer, but retains chronological order. This first version
supports **one sequential environment stream**. Non-boundary transitions must be
continuous (`previous.next_observation == current.observation`); interleaved
vector/asynchronous environments need a future explicit stream-ID contract.

`sample(batch_size, length=n)` uniformly samples distinct eligible **starting
positions**, not episodes. A start is eligible after n observed transitions or
when a real terminated/truncated boundary certifies a shorter tail. Open tails
with fewer than n steps are ineligible. A ring overwrite removes old starts;
retained windows only use retained, consecutive transitions and never cross an
episode boundary. `num_sequences(length)` counts candidates without consuming
RNG and lets training loops express warm-up. `len(buffer)` counts stored steps.

SequenceBatch tensors have shapes:

|Field|Shape / dtype|
|---|---|
|observation, next_observation|(B,T,*observation_shape), original dtype|
|action|(B,T) integer or (B,T,A) floating|
|reward, optional log_prob|(B,T), original dtype|
|terminated, truncated, valid|(B,T), bool|
|horizon|(B,), long|

Each valid row is a nonempty prefix. Padding is zero and has no experience
meaning. Real termination and external truncation are preserved independently;
the collector supplies the actual final observation, not an autoreset observation.
Short rows end at a real boundary. Storage and sampled tensors do not alias the
caller or each other, and have no graph. `.to(device, dtype=...)` casts floating
fields only, preserving integer states/actions, bool masks and long horizons.
SequenceBatch represents replay rows, not a generic (T,B) trajectory framework.
Buffer performs no discounting, model evaluation or target calculation.

### NStepTargetGenerator and DQN integration

`NStepTargetGenerator(gamma).generate(rewards, next_value, terminated, valid)`
accepts rewards/flags/mask (B,T) and externally evaluated next_value (B,).
The nonempty prefix mask defines m. Shapes/devices match without broadcasting;
rewards and next_value share a floating dtype. An interior terminal step is invalid.

For each row it computes
`sum(k=0..m-1, gamma**k * reward[k]) + gamma**m * (1-last_terminated) * next_value`.
Rewards are folded backward and bootstrap is discounted separately, preserving
the existing Dyna-DQN example's reward-sum arithmetic. Masked padding is removed
before arithmetic (including NaN/Inf padding); terminal bootstrap is masked even
if its supplied value is nonfinite. Truncated-only rows retain bootstrap. Caller
chooses the endpoint value (max Q, actual next-action Q, V, etc.) and manages
no_grad/detach. The generator preserves gradients and owns no Model/Buffer/Optimizer.
It deliberately does not inherit TD0's incompatible one-step interface.

DQNLearner preserves its existing five positional constructor arguments and
`learn(TransitionBatch)`, `optimize(prediction,target)`, `sync_target()` behavior.
An optional keyword-only `n_step_target_generator` supplies the new component;
when omitted, it defaults to the existing TD0 generator's gamma. Both gamma values
must agree at construction. Configure gamma before constructing the Learner.

`learn(SequenceBatch)` performs: first state/action -> Online Q gather -> last
valid next observation -> current Target eval/no_grad max Q -> n-step target ->
existing optimize. No bootstrap Q is stored in replay. Short horizons are evaluated
with their actual m. Existing one-step evaluation/optimizer order is unchanged.
The same Learner/Online/Target/Optimizer can alternate real n-step batches and
planning one-step batches; synchronization remains an external schedule.

The old `examples/gridworld_dyna_dqn.py` NStepExperience,
make_n_step_experience and learn_real_n_step remain intact as regression references.
Their example-only metadata deque is not silently replaced in old experiments.
New notebooks use the library SequenceReplayBuffer path. No new BaseLearner,
Runner, Agent, Planner or WorldModel framework is introduced.

Random replay of an n-step sequence is **not online on-policy SARSA**. Existing
SARSA windows preserve selected-action timing and flush order. Tree Backup and
Q(sigma) additionally need intermediate Q/policy/sigma/importance data; this
reward-only generator does not automatically implement those algorithms.

### Gymnasium tutorials and supported scope

`notebooks/gymnasium/00` through `16` independently construct components and call
Gymnasium reset/step in visible loops. Default mode is FULL; set
`MODULARRL_FAST=1` for a small API/shape smoke budget. Evaluation has a separate
environment, never calls learn, and all environments are closed. Plotting/evaluation
helpers contain no training loop or return formula. Notebook source records API
friction and leaves editable evaluation tables.

- Library APIs: DP, MC prediction/control, TD(0) prediction, tabular Q-learning,
  SARSA, DQN, Replay/Sequence replay, TD0/MC/n-step targets and existing UpdateRules.
- Example composition support: Expected SARSA, Double Q, online n-step SARSA,
  fixed-target off-policy n-step prediction, Tree Backup, Q(sigma), TabularWorldModel
  and Dyna planning. These notebooks reuse examples functions rather than copying
  their return equations or claiming installed public Learner classes.
- Fixed-target Tree Backup/Q(sigma) notebooks are policy evaluation, not control.
  Q(sigma) retains the earlier documented Eq. (7.17) body boundary interpretation;
  its discrepancy with the textbook pseudocode is not claimed to be an officially
  acknowledged erratum.
- Dyna-DQN uses FrozenLake's integer IDs and an observed-pair tabular sample model.
  Learned continuous-observation World Models, CartPole Dyna-DQN, prioritized replay
  and all-action planning are not provided.

See [GYMNASIUM_TUTORIALS.md](GYMNASIUM_TUTORIALS.md) for the notebook list, API
examples and actual validation/performance results. Existing DLFS4 notebooks and
all old algorithm examples remain regression references.

Initial integration checks: 72 new Buffer/target/learning cases passed (14 CUDA
cases skipped on this host); the final non-notebook suite passed 2,086 cases
(149 CUDA skips). All 17 Gymnasium tutorials passed FAST Run All twice, comparing
text/numeric output and 35 rendered PNGs per run. Full-suite and FULL n-step
measurements are recorded in the tutorial guide after completion.

FULL examples on seed 7 completed: FrozenLake Q-learning achieved 100/100 greedy
successes after 1,500 episodes; CartPole 1-step DQN improved mean evaluation return
9.2 -> 186.55 and n=3 replay DQN improved 9.2 -> 287.0 after 400 episodes each.
Their environment-step budgets differ (35,232 vs 61,342); no statistical superiority
claim follows from this single-seed measurement.

Existing DLFS4 04–08 each passed Run All twice in one kernel. Their executed
sources and text/numeric/PNG payloads also match Phase 1's executed references
(40 PNGs). Existing notebook files, examples and pre-existing user work retain
their starting hashes. The 18 existing CLI smoke examples passed unchanged.

The final collected suite has 2,257 cases. Disjoint verification of the latest
Core (2,086 passed / 149 CUDA skips), existing notebooks (5 passed) and new
notebooks (17 passed) covers all cases: 2,108 passed / 149 skipped. The long
existing-notebook run itself reported 2,087 passed / 145 skipped in 3,100.31s;
B=1 parameter cases added after its collection are covered by the final Core
rerun. Exact commands and scope accounting are in GYMNASIUM_TUTORIALS.md.

## 공개 API 간소화와 Learner 승격

representations/estimators/selectors/buffers/targets/updates/learners는 실제 공개 모듈이다.
짧은 이름은 기존 클래스를 그대로 re-export한다. 루트에는 Model을 공개하고 같은 이름의
Estimator와 UpdateRule은 각 네임스페이스에 둔다. 클래스 이름·모듈·state_dict·Pickle 경로,
Model의 forward/act/evaluate_q/encode와 기존 생성자는 유지한다. 전체 이름과 시그니처는
[API.md](API.md)에 정리한다.

학습 계산은 ExpectedSARSALearner, DoubleQLearner, NStepSARSALearner,
OffPolicyNStepSARSALearner, TreeBackupLearner, QSigmaLearner로 이동한다. 패키지는
examples/runpy에 의존하지 않는다. 설치 wheel만 사용하는 독립 프로세스에서 여섯 API를
검증한다. 예제 learn 함수는 위임하고 dc2ecdc의 독립 수학·Online 스케줄 참조는 tests/_legacy_td에
보존한다. 별도 초기 상태에서 Target/error/table/실제 행동/환경·행동 RNG/Flush를 비교한다.

ExpectedSARSA는 Model의 현재 Selector로 한 estimate의 pi와 Q를 계산한다. DoubleQ는 두
Model의 storage 독립성을 확인하고, estimate/act로 Q1+Q2의 명시적 행동 경로를 제공한다.
update_first는 외부 RNG가 정하며 learn은 선택 Model의 argmax와 다른 Model의 평가를 조합한다.
Model/Estimator에 복수 테이블 결합 책임을 추가하지 않는다.

네 multi-step Learner는 외부의 연속적인 owned B=1 TransitionBatch window를 받아 head 하나만
갱신한다. 공통 private window 검증만 재사용한다. 실제 next action의 사전 선택·재사용과
수집 당시 log_prob/sigma, window pop·Tail Flush 시점은 외부 책임이다. TD0의 기존 역순 계산
순서를 유지해 산술 재배열로 실험 결과를 바꾸지 않는다. 무작위 Replay는 Online window와 다르다.
TreeBackup/QSigma의 중간 기대값·보정 계수를 reward-only NStep 계산에 강제하지 않는다.
식 (7.17)의 본문 경계, σ=0 TreeBackup, n=1 Expected SARSA, σ=1 Control Variate와
중복 Importance Ratio 금지 계약은 그대로다. 의사코드 차이는 공식 인정 오류로 단정하지 않는다.

DQN은 기존 TD0 + keyword n_step_target_generator 계약에 더해 네 번째 인자로 NStep을 받는다.
이 경우 제공된 NStep은 sequence 경로에, 같은 gamma의 TD0는 1-step 경로에 사용한다.
optimize/sync_target 및 Online/Target/Optimizer의 공유·Gradient 경계는 바꾸지 않는다.
Dyna의 Real n-step과 Planning 1-step은 같은 Learner를 재사용한다. WorldModel/Planner와
환경·학습 스케줄은 examples에 남기며 새로운 범용 Learner/Factory/Runner를 만들지 않는다.
