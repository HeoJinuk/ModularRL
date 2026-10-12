# ModularRL 아키텍처와 데이터 계약

ModularRL은 연구·교육용 강화학습 라이브러리를 위한 Model·경험 데이터·기본
ReplayBuffer·TD(0) TargetGenerator·Tabular/Neural UpdateRule·TabularQLearner·DQNLearner 참조 구현이다.
현재 Tabular Q-learning은 FrozenLake, 기본 DQN은 CartPole 예제에서 학습할 수 있다.
Runner, Agent, 다른 Neural Learner, WorldModel/Planner는 아직 포함되지 않는다.
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
행동을 Learner에 별도로 전달하는 계약이 필요하다. 무작위 Replay의 이웃 행이나 현재
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

프로젝트 루트에서 Gymnasium을 core 의존성이 아닌 `examples` extra로 설치한다. 렌더링하지 않으므로
pygame 등의 GUI 의존성은 필요하지 않다.

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
처리·환경 close 테스트로 검증한다. Gymnasium이 없으면 해당 smoke 테스트만 skip한다.

### DQNLearner와 실제로 겹치는 코드

두 Learner는 Batch.to로 floating dtype·device를 준비하고 이산 action을 long으로 변환한다.
Batch의 reward/종료 shape·dtype 검사와 no_grad에서 action별 max Q -> TD0TargetGenerator를
호출하는 흐름도 겹친다. 반면 Tabular는 정수 state indices와 테이블의 입력 순서 갱신을,
DQN은 feature 관측과 미분 가능한 gather prediction·Optimizer·독립 Target Model을 사용한다.
이 작은 반복을 숨기기 위해 공통 BaseLearner나 Validation Framework를 추가하지 않는다.

## DQNLearner와 CartPole 검증

`learner/dqn.py`의 DQNLearner는 외부 Online Model, Target Model, Optimizer,
TD0TargetGenerator, TDLossUpdateRule을 조합하는 일반 Python 객체다. Buffer와
ActionSelector는 학습에서 사용하지 않는다. 수집·샘플링·평가는 외부 호출자의 책임이다.

| API / 입력 | 계약 |
|---|---|
| `DQNLearner(online_model, target_model, optimizer, target_generator, update_rule)` | 독립된 동일 구조 Model, Online 전용 Optimizer, 기존 Target/UpdateRule |
| `learn(batch: TransitionBatch) -> TDLossOutput` | 한 번의 backward/step, 반환값은 detach된 업데이트 직전 통계 |
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
         -> TDLossUpdateRule.compute(prediction, target)
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
Gymnasium이 없으면 환경 Smoke Test만, CUDA가 없으면 CUDA 검사만 skip한다.

Double DQN의 변경 지점은 no_grad Bootstrap 선택 경로다. Online next Q의 argmax 행동을
Target next Q에서 gather해 next_value로 전달하면 된다. 기존 TargetGenerator,
TDLossUpdateRule, 반환 계약과 Optimizer 절차는 재사용할 수 있다. 이번 구현에 조건문이나
Double DQN 옵션을 미리 추가하지 않는다.

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
