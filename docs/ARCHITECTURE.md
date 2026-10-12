# ModularRL 아키텍처와 데이터 계약

ModularRL은 연구·교육용 강화학습 라이브러리를 위한 Model·경험 데이터·기본
ReplayBuffer 참조 구현이다. 현재 저장소에는 Runner, Agent, Learner 전체와
학습 알고리즘이 포함되지 않는다.
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
정수 reward를 floating target 계산에 사용할 때는 TargetGenerator/호출자가 적절한
floating dtype을 명시한다. 정수로 저장된 연속 관측은 기존 InputAdapter의 명시적인
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

## TargetGenerator 구현 전에 결정할 사항

- 알고리즘별 입력·출력: one-step V/Q target, Expected SARSA/Importance Sampling의 Policy 확률과 Behavior log_prob 사용 규칙.
- Discount와 수치 dtype: gamma의 전달 방식, reward·target·mask를 계산할 floating dtype, Target 평가의 gradient 경계.
- 종료 정책: Time-limit 외 truncation의 의미, Bootstrap mask와 trace 경계의 분리, Runner의 실제 마지막 관측 제공 계약.
- Trajectory 확장: `(T, B, ...)`의 정렬·에피소드 경계, GAE/n-step 계산에 필요한 순서 정보와 On-policy Buffer 계약.
- Target Model 및 복원: 현재/Target Model 선택·갱신 책임, Buffer 내용·원형 위치·전용 RNG의 향후 checkpoint 범위.

## Learner 구현 전에 결정할 사항

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
