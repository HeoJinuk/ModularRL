# ModularRL — Model Contract v0.2

> This is a **reference implementation of the Model layer, experience contracts, basic ReplayBuffer and TD(0) targets**, not the entire ModularRL framework. It does not include a complete Learner, Runner, UpdateRule, or training algorithms.

## Layout

```text
modular_rl/
├── data/
│   ├── outputs.py                  # ValueOutput / EstimatorOutput / ActionOutput
│   └── experience.py               # Transition / TransitionBatch
├── model/
│   ├── rl_model.py                 # Model.forward / Model.act / Model.evaluate_q
│   └── input_adapter.py           # NumPy, Tensor, scalars → Tensor
└── component/
    ├── representation/
    │   ├── base.py                # BaseRepresentation
    │   ├── identity.py            # IdentityRepresentation
    │   └── mlp.py                 # MLPRepresentation
    ├── buffer/
    │   ├── base.py                # add / sample / clear / len
    │   └── replay.py              # Uniform replay with owned CPU snapshots
    ├── target_generator/
    │   ├── base.py                # Tensor inputs → target Tensor
    │   └── td0.py                 # One-step TD formula; caller supplies next_value
    ├── estimator/
    │   ├── base.py                # BaseEstimator, BasePolicy/Value and compatible roles
    │   ├── actor_critic.py        # Policy + Value composition
    │   ├── policy/
    │   │   ├── categorical.py     # CategoricalPolicyEstimator
    │   │   ├── gaussian.py        # Unbounded diagonal Gaussian
    │   │   └── deterministic.py   # Action Tensor head
    │   └── value/
    │       ├── state_value.py     # V(s)
    │       ├── discrete_q.py      # Q(s, all discrete actions)
    │       ├── continuous_q.py    # Q(s,a), via evaluate(features, actions)
    │       ├── advantage.py       # DiscreteAdvantageEstimator
    │       ├── tabular_v.py       # State-indexed V table
    │       └── tabular_q.py       # State-indexed Q table
    └── action_selector/
        ├── base.py                # BaseActionSelector
        ├── greedy.py              # Discrete Q greedy with uniform tie-breaking
        ├── epsilon_greedy.py      # Discrete Q epsilon-greedy
        └── policy_sampling.py     # Categorical policy sampling
```

`__init__.py` files provide shorter import paths. To avoid the duplicated names in the first reference, `data/model.py` is now `data/outputs.py` and `model/model.py` is now `model/rl_model.py`. The public import `from modular_rl.model import Model` remains unchanged.

## Input / output contracts

Experience data and replay are independent of Model. A `Transition` contains
observation, action, reward, next_observation, terminated, truncated and an
optional behavior `log_prob`. `ReplayBuffer.add()` copies values into detached
CPU tensors and preserves dtypes. `sample(B)` returns a `TransitionBatch` with
leading shape `(B, ...)`, uniformly sampled without replacement. Capacity
overflow replaces the oldest transition, and a private RNG supports `seed=`.
Each buffer uses consistent shapes/dtypes and either always includes or always
omits log_prob. Missing probabilities are never filled with a fabricated value.

`batch.to(device, dtype=...)` moves data and casts only floating fields, keeping
integer state indices/actions and boolean flags intact. Learner chooses the
training device and dtype. Ordered on-policy `(T, B, ...)` trajectories need a
future trajectory buffer. See [Architecture](docs/ARCHITECTURE.md) for the full
experience, bootstrap and Runner boundary contracts.

`TD0TargetGenerator(gamma).generate(reward, next_value, terminated)` computes
`reward + gamma * (1 - terminated) * next_value`. All three tensors must have
the same shape and device; reward/next_value share a floating dtype and
terminated is bool. Scalar `()`, replay `(B,)` and other same-shaped batches
are supported without broadcasting. The caller chooses the bootstrap value
for TD prediction, SARSA or Q-learning and controls `no_grad`/`detach`.
Time-limit truncation can bootstrap; trajectory boundaries are a separate concern.

```python
import torch
from modular_rl.component.target_generator import TD0TargetGenerator

target = TD0TargetGenerator(gamma=0.9).generate(
    reward=torch.tensor([1.0, 2.0]),
    next_value=torch.tensor([10.0, 20.0]),
    terminated=torch.tensor([False, True]),
)
torch.testing.assert_close(target, torch.tensor([10.0, 2.0]))
```

| Stage | Input | Output |
|---|---|---|
| `InputAdapter` | NumPy / Torch / numeric scalar | Torch Tensor on the Model's device |
| `Representation.forward` | Tensor | Tensor |
| `Estimator.forward` | feature Tensor | `EstimatorOutput` |
| `ActionSelector.select` | `EstimatorOutput` | `ActionOutput` |
| `Model.forward` | environment observation | `EstimatorOutput` (gradients enabled) |
| `Model.act` | environment observation | `ActionOutput` (`no_grad`) |

The semantic output types are defined in `data/outputs.py`:

```python
@dataclass(frozen=True, slots=True)
class ValueOutput:
    v: object | None = None
    q: tuple[object, ...] = ()
    advantage: object | None = None

@dataclass(frozen=True, slots=True)
class EstimatorOutput:
    policy: object | None = None
    values: ValueOutput = field(default_factory=ValueOutput)

@dataclass(frozen=True, slots=True)
class ActionOutput:
    action: torch.Tensor
    log_prob: torch.Tensor | None = None
```

`q` stores a tuple: empty when no Q is estimated, or one or more Q predictions. Each entry can carry scalar Q values or a richer distributional value representation in a future implementation. Consumers must validate the payload representation they support. `policy` is one semantic field, whether stochastic or deterministic. `advantage` means **directly estimated** A(s,a); GAE is a TargetGenerator calculation rather than an estimator field.

## MLP contract — same shape/configuration as the previous implementation

```python
from torch import nn
from modular_rl.component.representation import MLPRepresentation

rep = MLPRepresentation(
    input_dim=4,
    hidden_dims=(128, 64),
    activation=nn.ReLU,
)
assert rep.output_dim == 64
```

- Layers follow `Linear → activation` for **every** element of `hidden_dims`, including the final layer, matching the provided implementation.
- `forward()` takes only `torch.Tensor` (e.g., `(4,)`, `(32,4)`, `(T,B,4)`) and preserves every leading dimension. It has no NumPy conversion or device-moving code.
- Both dtype and device conversion happen **once at the Model boundary** via `InputAdapter`.
- Integer state indices remain `torch.long`; floating observations are converted to the Model's floating dtype (normally `float32`, or `float64` after `.double()`). For a Box environment with integer-valued observations, use `InputAdapter(dtype=torch.float32)` explicitly.
- `BaseRepresentation` already inherits `torch.nn.Module`; subclasses do **not** also inherit from `nn.Module`.

## Estimators

`BaseEstimator` defines only `forward(features) -> EstimatorOutput`.
`BasePolicyEstimator` and `BaseValueEstimator` are aliases of the existing
`PolicyEstimator` and `ValueEstimator` roles. Existing subclasses, imports,
and checkpoint keys remain compatible; no extra inheritance layer is added.
The existing `AdvantageEstimator` remains an abstract role, with the concrete
discrete implementation named `DiscreteAdvantageEstimator`.

In this table, `...` denotes all leading batch dimensions, `F` is feature
dimension, and `A` is the number of discrete actions or continuous coordinates.
Neural heads take floating-point tensors; tabular heads take `torch.long`
state indices. Estimators do not convert NumPy inputs or move devices.

| Estimator | Input | Output in `EstimatorOutput` |
|---|---|---|
| `TabularVEstimator` | state indices `(...)` | `values.v (...)` |
| `TabularQEstimator` | state indices `(...)` | `values.q = (q (..., A),)` |
| `StateValueEstimator` | features `(..., F)` | `values.v (...)` |
| `DiscreteQEstimator` | features `(..., F)` | `values.q = (q (..., A),)` |
| `ContinuousQEstimator.evaluate` | features `(..., F)`, actions `(..., A)` | `values.q = (q (...),)` |
| `DiscreteAdvantageEstimator` | features `(..., F)` | `values.advantage (..., A)` |
| `CategoricalPolicyEstimator` | features `(..., F)` | `policy = Categorical`, batch `(...)`, scalar event |
| `GaussianPolicyEstimator` | features `(..., F)` | `policy = Independent(Normal, 1)`, batch `(...)`, event `(A,)` |
| `DeterministicPolicyEstimator` | features `(..., F)` | `policy = Tensor (..., A)` |
| `ActorCriticEstimator` | shared features `(..., F)` | child's policy and child's complete `ValueOutput` |

Tabular values are mutable buffers, not gradient-trained parameters. Neural
V, discrete Q, advantage, and policy means default to linear heads over the
supplied Representation. All seven neural estimators accept a keyword-only
`network: nn.Module | None = None`, so a custom network can replace the default
while keeping the same output semantics. Continuous Q defaults to a joint
`Linear -> ReLU -> Linear` network with `hidden_dim=128`; `hidden_dim` applies
only to the default network. Its batch
dimensions must match exactly; it does not broadcast actions across states.
`forward(features)` fails with a message to use `evaluate(features, actions)`;
through Model, use `model.evaluate_q(observation, actions)`.

All seven neural estimators require positive integer dimensions for both
default and custom networks (NumPy integers are accepted; bools are not).
These configuration checks run once in the constructor.

Gaussian standard deviations are learned through one `log_std` parameter per
action coordinate, shared across states. `log_prob(action)` and `entropy()`
sum over the action event and return shape `(...)`. The Gaussian and
deterministic policies are unbounded. Action bounds, tanh-squashed policies,
and their log-probability corrections need a separate implementation.
No estimator samples an action.

`log_std_init` must be finite. Before creating the Gaussian, the small
state-independent scale vector is checked so that its variance is at least
`torch.finfo(dtype).tiny` and at most `torch.finfo(dtype).max / 2`. Unsafe scales
raise a clear error instead of clamping or mutating `log_std`; valid estimates
keep their original mathematical behavior. This avoids scale underflow,
variance overflow, and an overflowing `2 * variance` denominator in Normal's
log probability. It does not guarantee finite results for all possible means
and action inputs.

`ActorCriticEstimator` shares its input features between its two children and
passes the policy object and value payload through without converting them.
It requires a policy from its policy child and at least one of V, Q, or
advantage from its value child. It contains no Representation, optimizer,
loss, or training procedure. Action-conditioned Q evaluation remains separate
from this wrapper's `forward(features)` path.

ContinuousQEstimator cannot be the value child in this feature-only wrapper:
Q(s,a) also needs actions, and the wrapper does not expose `evaluate()`.
Use separate Policy and Critic Models, evaluate the Policy with `forward()`,
and supply actions explicitly to the Critic's `evaluate_q()`. A Representation
instance can be shared if callers manage its shared parameters consistently.

The advantage head directly parameterizes an estimate to be trained with
appropriate advantage targets. It does not compute GAE or imply that a
dueling network head represents the true policy advantage.

Existing action selectors support scalar discrete Q tensors and Categorical
policies. Gaussian and deterministic action tensors require a compatible
continuous ActionSelector; they are usable through `Model.forward()` now.
Attaching a Categorical selector to these policies raises a clear type error.

For example, both the default and a custom Q head use the same contract:

```python
from torch import nn
from modular_rl.component.estimator import DiscreteQEstimator

default_q = DiscreteQEstimator(feature_dim=64, n_actions=4)
my_network = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 4))
custom_q = DiscreteQEstimator(feature_dim=64, n_actions=4, network=my_network)
```

Custom networks preserve all leading batch dimensions and return a Tensor.
StateValue and ContinuousQ networks return `(..., 1)` before the estimator
removes the final singleton axis. Gaussian networks return only the mean;
the estimator retains its learned, state-independent `log_std`.
Networks are registered directly as `self.network`, including their parameters
and buffers in Model's `parameters()`, `state_dict()`, and device/dtype moves.
Constructors do not run trial forwards. Each estimate checks only the output
Tensor and shape in a separate `_validate_output()` method; there are no full
tensor scans or implicit conversions.

See [Architecture](docs/ARCHITECTURE.md) for each network's input/output shape,
Gaussian and continuous-Q examples, and checkpoint compatibility details.

## Discrete action selection

`GreedyActionSelector` and `EpsilonGreedyActionSelector` consume exactly one
scalar discrete Q tensor in `values.q`, with shape `(..., A)`. The last axis
is actions and the preceding axes are batch dimensions. Multiple critics
must be combined explicitly before selection; distributional payloads need
a compatible selector or an explicit conversion to scalar Q values.
Selectors check only the semantic connection: exactly one Q tensor with
shape `(..., A)` and a nonempty action axis, or a Categorical policy.
They trust the internal `EstimatorOutput` contract and do not recheck Q
dtype/layout/finiteness or a distribution's internal probability validity.
Estimators provide valid scalar Q values; PyTorch handles distribution
construction and validation. Epsilon is checked when configured, not on
every selection. Selectors do not convert NumPy inputs.

For epsilon-greedy selection, if `M` is the set of maximal Q actions:

```text
P(a|s) = epsilon/A + (1-epsilon) * I(a in M)/|M|
```

Greedy selection uses the same rule with epsilon zero. Sampling shares
greedy mass uniformly among ties and uses PyTorch's RNG. Both selectors
return the sampled action's behavior `log_prob`. With `deterministic=True`,
both choose the smallest index among maximal Q actions, ignoring epsilon,
and return `log_prob=None`. Calling `.eval()` alone does not request
deterministic selection; pass the flag explicitly to `select()` or `Model.act()`.

`CategoricalActionSelector` uses only `EstimatorOutput.policy`, which must
be a `torch.distributions.Categorical`. It samples the policy and returns
`policy.log_prob(action)`, or selects the first mode with no log probability
in deterministic mode. Actor-critic values do not affect selection.
Policy estimators produce the distribution without sampling actions.

All three selectors expose `probabilities(estimate)` with shape `(..., A)`
for Expected SARSA and importance sampling. This returns the sampling
policy's probabilities, including epsilon exploration; it is not changed
by a previous deterministic call. Actions and log probabilities have shape
`(...)` (scalar tensors for a single observation). `probabilities()` is not
required by `BaseActionSelector` or future continuous selectors.

## Examples

```python
import numpy as np
from modular_rl.component.representation import IdentityRepresentation, MLPRepresentation
from modular_rl.component.estimator import TabularQEstimator, DiscreteQEstimator
from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.model import Model

# Discrete state index → Tabular Q(s,·)
tabular = Model(IdentityRepresentation(), TabularQEstimator(16, 4),
                EpsilonGreedyActionSelector(0.1))
tabular.forward(5).values.q[0]  # shape (4,)
tabular.act(5).action          # scalar action tensor

# NumPy observation → Tensor MLP → Q(s,·)
rep = MLPRepresentation(4, hidden_dims=(128, 64))
dqn = Model(rep, DiscreteQEstimator(rep.output_dim, 2),
            EpsilonGreedyActionSelector(0.1))
dqn.forward(np.zeros(4, dtype=np.float32)).values.q[0]  # shape (2,)
```

Run all examples and tests from the project root:

```bash
python -m examples.quickstart
python -m pytest -q
```

## Scope and migration

- The original standalone reference had 17 tests. This version adds the provided MLP API regression cases and moved-file import coverage.
- Old internal imports must be updated, e.g. `from modular_rl.model.model import Model` → `from modular_rl.model import Model` and `from modular_rl.component.representation.basic import MLPRepresentation` → `from modular_rl.component.representation import MLPRepresentation`.
- The public re-export `from modular_rl.model import Model, ValueOutput, EstimatorOutput, ActionOutput` remains available. `Representation`, `Estimator`, and `ActionSelector` are temporary aliases of the new `Base*` names for imports from their `base.py` modules.
- `Model.act` returns a Torch tensor in `ActionOutput`; Runner is responsible for converting it into Gymnasium's expected action format. The Model contract is not a complete algorithm training implementation.
- Action-conditioned critics `Q(s,a)` use the optional `Model.evaluate_q` path. Distributional critics require a compatible selector and update rule; no such algorithm is claimed implemented here.
- Runtime shape/device checks are included for the implemented paths. Comprehensive cross-component static compatibility and vector-environment training are outside this reference's scope.
