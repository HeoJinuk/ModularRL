# ModularRL — Model Contract v0.2

> This is a **reference implementation of the Model layer, experience contracts, ReplayBuffer, TD/MC targets, TD UpdateRules, TabularQLearner, DQNLearner, tabular DP, MC/TD(0) prediction and on-policy MC control**, not the entire ModularRL framework. Examples cover FrozenLake Q-learning, CartPole DQN, Sutton Gridworld DP/MC/TD comparison and the book's Gridworld MC control. Runner, Agent, SARSA, off-policy MC control and other Neural Learners are not implemented.

## Layout

```text
modular_rl/
├── data/
│   ├── outputs.py                  # ValueOutput / EstimatorOutput / ActionOutput
│   ├── experience.py               # Transition / TransitionBatch
│   ├── episode.py                  # Owned, ordered Transition snapshots
│   ├── updates.py                  # TDLossOutput
│   └── mdp.py                      # TabularMDP / joint TransitionOutcome atoms
├── model/
│   ├── rl_model.py                 # Model.forward / Model.act / Model.evaluate_q
│   └── input_adapter.py           # NumPy, Tensor, scalars → Tensor
├── learner/
│   ├── tabular_q.py               # External replay batch → Q-learning update
│   ├── dqn.py                     # Online gather → target bootstrap → optimizer step
│   ├── mc_prediction.py           # Complete episode → first/every sample average
│   ├── mc_control.py              # Complete episode → state-action sample averages
│   └── tabular_td0.py             # External batch → next V → TD target → V update
├── dp/
│   └── algorithms.py              # Policy evaluation/improvement/iteration, value iteration
├── environment/
│   ├── gridworld.py               # Sutton kernel and Gymnasium environment constructor
│   ├── dlfs4_gridworld.py         # Book's 3x4 kernel, coordinates and environment constructor
│   ├── tabular_env.py            # Gymnasium Env sampling from a TabularMDP
│   ├── episode_collection.py      # reset/step → ordered Episode, external action callback
│   └── episode_sampler.py         # Legacy sampler API retained for compatibility
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
    │   ├── td0.py                 # One-step TD formula; caller supplies next_value
    │   └── monte_carlo.py         # Reverse complete-episode returns, no bootstrap
    ├── update_rule/
    │   ├── tabular.py             # Sequential in-place V/Q updates
    │   └── td_loss.py             # TD error + mean neural loss; no optimizer calls
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

## Gymnasium environments

Gymnasium is a required dependency (`pip install -e .`). The existing `examples`
extra remains accepted for installation compatibility; notebook tools remain optional.
`TabularMDP` holds a known transition model for pure NumPy DP. `TabularEnv(mdp,
start_state=None, render_mode=None)` samples that same model through Gymnasium.
It exposes `Discrete` state/action spaces, `reset`, `step`, `render` and `close`.
Policies, Models and Learners stay outside the environment.

```python
from modular_rl.data import Transition
from modular_rl.environment import make_dlfs4_gridworld_env

env = make_dlfs4_gridworld_env(render_mode="ansi", max_episode_steps=200)
try:
    observation, info = env.reset(seed=42, options={"start_state": 7})
    action = 0  # Or int(model.act(observation).action.item()).
    next_observation, reward, terminated, truncated, info = env.step(action)
    transition = Transition(observation, action, reward, next_observation, terminated, truncated)
    print(env.render())
    # TD/Q: learner.learn(transition.to_batch()); continue until either flag.
finally:
    env.close()
```

`make_sutton_gridworld_env()` samples a nonterminal start uniformly and defaults
to a 10,000-step `TimeLimit`. `make_dlfs4_gridworld_env()` starts at the book's
fixed state 7 and defaults to 2,000 steps. Both accept `render_mode` and
`max_episode_steps=None` to omit the wrapper. `env.unwrapped.mdp` exposes the
known kernel explicitly for DP. Original `sutton_gridworld()` and
`dlfs4_gridworld()` still return `TabularMDP`.

Seed the first reset, then use `reset()` to continue the environment's own RNG
for starts/outcomes. Policy RNG is separate; `action_space.sample()` requires
`action_space.seed(...)` for reproducibility. Options override the start for one
reset only. Natural termination requires reset before another step. The base Env
never auto-resets or truncates; external `TimeLimit` owns the cap. Callers reset
after **either** boundary and retain the actual final observation. At a natural
ending exactly on the cap, official TimeLimit can report both flags True.
`ansi` returns readable state IDs; `human` prints them on reset/step/render.

For tabular MC, `collect_episode(env, select_action, seed=None, options=None)`
collects an ordered, owned `Episode` using an external action callback. It stops
at either flag and leaves truncated fragments explicit. MC rejects truncated-only
episodes; TD/Q bootstrap unless terminated. FrozenLake and CartPole use the same
Gymnasium reset/step API, with their existing learning loops unchanged.

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

`TabularVUpdateRule(alpha).update(v_table, states, target)` and
`TabularQUpdateRule(alpha).update(q_table, states, actions, target)` update the
original floating table under `no_grad` and return signed errors. Indices are
long, scalar `()` or batch `(B,)`; target matches their shape and the table's
dtype/device. Targets are copied before writing, so table aliases remain safe.
Duplicate indices are processed in input order with errors recomputed before
each row update. Alpha accepts real scalars, including NumPy, in `(0, 1]`.

`TDLossUpdateRule(loss_fn=None).compute(prediction, target, *, weights=None)`
returns a frozen `TDLossOutput(td_error, loss)`. Default loss is unreduced MSE;
an injected callable must return a floating loss Tensor of the input shape.
Optional weights are keyword-only and use `mean(weights * per_sample_loss)`
without normalization. Inputs share shape, floating dtype and device. Graphs
are retained; Learner controls target detachment, backward and optimizer steps.

```python
import torch
from modular_rl.component.update_rule import TabularQUpdateRule, TDLossUpdateRule

q_table = torch.zeros(1, 1)
errors = TabularQUpdateRule(alpha=0.5).update(
    q_table, torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([1.0, 3.0]),
)
torch.testing.assert_close(errors, torch.tensor([1.0, 2.5]))
assert q_table.item() == 1.75

prediction = torch.tensor([1.0, 3.0], requires_grad=True)
result = TDLossUpdateRule().compute(prediction, torch.tensor([3.0, 1.0]))
torch.testing.assert_close(result.td_error, torch.tensor([2.0, -2.0]))
assert result.loss.item() == 4.0 and prediction.grad is None
result.loss.backward()  # Called by Learner, outside the UpdateRule.
```

`TabularQLearner(model, target_generator, update_rule).learn(batch)` accepts a
`TransitionBatch` of shape `(B,)` and returns signed TD errors in the same row
order. Model must use TabularQEstimator and preserve table state indices through
its Representation. ActionSelector is optional. Learner prepares device/dtype,
selects max Q(next_state, ·), delegates target creation and applies the tabular
update. It has no Buffer or Optimizer. Sampling and environment handling stay
in the caller; all Batch targets are fixed before sequential updates begin.

Run an explicit, seeded FrozenLake-v1 loop without a Runner or Agent:

```bash
python -m pip install -e ".[examples]"
python -m examples.frozenlake_tabular_q --episodes 1000 --eval-episodes 100 --seed 7
```

The example uses the non-slippery 4x4 map, prints before/after greedy evaluation,
episode boundaries and Q-table changes, and closes the environment in finally.
It converts environment rewards to float before creating Transition while
preserving terminated/truncated independently. Math is checked in deterministic
unit tests; sparse-reward evaluation success is reported without a test threshold.
FrozenLake smoke tests use the required Gymnasium dependency.

`DQNLearner(online_model, target_model, optimizer, target_generator, update_rule)`
uses the same external `learn(batch)` entry point. It gathers one discrete Q
prediction per action, builds a no-grad bootstrap with the target model, computes
the existing TD loss and runs zero_grad/backward/step. Its `TDLossOutput` contains
detached pre-update errors `(B,)` and a scalar loss. Optimizer parameters belong
only to the online model; the independent target model is frozen and kept in eval
mode. Call `sync_target()` explicitly for the initial copy and scheduled hard
synchronization of parameters and buffers. Build/move both compatible models
before constructing the optimizer. The learner never chooses actions or samples
replay. See [Architecture](docs/ARCHITECTURE.md) for shape and checkpoint contracts.

```bash
python -m examples.cartpole_dqn
# Short environment/optimizer smoke run, without a learning-performance assertion:
python -m examples.cartpole_dqn --episodes 4 --eval-episodes 2 --max-steps 5 --batch-size 2 --warmup-steps 2
```

The explicit CartPole loop owns epsilon decay, warm-up, replay sampling and target
synchronization. It reports episode rewards, losses, signed/absolute TD error
statistics and greedy evaluation before/after training. Tests check deterministic
updates and short real environment loops; full training reward is an example result.

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

## Tabular dynamic programming

DP computes full expectations from a known `TabularMDP`; it does not sample
ReplayBuffer or use the inference `Model`. `TransitionOutcome` preserves the
joint next-state, reward and termination distribution for every state/action.
The algorithms use CPU NumPy float64 values `(S,)` and policy probabilities
`(S, A)`, synchronous sweeps and uniform probability among exactly tied greedy
actions. `DPResult` reports values, policy, convergence, iteration counts and
the last value change. Reaching a limit returns `converged=False`; gamma=1
does not guarantee convergence for an improper policy.

```python
from modular_rl.dp import policy_evaluation, policy_iteration, uniform_policy, value_iteration
from modular_rl.environment import sutton_gridworld

mdp = sutton_gridworld()
evaluation = policy_evaluation(mdp, uniform_policy(mdp), gamma=1, theta=1e-10)
policy_result = policy_iteration(mdp, gamma=1, theta=1e-10)
value_result = value_iteration(mdp, gamma=1, theta=1e-10)
assert evaluation.converged and policy_result.converged and value_result.converged
print(evaluation.values.reshape(4, 4))
```

```bash
python -m examples.gridworld_dp
```

The example prints random-policy values, both optimal value tables, all selected
optimal actions, convergence and reference errors. Gridworld's kernel and policy
rows can later be reused for MC/TD comparison under the same fixed policy.
See [Architecture](docs/ARCHITECTURE.md) for APIs, terminal handling and limits.

## Tabular Monte Carlo prediction

`Episode(transitions)` snapshots the existing Transition fields into owned,
detached CPU tensors while preserving time order, continuity and boundary flags.
It represents one nonempty episode, including an explicitly open/truncated one,
rather than shuffled replay or a general trajectory batch.

`MonteCarloReturnGenerator(gamma).generate(rewards)` computes returns `(T,)` in
reverse. `MCPredictionLearner(model, return_generator, visit_mode="first")` uses
TabularVEstimator and learns only naturally terminated episodes. It selects first
visits in chronological order, or every visit, then applies `V[s] += (G-V[s])/count[s]`.
`learn(episode)` returns all discounted targets `(T,)` with no graph.
It never queries an MDP or chooses the behavior policy. The final next state is
not another visit. Save its `visit_counts` separately alongside Model.state_dict,
with the same gamma, visit mode and policy when resuming.

New examples collect with `make_sutton_gridworld_env()` and
`collect_episode(env, select_action, seed=...)`. Environment starts/outcomes use
seed; the external policy RNG uses seed+1. The unchanged
`TabularEpisodeSampler(mdp, policy, seed=...).sample(start_state, max_steps=...)`
API remains available for compatibility. Both paths preserve explicit cutoffs;
the example aborts at a truncated-only episode instead of silently learning a
partial return or selecting only completed short episodes.

```bash
python -m examples.gridworld_mc
python -m examples.gridworld_mc --episodes 100 --seed 7
```

Both visit modes receive identical episodes under the uniform policy. The example
prints episode/step totals, state visit/update counts, estimated V and DP-reference
RMSE over the 14 nonterminal states. In the earlier legacy Sampler experiment
with seed=7 and 20,000 episodes, RMSE was
0.199544 (first) and 0.283310 (every), over 365,007 environment steps. These are
historical results: the Gymnasium path has different RNG draw order and is not
expected to generate the same episodes for that seed. Tests check distributions,
mathematics and repeatability without requiring monotonically decreasing RMSE.
See [Architecture](docs/ARCHITECTURE.md) for contracts and checkpoint responsibilities.

## Tabular on-policy Monte Carlo control

`TabularMCControlLearner(model, return_generator, visit_mode="first").learn(episode)`
updates the existing TabularQEstimator table and returns all discounted returns
`(T,)`, detached, on the table's dtype/device. First/every selection is by the
**(state, action)** pair, in chronological order. It uses incremental sample
averages and long `visit_counts` `(S,A)`; save these separately with Model.state_dict,
gamma and visit mode. The final next observation is not another visit.

```python
from modular_rl.component.action_selector import EpsilonGreedyActionSelector
from modular_rl.component.estimator import TabularQEstimator
from modular_rl.component.representation import IdentityRepresentation
from modular_rl.component.target_generator import MonteCarloReturnGenerator
from modular_rl.environment import make_dlfs4_gridworld_env, collect_episode
from modular_rl.learner import TabularMCControlLearner
from modular_rl.model import Model

model = Model(IdentityRepresentation(), TabularQEstimator(11, 4),
              EpsilonGreedyActionSelector(epsilon=0.1))
learner = TabularMCControlLearner(model, MonteCarloReturnGenerator(gamma=0.9))
env = make_dlfs4_gridworld_env()
try:
    episode = collect_episode(env, lambda state: model.act(state).action.item(), seed=7)
    returns = learner.learn(episode)
finally:
    env.close()
```

Set `torch.manual_seed(...)` for behavior-action reproducibility as well as the
environment seed. Keep Q and epsilon fixed during collection. After learning,
the same Model.act uses updated Q in the next episode. Learner owns no environment,
policy matrix, sampler or optimizer. Truncated-only episodes are rejected.
Evaluation uses `model.act(..., deterministic=True)` without exploration or
learning. Change epsilon at episode boundaries in the external loop if desired.

```bash
python -m examples.gridworld_mc_control --episodes 1000 --seed 7 --epsilon 0.1
```

With gamma=.9, seed=7 and 1,000 first-visit episodes, training collected 5,672
steps with no cutoff. Before training, greedy evaluation hit the 20-step cap;
after training it reached the goal in 5 steps with reward +1, matching the DP
optimal path length. The last 100 training episodes had mean reward 1.0.
Fixed epsilon continues exploration; these measured results do not assert all Q
values equal Q* or impose a stochastic score threshold in pytest.

## Tabular TD(0) prediction and DP/MC comparison

`TabularTD0PredictionLearner(model, target_generator, update_rule).learn(batch)`
combines a TabularVEstimator Model with the existing TD0TargetGenerator and
TabularVUpdateRule. It returns signed TD errors `(B,)` without a graph. Integer
states are normalized to long, and floating rewards follow the live table's
dtype/device. It does not choose actions, sample a buffer or use an MDP.

All targets are evaluated before any batch write. With `B>1`, targets are frozen
while duplicate states update in input order. For online TD, pass one transition
at a time with `B=1`. Only termination masks bootstrap; a time-limit truncation
uses the actual final observation supplied by the collector.

```bash
python -m examples.gridworld_td
python -m examples.gridworld_td --episodes 5000 --seeds 7 11 19 --alphas 0.01 0.05 0.1
python -m examples.gridworld_td --episodes 20 --log-every 5
```

The independent first/every-visit MC and three TD models receive identical
completed episodes under the uniform policy, gamma=1 and uniform nonterminal
starts. TD processes each already collected episode chronologically with B=1;
this example does not implement a live streaming Runner. A cutoff aborts before
any learner processes it. The example reports values, RMSE over 14 nonterminal
states, episode/step checkpoints, alpha, seed and final RMSE mean/std across seeds.
The Python `train_and_compare()` result also exposes these checkpoints as `history`.
Constant-alpha TD retains sampling variability; no ordering of MC/TD performance
or monotonic RMSE is imposed by tests. DP remains pure NumPy Bellman evaluation.

Gymnasium-path verification with seed=7 and 1,000 episodes collected 17,609
steps with no cutoff. Both MC and TD examples produced the same MC values:
first RMSE=0.758770, every=0.600755. TD RMSE was 9.454563 / 1.738354 / 1.741590
for alpha=0.01 / 0.05 / 0.1. These are measured results, not performance thresholds.

## Jupyter Notebooks — Deep Learning from Scratch 4

Chapters 4–6 tutorials are in [notebooks/dlfs4](notebooks/dlfs4/README.md).
They use the book's **3×4** GridWorld, separate from Sutton's 4×4 environment.
DP, MC Prediction, TD(0) Prediction and Q-learning call ModularRL's public APIs.
The source correspondence table records the sample-average MC Control extension
and explicitly lists unsupported off-policy MC Control and SARSA exercises.

From the project root, install the optional stack and register/select its kernel:

```bash
conda activate modularrl-py312
python -m pip install -e ".[notebooks,examples]"
python -m ipykernel install --user --name modularrl-py312 --display-name "Python 3.12 (ModularRL)"
jupyter lab notebooks/dlfs4
```

Select **Python 3.12 (ModularRL)** in each notebook, then Run All. No installation or
sys.path discovery runs inside notebooks. `Transition.to_batch()` provides owned,
detached B=1 tensors for `learner.learn(transition.to_batch())`, retaining dtype and
Tensor device; the Learner still selects the training device/floating dtype.

```bash
python -m pytest -q tests/test_dlfs4_notebooks.py
```

The test runs each notebook twice in the same kernel and compares text/numeric/PNG
outputs. Notebook dependencies are optional for the core library.

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
