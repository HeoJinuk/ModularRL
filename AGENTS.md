# ModularRL 개발 맥락

## 목적과 범위
- 연구 및 교육용 강화학습 라이브러리. Sutton & Barto의 Tabular RL부터 DQN, A2C, PPO, SAC, Dyna, World Model 계열까지 확장 가능한 구조를 지향한다. 목표 알고리즘을 모두 구현된 것으로 간주하지 않는다.
- 초심자가 이해할 수 있는 단순한 Base Class와 연구자가 자유롭게 교체할 수 있는 명시적인 입출력 계약을 유지한다.
- GUI는 현재 범위 밖이다. 컴포넌트의 독립성과 교체 가능성은 핵심 요구사항이다.
- 새로운 추상 계층, 불필요한 Registry 또는 GUI 코드를 도입하지 않는다.

## 현재 저장소의 범위
- 현재는 Model 계약과 경험 데이터(Transition/TransitionBatch), 기본 ReplayBuffer의 참조 구현이다. Runner, Agent, Learner 전체, TargetGenerator, UpdateRule, WorldModel/Planner와 학습 알고리즘은 아직 포함하지 않는다.
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
- `data/outputs.py`
- `model/rl_model.py`, `model/input_adapter.py`
- `component/representation/{base,identity,mlp}.py`
- `component/estimator/{base,actor_critic}.py`
- `component/estimator/policy/`, `component/estimator/value/`
- `component/action_selector/`

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

## Git 작업 원칙
- 작업 전 branch와 `git status`를 확인하고 기존 수정·미추적·stage 상태를 파악한다. 사용자의 기존 변경을 삭제하거나 덮어쓰지 않는다.
- 작업 후 전체 pytest와 관련 예제를 실행하고 통과한 변경을 검토한다. 실패하면 원인을 보고하고 임의로 커밋하지 않는다.
- 이번 작업과 관련된 파일 경로만 명시적으로 stage한다. `git add .`, `git add -A`를 사용하지 않고 관련 없는 기존 변경을 포함하지 않는다.
- 비밀번호·인증서·.env·대용량 데이터와 cache·임시 작업 파일은 커밋하지 않는다.
- 의미 있는 Conventional Commit 메시지를 사용하고 commit hash와 최종 Git 상태를 보고한다. 사용자 변경과 충돌하면 커밋 전에 확인한다.
- 별도 요청 없이 push하지 않으며 `--force`, `reset --hard` 등 파괴적 명령을 사용하지 않는다.
