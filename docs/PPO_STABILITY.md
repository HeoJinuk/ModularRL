# PPO 선택적 안정화: 계약과 CartPole 실험

## 공개 API와 기본값 호환성

```python
learner = L.PPO(model, optimizer, T.GAE(.99, .95), U.PPO(.2, .5, .01),
                epochs=4, minibatch_size=64, seed=7,
                max_grad_norm=None, target_kl=None)
```

두 옵션은 기본 None이며 0/bool/NaN/Inf를 허용하지 않는 양의 유한 실수다.
기존 89aa872의 learn 계산을 고정 참조로 두고 두 번의 연속 학습에서 여섯 Loss/진단 지표,
Gradient, Parameter, SGD Momentum과 Shuffle/행동 RNG를 정확히 비교한다.
옵션을 끈 경로의 값은 float32/float64 CPU에서 완전히 같다. CUDA는 미사용 환경으로 skip한다.
기본 하이퍼파라미터나 U.PPO 수식은 변경하지 않았다.

Clipping은 backward 뒤 optimizer.step 앞에서 Optimizer가 관리하는 학습 가능한 고유 Parameter의
공동 L2 Norm에 적용한다. Actor/Critic/공유 Representation을 따로 Clip하지 않는다.
비유한 Norm은 그 Minibatch Step 전에 명시적으로 실패한다. 이전 완료 Step을 되돌리는 기능은 아니다.

KL 진단은 다음 Epoch가 남아 있을 때 Epoch 종료 뒤 전체 Rollout을 eval/no_grad로 재평가한다.
`log_ratio = current_log_prob - collected_old_log_prob`에 대해
`mean(exp(log_ratio)-1-log_ratio)`를 사용한다. U.PPO의 근사 정의와 같다.
`KL > target_kl`이면 남은 Epoch를 중단하며 별도 1.5배 문턱은 사용하지 않는다.
BN Running Stats를 바꾸거나 Dropout을 실행하지 않고 모든 하위 모듈의 원래 모드를 복원한다.
진단을 위한 행동 샘플링/Shuffle은 없다. 마지막 Epoch 뒤 불필요한 판정도 하지 않는다.

PPOOutput은 기존 여섯 Tensor와 optimizer_steps에 기본값 있는 epochs_completed/early_stopped를 추가한다.
기존 7개 positional 생성은 유지한다. 여섯 Tensor는 실제 처리한 Minibatch 표본 수로 가중 평균한다.
따라서 반환 approx_kl은 학습 중 진단의 평균이고 최종 Epoch 정책의 전체 Rollout KL 자체가 아니다.
old value/log_prob, GAE/Returns와 정규화 Advantage는 첫 업데이트 전에 고정한다.
Checkpoint 재개는 Model/Optimizer/Shuffle RNG와 두 옵션 포함 설정을 함께 복원한다.

## 비교 조건

실험은 [experiments/ppo_stability.py](../experiments/ppo_stability.py)의 명시적 수집 루프를 사용한다.
전체 원시 지표는 [ppo_stability.json](../experiments/results/ppo_stability.json)에 있다.

```sh
python -m experiments.ppo_stability --seeds 7 11 19 23 42 --output experiments/results/ppo_stability.json
```

각 설정은 같은 Seed, Network, 초기화·행동·환경·Shuffle Seed, 학습률과 환경 전이 예산을 사용한다.
업데이트가 달라지면 정책과 이후 경험도 달라지므로 같은 Seed가 같은 경험 전체를 의미하지는 않는다.

- CartPole-v1, Async SAME_STEP/copy=True, N=4, Horizon=128, 총 81,920 환경 전이(160 Rollout).
- MLP Representation 4->64 Tanh, Categorical(2)와 Scalar V. 동일한 eval Forward로 수집/재평가.
- Adam lr=.0003, gamma=.99, lambda=.95, epochs=4, minibatch_size=64, Advantage 정규화.
- U.PPO clip_epsilon=.2, value_coef=.5, entropy_coef=.01.
- baseline: 두 옵션 None; clip: max_grad_norm=.5; kl: target_kl=.02; clip_kl: 둘 다.
- 학습 전후 별도 Greedy 평가 각 20 Episode, 평가 Seed는 training_seed+1000+i.
- 실제 Optimizer/완료 Epoch/조기 종료/표본 사용 수를 기록한다. Norm 계측은 RNG/Gradient를 바꾸지 않는다.

Return의 설정별 평균/표준편차는 다섯 Seed 사이의 population 통계(ddof=0)다.
단일 Seed에서 반드시 향상한다거나 이 옵션이 모든 환경을 안정화한다고 가정하지 않는다.

## 실제 결과

학습 전후 Return은 다음과 같다. 각 값은 해당 Seed의 Greedy 평가 20 Episode 평균이다.

|Seed|학습 전(공통)|baseline|clip|kl|clip_kl|
|---|---|---|---|---|---|
|7|9.20|415.25|421.75|415.25|368.40|
|11|10.60|483.70|314.75|289.75|314.75|
|19|9.35|427.65|465.25|427.65|465.25|
|23|9.20|147.50|138.25|147.50|486.40|
|42|9.15|197.50|466.45|413.45|466.45|

모든 실행은 환경 전이 81,920개/160 Rollout이다. 다음 Step/Epoch 범위는 각 설정의 Seed별 실제 합계 범위이며
조기 종료는 다섯 Seed 전체의 발생 Rollout 수다. Return 평균±표준편차는 Seed 간 통계다.

|설정|평가 Return 평균±표준편차|Optimizer Step 범위|완료 Epoch 범위|조기 종료 합계|
|---|---|---|---|---|
|baseline|334.32 ± 135.05|5120–5120|640–640|0|
|clip|361.29 ± 124.45|5120–5120|640–640|0|
|kl|338.72 ± 107.99|5104–5120|638–640|2|
|clip_kl|420.25 ± 66.86|5112–5120|639–640|2|

KL/Clip Fraction은 실제 처리 표본 수로 Seed 간 가중 평균하고, Norm 평균은 실제 Optimizer Step 수로
가중 평균한다. 최대 Norm은 전체 다섯 Seed의 최대값이다. Clipping 뒤 0.50000006 수준의 차이는 float32 반올림이다.

|설정|학습 approx_kl|Clip Fraction|Norm 평균 전→후|Norm 최대 전→후|
|---|---|---|---|---|
|baseline|0.003111|0.029670|12.5209 → 12.5209|58.7189 → 58.718948|
|clip|0.002609|0.023241|12.5083 → 0.5000|60.0056 → 0.500000|
|kl|0.003094|0.029573|12.5553 → 12.5553|58.7189 → 58.718948|
|clip_kl|0.002558|0.022011|12.5174 → 0.5000|60.0056 → 0.500000|

기본 PPO의 다섯 Seed Return은 89aa872 실험과 정확히 일치했다. Clipping은 전체 Norm을 지정 범위로
제한했고 KL 제한은 일부 Rollout의 다음 Epoch를 생략했다. 평가 Return 변화는 Seed마다 달랐다.
이 작은 CartPole 비교만으로 일반적인 성능 우위를 주장하지 않으며 기본값을 바꾸지 않는다.

## 결정적 검증

[tests/test_ppo_stability.py](../tests/test_ppo_stability.py)는 다음을 검증한다.

- 옵션 None의 기준 구현과 정확한 동등성, 공동 Clip 전후 Norm과 고유/동결/미관리 Parameter 구분.
- 비유한 Gradient에서 Step 미실행, NumPy 실수 설정과 잘못된 옵션 거절.
- 작은 KL 문턱에서 1 Epoch/2 Step으로 중단, 큰 문턱에서는 기존과 동일한 전 Epoch 실행.
- 마지막 작은 Minibatch를 포함한 실제 표본 분모, Target 한 번 생성과 Snapshot 불변.
- BN/Dropout 혼합 모드·Buffer·RNG 복원 및 U.PPO와 KL 수식 일치.
- Model/Optimizer/Shuffle Checkpoint 재개와 기존 PPOOutput 생성 호환성.

전체 테스트와 배포 검증 결과는 [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md)에 기록한다.
