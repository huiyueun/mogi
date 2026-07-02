# Mosquito Trajectory Noise Removal Ablation

## 개요

이 실험은 모기 3D 비행 궤적 예측 문제에서, 명시적인 결측치는 없지만 센서 노이즈처럼 보이는 비정상 trajectory를 학습 데이터에서 제거했을 때 성능이 좋아지는지 확인하기 위한 ablation 실험이다.

사용한 기준 문서는 `codex_mosquito_noise_removal_instructions.md`이며, 모델 구조는 Private LB 1st 코드의 `AttnGRU`, `ODEModel`, `HyperPhysics` 구성을 사용했다.

중요한 정책은 다음과 같다.

- train set에서만 noise sample을 제거한다.
- validation set은 제거하지 않고 그대로 평가한다.
- test set은 어떤 row도 제거하지 않는다.
- test set에는 quality flag만 생성한다.

## 데이터 구조

각 sample은 400ms 동안 관측된 3D trajectory이다.

- timestep 수: 11개
- timestep 간격: 40ms
- 좌표: `x`, `y`, `z`
- target: 관측 이후의 미래 3D 위치

데이터에는 `NaN`, 빈 값, `inf`, 중복 trajectory 같은 명시적인 결측은 없었다. 대신 다음과 같은 암묵적 noise 가능성을 고려했다.

- 반복 좌표
- 거의 움직이지 않는 step
- 한 점만 튀는 spike
- 비정상적으로 큰 step distance
- 큰 acceleration 또는 jerk
- 최근 trajectory 흐름과 target 위치가 맞지 않는 경우

## Noise Criteria

### Criteria A: Conservative

Criteria A는 매우 의심스러운 sample만 제거하는 보수적인 기준이다.

사용한 주요 feature는 다음과 같다.

- `min_step`
- `max_step`
- `median_step`
- `max_step_over_median`
- `zero_step_count`
- `tiny_step_lt_1e4`
- `max_acc`
- `max_jerk`
- `max_interp_resid`
- `max_interp_resid_over_med_step`
- `straightness`
- `target_minus_cv80`
- `future_v_change_norm`

제거 대상은 반복 좌표, 거의 0에 가까운 이동, 매우 큰 step, 큰 acceleration/jerk, target consistency가 낮은 sample이다.

### Criteria B: Strict

Criteria B는 Criteria A를 포함하고, spike성 noise를 더 적극적으로 제거한다.

추가 조건은 다음과 같다.

```python
max_interp_resid_over_med_step > 1.5
max_step_over_median > 3.0
```

즉, 전체 trajectory 흐름 대비 특정 한 점이 튀거나, median step 대비 특정 step이 지나치게 큰 경우를 더 강하게 제거한다.

## Train / Validation Split

전체 train data는 5-fold로 나누었다. split은 random seed 기반이 아니라 sample id 기반 deterministic hash 방식이다.

```python
fold_id = md5(sample_id) % 5
```

이번 실험 설정은 다음과 같다.

- `folds = 5`
- `val_fold = 0`
- validation rows: 2020
- origin train rows: 7980

각 설정에서 같은 validation fold를 사용했다. Noise 제거는 validation fold가 아닌 train fold에만 적용했다.

```python
val_mask = fold_id == 0
train_mask = fold_id != 0
train_mask = train_mask & ~noise_remove_mask
```

이 방식으로 validation set 자체의 난이도와 분포를 동일하게 유지하면서, train data cleaning 효과만 비교했다.

## Test Set Handling

test set은 제거하지 않는다. 대회 제출에서는 모든 test sample에 대해 예측값이 필요하기 때문이다.

따라서 test set에 대해서는 다음만 수행한다.

- 동일한 trajectory quality feature 계산
- `remove_criteria_a_conservative` flag 생성
- `remove_criteria_b_strict` flag 생성
- `test_quality_report_with_noise_flags.csv` 저장

추후 실험에서는 suspicious test sample에 대해 원본 trajectory 예측과 보정 trajectory 예측을 blending할 수 있지만, 이번 비교에서는 test row 제거를 하지 않았다.

## Model Modules

Private LB 1st 코드의 세 가지 모델 module을 사용했다.

### AttnGRU

양방향 GRU와 attention pooling을 사용하는 sequence model이다. 11개 timestep의 trajectory 흐름을 학습해 target 3D 좌표를 예측한다.

### ODEModel

Neural ODE 방식의 모델이다. trajectory의 동역학적 흐름을 반영하기 위해 ODE 기반 예측 구조를 사용한다.

### HyperPhysics

물리 gray-box 성격의 모델이다. 최근 이동 방향, 속도, 회전 성분 등을 기반으로 trajectory 이후 위치를 예측한다.

원래 Private LB full recipe는 각 모델을 10개씩 학습하여 총 30-model ensemble을 구성한다.

이번 ablation은 빠른 비교를 위해 축소 설정으로 실행했다.

- `n_each = 1`
- `GRU/ODE epochs = 8`
- `HyperPhysics epochs = 3`
- device: CUDA
- GPU: NVIDIA GeForce GTX 1660

## Evaluation Metric

예측 좌표와 정답 좌표의 Euclidean distance를 계산했다.

```python
error = ||prediction - target||
hit = mean(error <= 0.01)
```

`hit`은 예측값이 정답 반경 `0.01` 안에 들어온 비율이다.

## Validation Results

아래 결과는 official LB score가 아니라 train 내부 validation split에서 측정한 결과이다.

| setting | train rows | removed total | removed from train fold | validation rows | hit | hits | mean distance | median distance | q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| origin | 7980 | 0 | 0 | 2020 | 0.071782 | 145 | 0.019154 | 0.015279 | 0.028998 |
| criteria A | 7949 | 37 | 31 | 2020 | 0.066832 | 135 | 0.019376 | 0.015534 | 0.029179 |
| criteria B | 7799 | 219 | 181 | 2020 | 0.073762 | 149 | 0.019179 | 0.015310 | 0.029034 |

## Result Interpretation

Hit score 기준으로는 Criteria B가 가장 좋은 결과를 보였다.

- Origin hit: `0.071782`
- Criteria A hit: `0.066832`
- Criteria B hit: `0.073762`

Criteria A는 37개 sample을 noise로 판단했고, 그중 validation fold를 제외한 train fold에서 31개를 제거했다. 그러나 이번 split에서는 origin보다 hit score가 낮았다.

Criteria B는 219개 sample을 noise로 판단했고, train fold에서 181개를 제거했다. 그 결과 hit score는 origin보다 약간 높았다.

다만 mean distance 기준으로는 origin이 가장 낮다.

- Origin mean distance: `0.019154`
- Criteria B mean distance: `0.019179`

따라서 최종 선택은 목표 metric에 따라 달라질 수 있다.

- 반경 `0.01` 안에 들어오는 hit score를 중시하면 Criteria B가 가장 좋은 후보이다.
- 평균 거리 자체를 중시하면 origin도 충분히 경쟁력이 있다.

이번 대회 metric이 hit 성격에 가깝다면 Criteria B 기반 train cleaning을 추가 실험할 가치가 있다.

## Files

업로드한 주요 파일은 다음과 같다.

- `codex_mosquito_noise_removal_instructions.md`
- `Private_LB_1st_noise_ablation_A_B_gpu.ipynb`
- `README.md`

실험 결과 파일은 local 환경에서 다음 경로에 저장되었다.

```text
private_lb_noise_ablation_gpu_n1_e8/private_lb_noise_ablation_results.csv
```

## 주의사항

이번 결과는 빠른 validation 비교용 축소 실험이다. Full Private LB recipe인 30-model ensemble 결과는 아니다.

Full setting으로 비교하려면 다음처럼 설정을 바꿔야 한다.

```text
n_each = 10
gru_ode_epochs = 55
h_epochs = 12
```

또한 `Data/` 폴더는 데이터 라이선스와 용량 문제 때문에 GitHub에 업로드하지 않는 것이 좋다.
