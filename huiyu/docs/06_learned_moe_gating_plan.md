# Learned MoE Gating 실험 계획

작성일: 2026-07-02

## 아이디어

기존 최종 조합은 모든 test 샘플에 같은 비율을 적용했다.

```text
final = 0.80 * GOH30_ODE_heavy + 0.20 * second_final
```

새 아이디어는 샘플마다 궤적 특성에 따라 second expert 비율을 다르게 선택하는 것이다.

```text
final_i = (1 - w_i) * GOH30_ODE_heavy_i + w_i * second_i
```

여기서 `w_i`는 궤적 feature와 expert disagreement를 보고 학습한다.

## 중요한 제약

2등 코드는 두 부분으로 나뉜다.

```text
second_final = 0.60 * second_base + 0.40 * second_phys
```

- `second_base`: 약 40개 멤버 앙상블 결과가 노트북에 test 예측으로 압축 내장되어 있음
- `second_phys`: 회전물리 3멤버이며 재학습 가능

2등 노트북 설명상 `second_base` 전체 재학습은 15~20시간 수준이라, 반나절 실험에서는 바로 OOF화하기 어렵다. 대신 `second_phys`는 5-fold OOF를 만들 수 있다.

따라서 이번 실험은 다음 구조로 진행한다.

1. train에서는 `GOH30_ODE_heavy OOF`와 `second_phys OOF` 사이의 최적 비율을 학습한다.
2. test에서는 학습된 gating을 `GOH30_ODE_heavy`와 `second_final` 또는 `second_phys`에 적용한다.

가장 실전적인 제출 후보는 `second_final`에 적용한 결과다. 다만 train에서 학습한 alt expert는 `second_phys`이므로 완전히 같은 expert를 학습한 것은 아니라는 점을 주의해야 한다.

## 1단계: 2등 phys OOF 생성

smoke test:

```bash
python huiyu/experiments/run_second_phys_oof.py \
  --folds 2 \
  --epochs 1 \
  --spec-limit 1 \
  --limit 120 \
  --device cpu \
  --out-dir outputs/second_phys_oof_smoke
```

full 실행:

```bash
.venv-cu128/bin/python huiyu/experiments/run_second_phys_oof.py \
  --folds 5 \
  --epochs 14 \
  --device cuda \
  --out-dir outputs/second_phys_oof_full
```

보수적으로 CPU 원본 설정에 맞추려면:

```bash
python huiyu/experiments/run_second_phys_oof.py \
  --folds 5 \
  --epochs 14 \
  --device cpu \
  --out-dir outputs/second_phys_oof_full
```

생성 파일:

- `outputs/second_phys_oof_full/oof_second_phys.npy`
- `outputs/second_phys_oof_full/pred_second_phys.npy`
- `outputs/second_phys_oof_full/second_phys_oof_metrics.csv`

## 2단계: learned second gating

smoke test:

```bash
python huiyu/experiments/learned_second_gating.py \
  --second-phys-dir outputs/second_phys_oof_smoke \
  --limit 120 \
  --folds 2 \
  --out-dir outputs/learned_second_gating_smoke
```

full 실행:

```bash
.venv-cu128/bin/python huiyu/experiments/learned_second_gating.py \
  --oof-dir outputs/goh30_oof_lite_gru_ode_h10_6_cuda \
  --second-phys-dir outputs/second_phys_oof_full \
  --folds 5 \
  --test-alt second_final \
  --out-dir outputs/learned_second_gating_full
```

비교용으로 `second_phys` test에 직접 적용:

```bash
.venv-cu128/bin/python huiyu/experiments/learned_second_gating.py \
  --oof-dir outputs/goh30_oof_lite_gru_ode_h10_6_cuda \
  --second-phys-dir outputs/second_phys_oof_full \
  --folds 5 \
  --test-alt second_phys \
  --out-dir outputs/learned_second_gating_phys_test
```

생성 파일:

- `learned_second_gating_oof_summary.csv`
- `oracle_weight_labels.csv`
- `meta_features.csv`
- 상위 후보 제출 CSV
- 후보별 test weight CSV

## 결과를 볼 때 기준

먼저 `learned_second_gating_oof_summary.csv`에서 다음을 본다.

1. `fixed_020`보다 learned 후보가 OOF hit을 올리는지
2. `oracle_grid`가 `fixed_020`보다 충분히 높은지
3. learned 후보의 `mean_w_alt`, `std_w_alt`, `min_w_alt`, `max_w_alt`가 과하게 흔들리지 않는지

해석 기준:

- `oracle_grid`만 높고 learned 후보가 낮으면, 샘플별 최적 비율은 존재하지만 feature로 일반화하지 못한 것이다.
- learned 후보가 `fixed_020`보다 높고 weight 분산이 적당하면, MoE gating 아이디어가 살아있다.
- learned 후보가 높더라도 `max_w_alt`가 0.40에 몰리거나 weight가 극단적이면 제출 리스크가 크다.

## 현재 smoke test 결과

120개 샘플 smoke 기준으로는 스크립트 경로가 정상 동작했다.

- 2등 phys OOF runner 정상 실행
- learned gating 정상 실행
- smoke 결과는 샘플 수가 너무 작아 성능 판단에는 쓰지 않는다.

## 최종 판단

이 실험은 기존 rule-based regime gating의 상위 버전이다.

rule-based:

```text
hard_turn/high_noise/vertical_change이면 second 비율 증가
```

learned gating:

```text
궤적 feature + expert disagreement를 보고 second 비율을 학습
```

단, 이번 반나절 버전은 `second_base` OOF가 없기 때문에 완전한 2등 final OOF gating은 아니다. full `second_phys` OOF 결과가 좋으면, 다음 단계로 `second_base`를 재현하거나 더 긴 시간으로 2등 base 계열 OOF를 만드는 것이 맞다.

