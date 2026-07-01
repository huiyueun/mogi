# 모기 궤적 예측 실험 진행 정리

작성일: 2026-07-01

## 0. 요약

현재 확인된 최고 제출:

| 제출 파일 | 공개 점수 | 비공개 점수 |
| --- | ---: | ---: |
| `outputs/second_place_expert_blends/confirmed_ode_heavy_800_second200.csv` | 0.7034 | 0.7042 |

현재 최고 절차:

- 1등 GOH30 코드에서 구성요소별 예측을 생성
- GOH30을 ODE 강화 비율로 다시 섞음
  - GRU 20%
  - ODE 60%
  - 물리 기반 H 모델 20%
- 2등 코드의 최종 예측을 노트북 내 압축값에서 복원
- 최종 제출:
  - 80% GOH30 ODE 강화 예측
  - 20% 2등 코드 예측

핵심 결론:

- GOH30 내부 변경 중 리더보드에서 실제로 살아남은 것은 ODE 강화뿐이었다.
- H 강화, KMeans 전문가 혼합, 학습형 스태킹, 구간별 가중치, 클러스터 특징 주입은 외부폴드 검증에서는 좋아 보였지만 리더보드로 전이되지 않았다.
- TCN과 경량 트랜스포머도 외부폴드 검증에서는 개선됐지만 리더보드에서는 개선되지 않았다.
- 가장 큰 개선은 새 모델 학습이 아니라, 리더보드에서 이미 검증된 2등 코드 예측과의 예측 블렌딩에서 나왔다.

## 1. 베이스라인 재현

베이스 코드:

- `best_solve/[Private_LB 1st] 코드 공유.ipynb`

모델 구조:

- GOH30 = GRU 10개 + 신경 ODE 10개 + 물리 기반 H 모델 10개
- 원본 제출은 30개 모델을 등가중 평균
- 구조별 원본 비율:
  - GRU 33.3%
  - ODE 33.3%
  - 물리 기반 H 모델 33.3%

생성된 주요 산출물:

- `models_goh30/phaseG_full_0.pt` ... `phaseG_full_9.pt`
- `models_goh30/phaseODE_full_0.pt` ... `phaseODE_full_9.pt`
- `models_goh30/phaseH_full_0.pt` ... `phaseH_full_9.pt`
- `submission_GOH30.csv`

커밋:

- `b9c6bd1 Add GOH30 base submission`
- 커밋에는 `submission_GOH30.csv`만 포함
- 모델 가중치와 실험 출력물은 의도적으로 제외

제출 점수:

| 제출 파일 | 공개 점수 | 비공개 점수 |
| --- | ---: | ---: |
| `submission_GOH30.csv` | 0.7020 | 0.7025 |

## 2. 등속 외삽 / Kalman 약한 베이스 실험

스크립트:

- `huiyu/experiments/turn_phase_residual_experiment.py`

아이디어:

- 등속 외삽:
  - `pred = last + 2 * (last - prev)`
- Kalman 등속 모델, 잔차 모델, 회전/단계 특징, 구간 게이트를 추가

외부폴드 검증 결과:

| 후보 | 검증 적중률 |
| --- | ---: |
| 등속 외삽 | 0.5788 |
| Kalman 등속 모델 | 0.5964 |
| 등속 잔차 + 회전/단계 | 0.5969 |
| Kalman 잔차 + 회전/단계 | 0.5985 |
| 등속/Kalman 잔차 블렌딩 | 0.6025 |
| 블렌딩 + 고속 구간 Kalman 게이트 | 0.6124 |

결론:

- 약한 등속/Kalman 베이스에서는 구간 기반 구분이 도움이 됐다.
- 하지만 같은 아이디어는 강한 GOH30에는 거의 전이되지 않았다.

## 3. GOH30 Kalman 후보정

스크립트:

- `huiyu/experiments/goh30_regime_submissions.py`

생성 후보:

- `outputs/goh30_regime_submissions/case00_goh30_original.csv`
- `case01_high_speed_kalman_005.csv`
- `case02_high_speed_kalman_010.csv`
- `case03_high_speed_kalman_015.csv`
- `case04_high_speed_kalman_020.csv`

제출 점수:

| 제출 파일 | 공개 점수 | 비공개 점수 |
| --- | ---: | ---: |
| GOH30 원본 | 0.7020 | 0.7025 |
| 고속 구간 Kalman 5% | 0.7022 | 0.7026 |
| 고속 구간 Kalman 10% | 0.7020 | 0.7021 |
| 고속 구간 Kalman 15% | 0.7016 | 0.7018 |
| 고속 구간 Kalman 20% | 0.7018 | 0.7019 |

결론:

- Kalman 블렌딩 신호는 매우 약했다.
- 5%는 아주 조금 좋아졌지만 신뢰하기 어려운 수준이었다.
- 더 큰 Kalman 비율은 점수를 떨어뜨렸다.

## 4. 외부폴드 간이 검증 절차

목표:

- 제출 제한에 의존하지 않고 로컬에서 후보를 먼저 거르기
- GOH30의 GRU/ODE/H 구성요소와 앙상블 아이디어를 빠르게 검증

메인 스크립트:

- `huiyu/experiments/goh30_oof_lite.py`

기능:

- 3개 폴드 기반 간이 외부폴드 검증
- GRU 외부폴드 예측
- ODE 외부폴드 예측
- `--include-h`로 물리 기반 H 모델 외부폴드 예측
- `--include-tcn`으로 TCN 전문가 외부폴드 예측
- `--include-transformer`로 경량 트랜스포머 외부폴드 예측
- CV/Kalman 기준 예측
- 구간별 점수 보고서

GPU 환경:

- `.venv-cu128`
- PyTorch `2.11.0+cu128`
- RTX 5070 지원 확인
  - 장치 연산 능력 `(12, 0)`
  - `sm_120` 포함

## 5. GRU/ODE 간이 외부폴드 검증 결과

명령:

```bash
python huiyu/experiments/goh30_oof_lite.py \
  --folds 3 \
  --epochs 10 \
  --device cuda \
  --out-dir outputs/goh30_oof_lite_gru_ode10_cuda
```

주요 결과:

| 후보 | 검증 적중률 |
| --- | ---: |
| GRU | 0.6263 |
| ODE | 0.6316 |
| GRU/ODE 50:50 | 0.6298 |
| GRU15/ODE85 | 0.6319 |

결론:

- 간이 외부폴드 검증에서는 ODE가 GRU보다 강했다.
- GRU/ODE 50:50보다 ODE 강화 비율이 나았다.

## 6. GOH30 구성요소 예측과 ODE 강화

스크립트:

- `huiyu/experiments/goh30_component_submissions.py`

생성한 구성요소 예측:

- `outputs/goh30_component_submissions/pred_gru.npy`
- `pred_ode.npy`
- `pred_h.npy`
- `pred_equal.npy`

생성 제출 후보:

- `case00_equal_goh30.csv`
- `case01_ode_heavy_g25_o50_h25.csv`
- `case02_ode_heavy_g20_o60_h20.csv`
- `case03_ode_heavy_g15_o65_h20.csv`
- `case04_ode_heavy_g15_o70_h15.csv`

제출 점수:

| 제출 파일 | 공개 점수 | 비공개 점수 |
| --- | ---: | ---: |
| `case02_ode_heavy_g20_o60_h20.csv` | 0.7018 | 0.7033 |

결론:

- ODE 강화는 실제 리더보드로 전이된 첫 개선이었다.
- 공개 점수는 약간 낮아졌지만 비공개 점수는 원본 대비 +0.0008 개선됐다.

## 7. PPT 아이디어 적용 결과

PPT 아이디어:

- 장면 클러스터링
- 노이즈 기준
- 전문가 혼합
- 전문가별 가중치

적용 방식:

| PPT 아이디어 | 구현 |
| --- | --- |
| 장면 클러스터링 | 궤적 특징 기반 KMeans |
| 노이즈 기준 | 고속, 고노이즈, 고가속, 급회전, 수직 변화 |
| 전문가 혼합 | GRU/ODE/H 비율 조정 |
| 전문가 게이팅 | 클러스터별 GRU/ODE/H 가중치 |
| 안정성 확인 | 폴드별 외부폴드 검증, 시드별 비교 |

결론:

- 약한 베이스에서는 의미가 있었지만, GOH30 위에서는 대부분 리더보드 개선으로 이어지지 않았다.

## 8. GRU/ODE 클러스터 전문가 혼합

스크립트:

- `huiyu/experiments/search_cluster_moe.py`
- `huiyu/experiments/make_cluster_moe_submissions.py`

외부폴드 검증 결과:

| 후보 | 검증 적중률 |
| --- | ---: |
| ODE | 0.6316 |
| 전체 GRU15/ODE85 | 0.6319 |
| KMeans5 클러스터 전문가 혼합 | 0.6324 |
| KMeans6 클러스터 전문가 혼합 | 0.6325 |
| KMeans8 클러스터 전문가 혼합 | 0.6327 |

결론:

- 외부폴드 검증에서는 클러스터별 가중치가 좋아 보였다.
- 하지만 이후 GOH 전체 K8 전문가 혼합이 리더보드에서 크게 실패하면서 클러스터 게이팅은 제출 우선순위에서 제외했다.

## 9. H 포함 간이 외부폴드 검증과 H 강화 실패

명령:

```bash
python huiyu/experiments/goh30_oof_lite.py \
  --folds 3 \
  --epochs 10 \
  --include-h \
  --h-epochs 6 \
  --device cuda \
  --out-dir outputs/goh30_oof_lite_gru_ode_h10_6_cuda
```

외부폴드 검증 결과:

| 후보 | 검증 적중률 |
| --- | ---: |
| H 단독 | 0.6533 |
| GOH-lite 등가중 | 0.6432 |
| G20/O60/H20 | 0.6400 |
| G15/O65/H20 | 0.6403 |
| H 강화 G10/O05/H85 | 0.6544 |

결론:

- 간이 외부폴드 검증에서는 H가 매우 강해 보였다.
- 하지만 H 강화/K8 전문가 혼합은 실제 리더보드에서 실패했다.

## 10. H 포함 클러스터 전문가 혼합

스크립트:

- `huiyu/experiments/search_full_moe_from_oof.py`
- `huiyu/experiments/stabilize_moe_candidates.py`

외부폴드 검증 결과:

| 후보 | 시드42 검증 | 시드777 검증 |
| --- | ---: | ---: |
| 등가중 | 0.6432 | 0.6437 |
| H 단독 | 0.6533 | 0.6528 |
| H 강화 | 0.6544 | 0.6535 |
| KMeans5 GOH 전문가 혼합 | 0.6563 | 0.6559 |
| KMeans6 GOH 전문가 혼합 | 0.6565 | 0.6562 |
| KMeans8 GOH 전문가 혼합 | 0.6570 | 0.6572 |

제출 결과:

| 제출 파일 | 공개 점수 | 비공개 점수 |
| --- | ---: | ---: |
| `avg_seed42_seed777_k8.csv` | 0.6966 | 0.6965 |

결론:

- K8 전문가 혼합은 외부폴드 검증에서는 매우 강했지만 리더보드에서 크게 무너졌다.
- 외부폴드 검증과 리더보드 분포 차이가 크다는 중요한 신호였다.
- 이후 H 강화와 클러스터 게이팅은 폐기했다.

## 11. 학습형 외부폴드 스태킹

스크립트:

- `huiyu/experiments/learned_oof_stacking.py`

아이디어:

- 외부폴드 예측과 궤적 특징으로 샘플별 GRU/ODE/H 가중치를 학습
- 내부 스태킹용 폴드 분할로 메타 모델 과적합을 줄이려 시도

결과:

| 후보 | 평균 검증 | 표준편차 | 최솟값 | 최댓값 |
| --- | ---: | ---: | ---: | ---: |
| HGB 전체 사전비율 alpha25 | 0.65485 | 0.00035 | 0.6546 | 0.6551 |
| HGB H85 사전비율 alpha50 | 0.65460 | 0.00014 | 0.6545 | 0.6547 |
| 전체 H 강화 | 0.65440 | 0.00000 | 0.6544 | 0.6544 |

결론:

- 외부폴드 검증에서는 H 강화보다 약간 좋았다.
- 하지만 K8 전문가 혼합보다 약했고, H 강화 계열이 리더보드에서 실패했기 때문에 제출 후보에서 제외했다.

## 12. 학습 전 샘플 가중치 / 클러스터 특징 주입

추가한 옵션:

- `--weight-mode regime-soft`
- `--weight-mode regime-strong`
- `--weight-mode hard-final`
- `--interior-weight`
- `--cluster-feature-mode`
- `--cluster-weight-mode`

대표 결과:

| 실험 | ODE | GRU | GRU/ODE |
| --- | ---: | ---: | ---: |
| 기존 GRU/ODE 외부폴드 검증 | 0.6316 | 0.6263 | 0.6298 |
| 약한 구간 가중치 + 내부 구간 0.7 | 0.6316 | 0.6254 | 0.6286 |
| 클러스터 k6 특징 + 약한 가중치 | 0.6311 | 0.6240 | 0.6284 |

결론:

- 구간별 샘플 가중치는 개선이 없었다.
- 클러스터 특징 주입과 클러스터별 샘플 가중치도 개선되지 않았다.
- 학습 전 단계의 PPT 아이디어는 GOH30에는 효과가 없었다.

## 13. TCN 전문가

스크립트:

- `huiyu/experiments/train_tcn_submissions.py`
- `huiyu/experiments/stabilize_tcn_candidates.py`

외부폴드 검증 명령:

```bash
.venv-cu128/bin/python huiyu/experiments/goh30_oof_lite.py \
  --folds 3 \
  --epochs 10 \
  --device cuda \
  --include-tcn \
  --out-dir outputs/goh30_oof_tcn10_cuda
```

외부폴드 검증 결과:

| 후보 | 검증 적중률 |
| --- | ---: |
| TCN | 0.6425 |
| ODE 90% + TCN 10% | 0.6332 |
| ODE 95% + TCN 5% | 0.6331 |
| ODE | 0.6316 |

전체 학습:

- TCN 5개 시드, 30 에폭
- 시드3000 묶음 + 시드4000 묶음
- 총 10개 TCN 모델 평균

제출 결과:

| 제출 파일 | 공개 점수 | 비공개 점수 |
| --- | ---: | ---: |
| `confirmed_ode_heavy_900_tcn100.csv` | 0.7022 | 0.7030 |

결론:

- TCN은 외부폴드 검증에서는 좋은 새 전문가처럼 보였다.
- 하지만 전체 학습 후 리더보드에서는 ODE 강화보다 낮았다.
- TCN 10%는 더 제출하지 않는 것이 맞다.

## 14. 경량 트랜스포머 전문가

구현:

- `huiyu/experiments/goh30_oof_lite.py`
- 옵션: `--include-transformer`

외부폴드 검증 결과:

| 후보 | 검증 적중률 |
| --- | ---: |
| 경량 트랜스포머 | 0.6421 |
| ODE 90% + 경량 트랜스포머 10% | 0.6332 |
| ODE 95% + 경량 트랜스포머 5% | 0.6325 |
| ODE | 0.6316 |

결론:

- 경량 트랜스포머는 TCN과 거의 같은 수준의 외부폴드 검증 신호였다.
- TCN이 같은 수준의 외부폴드 검증 신호로 리더보드 개선에 실패했기 때문에, 경량 트랜스포머는 전체 학습/제출로 가지 않았다.

## 15. 2등 코드 전문가 블렌딩

스크립트:

- `huiyu/experiments/blend_second_place_expert.py`

소스:

- `best_solve/_[private 2nd] 코드 공유.ipynb`
- 노트북 안에 기본 모델/물리 모델 테스트 예측값이 zlib+base64 형식으로 압축 저장되어 있음
- 학습 없이 2등 최종 예측 복원 가능

2등 복원 예측 점수:

| 예측 | 공개 점수 | 비공개 점수 |
| --- | ---: | ---: |
| 2등 복원 예측 | 0.7022 | 0.7031 |

생성 후보:

- `outputs/second_place_expert_blends/second_place_restored.csv`
- `outputs/second_place_expert_blends/confirmed_ode_heavy_975_second025.csv`
- `outputs/second_place_expert_blends/confirmed_ode_heavy_950_second050.csv`
- `outputs/second_place_expert_blends/confirmed_ode_heavy_900_second100.csv`
- `outputs/second_place_expert_blends/confirmed_ode_heavy_850_second150.csv`
- `outputs/second_place_expert_blends/confirmed_ode_heavy_800_second200.csv`
- `outputs/second_place_expert_blends/confirmed_ode_heavy_750_second250.csv`
- `outputs/second_place_expert_blends/confirmed_ode_heavy_700_second300.csv`

ODE 강화 대비 이동량:

| 후보 | 평균 이동 | 90% 분위 이동 | 최대 이동 | 1cm 초과 |
| --- | ---: | ---: | ---: | ---: |
| 2등 단독 | 0.001675 | 0.003132 | 0.017793 | 74 |
| 2등 10% | 0.000168 | 0.000313 | 0.001779 | 0 |
| 2등 20% | 0.000335 | 0.000627 | 0.003559 | 0 |
| 2등 30% | 0.000503 | 0.000940 | 0.005338 | 0 |

제출 결과:

| 제출 파일 | 공개 점수 | 비공개 점수 |
| --- | ---: | ---: |
| `confirmed_ode_heavy_800_second200.csv` | 0.7034 | 0.7042 |

결론:

- 2등 전문가 블렌딩이 최종 최고 성과를 만들었다.
- 2등 자체가 리더보드 검증된 예측이라 TCN/트랜스포머보다 신뢰도가 높았다.
- 현재 최고는 `ODE 강화 80% + 2등 20%`이다.

## 16. 구현 스크립트 목록

핵심 외부폴드 검증 / 예측:

- `huiyu/experiments/goh30_oof_lite.py`
- `huiyu/experiments/goh30_component_submissions.py`
- `huiyu/experiments/train_tcn_submissions.py`

탐색 / 전문가 혼합:

- `huiyu/experiments/search_oof_blends.py`
- `huiyu/experiments/search_cluster_moe.py`
- `huiyu/experiments/search_goh_lite_weights.py`
- `huiyu/experiments/search_full_moe_from_oof.py`
- `huiyu/experiments/learned_oof_stacking.py`

제출 생성:

- `huiyu/experiments/make_cluster_moe_submissions.py`
- `huiyu/experiments/make_h_heavy_submissions.py`
- `huiyu/experiments/stabilize_moe_candidates.py`
- `huiyu/experiments/stabilize_tcn_candidates.py`
- `huiyu/experiments/blend_second_place_expert.py`

보고:

- `huiyu/experiments/foldwise_oof_report.py`

## 17. 남은 후보

실제 점수 개선 관점에서는 추가 여지가 크지 않다.

가능한 후속 작업:

- 종료된 대회 연습 목적이라면 2등 블렌딩 비율 25%, 30%를 추가 제출해 점수 곡선 확인
- 3등/9등/12등 코드 예측을 복원하거나 재학습해 추가 리더보드 검증 전문가로 사용
- GOH30 + 2등 전문가 + 최종 블렌딩을 한 번에 실행하는 단독 실행 스크립트 작성

현재 추천:

- 실전 제출 기준으로는 `confirmed_ode_heavy_800_second200.csv`를 최고 결과로 유지
- 추가 제출 여유가 있으면 25% 블렌딩, 그다음 30% 블렌딩 순서로 확인
