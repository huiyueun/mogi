# 1등/2등 예측 차이 분석 보기 가이드

작성일: 2026-07-01

## 목적

이 문서는 1등 GOH30 원본, ODE 강화 예측, 2등 코드 복원 예측, 최종 블렌딩 예측을 비교해서 어디를 보면 되는지 정리한 것이다.

이번 분석의 핵심 질문은 다음이다.

- 1등 원본과 2등 예측은 어떤 궤적에서 많이 달라지는가?
- ODE 강화는 1등 원본에서 얼마나 움직였는가?
- 최종 `ODE 강화 80% + 2등 20%`는 예측점을 어떤 방향으로 얼마나 이동시키는가?
- 차이가 큰 샘플들이 급회전, 고가속, 고노이즈 같은 구간에 몰리는가?

분석 결과 폴더:

- `outputs/top_expert_comparison/`

## 먼저 볼 요약

전체 예측 차이 요약:

| 비교 | 평균 거리 | 중앙값 | 90% 분위 | 99% 분위 | 최대 | 1cm 초과 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1등 원본 vs 2등 | 0.001532 | 0.001088 | 0.002766 | 0.009013 | 0.034634 | 81 |
| ODE 강화 vs 2등 | 0.001675 | 0.001212 | 0.003132 | 0.009099 | 0.017793 | 74 |
| 1등 원본 vs ODE 강화 | 0.000444 | 0.000292 | 0.000838 | 0.002786 | 0.017177 | 7 |
| 1등 원본 vs 최종 블렌딩 | 0.000412 | 0.000264 | 0.000725 | 0.002848 | 0.020631 | 10 |
| ODE 강화 vs 최종 블렌딩 | 0.000335 | 0.000242 | 0.000627 | 0.001820 | 0.003559 | 0 |

해석:

- 1등 원본과 2등은 평균적으로 약 `0.00153m` 다르다.
- ODE 강화는 1등 원본에서 평균 `0.00044m` 정도만 움직인다.
- 최종 블렌딩은 ODE 강화에서 2등 방향으로 평균 `0.00034m`만 움직이는 약한 보정이다.
- 최종 블렌딩은 ODE 강화 대비 1cm를 넘게 움직인 샘플이 없다. 즉 큰 점프가 아니라 작은 방향 보정이다.

## 그림 1: 전체 차이 분포

파일:

- `outputs/top_expert_comparison/prediction_distance_hist.png`

![전체 예측 차이 분포](../../outputs/top_expert_comparison/prediction_distance_hist.png)

볼 것:

- 대부분 샘플은 1등과 2등이 가깝다.
- 일부 샘플에서만 차이가 크게 벌어진다.
- 이 크게 벌어지는 샘플들이 다음 분석 대상이다.

## 그림 2: 구간별 1등 원본 vs 2등 차이

파일:

- `outputs/top_expert_comparison/original_vs_second_regime_mean_distance.png`

![구간별 평균 차이](../../outputs/top_expert_comparison/original_vs_second_regime_mean_distance.png)

구간별 평균 차이:

| 구간 | 평균 거리 |
| --- | ---: |
| 전체 | 0.001532 |
| 고속 | 0.001981 |
| 급회전 | 0.002645 |
| 최근 회전 | 0.003125 |
| 고가속 | 0.003287 |
| 고노이즈 | 0.002770 |
| 수직 변화 | 0.001776 |
| 낮은 직선성 | 0.002477 |

해석:

- 1등과 2등의 차이는 전체 평균보다 `최근 회전`, `고가속`, `고노이즈`, `급회전`에서 훨씬 크다.
- 즉 2등 코드는 안정적인 직선 궤적보다, 움직임이 불안정하거나 방향이 바뀌는 구간에서 1등과 다른 판단을 하는 것으로 보인다.

## 그림 3: 구간별 방향 차이

파일:

- `outputs/top_expert_comparison/original_vs_second_direction_by_regime.png`

![구간별 방향 차이](../../outputs/top_expert_comparison/original_vs_second_direction_by_regime.png)

볼 것:

- 막대는 `2등 예측 - 1등 원본 예측`의 평균 방향이다.
- 특정 구간에서 `dx`, `dy`, `dz` 중 하나가 일관되게 치우치는지 본다.
- 방향성이 뚜렷하면 단순 앙상블이 아니라 후처리 규칙의 힌트가 될 수 있다.

현재 수치상으로는 큰 전역 방향 편향은 약하다. 다만 고가속/최근 회전 구간에서는 x/y 방향 차이가 상대적으로 커진다.

## 먼저 열어볼 Top case 그림

폴더:

- `outputs/top_expert_comparison/top_case_plots/`

우선 아래 10개만 먼저 보면 된다.

| 순위 | 샘플 | 그림 |
| ---: | --- | --- |
| 1 | `TEST_01506` | [열기](../../outputs/top_expert_comparison/top_case_plots/001_TEST_01506.png) |
| 2 | `TEST_08881` | [열기](../../outputs/top_expert_comparison/top_case_plots/002_TEST_08881.png) |
| 3 | `TEST_06159` | [열기](../../outputs/top_expert_comparison/top_case_plots/003_TEST_06159.png) |
| 4 | `TEST_09000` | [열기](../../outputs/top_expert_comparison/top_case_plots/004_TEST_09000.png) |
| 5 | `TEST_08250` | [열기](../../outputs/top_expert_comparison/top_case_plots/005_TEST_08250.png) |
| 6 | `TEST_05754` | [열기](../../outputs/top_expert_comparison/top_case_plots/006_TEST_05754.png) |
| 7 | `TEST_09191` | [열기](../../outputs/top_expert_comparison/top_case_plots/007_TEST_09191.png) |
| 8 | `TEST_05719` | [열기](../../outputs/top_expert_comparison/top_case_plots/008_TEST_05719.png) |
| 9 | `TEST_02199` | [열기](../../outputs/top_expert_comparison/top_case_plots/009_TEST_02199.png) |
| 10 | `TEST_04292` | [열기](../../outputs/top_expert_comparison/top_case_plots/010_TEST_04292.png) |

각 그림에서 볼 것:

- 검은 선: 과거 관측 궤적
- 파란 점: 1등 GOH30 원본
- 주황 점: ODE 강화 예측
- 초록 점: 2등 코드 복원 예측
- 빨간 점: 최종 `ODE 강화 80% + 2등 20%`

확인 질문:

- 2등 예측은 1등보다 더 멀리 가는가, 덜 가는가?
- 2등 예측은 회전 방향을 더 따라가는가, 덜 따라가는가?
- ODE 강화는 원본과 2등 사이에 있는가, 아니면 다른 방향으로 움직이는가?
- 최종 빨간 점은 주황 점에서 초록 점 방향으로 아주 조금 이동한 형태인가?
- z축 그림에서 2등이 높이 방향을 다르게 보정하는가?

## Top case 표에서 볼 컬럼

파일:

- `outputs/top_expert_comparison/top50_original_vs_second.csv`

먼저 볼 컬럼:

- `id`
- `original_vs_second_dist`
- `original_vs_second_dx`
- `original_vs_second_dy`
- `original_vs_second_dz`
- `hard_turn_regime`
- `recent_turn_regime`
- `high_acc`
- `high_noise`
- `vertical_change_regime`
- `low_straightness`

해석 방법:

- `original_vs_second_dist`가 클수록 1등과 2등 판단이 많이 다른 샘플이다.
- `hard_turn_regime`, `recent_turn_regime`, `high_acc`, `high_noise`가 `True`로 많이 몰리면 회전/가속/노이즈 구간에서 두 코드가 다르게 본다는 뜻이다.
- `dx`, `dy`, `dz`는 2등이 1등 대비 어느 방향으로 이동했는지 보여준다.

## 현재까지의 1차 결론

현재 결과만 놓고 보면 다음 정도는 말할 수 있다.

- 1등 원본과 2등 예측은 대부분의 샘플에서는 가깝다.
- 하지만 회전, 가속, 노이즈가 큰 구간에서는 두 예측의 차이가 커진다.
- 최종 블렌딩은 ODE 강화 예측을 2등 방향으로 약하게 이동시키는 방식이다.
- 따라서 2등 코드의 유용한 힌트는 전체 구간보다는 `회전/가속/노이즈가 큰 궤적을 어떻게 보정하는지`에 있을 가능성이 크다.

## 다음에 보면 좋은 것

그림을 본 뒤 다음 중 하나가 보이면 추가 분석 가치가 있다.

- 2등이 급회전에서 항상 덜 꺾거나 더 꺾는 패턴
- 2등이 고가속 구간에서 예측 거리를 더 길게 또는 짧게 잡는 패턴
- 2등이 고노이즈 구간에서 z축을 더 안정적으로 보정하는 패턴
- ODE 강화와 2등이 같은 방향으로 움직이는 샘플과 반대 방향으로 움직이는 샘플의 차이

그다음 단계는 정답이 있는 외부폴드 검증에서 같은 분석을 하는 것이다. 그러면 단순히 "다르다"를 넘어서 "어느 쪽이 더 맞다"까지 볼 수 있다.
