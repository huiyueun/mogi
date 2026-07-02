# 50:50 블렌드가 좋아진 이유 분석

작성일: 2026-07-02

## 배경

기존 최고 조합은 다음이었다.

```text
0.80 * GOH30_ODE_heavy + 0.20 * second_final
```

이후 second 비율을 올려 제출한 결과:

| second_final 비율 | Public | Private |
| ---: | ---: | ---: |
| 20% | 0.7034 | 0.7042 |
| 40% | 0.7032 | 0.7045 |
| 50% | 0.7052 | 0.7053 |
| 60% | 0.7044 | 0.7049 |

즉 실제 LB에서는 50:50이 가장 좋았다.

## 분석 실행

```bash
python huiyu/experiments/analyze_blend_ratio_behavior.py \
  --ratios 0.20,0.40,0.50,0.60,1.00 \
  --out-dir outputs/blend_ratio_behavior
```

생성 파일:

- `outputs/blend_ratio_behavior/blend_ratio_behavior_by_regime.csv`
- `outputs/blend_ratio_behavior/blend_ratio_pairwise_steps.csv`
- `outputs/blend_ratio_behavior/ratio_turn_delta_vs_ode.png`
- `outputs/blend_ratio_behavior/ratio_z_delta_vs_ode.png`
- `outputs/blend_ratio_behavior/ratio_shift_from_ode.png`
- `outputs/blend_ratio_behavior/ratio_cv_z_rate.png`

## 전체 변화

비율을 올릴수록 예측은 ODE-heavy에서 second_final 방향으로 선형 이동한다. 중요한 것은 그 이동이 어떤 물리적 행동 차이를 만들었는지다.

| second 비율 | ODE-heavy에서 평균 이동 | 회전각 변화 | z 이동 변화 | CV-z에 더 가까운 비율 |
| ---: | ---: | ---: | ---: | ---: |
| 20% | 0.000335 | -0.199도 | -0.000010 | 0.6065 |
| 40% | 0.000670 | -0.368도 | -0.000018 | 0.5937 |
| 50% | 0.000838 | -0.440도 | -0.000022 | 0.5873 |
| 60% | 0.001005 | -0.502도 | -0.000026 | 0.5807 |
| 100% | 0.001675 | -0.640도 | -0.000036 | 0.5562 |

해석:

- second 비율을 올리면 GOH30 ODE-heavy보다 덜 꺾는 방향으로 간다.
- z 이동도 조금 더 작게 잡는다.
- 다만 CV-z 기준 안정화 비율은 20%에서 가장 높고, 비율이 커질수록 낮아진다.
- 따라서 second 비율을 올리는 것은 단순히 “항상 더 안정적”이 아니라, 회전 완화와 z 안정화 사이의 균형을 바꾸는 일이다.

## 50%가 의미 있는 지점

50%는 ODE-heavy와 second_final 사이의 정확한 중간점이다.

```text
0.50 * GOH30_ODE_heavy + 0.50 * second_final
```

이때 행동 변화는 다음 정도다.

- ODE-heavy에서 평균 `0.000838` 이동
- 회전각을 평균 `-0.440도` 낮춤
- z 이동을 평균 `-0.000022` 줄임
- hard_turn에서 회전각을 `-0.994도` 낮춤
- high_noise에서 회전각을 `-0.780도` 낮춤
- vertical_change에서 z 이동을 `-0.000060` 줄임

즉 50%는 GOH30의 동역학 과반응을 꽤 강하게 누르지만, 아직 second_final 단독까지는 가지 않는 균형점이다.

## 20%와 비교했을 때 50%의 차이

20%는 안전한 보정이었다.

```text
20%: ODE-heavy를 평균 0.000335만 이동
```

하지만 50%는 다음처럼 더 강한 보정을 한다.

```text
50%: ODE-heavy를 평균 0.000838 이동
```

regime별로 보면 50%는 특히 다음 구간에서 더 많이 움직인다.

| 구간 | 20% 평균 이동 | 50% 평균 이동 |
| --- | ---: | ---: |
| hard_turn | 0.000572 | 0.001429 |
| recent_turn | 0.000658 | 0.001646 |
| high_acc | 0.000692 | 0.001731 |
| high_noise | 0.000594 | 0.001485 |
| low_straightness | 0.000540 | 0.001351 |

즉 50%는 우리가 처음에 2등 expert가 가치 있다고 본 구간에서 실제로 더 강하게 작동한다.

## 왜 60%부터 내려갔을까

60%는 50%보다 더 second_final 쪽으로 간다.

| 비교 | 전체 평균 추가 이동 | hard_turn 추가 이동 | high_noise 추가 이동 |
| --- | ---: | ---: | ---: |
| 50% -> 60% | 0.000168 | 0.000286 | 0.000297 |

추가 이동 자체는 작지만, 이 문제는 1cm hit 경계 근처 샘플이 중요하다. 그래서 50%에서 살린 샘플 일부가 60%에서 다시 경계 밖으로 밀렸을 수 있다.

60%의 행동 변화:

- 회전각을 평균 `-0.502도`까지 낮춤
- hard_turn에서는 `-1.158도`까지 낮춤
- low_straightness에서는 `-1.162도`까지 낮춤
- high_acc에서는 step을 더 길게 잡는 경향이 커짐

즉 60%는 50%보다 더 보수적으로 덜 꺾는 방향으로 가지만, 일부 샘플에서는 그 보정이 과해진 것으로 보인다.

## 왜 100%는 조심해야 하나

100%는 사실상 2등 단독 제출이다.

```text
100% second_final = second_place_restored.csv
```

파일 비교 결과:

```text
max_abs_diff = 0.0
mean_abs_diff = 0.0
```

따라서 100%는 새 앙상블이 아니라 2등 코드 복원값이다. OOF에서는 `second_phys` 기준으로 높게 보였지만, test 제출에서 100%는 `second_final` 단독이므로 해석을 조심해야 한다.

## 핵심 해석

50:50이 좋아진 이유는 다음으로 정리할 수 있다.

1. GOH30 ODE-heavy는 private 쪽에서 일부 구간에 과반응했을 가능성이 있다.
2. second_final은 그 과반응을 덜 꺾는 방향과 약한 z 안정화 방향으로 보정했다.
3. 20%는 보정이 너무 약했고, 40~50%에서 보정 강도가 충분해졌다.
4. 60%부터는 second_final 쪽으로 너무 많이 가서 일부 샘플을 다시 손해 보게 만들었다.
5. 따라서 50%는 두 expert의 오류가 가장 잘 상쇄된 균형점으로 보인다.

한 줄 결론:

> 50:50은 GOH30의 강한 동역학 예측과 2등의 보수적 smoothing/다른 물리 가정을 동등하게 섞어, hit 경계 근처 샘플을 가장 많이 살린 지점으로 보인다.

