# 2등 코드 구성요소 분석

## 목적

1등 GOH30 결과와 2등 코드 결과를 섞었을 때 점수가 오른 이유를 보기 위해, 2등 예측을 내부 구성요소별로 나눠서 비교했다.

2등 코드는 노트북 안에 압축 저장된 두 예측을 사용한다.

- `second_base`: 2등 코드의 base 예측
- `second_phys`: 2등 코드의 물리/회전 보정 예측
- `second_final`: `0.60 * second_base + 0.40 * second_phys`

이번 분석의 핵심 질문은 세 가지다.

- 2등이 1등보다 덜 꺾는 성향은 base에서 오는가, phys에서 오는가?
- z축을 더 안정적으로 잡는 성향은 base에서 오는가, phys에서 오는가?
- ODE-heavy와 다른 방향으로 움직이는 다양성은 어느 구성요소에서 오는가?

## 실행

```bash
python huiyu/experiments/analyze_second_components.py \
  --out-dir outputs/second_component_analysis
```

생성 파일:

- `outputs/second_component_analysis/second_component_pairwise_summary.csv`
- `outputs/second_component_analysis/second_component_pairwise_by_regime.csv`
- `outputs/second_component_analysis/second_component_behavior_by_regime.csv`
- `outputs/second_component_analysis/component_turn_angle_delta_by_regime.png`
- `outputs/second_component_analysis/component_step_delta_by_regime.png`
- `outputs/second_component_analysis/component_z_move_delta_by_regime.png`
- `outputs/second_component_analysis/component_ode_alignment_by_regime.png`
- `outputs/second_component_analysis/predictions/second_base.csv`
- `outputs/second_component_analysis/predictions/second_phys.csv`
- `outputs/second_component_analysis/predictions/second_final.csv`

## 전체 요약

아래 값은 모두 GOH30 원본 예측을 기준으로 한 차이다.

| 구성요소 | 회전각 변화 | 이동거리 변화 | z 이동 변화 | CV z에 더 가까운 비율 | ODE-heavy 보정과 평균 cos |
| --- | ---: | ---: | ---: | ---: | ---: |
| `second_base` | -0.949도 | -0.000015 | -0.000060 | 0.552 | -0.139 |
| `second_phys` | +0.240도 | -0.000009 | -0.000033 | 0.490 | -0.357 |
| `second_final` | -0.573도 | -0.000063 | -0.000070 | 0.551 | -0.239 |

해석:

- `second_base`는 GOH30 원본보다 확실히 덜 꺾는다.
- `second_phys`는 오히려 평균적으로 더 꺾는다.
- `second_final`은 base 60%, phys 40%라서 덜 꺾는 성향은 유지하되, phys가 일부 반대 성향을 섞어준다.
- ODE-heavy와 가장 반대 방향으로 움직이는 것은 `second_phys`다.

## 질문별 결론

### 1. 덜 꺾는 성향은 어디서 오는가?

주로 `second_base`에서 온다.

regime별 회전각 변화:

| 구간 | `second_base` | `second_phys` | `second_final` |
| --- | ---: | ---: | ---: |
| 전체 | -0.949도 | +0.240도 | -0.573도 |
| hard_turn | -2.624도 | +0.328도 | -1.474도 |
| recent_turn | -3.481도 | +1.676도 | -1.419도 |
| high_acc | -3.073도 | +1.608도 | -1.201도 |
| high_noise | -2.283도 | +0.617도 | -1.159도 |

즉 2등 최종 결과가 GOH30보다 덜 꺾는 것처럼 보인 이유는 `base`의 영향이 크다. `phys`는 이름과 다르게 보수적으로 누르는 역할만 하는 것이 아니라, 회전/가속 구간에서는 더 크게 꺾는 방향을 일부 제공한다.

### 2. z축 안정화는 어디서 오는가?

대체로 `second_base`와 `second_final` 쪽이 더 안정적이다.

`second_base`는 전체에서 CV z에 더 가까운 비율이 0.552이고, hard_turn/recent_turn/high_acc/high_noise에서도 0.58~0.62 수준이다. 특히 vertical_change에서는 z 이동량 자체를 줄이는 경향이 크다.

`second_phys`도 평균 z 이동량은 조금 줄지만, CV z 기준으로는 전체 0.490이라 안정화 방향이 일관적이라고 보긴 어렵다. recent_turn/high_acc에서는 오히려 z 이동량을 늘리는 쪽도 보인다.

따라서 “z축을 더 안정적으로 잡는 2등의 성향”은 `phys`보다는 `base` 또는 `base`가 많이 섞인 `final`에서 온다고 보는 게 맞다.

### 3. ODE-heavy와 다른 방향성은 어디서 오는가?

가장 강한 다양성은 `second_phys`에서 온다.

ODE-heavy가 GOH30 원본에서 움직인 방향과 각 구성요소가 움직인 방향의 평균 cosine은 다음과 같다.

| 구성요소 | 전체 평균 cos | 같은 방향 비율 |
| --- | ---: | ---: |
| `second_base` | -0.139 | 0.394 |
| `second_phys` | -0.357 | 0.269 |
| `second_final` | -0.239 | 0.334 |

`second_phys`는 ODE-heavy와 가장 반대 방향으로 움직인다. 그래서 2등 코드를 외부 expert로 섞었을 때 GOH30 내부 비율 조정과 다른 효과가 나온 것으로 볼 수 있다.

## 1등 + 2등 앙상블 관점

현재 가장 좋은 제출 확인 결과는 다음이었다.

| 제출 | Public | Private |
| --- | ---: | ---: |
| GOH30 원본 | 0.7020 | 0.7025 |
| ODE-heavy | 0.7018 | 0.7033 |
| ODE-heavy 80% + 2등 final 20% | 0.7034 | 0.7042 |

이번 분석 기준으로 보면 이 결과는 단순히 “2등 전체가 좋다”라기보다, 2등 안의 두 성향이 GOH30에 없는 방향을 제공했기 때문일 가능성이 있다.

- `second_base`: 덜 꺾고, z를 더 안정적으로 잡는 보수적 expert
- `second_phys`: ODE-heavy와 반대 방향성이 강한 diversity expert
- `second_final`: base와 phys를 60:40으로 섞어 과한 변화를 줄인 안정형 expert

즉 2등 코드를 하나의 외부 expert로 둔 것도 맞지만, 더 정확히는 `base expert`와 `phys expert` 두 개로 나눠서 볼 수 있다.

## 다음 실험 후보

제출 전에 로컬/OOF에서 먼저 볼 만한 후보는 다음이다.

- `0.80 * ODE-heavy + 0.20 * second_final`
  - 이미 제출에서 가장 좋았던 조합.
- `0.80 * ODE-heavy + 0.12 * second_base + 0.08 * second_phys`
  - 2등 final의 60:40 구조를 유지하되 명시적으로 분리한 형태.
- `0.85 * ODE-heavy + 0.10 * second_base + 0.05 * second_phys`
  - 2등 영향이 과한지 확인하는 보수형.
- `0.80 * ODE-heavy + 0.15 * second_base + 0.05 * second_phys`
  - 덜 꺾는/z 안정화 성향을 더 믿는 형태.
- `0.85 * ODE-heavy + 0.05 * second_base + 0.10 * second_phys`
  - ODE-heavy와 반대 방향 diversity를 더 믿는 형태. 다만 리스크는 더 크다.

주의할 점은, 현재 분석은 test 예측 간 차이를 본 것이고 정답을 직접 본 것은 아니다. 그래서 “왜 다른가”를 좁히는 데는 의미가 있지만, 어떤 조합이 무조건 더 맞는지는 OOF나 제출 결과로만 확인할 수 있다.

