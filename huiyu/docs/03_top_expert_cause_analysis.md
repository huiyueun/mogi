# 1등/2등 예측 차이의 원인 가설 분석

작성일: 2026-07-01

## 목적

앞선 분석에서 다음 관찰이 나왔다.

- 2등은 급회전/고노이즈/낮은 직선성 구간에서 1등보다 덜 꺾는 경향이 있다.
- 2등은 최근 회전/고가속 구간에서는 예측 거리를 조금 더 길게 잡는 경향이 있다.
- 2등은 z축 이동을 평균적으로 약간 더 작게 잡고, 특히 수직 변화 구간에서 그 경향이 크다.
- ODE 강화와 2등은 대체로 같은 방향 보정이 아니다.

이 문서는 이 차이가 코드 구조상 어디서 생겼을 가능성이 큰지 정리한 것이다.

중요한 전제:

- 지금 분석은 test 정답 없이 예측끼리 비교한 것이다.
- 따라서 "2등이 왜 더 맞는가"가 아니라 "2등이 왜 다르게 예측하는가"에 대한 원인 가설이다.
- 2등 코드의 최종 base 예측은 압축 내장된 약 40개 멤버 앙상블 결과라 내부 멤버별 원인을 완전히 분해할 수는 없다.

## 결론 요약

가장 그럴듯한 설명은 다음이다.

> GOH30은 등속 base 위에 최근 속도/가속/저크/각속도 피처와 GRU/ODE/H 물리 모델을 적극적으로 사용한다.  
> 반면 2등 코드는 Kalman 등속 baseline, 노이즈/직선성 피처, 잔차 크기 제한, conservative DE 블렌드, 회전물리 3멤버 블렌딩이 섞여 있어 불안정 구간에서 더 보수적으로 예측할 가능성이 크다.

즉 2등은 GOH30의 단순 ODE 강화와 같은 방향의 보정이 아니라, 노이즈/회전/z축 변화에 대해 조금 더 완충된 외부 전문가 예측으로 보인다.

## 코드 구조 비교

| 항목 | GOH30 1등 | 2등 코드 |
| --- | --- | --- |
| 기본 기준 예측 | 등속 외삽 `last + 2*(last-prev)` | Kalman 등속 예측을 여러 멤버의 baseline으로 사용 |
| 좌표계 | 마지막 xy 속도 기준 yaw 정렬 | yaw/local frame, 회전물리에서는 forward/right/up 프레임 |
| 주요 피처 | 상대위치, 속도, 가속, 저크, 각속도, 속력/가속/직선성/노이즈 | 속도, 가속, jerk, 직선성, turn, 다중 노이즈 추정, Kalman 잔차 |
| 모델 구성 | GRU 10 + Neural ODE 10 + HyperPhysics 10 | base 약 40멤버 DE conservative 블렌드 + 회전물리 3멤버 |
| 최종 블렌드 | 원본은 30모델 등가중, 실험은 G20/O60/H20 | 최종은 base 60% + 회전물리 40% |
| 보수적 장치 | EMA, Y-flip TTA, soft-hit, damping | Kalman baseline, `tanh` 잔차 제한, conservative DE, 평활 노이즈 피처, 물리 prior |

## 관찰 1: 2등은 급회전/고노이즈에서 덜 꺾는다

관찰 수치:

| 구간 | 1등 평균 각도 | 2등 평균 각도 | 2등-1등 |
| --- | ---: | ---: | ---: |
| 급회전 | 6.78도 | 5.31도 | -1.47도 |
| 최근 회전 | 9.59도 | 8.26도 | -1.33도 |
| 고노이즈 | 6.03도 | 4.88도 | -1.15도 |
| 낮은 직선성 | 6.34도 | 4.88도 | -1.47도 |

가능한 원인:

1. 2등의 Kalman baseline
   - 2등 §1 공통 코어에 `kalman_predict()`가 있다.
   - 주석상 여러 멤버가 공유하는 잔차 baseline으로 사용된다.
   - Kalman 등속 모델은 관측의 순간 튐을 상태 추정으로 완충하므로 급격한 방향 변화에 덜 끌려갈 수 있다.

2. 2등의 잔차 크기 제한
   - 2등 Pool A의 `GRUModelMultiAux`는 main head 출력에 `tanh`를 적용하고 `main_scale_cm / 100.0`을 곱한다.
   - 기본값은 `main_scale_cm=2.0`, 즉 Kalman 기준 잔차를 약 2cm 범위로 제한하는 구조다.
   - 이 구조는 큰 방향 전환 잔차를 과하게 내기 어렵게 만든다.

3. 2등의 conservative DE 블렌드
   - 2등 노트북 §6에는 base가 "DE conservative 블렌드"라고 설명되어 있다.
   - 여러 멤버를 보수적으로 섞으면 개별 모델의 급격한 회전 예측이 평균화될 수 있다.

4. 2등 회전물리 멤버의 prior와 게이팅
   - 2등 회전물리 모델은 `PriorBiasedLinear`로 prior bias에서 시작한다.
   - `theta`, `speed`, 감쇠항, EMA 속도/가속을 사용한다.
   - 회전물리 자체는 회전을 모델링하지만, prior와 감쇠가 있어 노이즈성 회전을 그대로 따라가기보다 제한된 회전을 만들 가능성이 있다.

GOH30 쪽 근거:

- [goh30_component_submissions.py](/home/huiyu/Workspace/summer/huiyu/experiments/goh30_component_submissions.py:135)는 등속 base 잔차 학습을 사용한다.
- [goh30_component_submissions.py](/home/huiyu/Workspace/summer/huiyu/experiments/goh30_component_submissions.py:160)는 속도, 가속, 저크, 각속도를 시퀀스 피처로 넣는다.
- [goh30_component_submissions.py](/home/huiyu/Workspace/summer/huiyu/experiments/goh30_component_submissions.py:291)의 ODE 모델은 학습된 가속도장을 RK4로 적분한다.
- [goh30_component_submissions.py](/home/huiyu/Workspace/summer/huiyu/experiments/goh30_component_submissions.py:329)의 HyperPhysics는 선회 물리를 명시적으로 모델링한다.

해석:

- GOH30은 최근 운동 변화 신호를 피처와 모델 구조에 적극적으로 넣는다.
- 2등은 Kalman/제한 잔차/보수 앙상블 때문에 급격한 방향 변화에 더 보수적으로 반응했을 가능성이 크다.

## 관찰 2: 2등은 최근 회전/고가속에서 예측 거리를 조금 더 길게 잡는다

관찰 수치:

| 구간 | 2등이 더 길게 예측한 비율 | 평균 거리 차이 |
| --- | ---: | ---: |
| 최근 회전 | 57.3% | +0.000143 |
| 고가속 | 57.6% | +0.000172 |
| 고속 | 50.2% | +0.000123 |
| 급회전 | 49.2% | -0.000175 |
| 고노이즈 | 45.9% | -0.000144 |

가능한 원인:

1. 2등의 ControlHead 계열 모델
   - 2등 §4의 `ControlHead`는 `v_scale * init_vel * T + 0.5 * a * T^2` 형태로 변위를 만든다.
   - `v_scale`은 초기 속도 신뢰도를 학습하는 파라미터다.
   - 최근 회전/고가속에서 초기 속도를 어느 정도 유지하면 1등보다 예측 거리가 길어질 수 있다.

2. 2등의 Neural ODE 계열 모델
   - 2등 §3의 `NeuralODEModel`도 위치/속도 6차원 상태를 두고 `dp/dt = v`, `dv/dt = -damping*v + a_neural` 형태로 적분한다.
   - damping이 있지만 init velocity를 명시적으로 들고 가므로, 최근 회전/가속 구간에서 속도 성분을 유지하는 예측이 나올 수 있다.

3. 구간별로 다른 효과
   - 고가속/최근회전에서는 2등이 조금 더 길게 간다.
   - 하지만 급회전/고노이즈에서는 오히려 더 짧게 잡는다.
   - 따라서 2등이 항상 멀리 가는 모델은 아니고, 노이즈와 가속을 구분해서 보수/전진 성향이 달라지는 것으로 보인다.

해석:

- 2등은 "불안정하면 무조건 짧게"가 아니다.
- 회전/가속이 있지만 속도 신호가 유효하다고 판단되는 경우에는 초기 속도 기반 이동을 더 유지하는 듯하다.
- 반대로 노이즈성 불안정이나 급격한 꺾임에서는 더 짧고 덜 꺾는 방향으로 누르는 것으로 보인다.

## 관찰 3: 2등은 z축 이동을 조금 더 안정화한다

관찰 수치:

| 구간 | 2등이 z 이동을 더 작게 잡은 비율 | 평균 z 이동 차이 |
| --- | ---: | ---: |
| 전체 | 53.0% | -0.000070 |
| 고노이즈 | 53.8% | -0.000143 |
| 수직 변화 | 59.8% | -0.000193 |
| 낮은 직선성 | 53.9% | -0.000126 |

가능한 원인:

1. Kalman baseline의 축별 상태 추정
   - 2등 `kalman_predict()`는 x/y/z 각 축을 독립적인 등속 상태로 필터링한다.
   - z축이 노이즈성으로 튀면 등속 상태 추정이 완충 역할을 할 수 있다.

2. 2등 노이즈 피처
   - 2등은 `noise_poly2`, `noise_savgol`, `noise_loo_subset`을 사용한다.
   - Savitzky-Golay 평활 잔차와 다항 잔차를 노이즈 피처로 넣는다.
   - 이런 피처는 흔들림이 큰 구간에서 모델이 과한 z 변화를 덜 믿게 하는 데 쓰였을 가능성이 있다.

3. 회전물리 프레임의 z 유지/분리
   - 2등 공통 회전 함수는 xy 회전 시 z를 유지한다.
   - 회전물리에서는 forward/right/up 프레임을 만들고, z/수직 성분을 로컬 프레임에서 다룬다.
   - 이것이 z축을 xy 회전과 분리해서 더 안정적으로 다루는 효과를 냈을 수 있다.

GOH30 쪽 대비:

- GOH30도 z를 포함한 3D 피처와 잔차를 학습하지만, 속도/가속/저크/각속도 피처가 모두 3D로 들어간다.
- 그래서 z축 변화가 최근 동역학 신호로 강하게 들어가면 2등보다 더 반응할 수 있다.

해석:

- z축 안정화는 2등의 Kalman baseline + 노이즈 피처 + 회전물리 프레임 설계가 같이 만든 효과일 가능성이 있다.
- 다만 평균 차이는 매우 작으므로, 강한 z축 전용 후처리가 있었다고 단정하기보다는 약한 안정화 성향으로 보는 것이 맞다.

## 관찰 4: ODE 강화와 2등은 같은 방향 보정이 아니다

관찰 수치:

| 구간 | 같은 방향 비율 | 강한 같은 방향 비율 | 평균 코사인 |
| --- | ---: | ---: | ---: |
| 전체 | 33.4% | 15.7% | -0.239 |
| 고속 | 37.3% | 17.5% | -0.170 |
| 급회전 | 31.6% | 17.0% | -0.273 |
| 최근 회전 | 38.3% | 21.1% | -0.179 |
| 고가속 | 38.2% | 20.6% | -0.169 |
| 고노이즈 | 33.9% | 18.0% | -0.230 |

가능한 원인:

1. ODE 강화는 GOH30 내부 비율 조정이다.
   - [blend_second_place_expert.py](/home/huiyu/Workspace/summer/huiyu/experiments/blend_second_place_expert.py:57)에서 GOH30 컴포넌트는 GRU/ODE/H 예측으로 나뉜다.
   - [blend_second_place_expert.py](/home/huiyu/Workspace/summer/huiyu/experiments/blend_second_place_expert.py:61)의 confirmed ODE 강화는 `0.20*GRU + 0.60*ODE + 0.20*H`다.
   - 즉 ODE 강화는 같은 GOH30 피처/학습 체계 안에서 동역학 모델 비중만 키운 것이다.

2. 2등은 다른 baseline과 다른 앙상블이다.
   - [blend_second_place_expert.py](/home/huiyu/Workspace/summer/huiyu/experiments/blend_second_place_expert.py:63)는 2등 base 예측을 복원한다.
   - [blend_second_place_expert.py](/home/huiyu/Workspace/summer/huiyu/experiments/blend_second_place_expert.py:64)는 2등 회전물리 예측을 복원한다.
   - [blend_second_place_expert.py](/home/huiyu/Workspace/summer/huiyu/experiments/blend_second_place_expert.py:65)는 2등 최종 예측을 `base 60% + phys 40%`로 만든다.

해석:

- ODE 강화는 GOH30 내부에서 더 동역학적으로 가는 보정이다.
- 2등은 Kalman 잔차, conservative DE, 회전물리 블렌드가 섞인 별도 시스템이다.
- 그래서 둘이 점수상으로 모두 좋아 보여도 예측점 이동 방향은 서로 다를 수 있다.
- 이 점 때문에 2등을 외부 전문가로 섞었을 때 다양성 효과가 생겼다고 볼 수 있다.

## 원인 가설 우선순위

현재 증거 기준으로 가능성이 높은 순서:

1. 2등의 Kalman baseline과 잔차 제한이 급회전/노이즈 구간에서 보수성을 만든다.
2. 2등의 conservative DE 블렌드가 극단적 예측을 평균화한다.
3. 2등의 노이즈 피처가 고노이즈와 z축 변화 구간에서 과한 이동을 줄인다.
4. 2등 회전물리 40% 블렌드가 GOH30과 다른 방향의 물리적 보정을 추가한다.
5. GOH30은 최근 속도/가속/저크/각속도 피처와 ODE/H 모델 때문에 최근 변화에 더 민감하다.

## 아직 확인이 필요한 부분

2등 최종 base는 압축 내장된 약 40멤버 앙상블 결과다. 따라서 지금은 다음을 직접 분해하지 못했다.

- base 안에서 Kalman 잔차 GRU가 얼마나 기여했는지
- base 안에서 Neural ODE, ControlHead, 기타 멤버가 각각 어떤 방향성을 냈는지
- 회전물리 3멤버만 따로 봤을 때 위 패턴이 얼마나 강한지
- train 외부폴드에서 실제 정답 기준으로 어느 구간에서 2등 방식이 더 맞는지

## 다음 검증 제안

가장 의미 있는 다음 분석은 2등을 세 부분으로 분해해서 같은 진단을 다시 하는 것이다.

1. 2등 base 단독
2. 2등 회전물리 단독
3. 2등 최종 `base 60% + 회전물리 40%`

이미 `blend_second_place_expert.py`에서 base와 phys를 복원하고 있으므로, 분석 스크립트에 이 둘을 별도 입력으로 넣으면 된다.

보고 싶은 질문:

- 덜 꺾는 성향은 base에서 오는가, 회전물리에서 오는가?
- z축 안정화는 base에서 오는가, 회전물리에서 오는가?
- 2등 최종이 GOH30과 다른 방향인 이유가 phys 40% 때문인가, base 자체 때문인가?

이것까지 보면 "2등 코드의 어떤 내부 구현이 차이를 만들었는지"를 지금보다 훨씬 더 좁힐 수 있다.
