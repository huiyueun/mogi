# 모기 궤적 예측 Feature / Pattern / Noise 정리

## 0. 전제

원본 데이터는 각 샘플마다 11개 시점의 `x`, `y`, `z` 좌표와 `timestep_ms`만 제공된다.

- 입력 시점: `-400ms, -360ms, ..., -40ms, 0ms`
- 예측 대상: 마지막 관측 시점 기준 `+80ms`의 `x`, `y`, `z`
- 좌표 단위: meter
- 제공되지 않는 정보: `scene_id`, 환경 라벨, 속도, 가속도, 센서 노이즈 라벨

따라서 상위권 해답들은 `x`, `y`, `z` 좌표에서 운동 feature를 직접 만들고, 이를 바탕으로 미래 위치를 예측했다.

## 1. X, Y, Z에서 확장한 Feature

### 1.1 위치 관련 Feature

좌표 자체와 위치 변화량을 사용한다.

- 마지막 위치: `p0 = p[-1]`
- 직전 위치: `p[-2]`
- 상대 위치: `p[t] - p0`
- 전체 변위: `p[-1] - p[0]`
- 순변위 크기: `net_disp = ||p[-1] - p[0]||`
- 누적 이동 거리: `path_length = sum(||p[t] - p[t-1]||)`
- 직선성: `straightness = net_disp / path_length`

직선성은 궤적이 곧게 이동했는지, 많이 흔들리거나 회전했는지를 나타내는 핵심 feature다.

### 1.2 속도 Feature

좌표 차분으로 속도를 만든다. 시간 간격은 `40ms = 0.04s`다.

```python
v[t] = (p[t] - p[t-1]) / 0.04
```

주요 feature:

- `v[t]`
- `speed = ||v||`
- `mean_speed`
- `max_speed`
- `speed_std`
- 마지막 속도: `v_last`
- 마지막 속도 크기: `|v_last|`
- 속도 구간 one-hot

상위 풀이 중 일부는 마지막 속도를 다음 구간으로 나누어 사용했다.

```text
SPEED_BINS = [0.0, 0.3, 0.6, 0.9, 1.2, inf]
```

이는 모기 궤적을 느림, 보통, 빠름 같은 운동 regime으로 구분하는 역할을 한다.

### 1.3 가속도 Feature

속도의 차분으로 가속도를 만든다.

```python
a[t] = (v[t] - v[t-1]) / 0.04
```

주요 feature:

- `a[t]`
- `acc_magnitude = ||a||`
- `mean_acc`
- `max_acc`
- 마지막 가속도: `a_last`
- 마지막 가속도 크기: `|a_last|`
- 최근 평균 가속도: `|a_recent|`
- 고가속 flag: `high_acc = max_acc > 15`

일부 해답은 마지막 가속도 크기가 큰 샘플을 minority regime으로 보고 검증 fold를 균형 있게 나눴다.

```text
minority_mask = ||a_last|| >= 5.0
```

### 1.4 Jerk Feature

jerk는 가속도의 변화량이다.

```python
j[t] = (a[t] - a[t-1]) / 0.04
```

주요 feature:

- `j[t]`
- `jerk_last`
- `jerk_recent`
- `max_jerk`
- `jerk_magnitude = ||j||`

jerk는 갑작스러운 방향 변화나 가속 변화가 있는 궤적을 포착하는 데 사용된다.

### 1.5 방향 / 회전 Feature

모기 비행은 절대 좌표축보다 최근 진행 방향 기준으로 보는 것이 유리하다. 그래서 많은 해답이 방향과 회전 관련 feature를 만들었다.

주요 feature:

- `yaw_angle`
- 마지막 속도 방향 기준 회전 좌표계
- `theta`: 연속 속도 벡터 사이 각도
- `theta_mean`
- `theta_std`
- `theta_vel`
- `theta_acc`
- `theta_trend`
- `angular_velocity`
- `turn_cos`

`theta`는 급회전 여부를 나타내는 핵심 feature다.

```python
theta = arccos(cos(v[t], v[t-1]))
```

`turn_cos`는 마지막 진행 방향과 과거 평균 진행 방향이 얼마나 비슷한지를 나타낸다.

```text
hard_turn = turn_cos < 0.5
```

### 1.6 궤적 모양 Feature

궤적의 전체 형태를 요약하는 feature도 많이 사용된다.

- `straightness`: 직선성
- `curvature`: 곡률
- `net_disp / path_length`
- `z_speed`
- `z_acc`
- `clipping` 또는 `clip_flag`
- `high_speed`
- `high_acc`

예시:

```text
high_speed = |v_last| > 1.0
high_acc = max_acc > 15
```

### 1.7 좌표계 변환 Feature

상위권 해답들은 원본 world 좌표를 그대로 쓰기보다, 최근 운동 방향 기준으로 좌표계를 바꾸는 경우가 많았다.

사용된 좌표계:

- `yaw frame`
- `local heading frame`
- `Frenet frame`

Frenet frame 구성:

- `tangent`: 최근 진행 방향
- `normal`: 가속도의 수직 성분 방향
- `binormal`: `tangent x normal`

관련 feature:

- 진행 방향 성분: `parallel`
- 수직 방향 성분: `perpendicular`
- `parallel acceleration`
- `perpendicular acceleration`
- Frenet frame 기준 residual

이 방식은 모기가 어느 절대 방향으로 날고 있든, 진행 방향 기준으로 비슷한 패턴을 학습하게 만든다.

### 1.8 Baseline Residual Feature

좌표를 직접 예측하기보다 강한 물리 baseline을 먼저 만들고, 모델은 그 잔차만 학습하는 방식이 많았다.

대표 baseline:

```python
constant_velocity = p[-1] + 2 * (p[-1] - p[-2])
```

즉 마지막 40ms 이동량이 다음 80ms 동안 유지된다고 보는 등속 외삽이다.

다른 baseline:

- Constant Velocity baseline
- Constant Acceleration baseline
- Kalman filter baseline
- Frenet physics candidate
- turn / jerk / latency candidate

모델 target은 다음처럼 정의된다.

```python
residual = y_true - baseline_pred
```

이렇게 하면 모델이 전체 좌표를 새로 예측하지 않고, 물리적으로 그럴듯한 예측에서 얼마나 보정할지만 학습하게 된다.

## 2. 모기 비행 Pattern 구분

### 2.1 Scene ID는 사용하지 않음

데이터에는 `scene_id`, 장소, 환경 조건, 센서 종류 같은 정보가 없다.

상위권 해답들도 명시적으로 공간 또는 환경 scene을 추정해서 나누지는 않았다.

즉 주류 접근은 다음이 아니다.

```text
이 샘플은 실내 scene
이 샘플은 복도 scene
이 샘플은 야외 scene
```

대신 좌표 궤적에서 드러나는 운동 패턴을 기준으로 샘플을 구분했다.

### 2.2 속도 Regime

마지막 속도 또는 평균 속도에 따라 움직임을 구분한다.

예시:

```text
SPEED_BINS = [0.0, 0.3, 0.6, 0.9, 1.2, inf]
```

의미:

- 매우 느림
- 느림
- 보통
- 빠름
- 매우 빠름

이 값은 one-hot scalar feature로 들어가거나, 고속 샘플 flag로 사용된다.

```text
high_speed = |v_last| > 1.0
```

### 2.3 급회전 Regime

`theta`가 큰 샘플은 급회전 샘플로 본다.

일부 해답은 급회전 샘플을 더 자주 학습하도록 weighted sampling을 사용했다.

```python
theta_weights = 1.0 + 4.0 * clamp(theta_last / 1.0, 0, 1)
```

의미:

- 직진 샘플: weight 약 1
- 급회전 샘플: weight 최대 5

또 다른 구분:

```text
hard_turn = turn_cos < 0.5
```

### 2.4 고가속 Minority Regime

가속도가 큰 샘플은 일반 샘플보다 예측이 어렵다. 일부 해답은 이를 minority로 정의했다.

```text
minority_mask = ||a_last|| >= 5.0
```

이 mask는 `StratifiedKFold`에 사용되어 train/validation fold의 어려운 샘플 비율을 맞추는 데 쓰였다.

### 2.5 Candidate Family Regime

후보 선택형 해답은 미래 위치 후보를 여러 family로 나눴다.

대표 family:

- 기본 등속 후보
- 가속도 후보
- Frenet 후보
- turn 후보
- jerk 후보
- latency 후보

이 방식은 사실상 움직임 패턴별 분기다.

예를 들어:

- 직진에 가까운 샘플은 등속/Frenet 후보가 유리할 수 있음
- 급회전 샘플은 turn 후보가 유리할 수 있음
- 가속 변화가 큰 샘플은 jerk 후보가 유리할 수 있음
- 관측 지연처럼 보이는 샘플은 latency 후보가 유리할 수 있음

## 3. 노이즈 처리 방식

### 3.1 노이즈 샘플을 버리지는 않음

상위권 해답에서 노이즈가 커 보이는 샘플을 단순 삭제하는 방식은 주류가 아니었다.

대신 다음 방식이 많았다.

- 노이즈 크기를 추정한다.
- 추정된 노이즈를 scalar feature로 넣는다.
- Kalman filter 또는 smoothing으로 안정적인 baseline을 만든다.
- 모델의 보정폭을 제한해서 노이즈에 과하게 반응하지 않게 한다.

### 3.2 `noise_poly2`

2차 다항식으로 궤적을 fitting한 뒤, 원본과 fitting 곡선의 차이를 노이즈로 본다.

개념:

```python
fit = polyfit_2nd_order(t, x_y_z)
noise_poly2 = std(original - fit)
```

의미:

- 부드러운 2차 운동으로 설명되지 않는 흔들림
- 관측 jitter 또는 비정상적인 궤적 변화의 proxy

### 3.3 `noise_savgol`

Savitzky-Golay smoothing을 적용한 뒤, 원본과 smoothing 결과의 차이를 노이즈로 본다.

개념:

```python
smooth = savgol_filter(X, window_length=5, polyorder=2)
noise_savgol = std(X - smooth)
```

의미:

- 짧은 window에서 부드러운 곡선 대비 얼마나 흔들리는지
- 계산이 비교적 빠르며 test에도 적용 가능

### 3.4 `noise_loo`

Leave-one-out cubic spline 방식이다.

한 시점을 빼고 나머지 시점으로 spline을 만든 뒤, 빠진 점을 얼마나 잘 복원하는지 본다.

개념:

```text
각 중간 시점 k에 대해:
1. k번째 점을 제거
2. 나머지 점으로 cubic spline 보간
3. 보간값과 실제 k번째 점의 차이를 계산
4. 전체 차이의 RMS를 noise_loo로 사용
```

의미:

- 주변 점들로 설명되지 않는 관측 흔들림
- 계산량이 커서 일부 풀이에서는 train에만 쓰거나 `noise_savgol`로 대체

### 3.5 Kalman Filter 기반 노이즈 완화

일부 해답은 Kalman filter를 사용해 관측 노이즈와 실제 운동 불확실성을 분리했다.

주요 파라미터:

- `sigma_obs`: 관측 노이즈 크기
- `sigma_proc`: 실제 운동 변화 불확실성

예시:

```python
kalman_pred = kalman_predict(
    X,
    sigma_obs=0.30e-3,
    sigma_proc=1.0
)
```

이후 모델은 Kalman 예측의 잔차를 학습한다.

```python
residual_target = y_true - kalman_pred
```

의미:

- Kalman filter가 안정적인 물리 baseline 역할을 함
- 모델은 baseline이 놓친 비선형 움직임만 보정

### 3.6 과보정 방지

노이즈가 큰 샘플에서는 모델이 흔들림을 실제 운동으로 오해할 수 있다. 그래서 여러 해답이 보정폭을 제한했다.

사용된 방식:

- gradient clipping
- residual correction 제한
- candidate tiny correction
- boundary correction cap
- LGBM residual correction clip

예시:

```text
boundary correction cap = 8mm
LGBM residual correction <= +/- 0.0035m per axis
```

핵심은 노이즈를 제거하는 것이 아니라, 노이즈 때문에 예측이 과격하게 튀지 않도록 보정폭을 작게 유지하는 것이다.

## 4. 전체 요약

상위권 해답들의 공통 구조는 다음과 같다.

```text
x, y, z 좌표
-> 속도 / 가속도 / jerk / 방향 / 곡률 feature 생성
-> yaw, local, Frenet frame으로 좌표계 정렬
-> 등속 또는 Kalman 기반 baseline 생성
-> target을 baseline residual로 변환
-> 속도, 회전, 고가속, candidate family로 운동 pattern 구분
-> 노이즈는 삭제하지 않고 feature로 추정
-> 보정폭을 제한해 1cm hit boundary 근처에서 안정적으로 조정
```

즉 이 문제의 핵심은 `scene_id`를 찾는 것이 아니라, `x,y,z`에서 짧은 시간의 동역학을 최대한 안정적으로 복원하고, 1cm 명중률에 맞게 보수적으로 보정하는 것이다.
