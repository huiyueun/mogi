#!/usr/bin/env python
# coding: utf-8

# # DACON 모기 비행 궤적 예측 — Private LB 0.7035 재현 .ipynb
# 
# 11스텝(−400~0ms, 40ms 간격) 3D 위치로 **+80ms 위치**를 예측. 지표 **R-Hit@1cm**(정답과 ≤1cm 비율).
# 
# **최종 = GOH30**: GRU + Neural ODE + HyperPhysics 3개 아키텍처 각 10시드, 등가중 30모델 블렌드.
# 
# **이 노트북 하나로 raw 데이터 → 전처리 → 학습 → 제출까지 전 과정을 자체 완결** (외부 .py/.pt/.npz 불필요).
# 
# **공통 레시피:** cv_1step base + yaw회전 잔차학습 + Soft R-Hit 손실 + 내부전이 사전학습 + EMA + Y-flip TTA.
# 
# | 섹션 | 내용 |
# |---|---|
# | cell 1 | 설정 (import·device·상수·토글) |
# | cell 2 | 피처 엔지니어링 |
# | cell 3 | 전처리 (norm_stats + 캐시, raw에서 생성) |
# | cell 4 | 모델 정의 (GRU / ODE / HyperPhysics) |
# | cell 5 | 손실 + 학습 함수 |
# | cell 6 | 학습 실행 (30모델) |
# | cell 7 | 예측 + 블렌드 → 제출 |
# | cell 8 | 요약 |
# 

# ## 개발 환경
# - Python 3.11.15 (conda env: DA_project)
# - torch 2.12.0 · numpy 2.4.6 · pandas 3.0.3 · scikit-learn 1.8.0
# - macOS (Apple Silicon, arm64) / 디바이스: MPS (CUDA·CPU 자동 fallback)
# 
# ## RAW 데이터 파일 설정 위치
# **📂 폴더 구조** — `Data/`만 직접 배치하면 되고, `models_goh30/`(.pt)와 `submission_GOH30.csv`는 **실행 시 자동 생성**된다:
# 
# ```text
# <>/
# ├── GOH30_reproduce.ipynb         # 현재 재현 노트북
# │
# ├── Data/                         # [입력] 직접 배치 (DACON 원본, DATA_DIR=./Data)
# │   ├── train/                    #       학습 궤적 CSV 10,000개
# │   ├── test/                     #       테스트 궤적 CSV 10,000개
# │   └── train_labels.csv          #       정답 라벨
# │
# ├── models_goh30/                 # [생성] cell 1이 폴더 생성 → cell 6학습 실행시 .pt 생성 및 저장
# │   ├── phaseG_full_0-9.pt        #       GRU 10개
# │   ├── phaseODE_full_0-9.pt      #       ODE 10개
# │   └── phaseH_full_0-9.pt        #       HyperPhysics 10개   (합 30개)
# │
# └── submission_GOH30.csv          # [생성] cell 7 실행시 생성. 예측 단계 결과 = 최종 제출 .csv
# ```
# 
# ## 런타임
# - .ipynb에서는 모델 학습 병렬처리가 힘든 것 같습니다.
# - 첫 학습하는데 걸리는 시간은 약 2시간 10분입니다(MPS 기준).
# - 학습된 모델은 `./models_goh30/`에 저장되며, 모델 학습 후 cell 1를 `FROM_SCRATCH=False`로 수정 후 재실행 시 로 ~2분에 재현 가능.
# 
# 
# ## 모델 개요 — GOH30 (Private LB 0.7035)
# GRU + Neural ODE + HyperPhysics 3개 아키텍처를 각 10시드씩, 등가중 평균한 30모델 블렌드.
# - GRU: 양방향 GRU(h=128, 3층) + attention pooling
# - Neural ODE: GRU 인코더 → 댐핑 가속도장 RK4 적분(nsteps=4)
# - HyperPhysics: 물리 gray-box (roll 기반 Rodrigues 회전 + θ/speed 게이팅)
# 공통: cv_1step base, yaw회전 잔차학습, Soft R-Hit 손실, 내부전이 사전학습, EMA, Y-flip TTA.

# In[1]:


# ══════════════════════════════════════════════════════════════════════
# cell 0 · 의존성 설치  (최초 1회 실행 · 로컬에 이미 있으면 생략 가능)
#   검증 환경: Python 3.11.15 · torch 2.12.0 · numpy 2.4.6 · pandas 3.0.3
# ══════════════════════════════════════════════════════════════════════
# Prediction-only script extracted from the GOH30 notebook.
# Dependencies should already be installed in the active environment.

# 정확한 버전 고정이 필요하면 아래 사용:
# !pip install -q numpy==2.4.6 pandas==3.0.3 torch==2.12.0


# ## cell 1. 설정 (import · device · 토글 · 상수)
# 
# 라이브러리 import → 연산 디바이스 자동 선택(CUDA→MPS→CPU) → 실행 제어값 정의.
# 
# 설정값은 두 종류로 나뉜다.
# 
# - **토글** — 환경·실행 방식에 맞춰 자유롭게 바꾸는 스위치
#   - `DATA_DIR` : DACON 원본 데이터 위치
#   - `FROM_SCRATCH` : `True`=30모델 처음부터 학습(~2시간 10분) / `False`=`./models_goh30/` 학습본 로드(~2분)
#   - `MODELS_DIR` : 학습 가중치 저장·로드 폴더
# - **상수** — 하이퍼파라미터. **0.7035 재현을 위해 고정**(바꾸면 결과가 달라짐). 각 값의 의미는 아래 코드 주석 참고.
# 
# > 💡 set_seed()로 시드를 고정. 검증 환경(Apple Silicon MPS)에서는 처음부터 재학습해도 동일 점수(0.7035)가 재현됨. (다른 환경에선 부동소수점 차이로 미세하게 다를 수 있음)
# 

# In[2]:


import os, glob, random, time
from datetime import datetime
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler   # WeightedRandomSampler: H 학습의 θ-가중 오버샘플

# ── 토글 (실행 환경/방식에 맞춰 바꾸는 스위치) ──────────────────────────────
DATA_DIR     = os.environ.get('DATA_DIR', './open')   # DACON 원본 위치 (open/train, open/test, train_labels.csv). 환경변수로도 지정 가능
FROM_SCRATCH = False
MODELS_DIR   = os.environ.get('MODELS_DIR', './models_goh30')                         # 학습 가중치 저장·로드 폴더
OUT_DIR      = os.environ.get('OUT_DIR', './outputs/goh30_component_submissions')
os.makedirs(MODELS_DIR, exist_ok=True)                 # 폴더 없으면 생성
os.makedirs(OUT_DIR, exist_ok=True)
_model_count = len(glob.glob(f'{MODELS_DIR}/phase*_full_*.pt'))
assert _model_count == 30, f'{MODELS_DIR}에 30개 .pt 필요 (현재 {_model_count}개). 학습 완료 후 실행하세요.'

# 연산 디바이스 자동 선택: CUDA > MPS(Apple Silicon) > CPU
DEVICE = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))

# ── 상수 (우승 레시피 하이퍼파라미터 · 0.7035 재현 위해 고정 — 바꾸면 결과 달라짐) ──
DT = 0.04; PRED_DT = 0.08; CLIP_THR = 1.33     # 관측간격 40ms · 예측시점 +80ms · 센서 클리핑 속력기준(scalar clip_flag용)
SPEED_BINS = [0.0, 0.3, 0.6, 0.9, 1.2, np.inf]                       # scalar 피처의 속력 구간 one-hot(5칸) 경계
SIGMA = 0.02; RHIT_TAU = 0.0015; RHIT_W = 2.0; HW = 0.5; GW = 0.5     # GRU/ODE 손실: 가우시안폭 σ · R-Hit 날카로움 τ · R-Hit가중 · Huber/soft 가중
FLIP_PROB = 0.5; NOISE_STD = 0.02; Y_FLIP = [1, 4, 7, 10]; INTERIOR_E = [5, 6, 7, 8]   # 증강: y-flip 확률·입력노이즈 / Y_FLIP=좌우대칭 시 부호반전할 seq채널 / INTERIOR_E=내부전이 사전학습 지점 e
N_EACH = 10                                                          # 아키텍처당 시드 수 (GRU10 + ODE10 + H10 = 30모델)
GRU_ODE_EPOCHS = 55; H_EPOCHS = 12; EMA_DECAY = 0.9                  # 전체학습 에폭(GRU·ODE / HyperPhysics) · EMA 가중평균 감쇠율

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)         # random·numpy·torch 시드 동시 고정 → 재현성 확보

print('device:', DEVICE, '| FROM_SCRATCH:', FROM_SCRATCH, '| DATA_DIR:', DATA_DIR)


# ## cell 2. 피처 엔지니어링
# 
# 원본 `src/preprocess.py` + `src/features_clean.py` + `window_features`를 노트북에 인라인.
# 
# **핵심 설계**
# - **base = cv_1step** = `last + 2·(last−prev)` — 등속 가정 +80ms 예측. 모든 모델은 이 base의 *잔차*만 학습(난이도↓·안정성↑).
# - **yaw 정렬** — 마지막 속도의 xy 방향을 +x축으로 회전(z축). 절대 방위에 불변 → 모기가 어느 방향을 향하든 같은 패턴으로 학습 → 일반화↑.
# - **속도·가속·저크** = `np.gradient` 중앙차분 (Kalman 미사용).
# 
# **산출 피처**
# - `seq` (11, 13): 상대위치3 · 회전속도3 · 가속3 · 저크3 · 각속도1
# - `scalar` (22): 속력·가속·직선성·클리핑·방향일관성 등 14 + 추가 통계 8

# In[3]:


# ── raw CSV(11×xyz) 로드 ──
def load_sample(path):
    df = pd.read_csv(path)
    return df[['x', 'y', 'z']].to_numpy(dtype=np.float32)   # (11,3) 위치 배열

# ── yaw 회전행렬: xy 속도방향을 +x축에 정렬(z축 회전) → 방위 불변 ──
def yaw_rotation_matrix(velocity):
    vx, vy = float(velocity[0]), float(velocity[1])
    speed_xy = np.sqrt(vx ** 2 + vy ** 2)
    if speed_xy < 1e-6:   # 정지면 회전 없음(단위행렬)
        return np.eye(3, dtype=np.float32)
    cos_yaw, sin_yaw = vx / speed_xy, vy / speed_xy
    return np.array([[cos_yaw, sin_yaw, 0.0], [-sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)   # z축 회전행렬 (3,3)

# ── 시퀀스 피처 (11,13): 회전프레임 위치/속도/가속/저크/각속도 ──
def extract_seq_features(smoothed_pos, smoothed_vel, rot):
    last_pos = smoothed_pos[-1]
    rel_pos = (smoothed_pos - last_pos) @ rot.T   # 마지막 위치 기준 상대위치 → 회전프레임
    vel_rot = smoothed_vel @ rot.T   # 속도도 회전프레임 (11,3)
    accel = np.zeros_like(vel_rot)
    accel[1:-1] = (vel_rot[2:] - vel_rot[:-2]) / (2 * DT); accel[0] = accel[1]; accel[-1] = accel[-2]   # 가속도=속도 중앙차분(양끝 복제)
    jerk = np.zeros_like(accel)
    jerk[1:-1] = (accel[2:] - accel[:-2]) / (2 * DT); jerk[0] = jerk[1]; jerk[-1] = jerk[-2]   # 저크=가속도 중앙차분
    speed = np.linalg.norm(vel_rot, axis=1, keepdims=True)
    v_norm = vel_rot / (speed + 1e-12)
    cos_sim = (v_norm[:-1] * v_norm[1:]).sum(axis=1)
    angular_vel = np.concatenate([[cos_sim[0]], cos_sim])   # 연속 속도방향 코사인유사도(선회 정도)
    features = np.concatenate([rel_pos, vel_rot, accel, jerk, angular_vel[:, None]], axis=1)   # 3+3+3+3+1=13채널
    return features.astype(np.float32)

# ── 스칼라 피처: 궤적 요약 통계 ──
def extract_scalar_features(smoothed_pos, smoothed_vel):
    speeds = np.linalg.norm(smoothed_vel, axis=1); last_speed = float(speeds[-1])
    vel_diff = np.diff(smoothed_vel, axis=0) / DT; accel_mag = np.linalg.norm(vel_diff, axis=1)
    last_accel = float(accel_mag[-1]); mean_accel = float(accel_mag.mean())
    t = np.arange(len(smoothed_pos), dtype=np.float32); r2_list = []
    for dim in range(3):
        y = smoothed_pos[:, dim]; coeffs = np.polyfit(t, y, 1); y_pred = np.polyval(coeffs, t)
        ss_res = float(np.sum((y - y_pred) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2_list.append(1.0 - ss_res / (ss_tot + 1e-10))
    linearity = float(np.mean(r2_list)); clip_flag = float(last_speed > CLIP_THR)   # 직선성(R²) · 클리핑여부(속력>1.33)
    v_norm = smoothed_vel / (np.linalg.norm(smoothed_vel, axis=1, keepdims=True) + 1e-12)
    cos_sim_all = (v_norm[:-1] * v_norm[1:]).sum(axis=1)
    dir_consistency = float(cos_sim_all.mean()); delta_speed = float(speeds[-1] - speeds[-2])
    last_dir_change = float(cos_sim_all[-1])
    last_vel_norm = v_norm[-1]; last_accel_vec = vel_diff[-1]
    tangential = np.dot(last_accel_vec, last_vel_norm) * last_vel_norm
    last_normal_accel = float(np.linalg.norm(last_accel_vec - tangential))   # 법선(구심)가속 = 선회 강도
    speed_bin = np.zeros(5, dtype=np.float32)
    for k in range(5):   # 마지막 속력 → 5구간 one-hot
        if SPEED_BINS[k] <= last_speed < SPEED_BINS[k + 1]: speed_bin[k] = 1.0; break
    scalar = np.array([last_speed, last_accel, mean_accel, linearity, clip_flag,
                       dir_consistency, delta_speed, last_dir_change, last_normal_accel], dtype=np.float32)
    return np.concatenate([scalar, speed_bin])   # 9 + 5 = 14

# ── 가변길이 윈도우(L≥4) → 피처 (build_features_clean의 일반화) ──
def window_features(W):
    '''길이 L>=4 윈도우 → seq(L,13), scalar(22), rot, base, last_pos. (build_features_clean의 일반화)'''
    W = W.astype(np.float64); vel = np.gradient(W, DT, axis=0); rot = yaw_rotation_matrix(vel[-1])   # 속도=중앙차분, 회전=마지막 속도방향
    seq = extract_seq_features(W, vel, rot); b14 = extract_scalar_features(W, vel)
    sp = np.linalg.norm(vel, axis=1); steps = np.linalg.norm(np.diff(W, axis=0), axis=1); L = len(W)
    path = float(steps.sum()); net = float(np.linalg.norm(W[-1] - W[0])); straight = net / (path + 1e-8)
    t = np.arange(float(L)); noise = float(np.mean([(W[:, d] - np.polyval(np.polyfit(t, W[:, d], 2), t)).std() for d in range(3)]))
    k = min(4, L); acc_trend = float(np.polyfit(np.arange(float(k)), sp[-k:], 1)[0])
    sc = np.concatenate([b14, [float(sp.max()), float(sp.std()), float(sp[-3:].mean()), float(sp[-5:].mean()),   # 추가 8: 최대·표준편차·최근3/5평균 속력 등
                               path, straight, noise, acc_trend]]).astype(np.float32)
    base = (W[-1] + 2.0 * (W[-1] - W[-2])).astype(np.float32)   # base = cv_1step (+80ms 등속)
    return seq.astype(np.float32), sc, rot.astype(np.float32), base, W[-1].astype(np.float32)   # seq, scalar22, rot, base, last_pos

# ── L=11 전용 진입점(테스트 전처리) ──
def build_features_clean(X):
    '''L=11 전용 (window_features와 동일 산식). 테스트 전처리용.'''
    return window_features(X)

# ── norm_stats로 표준화 ──
def normalize(seq, scalar, stats):
    seq_n = ((seq - stats['seq_mean']) / stats['seq_std']).astype(np.float32)
    scal_n = ((scalar - stats['scalar_mean']) / stats['scalar_std']).astype(np.float32)
    return seq_n, scal_n

print('피처 함수 정의 완료')


# ## cell 3. 전처리 — norm_stats + 학습 캐시 (raw에서 직접 생성)
# 
# `Data/train`(10,000 궤적)에서 두 가지를 메모리에 생성.
# 
# 1. **norm_stats** — 전체 학습셋 피처의 평균/표준편차(seq 13 + scalar 22). 입력 표준화에 사용.
# 2. **학습 캐시** — 궤적당 5개 예시 = 총 50,000:
#    - **real** (e=10): 11스텝 전체 → 실제 라벨(+80ms)
#    - **interior** (e∈{5,6,7,8}): 앞부분만 잘라 내부 지점 `X[e+2]`를 타깃으로 = *내부 전이 사전학습*. 짧은 윈도우는 좌측 zero-pad + mask.
#    - 타깃은 base 기준 잔차를 yaw 프레임으로: `rot @ (tgt − base)`.
# 
# > 이 캐시로 GRU·ODE를 학습. (HyperPhysics는 raw 궤적을 직접 사용 — cell 4c/5 참고.)

# In[4]:


TRAIN_DIR = f'{DATA_DIR}/train'; TEST_DIR = f'{DATA_DIR}/test'
train_paths = sorted(glob.glob(f'{TRAIN_DIR}/*.csv'))
labels = pd.read_csv(f'{DATA_DIR}/train_labels.csv').sort_values('id').reset_index(drop=True)[['x', 'y', 'z']].to_numpy(np.float32)   # id 정렬 후 (x,y,z) 라벨
assert len(train_paths) == len(labels), (len(train_paths), len(labels))
print(f'train 궤적 {len(train_paths):,}개')

# ── (1) norm_stats: 전체 학습셋 피처의 평균/표준편차 ───────────────────
_SEQ, _SC = [], []
for p in train_paths:
    X = pd.read_csv(p)[['x', 'y', 'z']].to_numpy(); s, sc, *_ = build_features_clean(X); _SEQ.append(s); _SC.append(sc)   # 궤적별 피처
_SEQ = np.stack(_SEQ); _SC = np.stack(_SC)
STATS = {'seq_mean': _SEQ.reshape(-1, 13).mean(0), 'seq_std': _SEQ.reshape(-1, 13).std(0),   # 채널별 평균/표준편차
         'scalar_mean': _SC.mean(0), 'scalar_std': _SC.std(0)}
print('norm_stats 생성 (seq 13 + scalar 22)')

print('예측 전용 실행: 학습 캐시 생성 생략')


# ## cell 4. 모델 정의
# 
# 3개 아키텍처를 정의한다 — **GRU**(시퀀스), **Neural ODE**(물리 적분), **HyperPhysics**(선회 물리). 각 10시드 학습 → 등가중 30모델 블렌드.
# 
# > 서로 다른 귀납 편향(inductive bias)을 가진 모델을 섞어 **탈상관**시키는 것이 핵심. 세 모델이 서로 다른 메커니즘이라 블렌드 다양성이 확보된다.

# ### cell 4a. GRU — 양방향 GRU + attention pooling
# 
# 시퀀스를 양방향 GRU(h=128, 3층)로 인코딩 → **3종 풀링**(마지막 스텝 · mask 평균 · attention)을 이어붙이고 scalar를 결합해 base의 *잔차*(3D)를 예측.

# In[5]:


class AttnGRU(nn.Module):   # phaseG_full
    def __init__(self, seq_dim=13, scal_dim=22, h=128, nl=3, dr=0.15):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(seq_dim, h), nn.LayerNorm(h))   # 13→128 입력 투영
        self.gru = nn.GRU(h, h, nl, batch_first=True, bidirectional=True, dropout=dr if nl > 1 else 0)   # 양방향 GRU(128,3층)→256
        self.attn = nn.Linear(h*2, 1)   # attention 점수
        self.head = nn.Sequential(nn.Linear(h*6+scal_dim, 256), nn.GELU(), nn.Dropout(dr),   # 풀링3(768)+scalar22 → 3D 잔차
                                  nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 3))
    def forward(self, seq, scal, mask):
        x = self.proj(seq); out, _ = self.gru(x); last = out[:, -1, :]; m = mask.unsqueeze(-1)   # last=마지막 스텝
        mean = (out*m).sum(1)/m.sum(1).clamp(min=1)   # mask 평균 풀링(pad 제외)
        score = self.attn(out).squeeze(-1).masked_fill(mask < 0.5, -1e9)   # pad는 softmax에서 제외
        att = (torch.softmax(score, dim=1).unsqueeze(-1)*out).sum(1)   # attention 가중 풀링
        return self.head(torch.cat([last, mean, att, scal], -1))   # 3풀링+scalar → 잔차


# ### cell 4b. Neural ODE — 댐핑 가속도장을 RK4 적분
# 
# GRU로 시퀀스를 latent로 인코딩 → 신경망 가속도장 `a(x,v) = NN(x,v,latent) − damping·v` 를 RK4로 0.08초 적분해 변위(잔차)를 얻는다. 명시적 적분 구조가 GRU와 다른 귀납 편향을 준다.

# In[6]:


class ODEModel(nn.Module):   # phaseODE_full (train_phaseODE의 MaskedBiGRU와 동일)
    def __init__(self, seq_dim=13, scal_dim=22, h=128, nl=2, dr=0.15, latent=96, nsteps=4):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(seq_dim, h), nn.LayerNorm(h))
        self.gru = nn.GRU(h, h, nl, batch_first=True, bidirectional=True, dropout=dr if nl > 1 else 0)
        self.to_latent = nn.Sequential(nn.Linear(h*4+scal_dim, latent), nn.LayerNorm(latent), nn.GELU())   # GRU표현(256)+scalar → latent 96
        self.accel = nn.Sequential(nn.Linear(3+3+latent, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dr),   # 가속도장 NN: (pos3,vel3,latent)→acc3
                                   nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3))
        self.damping = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))   # 축별 댐핑(학습 파라미터)
        self.bias = nn.Parameter(torch.zeros(3))
        self.nsteps = nsteps; self.dt = 0.08/nsteps   # +80ms를 4스텝으로 적분
    # 미분방정식 우변(댐핑 가속도장)
    def _deriv(self, rpos, rvel, lat):
        a = self.accel(torch.cat([rpos, rvel, lat], -1))
        return rvel, -self.damping*rvel+a   # dx/dt=v, dv/dt=a_NN−damping·v
    def forward(self, seq, scal, mask):
        x = self.proj(seq); out, _ = self.gru(x); m = mask.unsqueeze(-1)
        mean = (out*m).sum(1)/m.sum(1).clamp(min=1)
        lat = self.to_latent(torch.cat([out[:, -1, :], mean, scal], -1))   # 시퀀스→latent
        rpos = torch.zeros(seq.size(0), 3, device=seq.device); rvel = torch.zeros_like(rpos)   # 잔차 위치/속도 0에서 시작
        for _ in range(self.nsteps):   # RK4 4스텝 적분
            dt = self.dt
            dp1, dv1 = self._deriv(rpos, rvel, lat)
            dp2, dv2 = self._deriv(rpos+0.5*dt*dp1, rvel+0.5*dt*dv1, lat)
            dp3, dv3 = self._deriv(rpos+0.5*dt*dp2, rvel+0.5*dt*dv2, lat)
            dp4, dv4 = self._deriv(rpos+dt*dp3, rvel+dt*dv3, lat)
            rpos = rpos+(dt/6)*(dp1+2*dp2+2*dp3+dp4)
            rvel = rvel+(dt/6)*(dv1+2*dv2+2*dv3+dv4)
        return rpos+self.bias   # 적분 변위(잔차)+bias


# ### cell 4c. HyperPhysics — 물리 gray-box (roll 기반 선회 모델)
# 
# 모기 선회의 물리(roll 뱅킹 턴)를 명시적으로 모델링한 gray-box 모델. GRU·ODE와 다른 귀납편향이라 블렌드에 함께 사용(탈상관).
# 
# **예측식**
# `pred = last + R · [ w_v · e^(−exp_v) · Rodrigues(v_ema, ω)  +  w_a · e^(−exp_a) · a_ema ]`
# 
# - `R` : forward·right·up 로컬 프레임 (3step 평균 heading 기준)
# - `v_ema, a_ema` : EMA로 평활한 로컬 속도/가속 (`temporal_net`이 EMA 계수 출력)
# - **`Rodrigues(v_ema, ω)`** : 각속도 ω로 속도벡터를 회전 = **뱅킹 턴**. ω = 과거 회전(omega_hist) + 신경망 보정(omega_delta)
# - `w_v, w_a, exp_v, exp_a` : `dynamics_net`이 속도·θ로 만든 감쇠 가중 (빠르거나 급선회면 보수적으로)
# - **게이팅** : θ(선회각)·속력이 임계값(θ_thr=1.087618, speed_thr=0.034583)을 넘을 때만 회전 발동 → 직진 구간 오차 방지
# 
# **보조 클래스/함수**
# - `SlidingWindowDataset` — 가변 윈도우 + 확장 타깃[4–10,12] + θ-가중(급선회 최대 5× 오버샘플)
# - `PriorBiasedLinear` — 마지막 층 0 초기화 + prior_bias에서 출발(안정적 물리 prior)
# - `ResBlock` · `rodrigues_rotate` · `extract_features`(24차원 물리 피처) · `_soft_hit_loss`(1.3cm soft R-Hit)

# In[7]:


# ── 학습 데이터셋: 가변 윈도우 + 확장 타깃 + θ-가중 오버샘플 ──
class SlidingWindowDataset(Dataset):
    def __init__(self, X, y, min_win=3, mode="extended", device="cpu"):
        X_tensor = torch.tensor(X, dtype=torch.float32); y_tensor = torch.tensor(y, dtype=torch.float32)
        windows = []
        for i in range(len(X)):
            targets = [4, 5, 6, 7, 8, 9, 10, 12] if mode == "extended" else [12, 10]   # 내부 지점들도 타깃(확장)
            for target_idx in targets:
                end_idx = target_idx - 2
                max_w = end_idx + 2 if mode == "extended" else (12 if target_idx == 12 else 10)
                for w in range(min_win, max_w):
                    windows.append((i, w, target_idx))
        X_list = []; y_list = []
        for i, w, target_idx in windows:
            X_orig = X_tensor[i]; end_idx = target_idx - 2
            pts = X_orig[end_idx - w + 1: end_idx + 1]
            target = y_tensor[i] if target_idx == 12 else X_orig[target_idx]
            if w < 11:
                v0 = pts[1] - pts[0]; n_pad = 11 - w
                js = torch.arange(n_pad, 0, -1, dtype=torch.float32)
                pad = pts[0:1] - js.unsqueeze(1) * v0.unsqueeze(0)
                X_padded = torch.cat([pad, pts], dim=0)
            else:
                X_padded = pts.clone()
            X_list.append(X_padded); y_list.append(target)
        self.X_all = torch.stack(X_list).to(device); self.y_all = torch.stack(y_list).to(device)
        diffs = self.X_all[:, 1:] - self.X_all[:, :-1]
        n1 = diffs[:, 1:].norm(dim=2).clamp(min=1e-8); n2 = diffs[:, :-1].norm(dim=2).clamp(min=1e-8)
        cos_t = ((diffs[:, 1:] * diffs[:, :-1]).sum(dim=2) / (n1 * n2)).clamp(-1, 1)
        theta_last = torch.acos(cos_t[:, -1])
        self.theta_weights = (1.0 + 4.0 * (theta_last / 1.0).clamp(0, 1)).cpu().numpy()   # 급선회(θ↑)일수록 최대 5× 가중

    def __len__(self): return len(self.X_all)
    def __getitem__(self, idx): return self.X_all[idx], self.y_all[idx]


# ───────────────────────── 원본 cell[5]: 피처/손실/블록 ─────────────────────────
# ── EMA로 로컬 속도(vl)·가속(al) 평활 ──
def _ema_va_local(diffs_local, alpha, beta):
    B, T, _ = diffs_local.shape
    one_m_a = 1.0 - alpha; one_m_b = 1.0 - beta
    vs = diffs_local.new_empty(B, T, 3); v = diffs_local[:, 0]; vs[:, 0] = v
    for t in range(1, T):
        v = alpha * diffs_local[:, t] + one_m_a * v; vs[:, t] = v
    vl = vs[:, -1]
    ad = vs[:, 1:] - vs[:, :-1]; a = ad[:, 0]
    for t in range(1, T - 1):
        a = beta * ad[:, t] + one_m_b * a
    return vl, a


# ── soft R-Hit 손실: 1.3cm 임계 sigmoid 근사 ──
def _soft_hit_loss(pred, target, thr=0.013012, k=408.348):
    return (1 - torch.sigmoid(-(torch.norm(pred - target, dim=1) - thr) * k)).mean()


# ── 24차원 물리 피처 + 로컬 프레임 R + diffs 추출 ──
def extract_features(X, mean_stats=None, std_stats=None, dir_net=None, heading_mode="3step"):
    device = X.device
    p_last = X[:, 10]; diffs = X[:, 1:] - X[:, :-1]
    n1 = diffs[:, 1:].norm(dim=2, keepdim=True) + 1e-8; n2 = diffs[:, :-1].norm(dim=2, keepdim=True) + 1e-8
    cos_t = ((diffs[:, 1:] * diffs[:, :-1]).sum(dim=2, keepdim=True) / (n1 * n2)).clamp(-1, 1)
    theta_seq = torch.acos(cos_t).squeeze(2)
    theta = theta_seq[:, -1:]; theta_mean = theta_seq.mean(1, keepdim=True); theta_std = theta_seq.std(1, keepdim=True)
    theta_vel = theta_seq[:, -1:] - theta_seq[:, -2:-1]
    theta_acc = theta_seq[:, -1:] - 2 * theta_seq[:, -2:-1] + theta_seq[:, -3:-2]
    theta_trend = theta_seq[:, -1:] - theta_seq[:, -3:].mean(1, keepdim=True)
    if dir_net is not None:
        speed_seq = diffs.norm(dim=2); state = torch.cat([speed_seq, theta_seq], dim=1)
        if dir_net[0].in_features == 29:
            z_speed_seq = diffs[:, :, 2].abs(); state = torch.cat([state, z_speed_seq], dim=1)
        weights = F.softmax(dir_net(state), dim=1); v_sm = (diffs * weights.unsqueeze(2)).sum(dim=1)
    else:
        v_sm = (3 * diffs[:, -1] + 2 * diffs[:, -2] + diffs[:, -3]) / 6.0 if heading_mode == "3step" else diffs[:, -1]
    fwd = v_sm / (v_sm.norm(dim=1, keepdim=True) + 1e-8)
    up_w = torch.zeros_like(fwd); up_w[:, 2] = 1.0
    up_w[fwd[:, 2].abs() > 0.99] = torch.tensor([0., 1., 0.], device=device)
    right = torch.cross(fwd, up_w, dim=1); right = right / (right.norm(dim=1, keepdim=True) + 1e-8)
    up = torch.cross(right, fwd, dim=1); up = up / (up.norm(dim=1, keepdim=True) + 1e-8)
    R = torch.stack([fwd, right, up], dim=2)   # 로컬 프레임(forward,right,up)
    v_last = diffs[:, -1]; v_prev1 = diffs[:, -2]; speed = v_last.norm(dim=1, keepdim=True)
    a_last = v_last - v_prev1; acc_mag = a_last.norm(dim=1, keepdim=True)
    v_local = torch.matmul(v_last.unsqueeze(1), R).squeeze(1)
    a_local = torch.matmul(a_last.unsqueeze(1), R).squeeze(1)
    X_local = torch.matmul(X - p_last.unsqueeze(1), R); p_std_local = X_local.std(1)
    v_local_abs = v_local.abs()
    jerk_g = diffs[:, -1] - 2 * diffs[:, -2] + diffs[:, -3]
    jerk_l = torch.matmul(jerk_g.unsqueeze(1), R).squeeze(1); jerk_mag = jerk_g.norm(dim=1, keepdim=True)
    features = torch.cat([v_local, a_local, speed, acc_mag, theta, theta_mean, theta_std, theta_trend,   # 24차원 물리 피처
                          theta_vel, theta_acc, p_std_local, v_local_abs, jerk_l, jerk_mag], dim=1)
    if mean_stats is None or std_stats is None:
        mean_stats = features.mean(0, keepdim=True); std_stats = features.std(0, keepdim=True) + 1e-8
    return (features - mean_stats) / std_stats, diffs, p_last, theta, theta_mean, theta_std, theta_seq, R, speed, mean_stats, std_stats


# ── 잔차 블록(LayerNorm+GELU) ──
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(0.15), nn.Linear(dim, dim))
        self.ln = nn.LayerNorm(dim)
    def forward(self, x): return self.ln(x + self.net(x))


# ── 마지막 층 0 + prior_bias에서 출발(물리 prior) ──
class PriorBiasedLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_bias):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer('prior_bias', prior_bias.clone().detach())
        with torch.no_grad():
            nn.init.zeros_(self.linear.weight); nn.init.zeros_(self.linear.bias)
    def forward(self, x): return self.linear(x) + self.prior_bias


# ── Rodrigues 회전: v를 각속도 w로 회전(뱅킹 턴) ──
def rodrigues_rotate(v, w):
    theta = w.norm(dim=1, keepdim=True); k = w / (theta + 1e-8)
    cos_t = torch.cos(theta); sin_t = torch.sin(theta)
    dot = (v * k).sum(dim=1, keepdim=True); cross = torch.cross(k, v, dim=1)
    return v * cos_t + cross * sin_t + k * dot * (1.0 - cos_t)


# ───────────────────────── 원본 cell[7]: HyperPhysics_xy2 ─────────────────────────
# ── HyperPhysics 본체 (예측식은 위 마크다운 cell 4c 참고) ──
class HyperPhysics_xy2(nn.Module):
    def __init__(self, input_dim=24, **kwargs):
        super().__init__()
        self.sh_thr = kwargs.pop('sh_thr', 0.013012); self.sh_k = kwargs.pop('sh_k', 408.348044)
        self.mse_w = kwargs.pop('mse_w', 129.172037); self.local_w = kwargs.pop('local_w', 0.050941)
        self.theta_thr = kwargs.pop('theta_thr', 1.087618); self.speed_thr = kwargs.pop('speed_thr', 0.034583)   # 회전 게이팅 임계값(원본 튜닝 고정)
        self.lr = 0.005400; self.wd = 0.005659
        self.register_buffer("mean_stats", torch.zeros(1, input_dim)); self.register_buffer("std_stats", torch.ones(1, input_dim))
        prior_dir = torch.tensor([-10., -10., -10., -10., -10., -10., -10., 0., 0.693, 1.098])
        self.dir_net = nn.Sequential(nn.Linear(29, 24), nn.LayerNorm(24), nn.GELU(), PriorBiasedLinear(24, 10, prior_dir))   # heading 가중(3step 방향) 학습
        prior_ema = torch.zeros(6)
        self.temporal_net = nn.Sequential(nn.Linear(9, 32), nn.LayerNorm(32), nn.GELU(), PriorBiasedLinear(32, 6, prior_ema))   # EMA 계수(alpha,beta) 출력
        prior_dyn = torch.tensor([0., 0., 0., 0., 0., 0.] + [-4.] * 24)
        self.dynamics_net = nn.Sequential(nn.Linear(input_dim, 96), nn.LayerNorm(96), nn.GELU(), ResBlock(96), PriorBiasedLinear(96, 30, prior_dyn))   # w_v,w_a,exp_v,exp_a 동역학 계수
        self.omega_w = nn.Parameter(torch.tensor([0.0, -0.5, -1.0]))
        self.omega_net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 48), nn.GELU(), nn.Linear(48, 3))   # 각속도 보정(omega_delta)
        with torch.no_grad():
            nn.init.normal_(self.omega_net[-1].weight, std=0.01); nn.init.zeros_(self.omega_net[-1].bias)
        self.diffusion_net = nn.Sequential(nn.Linear(input_dim, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, 3))

    def get_features(self, X, mean_stats=None, std_stats=None):
        return extract_features(X, mean_stats, std_stats, self.dir_net, heading_mode="3step")

    @staticmethod
    def _rotation_vector(d_prev, d_curr):
        n_prev = d_prev.norm(dim=1, keepdim=True).clamp(min=1e-8); n_curr = d_curr.norm(dim=1, keepdim=True).clamp(min=1e-8)
        d_hat_prev = d_prev / n_prev; d_hat_curr = d_curr / n_curr
        cross = torch.linalg.cross(d_hat_prev, d_hat_curr, dim=1); sin_t = cross.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cos_t = (d_hat_prev * d_hat_curr).sum(1, keepdim=True).clamp(-0.9999, 0.9999); theta = torch.atan2(sin_t, cos_t)
        speed_gate = torch.sigmoid((n_prev + n_curr) * 500 - 5)
        return cross / sin_t * theta * speed_gate

    def forward(self, features, diffs, p_last, theta, speed, R):
        B = diffs.shape[0]
        ema_raw = self.temporal_net(features[:, 8:17])
        alpha = torch.sigmoid(ema_raw[:, 0:3]) * 0.8 + 0.1; beta = torch.sigmoid(ema_raw[:, 3:6]) * 0.199 + 0.8   # EMA 계수 범위 제한
        dyn_raw = self.dynamics_net(features)
        w_v = 2.0 + dyn_raw[:, 0:3]; w_a = 1.0 + dyn_raw[:, 3:6]   # 속도/가속 기여 가중(prior 2.0/1.0)
        v_local_abs = features[:, 17:20]; v_local_abs2 = v_local_abs * v_local_abs; theta2 = theta * theta
        exp_v = (F.softplus(dyn_raw[:, 6:9]) * v_local_abs + F.softplus(dyn_raw[:, 9:12]) * v_local_abs2 +
                 F.softplus(dyn_raw[:, 12:15]) * theta + F.softplus(dyn_raw[:, 15:18]) * theta2)
        exp_a = (F.softplus(dyn_raw[:, 18:21]) * v_local_abs + F.softplus(dyn_raw[:, 21:24]) * v_local_abs2 +
                 F.softplus(dyn_raw[:, 24:27]) * theta + F.softplus(dyn_raw[:, 27:30]) * theta2)
        diffs_local = torch.matmul(diffs, R)   # 변위들을 로컬 프레임으로
        vl, al = _ema_va_local(diffs_local, alpha, beta)   # EMA 평활 속도/가속
        diff_speed = diffs_local.norm(dim=2)
        def rv_masked(ka, kb):
            rv = self._rotation_vector(diffs_local[:, ka], diffs_local[:, kb])
            valid = ((diff_speed[:, ka] > 1e-5) & (diff_speed[:, kb] > 1e-5)).float()
            return rv * valid.unsqueeze(1), valid
        ov1, vm1 = rv_masked(-2, -1); ov2, vm2 = rv_masked(-3, -2); ov3, vm3 = rv_masked(-4, -3)
        w_logits = self.omega_w.view(1, 3).expand(B, -1)
        masks = torch.stack([vm1, vm2, vm3], dim=1)
        w_attn = F.softmax(w_logits.masked_fill(masks == 0, -1e9), dim=1)
        omega_hist = (w_attn[:, 0].unsqueeze(1) * ov1 + w_attn[:, 1].unsqueeze(1) * ov2 + w_attn[:, 2].unsqueeze(1) * ov3)
        current_speed = speed.view(B, 1) if speed is not None else diff_speed[:, -1].unsqueeze(1)
        omega_speed_gate = torch.sigmoid(current_speed * 500 - 5)
        omega_delta = self.omega_net(features) * omega_speed_gate
        theta_scalar = theta.view(B, 1)
        theta_gate = torch.sigmoid((theta_scalar - self.theta_thr) * 10)   # θ가 임계 넘어야 회전 ON
        speed_gate_strong = torch.sigmoid((current_speed - self.speed_thr) * 200)   # 속력 임계 게이트
        rotation_gate = theta_gate * speed_gate_strong
        omega = (omega_hist + omega_delta) * rotation_gate   # 최종 각속도(게이팅 적용)
        v_rotated = rodrigues_rotate(vl, omega)   # 속도벡터를 ω로 회전 = 뱅킹 턴
        pred_local = (w_v * torch.exp(-exp_v)) * v_rotated + (w_a * torch.exp(-exp_a)) * al   # 로컬 잔차 = 감쇠(회전속도)+감쇠(가속)
        log_var = self.diffusion_net(features).clamp(min=-5.0, max=5.0)
        pred_global = p_last + torch.einsum('nij,nj->ni', R, pred_local)   # 글로벌: last + R·pred_local
        return pred_global, pred_local, log_var

    def compute_loss(self, pp, yr, pred_local=None, yr_local=None, log_var=None, **kwargs):
        sh = _soft_hit_loss(pp, yr, thr=self.sh_thr, k=self.sh_k)
        loss = sh + self.mse_w * F.mse_loss(pp, yr)   # soft-hit + MSE
        if pred_local is not None and yr_local is not None and log_var is not None:
            squared_error = (pred_local - yr_local) ** 2
            nll_loss = 0.5 * (torch.exp(-log_var) * squared_error + log_var)   # 로컬 잔차 NLL(불확실성 가중)
            loss = loss + self.local_w * nll_loss.mean()
        return loss


# ───────────────────────── 우리 fold OOF 래퍼 ─────────────────────────
# ── (참고용) OOF 검증 래퍼 — GOH30 재현 경로엔 미사용 ──
def load_train():
    paths = sorted((ROOT/'Data'/'train').glob('*.csv'))
    labs = pd.read_csv(ROOT/'Data'/'train_labels.csv').sort_values('id').reset_index(drop=True)[['x', 'y', 'z']].to_numpy(np.float32)
    X = np.stack([pd.read_csv(p)[['x', 'y', 'z']].to_numpy(np.float32) for p in paths])
    return X, labs


# ── 평가지표 R-Hit@1cm ──
def r_hit(p, t, thr=0.01): return float(np.mean(np.linalg.norm(p - t, axis=1) <= thr))


# ## cell 5. 손실 + 학습 함수
# 
# - **GRU/ODE** (`combined_loss`, `train_cache_seed`): base 잔차를 학습. 손실 = Huber + 가우시안-soft + **Soft R-Hit**(1cm sigmoid 근사, 가중 2.0). 캐시 전체로 55ep, Cosine LR, EMA, y-flip/노이즈 증강.
# - **HyperPhysics** (`train_h_seed`): 자체 손실(1.3cm soft-hit + MSE + 로컬 NLL). raw 궤적으로 12ep, StepLR(step4, γ0.6), EMA, θ-가중 오버샘플.
# 
# > Soft R-Hit는 "1cm 이내" 지표를 미분 가능하게 근사 → 평균(mean)이 아니라 **적중(mode)**을 직접 최적화. 0.7대 진입의 공통 기법.

# In[8]:


# ── GRU/ODE 손실: Huber + 가우시안-soft + Soft R-Hit ──
def combined_loss(pred,true):
    d=0.01; hub=F.huber_loss(pred,true,delta=d)/(0.5*d*d)   # ① Huber(1cm) 정규화
    d2=(pred-true).pow(2).sum(-1); soft=(1-torch.exp(-d2/(2*SIGMA**2))).mean()   # ② 가우시안-soft(σ=0.02)
    dd=torch.sqrt(d2+1e-12); sr=-torch.sigmoid((0.01-dd)/RHIT_TAU).mean()   # ③ Soft R-Hit: 1cm 적중 미분근사
    return HW*hub+GW*soft+RHIT_W*sr   # 0.5·Huber + 0.5·soft + 2.0·R-Hit

# ── 평가지표 R-Hit@1cm ──
def r_hit(p, t, thr=0.01):
    return float(np.mean(np.linalg.norm(p - t, axis=1) <= thr))


# In[9]:


# ── GRU/ODE 학습(캐시) → EMA 가중치 반환 ──
def train_cache_seed(seed, factory):
    '''GRU/ODE 전체데이터 학습 (캐시 사용) → EMA state_dict 반환.'''
    dev = DEVICE
    seq = torch.tensor(CACHE['seq']); scal = torch.tensor(CACHE['scal'])
    msk = torch.tensor(CACHE['mask']); tgt = torch.tensor(CACHE['tgt'])
    N = len(seq); idx_all = np.arange(N)
    torch.manual_seed(1000 + seed); np.random.seed(1000 + seed)   # 시드별 재현 고정
    model = factory().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)   # AdamW (lr 2e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=GRU_ODE_EPOCHS)   # Cosine LR (55ep)
    flip = torch.tensor(Y_FLIP, device=dev)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for ep in range(1, GRU_ODE_EPOCHS + 1):
        model.train(); np.random.shuffle(idx_all)
        for i in range(0, N, 256):
            b = idx_all[i:i + 256]
            s = seq[b].to(dev); c = scal[b].to(dev); mk = msk[b].to(dev); tg = tgt[b].to(dev)
            if torch.rand(1).item() < FLIP_PROB:   # 50% 확률 y-flip 증강(좌우대칭)
                s = s.clone(); s[:, :, flip] *= -1; tg = tg.clone(); tg[:, 1] *= -1   # seq y채널 + 타깃 y 부호반전
            s = s + torch.randn_like(s) * NOISE_STD * mk.unsqueeze(-1)   # 입력 노이즈 증강(유효 스텝만)
            opt.zero_grad(); loss = combined_loss(model(s, c, mk), tg); loss.backward()   # 잔차 예측 → combined_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()   # grad clip 0.5
        sch.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point: ema[k].mul_(EMA_DECAY).add_(v, alpha=1 - EMA_DECAY)
                else: ema[k] = v.detach().clone()
    return ema

# ── HyperPhysics 학습(raw 궤적) → EMA 가중치 반환 ──
def train_h_seed(seed, X, Y):
    '''HyperPhysics 전체데이터 학습 (raw 궤적 사용) → EMA state_dict 반환.'''
    dev = DEVICE; set_seed(1000 + seed)
    ds = SlidingWindowDataset(X, Y, min_win=3, mode="extended", device=dev)   # 가변윈도우 데이터셋
    loader = DataLoader(ds, batch_size=256, sampler=WeightedRandomSampler(ds.theta_weights, len(ds), replacement=True))   # θ-가중 오버샘플(급선회 강조)
    model = HyperPhysics_xy2().to(dev)
    with torch.no_grad():
        *_, mn, st = model.get_features(torch.tensor(X, dtype=torch.float32, device=dev))
        model.mean_stats.copy_(mn); model.std_stats.copy_(st)   # 피처 표준화 통계 주입
    opt = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=model.wd)   # AdamW (lr=model.lr 0.0054)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.6)   # StepLR (4ep마다 ×0.6)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for ep in range(1, H_EPOCHS + 1):
        model.train()
        for Xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            ft, df, pl, th, _, _, _, Rt, sp, _, _ = model.get_features(Xb, model.mean_stats, model.std_stats)   # 물리 피처/프레임 추출
            pp, pred_local, log_var = model(ft, df, pl, th, sp, Rt)
            yr_local = torch.matmul((yb - pl).unsqueeze(1), Rt).squeeze(1)   # 타깃 잔차를 로컬 프레임으로(NLL용)
            loss = model.compute_loss(pp, yb, pred_local, yr_local, log_var)   # soft-hit+MSE+NLL
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()   # grad clip 1.0
        sch.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point: ema[k].mul_(EMA_DECAY).add_(v, alpha=1 - EMA_DECAY)
                else: ema[k] = v.detach().clone()
    return ema

print('학습 함수 정의 완료')


# ## cell 6. 학습 실행 (30모델 = GRU 10 + ODE 10 + HyperPhysics 10)
# 
# cell 1의 `FROM_SCRATCH=True`면 처음부터 순차 학습(~2시간 10분, MPS) 후 `./models_goh30/`에 저장. `False`면 기존 .pt 30개를 로드(개수 assert 포함).
# 
# > 시드 0~9를 아키텍처마다 돌려 다양성 확보. 검증 환경(Apple Silicon MPS)에서는 처음부터 재학습해도 가중치가 재현됨. (다른 환경에선 부동소수점 차이로 미세하게 다를 수 있음)

# In[10]:


n = len(glob.glob(f'{MODELS_DIR}/phase*_full_*.pt'))
assert n == 30, f'{MODELS_DIR}에 30개 .pt 필요 (현재 {n}개). 학습 완료 후 실행하세요.'
print(f'기존 학습본 {n}개 로드 사용')


# ## cell 7. 예측 + 블렌드 → 제출
# 
# 테스트 10,000개를 30모델로 예측해 등가중 평균.
# - **GRU/ODE** (`predict_resid`): 잔차 예측 → base 더하고 yaw 역회전으로 글로벌 좌표 복원.
# - **HyperPhysics** (`predict_h`): 위치를 직접 출력.
# - **Y-flip TTA**: 입력을 좌우대칭으로 한 번 더 예측해 평균(대칭성 활용 → 안정화).
# - 30모델 평균 → `submission_GOH30.csv`.

# In[11]:


def _safe_norm(x, axis=-1, keepdims=False):
    return np.linalg.norm(x, axis=axis, keepdims=keepdims)


def kalman_cv_predict(x, sigma_obs=0.30e-3, sigma_proc=1.0, p0=1.0):
    n, t, _ = x.shape
    f = np.array([[1.0, DT], [0.0, 1.0]])
    f_pred = np.array([[1.0, PRED_DT], [0.0, 1.0]])
    q = sigma_proc**2 * np.array([[DT**4 / 4.0, DT**3 / 2.0], [DT**3 / 2.0, DT**2]])
    r = sigma_obs**2
    pred = np.zeros((n, 3), dtype=np.float64)

    for axis in range(3):
        z = x[:, :, axis]
        state = np.zeros((n, 2), dtype=np.float64)
        state[:, 0] = z[:, 0]
        cov = np.eye(2) * p0
        for step in range(1, t):
            state = state @ f.T
            cov = f @ cov @ f.T + q
            innovation = z[:, step] - state[:, 0]
            s = cov[0, 0] + r
            k = cov[:, 0] / s
            state = state + innovation[:, None] * k[None, :]
            cov = cov - np.outer(k, cov[0])
        pred[:, axis] = (state @ f_pred.T)[:, 0]
    return pred.astype(np.float32)


def noise_score(x):
    try:
        from scipy.signal import savgol_filter

        smooth = savgol_filter(x, window_length=5, polyorder=2, axis=1)
        return (x - smooth).std(axis=1).mean(axis=1)
    except Exception:
        t = np.arange(x.shape[1], dtype=np.float64)
        vand = np.vander(t, 3, increasing=False)
        out = np.zeros(len(x), dtype=np.float64)
        for axis in range(3):
            coef = np.linalg.lstsq(vand, x[:, :, axis].T, rcond=None)[0]
            fit = (vand @ coef).T
            out += (x[:, :, axis] - fit).std(axis=1)
        return out / 3.0


def regime_masks(x):
    d = np.diff(x, axis=1)
    v = d / DT
    a = np.diff(v, axis=1) / DT
    speed = _safe_norm(v)
    acc = _safe_norm(a)
    denom = _safe_norm(v[:, 1:]) * _safe_norm(v[:, :-1]) + 1e-12
    cos_theta = np.clip(np.sum(v[:, 1:] * v[:, :-1], axis=-1) / denom, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    noise = noise_score(x)
    return {
        "hard_turn": theta[:, -3:].max(axis=1) > 0.20,
        "recent_turn": theta[:, -2:].max(axis=1) > 0.20,
        "high_acc": acc.max(axis=1) > 15.0,
        "high_speed": speed[:, -1] > 1.0,
        "vertical_change": np.abs(a[:, -1, 2]) > np.quantile(np.abs(a[:, -1, 2]), 0.75),
        "high_noise": noise > np.quantile(noise, 0.75),
    }


def masked_blend(base_pred, alt_pred, mask, alt_weight):
    out = base_pred.copy()
    out[mask] = (1.0 - alt_weight) * base_pred[mask] + alt_weight * alt_pred[mask]
    return out


def write_submission(name, pred, ids):
    sub = pd.DataFrame({"id": ids, "x": pred[:, 0], "y": pred[:, 1], "z": pred[:, 2]})
    path = os.path.join(OUT_DIR, f"{name}.csv")
    sub.to_csv(path, index=False)
    print(f"저장: {path} {sub.shape}")


test_paths = sorted(glob.glob(f'{TEST_DIR}/*.csv'))
ids = [os.path.basename(p)[:-4] for p in test_paths]   # 파일명 = 제출 id
seqs, scals, rots, bases = [], [], [], []
for p in test_paths:
    X = pd.read_csv(p)[['x', 'y', 'z']].to_numpy()
    seq, sc22, rot, base, _ = build_features_clean(X); seq_n, sc_n = normalize(seq, sc22, STATS)   # 테스트 피처+표준화
    seqs.append(seq_n); scals.append(sc_n); rots.append(rot); bases.append(base)
seqT = torch.tensor(np.stack(seqs)); scalT = torch.tensor(np.stack(scals))
rotT = np.stack(rots); baseT = np.stack(bases); maskT = torch.ones(len(seqT), 11); flipT = torch.tensor(Y_FLIP)
X_test_raw = np.stack([load_sample(p) for p in test_paths]).astype(np.float32)
XtT = torch.tensor(X_test_raw)   # raw (H용)

def predict_resid(fp, factory):     # GRU/ODE: 잔차 + Y-flip TTA → 위치
    m = factory().to(DEVICE); m.load_state_dict(torch.load(fp, map_location=DEVICE, weights_only=False)['model_state']); m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqT), 256):
            s = seqT[i:i+256].to(DEVICE); c = scalT[i:i+256].to(DEVICE); mk = maskT[i:i+256].to(DEVICE)
            pr = m(s, c, mk).cpu().numpy(); sf = s.clone(); sf[:, :, flipT] *= -1   # 원본 예측 + y채널 반전본
            pf = m(sf, c, mk).cpu().numpy(); pf[:, 1] *= -1; out.append((pr + pf) / 2)   # 반전 예측 되돌려 평균(TTA)
    r = np.concatenate(out)
    return baseT + np.einsum('bij,bj->bi', rotT.transpose(0, 2, 1), r)   # 잔차 yaw 역회전 + base → 글로벌

def predict_h(fp):                  # HyperPhysics: 위치 직접 + Y-flip TTA
    m = HyperPhysics_xy2().to(DEVICE); m.load_state_dict(torch.load(fp, map_location=DEVICE, weights_only=False)['model_state']); m.eval()
    def fwd(Z):
        o = []
        with torch.no_grad():
            for i in range(0, len(Z), 256):
                b = Z[i:i+256].to(DEVICE)
                ft, df, pl, th, _, _, _, Rt, sp, _, _ = m.get_features(b, m.mean_stats, m.std_stats)
                pp, _, _ = m(ft, df, pl, th, sp, Rt); o.append(pp.cpu().numpy())
        return np.concatenate(o)
    pr = fwd(XtT); Xf = XtT.clone(); Xf[:, :, 1] *= -1; pf = fwd(Xf); pf[:, 1] *= -1   # H도 y-flip TTA
    return (pr + pf) / 2

# ── 30모델 예측 수집을 architecture별로 분리(GRU10 + ODE10 + H10) ──
preds_g = []
preds_ode = []
preds_h = []
for k in range(N_EACH):
    print(f"predict GRU {k}", flush=True)
    preds_g.append(predict_resid(f'{MODELS_DIR}/phaseG_full_{k}.pt', AttnGRU))
for k in range(N_EACH):
    print(f"predict ODE {k}", flush=True)
    preds_ode.append(predict_resid(f'{MODELS_DIR}/phaseODE_full_{k}.pt', ODEModel))
for k in range(N_EACH):
    print(f"predict H {k}", flush=True)
    preds_h.append(predict_h(f'{MODELS_DIR}/phaseH_full_{k}.pt'))

pred_g = np.mean(preds_g, axis=0)
pred_ode = np.mean(preds_ode, axis=0)
pred_h = np.mean(preds_h, axis=0)
pred_equal = (pred_g + pred_ode + pred_h) / 3.0

np.save(os.path.join(OUT_DIR, "pred_gru.npy"), pred_g)
np.save(os.path.join(OUT_DIR, "pred_ode.npy"), pred_ode)
np.save(os.path.join(OUT_DIR, "pred_h.npy"), pred_h)
np.save(os.path.join(OUT_DIR, "pred_equal.npy"), pred_equal)
pd.DataFrame({"id": ids}).to_csv(os.path.join(OUT_DIR, "ids.csv"), index=False)

write_submission("case00_equal_goh30", pred_equal, ids)

weight_cases = {
    "case01_ode_heavy_g25_o50_h25": (0.25, 0.50, 0.25),
    "case02_ode_heavy_g20_o60_h20": (0.20, 0.60, 0.20),
    "case03_ode_heavy_g15_o65_h20": (0.15, 0.65, 0.20),
    "case04_ode_heavy_g15_o70_h15": (0.15, 0.70, 0.15),
    "case05_ode_heavy_g10_o75_h15": (0.10, 0.75, 0.15),
    "case06_ode_heavy_g15_o85_h00": (0.15, 0.85, 0.00),
}
for name, (wg, wo, wh) in weight_cases.items():
    pred = wg * pred_g + wo * pred_ode + wh * pred_h
    write_submission(name, pred, ids)

print("wrote component predictions and ODE-heavy submissions")


# ## cell 8. 요약
# 
# raw 데이터 → 전처리 → 30모델 학습 → 블렌드 → `submission_GOH30.csv` 재현 notebook
# 
# | 단계 | 출력 |
# |---|---|
# | cell 2–3 | 피처 + norm_stats + 학습 캐시(5만) |
# | cell 4–5 | 3개 아키텍처 + 손실/학습 함수 |
# | cell 6 | 30개 가중치 → `./models_goh30/` |
# | cell 7 | `submission_GOH30.csv` (R-Hit@1cm 0.7035) |

# 
