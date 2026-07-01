from __future__ import annotations

import argparse
import json
import math
import os
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


DT = 0.04
PRED_DT = 0.08
R_HIT = 0.01
SIGMA = 0.02
RHIT_TAU = 0.0015
RHIT_W = 2.0
HW = 0.5
GW = 0.5
SPEED_BINS = [0.0, 0.3, 0.6, 0.9, 1.2, np.inf]
Y_FLIP = [1, 4, 7, 10]
INTERIOR_E = [5, 6, 7, 8]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_open_dir(root: Path) -> Path:
    open_dir = root / "open"
    if (open_dir / "train").exists() and (open_dir / "test").exists():
        return open_dir
    zip_path = root / "data" / "open.zip"
    if not zip_path.exists():
        raise FileNotFoundError("Expected open/ or data/open.zip")
    open_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(open_dir)
    return open_dir


def load_xyz(path: Path) -> np.ndarray:
    return pd.read_csv(path).sort_values("timestep_ms")[["x", "y", "z"]].to_numpy(np.float32)


def load_data(root: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, list[str]]:
    data_dir = ensure_open_dir(root)
    train_paths = sorted((data_dir / "train").glob("TRAIN_*.csv"))
    test_paths = sorted((data_dir / "test").glob("TEST_*.csv"))
    labels = pd.read_csv(data_dir / "train_labels.csv").set_index("id")
    ids = [p.stem for p in train_paths]
    test_ids = [p.stem for p in test_paths]
    x = np.stack([load_xyz(p) for p in train_paths])
    x_test = np.stack([load_xyz(p) for p in test_paths])
    y = labels.loc[ids][["x", "y", "z"]].to_numpy(np.float32)
    return x, y, ids, x_test, test_ids


def safe_norm(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    return np.linalg.norm(x, axis=axis, keepdims=keepdims)


def hit_rate(pred: np.ndarray, y: np.ndarray, radius: float = R_HIT) -> float:
    return float((safe_norm(pred - y) <= radius).mean())


def cv_predict(x: np.ndarray) -> np.ndarray:
    return x[:, -1] + (PRED_DT / DT) * (x[:, -1] - x[:, -2])


def kalman_cv_predict(
    x: np.ndarray,
    sigma_obs: float = 0.30e-3,
    sigma_proc: float = 1.0,
    p0: float = 1.0,
) -> np.ndarray:
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


def yaw_rotation_matrix(v: np.ndarray) -> np.ndarray:
    ang = math.atan2(float(v[1]), float(v[0]))
    c = math.cos(-ang)
    s = math.sin(-ang)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def extract_seq_features(w: np.ndarray, vel: np.ndarray, rot: np.ndarray) -> np.ndarray:
    rel = (w - w[-1]) @ rot.T
    v = vel @ rot.T
    acc = np.gradient(vel, DT, axis=0) @ rot.T
    jerk = np.gradient(np.gradient(vel, DT, axis=0), DT, axis=0) @ rot.T
    v1 = v[1:]
    v0 = v[:-1]
    denom = safe_norm(v1, keepdims=True) * safe_norm(v0, keepdims=True) + 1e-12
    omega = np.zeros((len(w), 1), dtype=np.float32)
    omega[1:, 0] = np.arccos(np.clip((v1 * v0).sum(1, keepdims=True) / denom, -1.0, 1.0))[:, 0]
    return np.concatenate([rel, v, acc, jerk, omega], axis=1).astype(np.float32)


def extract_scalar_features(w: np.ndarray, vel: np.ndarray) -> np.ndarray:
    speeds = safe_norm(vel)
    accel = np.gradient(vel, DT, axis=0)
    acc_mag = safe_norm(accel)
    last_speed = float(speeds[-1])
    last_accel = float(acc_mag[-1])
    mean_accel = float(acc_mag.mean())
    steps = safe_norm(np.diff(w, axis=0))
    path = float(steps.sum())
    net = float(safe_norm(w[-1] - w[0]))
    linearity = net / (path + 1e-8)
    clip_flag = float(last_speed > 1.33)
    v_norm = vel / (safe_norm(vel, keepdims=True) + 1e-12)
    cos_sim_all = (v_norm[:-1] * v_norm[1:]).sum(axis=1)
    dir_consistency = float(cos_sim_all.mean())
    delta_speed = float(speeds[-1] - speeds[-2])
    last_dir_change = float(cos_sim_all[-1])
    last_vel_norm = v_norm[-1]
    last_accel_vec = accel[-1]
    tangential = np.dot(last_accel_vec, last_vel_norm) * last_vel_norm
    last_normal_accel = float(safe_norm(last_accel_vec - tangential))
    speed_bin = np.zeros(5, dtype=np.float32)
    for k in range(5):
        if SPEED_BINS[k] <= last_speed < SPEED_BINS[k + 1]:
            speed_bin[k] = 1.0
            break
    base = np.array(
        [
            last_speed,
            last_accel,
            mean_accel,
            linearity,
            clip_flag,
            dir_consistency,
            delta_speed,
            last_dir_change,
            last_normal_accel,
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, speed_bin])


def window_features(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w64 = w.astype(np.float64)
    vel = np.gradient(w64, DT, axis=0)
    rot = yaw_rotation_matrix(vel[-1])
    seq = extract_seq_features(w64, vel, rot)
    b14 = extract_scalar_features(w64, vel)
    sp = safe_norm(vel)
    steps = safe_norm(np.diff(w64, axis=0))
    path = float(steps.sum())
    net = float(safe_norm(w64[-1] - w64[0]))
    straight = net / (path + 1e-8)
    t = np.arange(float(len(w64)))
    noise = float(np.mean([(w64[:, d] - np.polyval(np.polyfit(t, w64[:, d], 2), t)).std() for d in range(3)]))
    k = min(4, len(w64))
    acc_trend = float(np.polyfit(np.arange(float(k)), sp[-k:], 1)[0])
    scal = np.concatenate(
        [
            b14,
            [
                float(sp.max()),
                float(sp.std()),
                float(sp[-3:].mean()),
                float(sp[-5:].mean()),
                path,
                straight,
                noise,
                acc_trend,
            ],
        ]
    ).astype(np.float32)
    base = (w64[-1] + 2.0 * (w64[-1] - w64[-2])).astype(np.float32)
    return seq, scal, rot.astype(np.float32), base


def make_stats(x: np.ndarray, idx: np.ndarray) -> dict[str, np.ndarray]:
    seqs = []
    scals = []
    for i in idx:
        seq, scal, _, _ = window_features(x[i])
        seqs.append(seq)
        scals.append(scal)
    seq_arr = np.stack(seqs)
    scal_arr = np.stack(scals)
    return {
        "seq_mean": seq_arr.reshape(-1, 13).mean(0),
        "seq_std": seq_arr.reshape(-1, 13).std(0) + 1e-8,
        "scalar_mean": scal_arr.mean(0),
        "scalar_std": scal_arr.std(0) + 1e-8,
    }


def build_cache(
    x: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    stats: dict[str, np.ndarray],
    use_interiors: bool,
    sample_weights: np.ndarray | None = None,
    interior_weight: float = 1.0,
    extra_features: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    seqs = []
    scals = []
    masks = []
    tgts = []
    weights = []
    for sample_idx in idx:
        sample = x[sample_idx]
        examples = [(10, y[sample_idx])]
        if use_interiors:
            examples.extend((e, sample[e + 2]) for e in INTERIOR_E)
        for e, target in examples:
            w = sample[: e + 1]
            seq, scal, rot, base = window_features(w)
            seq_n = ((seq - stats["seq_mean"]) / stats["seq_std"]).astype(np.float32)
            scal_n = ((scal - stats["scalar_mean"]) / stats["scalar_std"]).astype(np.float32)
            if extra_features is not None:
                scal_n = np.concatenate([scal_n, extra_features[sample_idx].astype(np.float32)])
            pad = 11 - len(w)
            seq11 = np.zeros((11, 13), dtype=np.float32)
            seq11[pad:] = seq_n
            mask = np.zeros(11, dtype=np.float32)
            mask[pad:] = 1.0
            target_rot = (rot @ (target.astype(np.float32) - base)).astype(np.float32)
            seqs.append(seq11)
            scals.append(scal_n)
            masks.append(mask)
            tgts.append(target_rot)
            base_weight = 1.0 if sample_weights is None else float(sample_weights[sample_idx])
            weights.append(base_weight if e == 10 else base_weight * interior_weight)
    return {
        "seq": np.stack(seqs),
        "scal": np.stack(scals),
        "mask": np.stack(masks),
        "tgt": np.stack(tgts),
        "weight": np.asarray(weights, dtype=np.float32),
    }


def build_eval_features(
    x: np.ndarray,
    idx: np.ndarray,
    stats: dict[str, np.ndarray],
    extra_features: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    seqs = []
    scals = []
    rots = []
    bases = []
    for sample_idx in idx:
        seq, scal, rot, base = window_features(x[sample_idx])
        seqs.append(((seq - stats["seq_mean"]) / stats["seq_std"]).astype(np.float32))
        scal_n = ((scal - stats["scalar_mean"]) / stats["scalar_std"]).astype(np.float32)
        if extra_features is not None:
            scal_n = np.concatenate([scal_n, extra_features[sample_idx].astype(np.float32)])
        scals.append(scal_n)
        rots.append(rot)
        bases.append(base)
    return {
        "seq": np.stack(seqs),
        "scal": np.stack(scals),
        "mask": np.ones((len(idx), 11), dtype=np.float32),
        "rot": np.stack(rots),
        "base": np.stack(bases),
    }


class AttnGRU(nn.Module):
    def __init__(self, seq_dim: int = 13, scal_dim: int = 22, h: int = 128, nl: int = 3, dr: float = 0.15):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(seq_dim, h), nn.LayerNorm(h))
        self.gru = nn.GRU(h, h, nl, batch_first=True, bidirectional=True, dropout=dr if nl > 1 else 0)
        self.attn = nn.Linear(h * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(h * 6 + scal_dim, 256),
            nn.GELU(),
            nn.Dropout(dr),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, seq: torch.Tensor, scal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(seq)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        m = mask.unsqueeze(-1)
        mean = (out * m).sum(1) / m.sum(1).clamp(min=1)
        score = self.attn(out).squeeze(-1).masked_fill(mask < 0.5, -1e9)
        att = (torch.softmax(score, dim=1).unsqueeze(-1) * out).sum(1)
        return self.head(torch.cat([last, mean, att, scal], -1))


class ODEModel(nn.Module):
    def __init__(self, seq_dim: int = 13, scal_dim: int = 22, h: int = 128, nl: int = 2, dr: float = 0.15, latent: int = 96, nsteps: int = 4):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(seq_dim, h), nn.LayerNorm(h))
        self.gru = nn.GRU(h, h, nl, batch_first=True, bidirectional=True, dropout=dr if nl > 1 else 0)
        self.to_latent = nn.Sequential(nn.Linear(h * 4 + scal_dim, latent), nn.LayerNorm(latent), nn.GELU())
        self.accel = nn.Sequential(
            nn.Linear(3 + 3 + latent, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dr),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )
        self.damping = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        self.bias = nn.Parameter(torch.zeros(3))
        self.nsteps = nsteps
        self.dt = 0.08 / nsteps
        nn.init.zeros_(self.accel[-1].weight)
        nn.init.zeros_(self.accel[-1].bias)

    def _deriv(self, rpos: torch.Tensor, rvel: torch.Tensor, lat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.accel(torch.cat([rpos, rvel, lat], -1))
        return rvel, -self.damping * rvel + a

    def forward(self, seq: torch.Tensor, scal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(seq)
        out, _ = self.gru(x)
        m = mask.unsqueeze(-1)
        mean = (out * m).sum(1) / m.sum(1).clamp(min=1)
        lat = self.to_latent(torch.cat([out[:, -1, :], mean, scal], -1))
        rpos = torch.zeros(seq.size(0), 3, device=seq.device)
        rvel = torch.zeros_like(rpos)
        for _ in range(self.nsteps):
            dt = self.dt
            dp1, dv1 = self._deriv(rpos, rvel, lat)
            dp2, dv2 = self._deriv(rpos + 0.5 * dt * dp1, rvel + 0.5 * dt * dv1, lat)
            dp3, dv3 = self._deriv(rpos + 0.5 * dt * dp2, rvel + 0.5 * dt * dv2, lat)
            dp4, dv4 = self._deriv(rpos + dt * dp3, rvel + dt * dv3, lat)
            rpos = rpos + (dt / 6) * (dp1 + 2 * dp2 + 2 * dp3 + dp4)
            rvel = rvel + (dt / 6) * (dv1 + 2 * dv2 + 2 * dv3 + dv4)
        return rpos + self.bias


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.12):
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if y.size(-1) != x.size(-1):
            y = y[..., : x.size(-1)]
        return self.norm((x + y).transpose(1, 2)).transpose(1, 2)


class TCNModel(nn.Module):
    def __init__(self, seq_dim: int = 13, scal_dim: int = 22, h: int = 128, dr: float = 0.12):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(seq_dim, h), nn.LayerNorm(h), nn.GELU())
        self.blocks = nn.ModuleList([TCNBlock(h, d, dr) for d in [1, 2, 4, 8]])
        self.attn = nn.Linear(h, 1)
        self.head = nn.Sequential(
            nn.Linear(h * 3 + scal_dim, 192),
            nn.GELU(),
            nn.Dropout(dr),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, seq: torch.Tensor, scal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(seq).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        out = x.transpose(1, 2)
        m = mask.unsqueeze(-1)
        mean = (out * m).sum(1) / m.sum(1).clamp(min=1)
        score = self.attn(out).squeeze(-1).masked_fill(mask < 0.5, -1e9)
        att = (torch.softmax(score, dim=1).unsqueeze(-1) * out).sum(1)
        last = out[:, -1, :]
        return self.head(torch.cat([last, mean, att, scal], -1))


class TransformerLite(nn.Module):
    def __init__(
        self,
        seq_dim: int = 13,
        scal_dim: int = 22,
        h: int = 128,
        nhead: int = 4,
        nl: int = 3,
        dr: float = 0.12,
    ):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(seq_dim, h), nn.LayerNorm(h), nn.GELU())
        self.pos = nn.Parameter(torch.zeros(1, 11, h))
        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dr,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nl)
        self.attn = nn.Linear(h, 1)
        self.head = nn.Sequential(
            nn.Linear(h * 3 + scal_dim, 192),
            nn.GELU(),
            nn.Dropout(dr),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, seq: torch.Tensor, scal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(seq) + self.pos[:, : seq.size(1)]
        key_padding_mask = mask < 0.5
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        m = mask.unsqueeze(-1)
        mean = (out * m).sum(1) / m.sum(1).clamp(min=1)
        score = self.attn(out).squeeze(-1).masked_fill(mask < 0.5, -1e9)
        att = (torch.softmax(score, dim=1).unsqueeze(-1) * out).sum(1)
        last = out[:, -1, :]
        return self.head(torch.cat([last, mean, att, scal], -1))


class SlidingWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, min_win: int = 3, mode: str = "extended", device: torch.device | str = "cpu"):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        windows = []
        for i in range(len(x)):
            targets = [4, 5, 6, 7, 8, 9, 10, 12] if mode == "extended" else [12, 10]
            for target_idx in targets:
                end_idx = target_idx - 2
                max_w = end_idx + 2 if mode == "extended" else (12 if target_idx == 12 else 10)
                for w in range(min_win, max_w):
                    windows.append((i, w, target_idx))
        x_list = []
        y_list = []
        for i, w, target_idx in windows:
            x_orig = x_tensor[i]
            end_idx = target_idx - 2
            pts = x_orig[end_idx - w + 1 : end_idx + 1]
            target = y_tensor[i] if target_idx == 12 else x_orig[target_idx]
            if w < 11:
                v0 = pts[1] - pts[0]
                n_pad = 11 - w
                js = torch.arange(n_pad, 0, -1, dtype=torch.float32)
                pad = pts[0:1] - js.unsqueeze(1) * v0.unsqueeze(0)
                x_padded = torch.cat([pad, pts], dim=0)
            else:
                x_padded = pts.clone()
            x_list.append(x_padded)
            y_list.append(target)
        self.x_all = torch.stack(x_list).to(device)
        self.y_all = torch.stack(y_list).to(device)
        diffs = self.x_all[:, 1:] - self.x_all[:, :-1]
        n1 = diffs[:, 1:].norm(dim=2).clamp(min=1e-8)
        n2 = diffs[:, :-1].norm(dim=2).clamp(min=1e-8)
        cos_t = ((diffs[:, 1:] * diffs[:, :-1]).sum(dim=2) / (n1 * n2)).clamp(-1, 1)
        theta_last = torch.acos(cos_t[:, -1])
        self.theta_weights = (1.0 + 4.0 * (theta_last / 1.0).clamp(0, 1)).cpu().numpy()

    def __len__(self) -> int:
        return len(self.x_all)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x_all[idx], self.y_all[idx]


def _ema_va_local(diffs_local: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, t, _ = diffs_local.shape
    one_m_a = 1.0 - alpha
    one_m_b = 1.0 - beta
    vs = diffs_local.new_empty(diffs_local.shape)
    v = diffs_local[:, 0]
    vs[:, 0] = v
    for step in range(1, t):
        v = alpha * diffs_local[:, step] + one_m_a * v
        vs[:, step] = v
    vl = vs[:, -1]
    ad = vs[:, 1:] - vs[:, :-1]
    a = ad[:, 0]
    for step in range(1, t - 1):
        a = beta * ad[:, step] + one_m_b * a
    return vl, a


def _soft_hit_loss(pred: torch.Tensor, target: torch.Tensor, thr: float = 0.013012, k: float = 408.348) -> torch.Tensor:
    return (1 - torch.sigmoid(-(torch.norm(pred - target, dim=1) - thr) * k)).mean()


def extract_h_features(
    x: torch.Tensor,
    mean_stats: torch.Tensor | None = None,
    std_stats: torch.Tensor | None = None,
    dir_net: nn.Module | None = None,
    heading_mode: str = "3step",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = x.device
    p_last = x[:, 10]
    diffs = x[:, 1:] - x[:, :-1]
    n1 = diffs[:, 1:].norm(dim=2, keepdim=True) + 1e-8
    n2 = diffs[:, :-1].norm(dim=2, keepdim=True) + 1e-8
    cos_t = ((diffs[:, 1:] * diffs[:, :-1]).sum(dim=2, keepdim=True) / (n1 * n2)).clamp(-1, 1)
    theta_seq = torch.acos(cos_t).squeeze(2)
    theta = theta_seq[:, -1:]
    theta_mean = theta_seq.mean(1, keepdim=True)
    theta_std = theta_seq.std(1, keepdim=True)
    theta_vel = theta_seq[:, -1:] - theta_seq[:, -2:-1]
    theta_acc = theta_seq[:, -1:] - 2 * theta_seq[:, -2:-1] + theta_seq[:, -3:-2]
    theta_trend = theta_seq[:, -1:] - theta_seq[:, -3:].mean(1, keepdim=True)
    if dir_net is not None:
        speed_seq = diffs.norm(dim=2)
        state = torch.cat([speed_seq, theta_seq], dim=1)
        if dir_net[0].in_features == 29:
            z_speed_seq = diffs[:, :, 2].abs()
            state = torch.cat([state, z_speed_seq], dim=1)
        weights = F.softmax(dir_net(state), dim=1)
        v_sm = (diffs * weights.unsqueeze(2)).sum(dim=1)
    else:
        v_sm = (3 * diffs[:, -1] + 2 * diffs[:, -2] + diffs[:, -3]) / 6.0 if heading_mode == "3step" else diffs[:, -1]
    fwd = v_sm / (v_sm.norm(dim=1, keepdim=True) + 1e-8)
    up_w = torch.zeros_like(fwd)
    up_w[:, 2] = 1.0
    up_w[fwd[:, 2].abs() > 0.99] = torch.tensor([0.0, 1.0, 0.0], device=device)
    right = torch.cross(fwd, up_w, dim=1)
    right = right / (right.norm(dim=1, keepdim=True) + 1e-8)
    up = torch.cross(right, fwd, dim=1)
    up = up / (up.norm(dim=1, keepdim=True) + 1e-8)
    rot = torch.stack([fwd, right, up], dim=2)
    v_last = diffs[:, -1]
    v_prev1 = diffs[:, -2]
    speed = v_last.norm(dim=1, keepdim=True)
    a_last = v_last - v_prev1
    acc_mag = a_last.norm(dim=1, keepdim=True)
    v_local = torch.matmul(v_last.unsqueeze(1), rot).squeeze(1)
    a_local = torch.matmul(a_last.unsqueeze(1), rot).squeeze(1)
    x_local = torch.matmul(x - p_last.unsqueeze(1), rot)
    p_std_local = x_local.std(1)
    v_local_abs = v_local.abs()
    jerk_g = diffs[:, -1] - 2 * diffs[:, -2] + diffs[:, -3]
    jerk_l = torch.matmul(jerk_g.unsqueeze(1), rot).squeeze(1)
    jerk_mag = jerk_g.norm(dim=1, keepdim=True)
    features = torch.cat(
        [
            v_local,
            a_local,
            speed,
            acc_mag,
            theta,
            theta_mean,
            theta_std,
            theta_trend,
            theta_vel,
            theta_acc,
            p_std_local,
            v_local_abs,
            jerk_l,
            jerk_mag,
        ],
        dim=1,
    )
    if mean_stats is None or std_stats is None:
        mean_stats = features.mean(0, keepdim=True)
        std_stats = features.std(0, keepdim=True) + 1e-8
    return (features - mean_stats) / std_stats, diffs, p_last, theta, theta_mean, theta_std, theta_seq, rot, speed, mean_stats, std_stats


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(0.15), nn.Linear(dim, dim))
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x + self.net(x))


class PriorBiasedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_bias: torch.Tensor):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer("prior_bias", prior_bias.clone().detach())
        with torch.no_grad():
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.prior_bias


def rodrigues_rotate(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    theta = w.norm(dim=1, keepdim=True)
    k = w / (theta + 1e-8)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    dot = (v * k).sum(dim=1, keepdim=True)
    cross = torch.cross(k, v, dim=1)
    return v * cos_t + cross * sin_t + k * dot * (1.0 - cos_t)


class HyperPhysicsXY2(nn.Module):
    def __init__(self, input_dim: int = 24):
        super().__init__()
        self.sh_thr = 0.013012
        self.sh_k = 408.348044
        self.mse_w = 129.172037
        self.local_w = 0.050941
        self.theta_thr = 1.087618
        self.speed_thr = 0.034583
        self.lr = 0.005400
        self.wd = 0.005659
        self.register_buffer("mean_stats", torch.zeros(1, input_dim))
        self.register_buffer("std_stats", torch.ones(1, input_dim))
        prior_dir = torch.tensor([-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 0.0, 0.693, 1.098])
        self.dir_net = nn.Sequential(nn.Linear(29, 24), nn.LayerNorm(24), nn.GELU(), PriorBiasedLinear(24, 10, prior_dir))
        prior_ema = torch.zeros(6)
        self.temporal_net = nn.Sequential(nn.Linear(9, 32), nn.LayerNorm(32), nn.GELU(), PriorBiasedLinear(32, 6, prior_ema))
        prior_dyn = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [-4.0] * 24)
        self.dynamics_net = nn.Sequential(nn.Linear(input_dim, 96), nn.LayerNorm(96), nn.GELU(), ResBlock(96), PriorBiasedLinear(96, 30, prior_dyn))
        self.omega_w = nn.Parameter(torch.tensor([0.0, -0.5, -1.0]))
        self.omega_net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 48), nn.GELU(), nn.Linear(48, 3))
        with torch.no_grad():
            nn.init.normal_(self.omega_net[-1].weight, std=0.01)
            nn.init.zeros_(self.omega_net[-1].bias)
        self.diffusion_net = nn.Sequential(nn.Linear(input_dim, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, 3))

    def get_features(self, x: torch.Tensor, mean_stats: torch.Tensor | None = None, std_stats: torch.Tensor | None = None):
        return extract_h_features(x, mean_stats, std_stats, self.dir_net, heading_mode="3step")

    @staticmethod
    def _rotation_vector(d_prev: torch.Tensor, d_curr: torch.Tensor) -> torch.Tensor:
        n_prev = d_prev.norm(dim=1, keepdim=True).clamp(min=1e-8)
        n_curr = d_curr.norm(dim=1, keepdim=True).clamp(min=1e-8)
        d_hat_prev = d_prev / n_prev
        d_hat_curr = d_curr / n_curr
        cross = torch.linalg.cross(d_hat_prev, d_hat_curr, dim=1)
        sin_t = cross.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cos_t = (d_hat_prev * d_hat_curr).sum(1, keepdim=True).clamp(-0.9999, 0.9999)
        theta = torch.atan2(sin_t, cos_t)
        speed_gate = torch.sigmoid((n_prev + n_curr) * 500 - 5)
        return cross / sin_t * theta * speed_gate

    def forward(self, features: torch.Tensor, diffs: torch.Tensor, p_last: torch.Tensor, theta: torch.Tensor, speed: torch.Tensor, rot: torch.Tensor):
        batch = diffs.shape[0]
        ema_raw = self.temporal_net(features[:, 8:17])
        alpha = torch.sigmoid(ema_raw[:, 0:3]) * 0.8 + 0.1
        beta = torch.sigmoid(ema_raw[:, 3:6]) * 0.199 + 0.8
        dyn_raw = self.dynamics_net(features)
        w_v = 2.0 + dyn_raw[:, 0:3]
        w_a = 1.0 + dyn_raw[:, 3:6]
        v_local_abs = features[:, 17:20]
        v_local_abs2 = v_local_abs * v_local_abs
        theta2 = theta * theta
        exp_v = (
            F.softplus(dyn_raw[:, 6:9]) * v_local_abs
            + F.softplus(dyn_raw[:, 9:12]) * v_local_abs2
            + F.softplus(dyn_raw[:, 12:15]) * theta
            + F.softplus(dyn_raw[:, 15:18]) * theta2
        )
        exp_a = (
            F.softplus(dyn_raw[:, 18:21]) * v_local_abs
            + F.softplus(dyn_raw[:, 21:24]) * v_local_abs2
            + F.softplus(dyn_raw[:, 24:27]) * theta
            + F.softplus(dyn_raw[:, 27:30]) * theta2
        )
        diffs_local = torch.matmul(diffs, rot)
        vl, al = _ema_va_local(diffs_local, alpha, beta)
        diff_speed = diffs_local.norm(dim=2)

        def rv_masked(ka: int, kb: int) -> tuple[torch.Tensor, torch.Tensor]:
            rv = self._rotation_vector(diffs_local[:, ka], diffs_local[:, kb])
            valid = ((diff_speed[:, ka] > 1e-5) & (diff_speed[:, kb] > 1e-5)).float()
            return rv * valid.unsqueeze(1), valid

        ov1, vm1 = rv_masked(-2, -1)
        ov2, vm2 = rv_masked(-3, -2)
        ov3, vm3 = rv_masked(-4, -3)
        w_logits = self.omega_w.view(1, 3).expand(batch, -1)
        masks = torch.stack([vm1, vm2, vm3], dim=1)
        w_attn = F.softmax(w_logits.masked_fill(masks == 0, -1e9), dim=1)
        omega_hist = w_attn[:, 0].unsqueeze(1) * ov1 + w_attn[:, 1].unsqueeze(1) * ov2 + w_attn[:, 2].unsqueeze(1) * ov3
        current_speed = speed.view(batch, 1)
        omega_speed_gate = torch.sigmoid(current_speed * 500 - 5)
        omega_delta = self.omega_net(features) * omega_speed_gate
        theta_gate = torch.sigmoid((theta.view(batch, 1) - self.theta_thr) * 10)
        speed_gate_strong = torch.sigmoid((current_speed - self.speed_thr) * 200)
        omega = (omega_hist + omega_delta) * theta_gate * speed_gate_strong
        v_rotated = rodrigues_rotate(vl, omega)
        pred_local = (w_v * torch.exp(-exp_v)) * v_rotated + (w_a * torch.exp(-exp_a)) * al
        log_var = self.diffusion_net(features).clamp(min=-5.0, max=5.0)
        pred_global = p_last + torch.einsum("nij,nj->ni", rot, pred_local)
        return pred_global, pred_local, log_var

    def compute_loss(self, pp: torch.Tensor, yr: torch.Tensor, pred_local: torch.Tensor, yr_local: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        sh = _soft_hit_loss(pp, yr, thr=self.sh_thr, k=self.sh_k)
        loss = sh + self.mse_w * F.mse_loss(pp, yr)
        squared_error = (pred_local - yr_local) ** 2
        nll_loss = 0.5 * (torch.exp(-log_var) * squared_error + log_var)
        return loss + self.local_w * nll_loss.mean()


def combined_loss(pred: torch.Tensor, true: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    d = 0.01
    hub = F.huber_loss(pred, true, delta=d, reduction="none").mean(dim=1) / (0.5 * d * d)
    d2 = (pred - true).pow(2).sum(-1)
    soft = 1 - torch.exp(-d2 / (2 * SIGMA**2))
    dd = torch.sqrt(d2 + 1e-12)
    sr = -torch.sigmoid((0.01 - dd) / RHIT_TAU)
    row = HW * hub + GW * soft + RHIT_W * sr
    if weight is None:
        return row.mean()
    weight = weight / weight.mean().clamp(min=1e-8)
    return (row * weight).mean()


def training_sample_weights(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones(len(x), dtype=np.float32)
    masks = regime_masks(x)
    weights = np.ones(len(x), dtype=np.float32)
    if mode == "regime-soft":
        weights[masks["hard_turn"]] *= 1.20
        weights[masks["recent_turn"]] *= 1.10
        weights[masks["high_acc"]] *= 1.15
        weights[masks["vertical_change"]] *= 1.10
        weights[masks["high_noise"]] *= 0.85
    elif mode == "regime-strong":
        weights[masks["hard_turn"]] *= 1.45
        weights[masks["recent_turn"]] *= 1.20
        weights[masks["high_acc"]] *= 1.30
        weights[masks["vertical_change"]] *= 1.20
        weights[masks["high_noise"]] *= 0.70
    elif mode == "hard-final":
        hard = masks["hard_turn"] | masks["recent_turn"] | masks["high_acc"] | masks["vertical_change"]
        weights[hard] *= 1.35
        weights[masks["high_noise"] & ~hard] *= 0.80
    else:
        raise ValueError(f"unknown weight mode: {mode}")
    weights = np.clip(weights, 0.50, 2.00)
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)


def trajectory_cluster_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    d = np.diff(x, axis=1)
    v = d / DT
    a = np.diff(v, axis=1) / DT
    j = np.diff(a, axis=1) / DT
    speed = safe_norm(v)
    acc = safe_norm(a)
    jerk = safe_norm(j)
    denom = safe_norm(v[:, 1:]) * safe_norm(v[:, :-1]) + 1e-12
    theta = np.arccos(np.clip(np.sum(v[:, 1:] * v[:, :-1], axis=-1) / denom, -1.0, 1.0))
    net = safe_norm(x[:, -1] - x[:, 0])
    path = safe_norm(d).sum(axis=1)
    straight = net / (path + 1e-12)
    noise = noise_score(x)
    parts = [
        speed[:, -1],
        speed.mean(axis=1),
        speed.max(axis=1),
        speed.std(axis=1),
        acc[:, -1],
        acc.max(axis=1),
        acc.mean(axis=1),
        jerk.max(axis=1),
        theta[:, -1],
        theta[:, -3:].mean(axis=1),
        theta[:, -3:].max(axis=1),
        theta.mean(axis=1),
        theta.std(axis=1),
        straight,
        path,
        np.abs(a[:, -1, 2]),
        v[:, -1, 2],
        noise,
    ]
    names = [
        "cluster_last_speed",
        "cluster_mean_speed",
        "cluster_max_speed",
        "cluster_speed_std",
        "cluster_last_acc",
        "cluster_max_acc",
        "cluster_mean_acc",
        "cluster_max_jerk",
        "cluster_theta_last",
        "cluster_theta_recent_mean",
        "cluster_theta_recent_max",
        "cluster_theta_mean",
        "cluster_theta_std",
        "cluster_straightness",
        "cluster_path",
        "cluster_abs_z_acc_last",
        "cluster_z_speed_last",
        "cluster_noise",
    ]
    return np.column_stack(parts).astype(np.float32), names


def make_cluster_context(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    n_clusters: int,
    feature_mode: str,
    weight_mode: str,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray, pd.DataFrame]:
    if feature_mode == "none" and weight_mode == "none":
        return None, np.ones(len(x), dtype=np.float32), pd.DataFrame()

    raw_feats, feat_names = trajectory_cluster_features(x)
    scaler = StandardScaler().fit(raw_feats[train_idx])
    z = scaler.transform(raw_feats)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    train_clusters = km.fit_predict(z[train_idx])
    clusters = km.predict(z)

    extras = []
    if feature_mode in {"onehot", "onehot-scalar"}:
        onehot = np.eye(n_clusters, dtype=np.float32)[clusters] * 0.25
        extras.append(onehot)
    if feature_mode in {"scalar", "onehot-scalar"}:
        # Keep injected continuous features weak and standardized by the train fold.
        extras.append((z.astype(np.float32) * 0.25))
    extra = np.concatenate(extras, axis=1).astype(np.float32) if extras else None

    weights = np.ones(len(x), dtype=np.float32)
    rows = []
    if weight_mode != "none":
        cv = cv_predict(x)
        train_err = safe_norm(cv[train_idx] - y[train_idx])
        cluster_hits = []
        for c in range(n_clusters):
            local = train_clusters == c
            hit = float((train_err[local] <= R_HIT).mean()) if local.any() else 0.0
            cluster_hits.append(hit)
        med = float(np.median(cluster_hits))
        for c, hit in enumerate(cluster_hits):
            if weight_mode == "mild":
                w = 1.10 if hit < med else 0.95
            elif weight_mode == "strong":
                w = 1.20 if hit < med else 0.90
            else:
                raise ValueError(f"unknown cluster weight mode: {weight_mode}")
            weights[clusters == c] = w
            rows.append({"cluster": c, "train_n": int((train_clusters == c).sum()), "all_n": int((clusters == c).sum()), "cv_hit": hit, "cluster_weight": w})
        weights *= len(weights) / weights.sum()
    return extra, weights.astype(np.float32), pd.DataFrame(rows)


def train_model(
    cache: dict[str, np.ndarray],
    factory,
    epochs: int,
    seed: int,
    device: torch.device,
) -> nn.Module:
    set_seed(seed)
    model = factory().to(device)
    seq = torch.tensor(cache["seq"])
    scal = torch.tensor(cache["scal"])
    mask = torch.tensor(cache["mask"])
    tgt = torch.tensor(cache["tgt"])
    weight = torch.tensor(cache.get("weight", np.ones(len(seq), dtype=np.float32)))
    idx = np.arange(len(seq))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    flip = torch.tensor(Y_FLIP, device=device)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for _ in range(epochs):
        model.train()
        np.random.shuffle(idx)
        for start in range(0, len(idx), 256):
            b = idx[start : start + 256]
            s = seq[b].to(device)
            c = scal[b].to(device)
            m = mask[b].to(device)
            t = tgt[b].to(device)
            wb = weight[b].to(device)
            if torch.rand(1).item() < 0.5:
                s = s.clone()
                s[:, :, flip] *= -1
                t = t.clone()
                t[:, 1] *= -1
            s = s + torch.randn_like(s) * 0.02 * m.unsqueeze(-1)
            opt.zero_grad(set_to_none=True)
            loss = combined_loss(model(s, c, m), t, wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
        sch.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(0.9).add_(v, alpha=0.1)
                else:
                    ema[k] = v.detach().clone()
    model.load_state_dict(ema)
    model.eval()
    return model


def predict_model(model: nn.Module, feats: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    seq = torch.tensor(feats["seq"])
    scal = torch.tensor(feats["scal"])
    mask = torch.tensor(feats["mask"])
    flip = torch.tensor(Y_FLIP, device=device)
    out = []
    with torch.no_grad():
        for start in range(0, len(seq), 512):
            s = seq[start : start + 512].to(device)
            c = scal[start : start + 512].to(device)
            m = mask[start : start + 512].to(device)
            pr = model(s, c, m).cpu().numpy()
            sf = s.clone()
            sf[:, :, flip] *= -1
            pf = model(sf, c, m).cpu().numpy()
            pf[:, 1] *= -1
            out.append((pr + pf) / 2)
    residual = np.concatenate(out)
    return feats["base"] + np.einsum("bij,bj->bi", feats["rot"].transpose(0, 2, 1), residual)


def train_h_model(x: np.ndarray, y: np.ndarray, epochs: int, seed: int, device: torch.device) -> HyperPhysicsXY2:
    set_seed(seed)
    ds = SlidingWindowDataset(x, y, min_win=3, mode="extended", device=device)
    loader = DataLoader(ds, batch_size=256, sampler=WeightedRandomSampler(ds.theta_weights, len(ds), replacement=True))
    model = HyperPhysicsXY2().to(device)
    with torch.no_grad():
        *_, mn, st = model.get_features(torch.tensor(x, dtype=torch.float32, device=device))
        model.mean_stats.copy_(mn)
        model.std_stats.copy_(st)
    opt = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=model.wd)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.6)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            ft, df, pl, th, _, _, _, rot, sp, _, _ = model.get_features(xb, model.mean_stats, model.std_stats)
            pp, pred_local, log_var = model(ft, df, pl, th, sp, rot)
            yr_local = torch.matmul((yb - pl).unsqueeze(1), rot).squeeze(1)
            loss = model.compute_loss(pp, yb, pred_local, yr_local, log_var)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(0.9).add_(v, alpha=0.1)
                else:
                    ema[k] = v.detach().clone()
    model.load_state_dict(ema)
    model.eval()
    return model


def predict_h_model(model: HyperPhysicsXY2, x: np.ndarray, device: torch.device) -> np.ndarray:
    xt = torch.tensor(x, dtype=torch.float32)

    def fwd(z: torch.Tensor) -> np.ndarray:
        out = []
        with torch.no_grad():
            for start in range(0, len(z), 512):
                b = z[start : start + 512].to(device)
                ft, df, pl, th, _, _, _, rot, sp, _, _ = model.get_features(b, model.mean_stats, model.std_stats)
                pp, _, _ = model(ft, df, pl, th, sp, rot)
                out.append(pp.cpu().numpy())
        return np.concatenate(out)

    pr = fwd(xt)
    xf = xt.clone()
    xf[:, :, 1] *= -1
    pf = fwd(xf)
    pf[:, 1] *= -1
    return ((pr + pf) / 2).astype(np.float32)


def noise_score(x: np.ndarray) -> np.ndarray:
    t = np.arange(x.shape[1], dtype=np.float64)
    vand = np.vander(t, 3, increasing=False)
    out = np.zeros(len(x), dtype=np.float64)
    for axis in range(3):
        coef = np.linalg.lstsq(vand, x[:, :, axis].T, rcond=None)[0]
        fit = (vand @ coef).T
        out += (x[:, :, axis] - fit).std(axis=1)
    return out / 3.0


def regime_masks(x: np.ndarray) -> dict[str, np.ndarray]:
    d = np.diff(x, axis=1)
    v = d / DT
    a = np.diff(v, axis=1) / DT
    speed = safe_norm(v)
    acc = safe_norm(a)
    denom = safe_norm(v[:, 1:]) * safe_norm(v[:, :-1]) + 1e-12
    theta = np.arccos(np.clip(np.sum(v[:, 1:] * v[:, :-1], axis=-1) / denom, -1.0, 1.0))
    noise = noise_score(x)
    return {
        "all": np.ones(len(x), dtype=bool),
        "hard_turn": theta[:, -3:].max(axis=1) > 0.20,
        "recent_turn": theta[:, -2:].max(axis=1) > 0.20,
        "high_acc": acc.max(axis=1) > 15.0,
        "high_speed": speed[:, -1] > 1.0,
        "vertical_change": np.abs(a[:, -1, 2]) > np.quantile(np.abs(a[:, -1, 2]), 0.75),
        "high_noise": noise > np.quantile(noise, 0.75),
    }


def masked_blend(base: np.ndarray, alt: np.ndarray, mask: np.ndarray, w: float) -> np.ndarray:
    out = base.copy()
    out[mask] = (1 - w) * base[mask] + w * alt[mask]
    return out


def evaluate_candidates(preds: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    masks = regime_masks(x)
    rows = []
    for name, pred in preds.items():
        err = safe_norm(pred - y)
        row = {
            "candidate": name,
            "hit": hit_rate(pred, y),
            "mean_error": float(err.mean()),
            "median_error": float(np.median(err)),
        }
        for regime, mask in masks.items():
            row[f"hit_{regime}"] = hit_rate(pred[mask], y[mask])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("hit", ascending=False)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"
        supported = set(torch.cuda.get_arch_list())
        if supported and arch not in supported:
            print(
                f"CUDA device capability {arch} is not supported by this PyTorch build; falling back to CPU.",
                flush=True,
            )
            return torch.device("cpu")
    except Exception as exc:
        print(f"CUDA capability check failed ({exc}); falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device("cuda")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/goh30_oof_lite"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-ode", action="store_true")
    parser.add_argument("--include-tcn", action="store_true", help="Also train a lightweight TCN expert per fold.")
    parser.add_argument("--include-transformer", action="store_true", help="Also train a lightweight Transformer expert per fold.")
    parser.add_argument("--include-h", action="store_true", help="Also train one HyperPhysics model per fold.")
    parser.add_argument("--h-epochs", type=int, default=6)
    parser.add_argument("--sample-size", type=int, default=0, help="Use a deterministic subset for quick smoke runs.")
    parser.add_argument("--no-interiors", action="store_true", help="Train only on full-window labels, without interior transitions.")
    parser.add_argument(
        "--weight-mode",
        default="none",
        choices=["none", "regime-soft", "regime-strong", "hard-final"],
        help="GRU/ODE sample weighting before training.",
    )
    parser.add_argument(
        "--interior-weight",
        type=float,
        default=1.0,
        help="Extra multiplier for interior transition examples in the GRU/ODE cache.",
    )
    parser.add_argument("--cluster-k", type=int, default=6, help="Number of KMeans scene clusters.")
    parser.add_argument(
        "--cluster-feature-mode",
        default="none",
        choices=["none", "onehot", "scalar", "onehot-scalar"],
        help="Weak scene-cluster feature injection into GRU/ODE scalar features.",
    )
    parser.add_argument(
        "--cluster-weight-mode",
        default="none",
        choices=["none", "mild", "strong"],
        help="Weak cluster-level sample weighting based on train-fold CV difficulty.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Training device.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device={device} folds={args.folds} epochs={args.epochs}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    x, y, ids, x_test, test_ids = load_data(args.root)
    if args.sample_size > 0:
        rng = np.random.default_rng(args.seed)
        subset = np.sort(rng.choice(len(x), size=min(args.sample_size, len(x)), replace=False))
        x = x[subset]
        y = y[subset]
        ids = [ids[i] for i in subset]
        print(f"sample subset: {len(x)}", flush=True)

    cv = cv_predict(x)
    kalman = kalman_cv_predict(x)
    train_weights = training_sample_weights(x, args.weight_mode)
    if args.weight_mode != "none" or args.interior_weight != 1.0:
        print(
            "train weights:",
            f"mode={args.weight_mode}",
            f"interior={args.interior_weight}",
            f"min={train_weights.min():.3f}",
            f"mean={train_weights.mean():.3f}",
            f"max={train_weights.max():.3f}",
            flush=True,
        )
    oof_gru = np.zeros_like(y)
    oof_ode = np.zeros_like(y)
    oof_tcn = np.zeros_like(y)
    oof_transformer = np.zeros_like(y)
    oof_h = np.zeros_like(y)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    cluster_detail_frames = []

    for fold, (tr, va) in enumerate(kf.split(x), start=1):
        print(f"fold {fold}/{args.folds}: train={len(tr)} valid={len(va)}", flush=True)
        extra_features, cluster_weights, cluster_details = make_cluster_context(
            x,
            y,
            tr,
            args.cluster_k,
            args.cluster_feature_mode,
            args.cluster_weight_mode,
            args.seed + fold,
        )
        if not cluster_details.empty:
            cluster_details.insert(0, "fold", fold)
            cluster_detail_frames.append(cluster_details)
        fold_weights = train_weights * cluster_weights
        fold_weights *= len(fold_weights) / fold_weights.sum()
        stats = make_stats(x, tr)
        cache = build_cache(
            x,
            y,
            tr,
            stats,
            use_interiors=not args.no_interiors,
            sample_weights=fold_weights,
            interior_weight=args.interior_weight,
            extra_features=extra_features,
        )
        valid_feats = build_eval_features(x, va, stats, extra_features=extra_features)
        scal_dim = cache["scal"].shape[1]
        gru = train_model(cache, lambda: AttnGRU(scal_dim=scal_dim), args.epochs, args.seed + fold * 10, device)
        oof_gru[va] = predict_model(gru, valid_feats, device)
        if not args.skip_ode:
            ode = train_model(cache, lambda: ODEModel(scal_dim=scal_dim), args.epochs, args.seed + fold * 10 + 1, device)
            oof_ode[va] = predict_model(ode, valid_feats, device)
        else:
            oof_ode[va] = oof_gru[va]
        if args.include_tcn:
            tcn = train_model(cache, lambda: TCNModel(scal_dim=scal_dim), args.epochs, args.seed + fold * 10 + 3, device)
            oof_tcn[va] = predict_model(tcn, valid_feats, device)
        if args.include_transformer:
            trf = train_model(cache, lambda: TransformerLite(scal_dim=scal_dim), args.epochs, args.seed + fold * 10 + 4, device)
            oof_transformer[va] = predict_model(trf, valid_feats, device)
        if args.include_h:
            print(f"fold {fold}/{args.folds}: train H epochs={args.h_epochs}", flush=True)
            h_model = train_h_model(x[tr], y[tr], args.h_epochs, args.seed + fold * 10 + 2, device)
            oof_h[va] = predict_h_model(h_model, x[va], device)

    base = 0.5 * oof_gru + 0.5 * oof_ode
    masks = regime_masks(x)
    candidates = {
        "cv": cv,
        "kalman": kalman,
        "gru": oof_gru,
        "ode": oof_ode,
        "gru_ode_050_050": base,
    }
    if args.include_tcn:
        ode_heavy_no_h = 0.15 * oof_gru + 0.85 * oof_ode
        candidates["tcn"] = oof_tcn
        candidates["ode_tcn_950_050"] = 0.95 * oof_ode + 0.05 * oof_tcn
        candidates["ode_tcn_900_100"] = 0.90 * oof_ode + 0.10 * oof_tcn
        candidates["g15_o80_t05"] = 0.15 * oof_gru + 0.80 * oof_ode + 0.05 * oof_tcn
        candidates["g15_o75_t10"] = 0.15 * oof_gru + 0.75 * oof_ode + 0.10 * oof_tcn
        candidates["ode_heavy_no_h"] = ode_heavy_no_h
    if args.include_transformer:
        candidates["transformer"] = oof_transformer
        candidates["ode_trf_950_050"] = 0.95 * oof_ode + 0.05 * oof_transformer
        candidates["ode_trf_900_100"] = 0.90 * oof_ode + 0.10 * oof_transformer
        candidates["g15_o80_trf05"] = 0.15 * oof_gru + 0.80 * oof_ode + 0.05 * oof_transformer
        candidates["g15_o75_trf10"] = 0.15 * oof_gru + 0.75 * oof_ode + 0.10 * oof_transformer
    if args.include_h:
        candidates["h"] = oof_h
        candidates["goh_lite_equal"] = (oof_gru + oof_ode + oof_h) / 3.0
        candidates["goh_lite_g20_o60_h20"] = 0.20 * oof_gru + 0.60 * oof_ode + 0.20 * oof_h
        candidates["goh_lite_g15_o65_h20"] = 0.15 * oof_gru + 0.65 * oof_ode + 0.20 * oof_h
        candidates["goh_lite_g15_o70_h15"] = 0.15 * oof_gru + 0.70 * oof_ode + 0.15 * oof_h
    for w in [0.05, 0.10, 0.15, 0.20]:
        candidates[f"base_high_speed_kalman_{int(w * 100):03d}"] = masked_blend(base, kalman, masks["high_speed"], w)
    candidates["base_high_noise_kalman_005"] = masked_blend(base, kalman, masks["high_noise"], 0.05)
    candidates["base_high_speed_or_noise_kalman_010"] = masked_blend(
        base, kalman, masks["high_speed"] | masks["high_noise"], 0.10
    )

    report = evaluate_candidates(candidates, x, y)
    report.to_csv(args.out_dir / "oof_lite_scores.csv", index=False)
    if cluster_detail_frames:
        pd.concat(cluster_detail_frames, ignore_index=True).to_csv(args.out_dir / "cluster_weight_details.csv", index=False)
    np.save(args.out_dir / "oof_gru.npy", oof_gru)
    np.save(args.out_dir / "oof_ode.npy", oof_ode)
    if args.include_tcn:
        np.save(args.out_dir / "oof_tcn.npy", oof_tcn)
    if args.include_transformer:
        np.save(args.out_dir / "oof_transformer.npy", oof_transformer)
    if args.include_h:
        np.save(args.out_dir / "oof_h.npy", oof_h)
    np.save(args.out_dir / "oof_cv.npy", cv)
    np.save(args.out_dir / "oof_kalman.npy", kalman)
    meta = {
        "folds": args.folds,
        "epochs": args.epochs,
        "seed": args.seed,
        "skip_ode": args.skip_ode,
        "include_tcn": args.include_tcn,
        "include_transformer": args.include_transformer,
        "include_h": args.include_h,
        "h_epochs": args.h_epochs,
        "sample_size": args.sample_size,
        "no_interiors": args.no_interiors,
        "weight_mode": args.weight_mode,
        "interior_weight": args.interior_weight,
        "cluster_k": args.cluster_k,
        "cluster_feature_mode": args.cluster_feature_mode,
        "cluster_weight_mode": args.cluster_weight_mode,
        "ids": ids,
        "test_ids": test_ids,
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(report.head(12).to_string(index=False))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
