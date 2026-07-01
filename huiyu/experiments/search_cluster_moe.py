from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from goh30_oof_lite import hit_rate, kalman_cv_predict, load_data, safe_norm


DT = 0.04


def trajectory_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
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
    ]
    names = [
        "last_speed",
        "mean_speed",
        "max_speed",
        "speed_std",
        "last_acc",
        "max_acc",
        "mean_acc",
        "max_jerk",
        "theta_last",
        "theta_recent_mean",
        "theta_recent_max",
        "theta_mean",
        "theta_std",
        "straightness",
        "path",
        "abs_z_acc_last",
        "z_speed_last",
    ]
    return np.column_stack(parts).astype(np.float32), names


def hit_for(pred: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
    if len(idx) == 0:
        return float("nan")
    return hit_rate(pred[idx], y[idx])


def search_cluster_weights(
    y: np.ndarray,
    gru: np.ndarray,
    ode: np.ndarray,
    clusters: np.ndarray,
    weight_grid: np.ndarray,
) -> tuple[np.ndarray, dict[int, float], list[dict[str, float | int]]]:
    pred = np.zeros_like(y)
    best_weights: dict[int, float] = {}
    rows: list[dict[str, float | int]] = []
    for c in sorted(np.unique(clusters)):
        idx = np.where(clusters == c)[0]
        best = None
        for w_ode in weight_grid:
            cand = (1.0 - w_ode) * gru[idx] + w_ode * ode[idx]
            score = hit_rate(cand, y[idx])
            if best is None or score > best[0]:
                best = (score, float(w_ode), cand)
        assert best is not None
        score, w_ode, cand = best
        pred[idx] = cand
        best_weights[int(c)] = w_ode
        rows.append(
            {
                "cluster": int(c),
                "n": int(len(idx)),
                "best_w_ode": w_ode,
                "best_hit": score,
                "hit_gru": hit_for(gru, y, idx),
                "hit_ode": hit_for(ode, y, idx),
                "hit_050": hit_rate(0.5 * gru[idx] + 0.5 * ode[idx], y[idx]),
            }
        )
    return pred, best_weights, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--clusters", type=int, nargs="+", default=[3, 4, 5, 6, 8])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = args.out_dir or args.pred_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, *_ = load_data(args.root)
    gru = np.load(args.pred_dir / "oof_gru.npy")
    ode = np.load(args.pred_dir / "oof_ode.npy")
    kalman = np.load(args.pred_dir / "oof_kalman.npy") if (args.pred_dir / "oof_kalman.npy").exists() else kalman_cv_predict(x)

    feats, feat_names = trajectory_features(x)
    z = StandardScaler().fit_transform(feats)
    weight_grid = np.linspace(0.0, 1.0, 21)

    global_rows = []
    detail_frames = []
    base_050 = 0.5 * gru + 0.5 * ode
    global_rows.extend(
        [
            {"candidate": "gru", "hit": hit_rate(gru, y), "mean_error": float(safe_norm(gru - y).mean())},
            {"candidate": "ode", "hit": hit_rate(ode, y), "mean_error": float(safe_norm(ode - y).mean())},
            {"candidate": "gru_ode_050", "hit": hit_rate(base_050, y), "mean_error": float(safe_norm(base_050 - y).mean())},
            {"candidate": "kalman", "hit": hit_rate(kalman, y), "mean_error": float(safe_norm(kalman - y).mean())},
        ]
    )

    for k in args.clusters:
        km = KMeans(n_clusters=k, random_state=args.seed, n_init=20)
        clusters = km.fit_predict(z)
        pred, weights, detail_rows = search_cluster_weights(y, gru, ode, clusters, weight_grid)
        global_rows.append(
            {
                "candidate": f"kmeans{k}_cluster_best_gru_ode",
                "hit": hit_rate(pred, y),
                "mean_error": float(safe_norm(pred - y).mean()),
            }
        )
        detail = pd.DataFrame(detail_rows)
        detail.insert(0, "k", k)
        detail_frames.append(detail)
        np.save(out_dir / f"kmeans{k}_clusters.npy", clusters)
        np.save(out_dir / f"kmeans{k}_cluster_best_pred.npy", pred)

    global_df = pd.DataFrame(global_rows).sort_values(["hit", "mean_error"], ascending=[False, True])
    detail_df = pd.concat(detail_frames, ignore_index=True)
    global_df.to_csv(out_dir / "cluster_moe_summary.csv", index=False)
    detail_df.to_csv(out_dir / "cluster_moe_details.csv", index=False)
    pd.DataFrame({"feature": feat_names}).to_csv(out_dir / "cluster_features.csv", index=False)
    print(global_df.to_string(index=False))
    print()
    print(detail_df.sort_values(["k", "cluster"]).to_string(index=False))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
