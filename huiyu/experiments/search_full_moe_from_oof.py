from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from goh30_oof_lite import hit_rate, load_data, regime_masks, safe_norm
from search_cluster_moe import trajectory_features


def mix(g: np.ndarray, o: np.ndarray, h: np.ndarray, w: tuple[float, float, float]) -> np.ndarray:
    return w[0] * g + w[1] * o + w[2] * h


def score(pred: np.ndarray, y: np.ndarray) -> float:
    return hit_rate(pred, y)


def simplex(step: float) -> list[tuple[float, float, float]]:
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    out = []
    for wg in vals:
        for wo in vals:
            wh = 1.0 - wg - wo
            if wh >= -1e-9:
                out.append((round(float(wg), 10), round(float(wo), 10), round(float(wh), 10)))
    return out


def best_weight(g: np.ndarray, o: np.ndarray, h: np.ndarray, y: np.ndarray, idx: np.ndarray, weights: list[tuple[float, float, float]]) -> tuple[tuple[float, float, float], float]:
    best_w = weights[0]
    best_s = -1.0
    for w in weights:
        s = score(mix(g[idx], o[idx], h[idx], w), y[idx])
        if s > best_s:
            best_s = s
            best_w = w
    return best_w, best_s


def apply_cluster_weights(g: np.ndarray, o: np.ndarray, h: np.ndarray, clusters: np.ndarray, weights_by_cluster: dict[int, tuple[float, float, float]]) -> np.ndarray:
    pred = np.zeros_like(g)
    for c, w in weights_by_cluster.items():
        m = clusters == c
        pred[m] = mix(g[m], o[m], h[m], w)
    return pred


def write_submission(path: Path, ids: list[str], pred: np.ndarray) -> None:
    pd.DataFrame({"id": ids, "x": pred[:, 0], "y": pred[:, 1], "z": pred[:, 2]}).to_csv(path, index=False)
    print(f"wrote {path}")


def fold_rows(candidates: dict[str, np.ndarray], y: np.ndarray, folds: int, seed: int) -> pd.DataFrame:
    rows = []
    split = list(KFold(n_splits=folds, shuffle=True, random_state=seed).split(y))
    for name, pred in candidates.items():
        for i, (_, va) in enumerate(split, start=1):
            rows.append({"candidate": name, "fold": i, "hit": score(pred[va], y[va]), "n": len(va)})
        rows.append({"candidate": name, "fold": "all", "hit": score(pred, y), "n": len(y)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--oof-dir", type=Path, default=Path("outputs/goh30_oof_lite_gru_ode_h10_6_cuda"))
    parser.add_argument("--component-dir", type=Path, default=Path("outputs/goh30_component_submissions"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/goh30_full_moe_submissions"))
    parser.add_argument("--clusters", type=int, nargs="+", default=[5, 6, 8])
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    x, y, _, x_test, test_ids = load_data(args.root)
    og = np.load(args.oof_dir / "oof_gru.npy")
    oo = np.load(args.oof_dir / "oof_ode.npy")
    oh = np.load(args.oof_dir / "oof_h.npy")
    tg = np.load(args.component_dir / "pred_gru.npy")
    to = np.load(args.component_dir / "pred_ode.npy")
    th = np.load(args.component_dir / "pred_h.npy")

    weights = simplex(args.step)
    feats, _ = trajectory_features(x)
    test_feats, _ = trajectory_features(x_test)
    scaler = StandardScaler().fit(feats)
    z = scaler.transform(feats)
    zt = scaler.transform(test_feats)

    candidates_oof: dict[str, np.ndarray] = {
        "equal": mix(og, oo, oh, (1 / 3, 1 / 3, 1 / 3)),
        "h_heavy_g10_o05_h85": mix(og, oo, oh, (0.10, 0.05, 0.85)),
        "h_heavy_g05_o20_h75": mix(og, oo, oh, (0.05, 0.20, 0.75)),
        "h_only": oh,
    }
    candidates_test: dict[str, np.ndarray] = {
        "h_heavy_g10_o05_h85": mix(tg, to, th, (0.10, 0.05, 0.85)),
        "h_heavy_g05_o20_h75": mix(tg, to, th, (0.05, 0.20, 0.75)),
        "h_only": th,
    }

    detail_rows = []
    for k in args.clusters:
        km = KMeans(n_clusters=k, random_state=args.seed, n_init=20)
        c = km.fit_predict(z)
        ct = km.predict(zt)
        by_cluster: dict[int, tuple[float, float, float]] = {}
        for cluster in sorted(np.unique(c)):
            idx = np.where(c == cluster)[0]
            w, s = best_weight(og, oo, oh, y, idx, weights)
            by_cluster[int(cluster)] = w
            detail_rows.append({"k": k, "cluster": int(cluster), "n": len(idx), "test_n": int((ct == cluster).sum()), "wg": w[0], "wo": w[1], "wh": w[2], "hit": s})
        name = f"k{k}_cluster_goh_moe"
        candidates_oof[name] = apply_cluster_weights(og, oo, oh, c, by_cluster)
        candidates_test[name] = apply_cluster_weights(tg, to, th, ct, by_cluster)

    # Regime/noise gating on top of the strongest global H-heavy base.
    masks = regime_masks(x)
    test_masks = regime_masks(x_test)
    base_oof = candidates_oof["h_heavy_g10_o05_h85"].copy()
    base_test = candidates_test["h_heavy_g10_o05_h85"].copy()
    regime_weights = {
        "high_noise_h100": ("high_noise", (0.0, 0.0, 1.0)),
        "hard_turn_h100": ("hard_turn", (0.0, 0.0, 1.0)),
        "high_speed_o20_h75": ("high_speed", (0.05, 0.20, 0.75)),
        "vertical_h100": ("vertical_change", (0.0, 0.0, 1.0)),
    }
    for name, (mask_name, w) in regime_weights.items():
        po = base_oof.copy()
        pt = base_test.copy()
        po[masks[mask_name]] = mix(og[masks[mask_name]], oo[masks[mask_name]], oh[masks[mask_name]], w)
        pt[test_masks[mask_name]] = mix(tg[test_masks[mask_name]], to[test_masks[mask_name]], th[test_masks[mask_name]], w)
        candidates_oof[f"regime_{name}"] = po
        candidates_test[f"regime_{name}"] = pt

    summary = []
    for name, pred in candidates_oof.items():
        err = safe_norm(pred - y)
        summary.append({"candidate": name, "hit": score(pred, y), "mean_error": float(err.mean()), "median_error": float(np.median(err))})
    summary_df = pd.DataFrame(summary).sort_values(["hit", "mean_error"], ascending=[False, True])
    fold_df = fold_rows(candidates_oof, y, args.folds, args.seed)
    detail_df = pd.DataFrame(detail_rows)
    summary_df.to_csv(args.out_dir / "full_moe_oof_summary.csv", index=False)
    fold_df.to_csv(args.out_dir / "full_moe_foldwise.csv", index=False)
    detail_df.to_csv(args.out_dir / "cluster_goh_weights.csv", index=False)

    for name, pred in candidates_test.items():
        write_submission(args.out_dir / f"{name}.csv", test_ids, pred)
    print(summary_df.to_string(index=False))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
