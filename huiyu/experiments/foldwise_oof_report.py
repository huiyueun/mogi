from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from goh30_oof_lite import hit_rate, load_data, safe_norm


def row(candidate: str, fold: str | int, pred: np.ndarray, y: np.ndarray, idx: np.ndarray) -> dict[str, float | str | int]:
    p = pred[idx]
    t = y[idx]
    err = safe_norm(p - t)
    return {
        "candidate": candidate,
        "fold": fold,
        "n": int(len(idx)),
        "hit": hit_rate(p, t),
        "mean_error": float(err.mean()),
        "median_error": float(np.median(err)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--pred-dir", type=Path, default=Path("outputs/goh30_oof_lite_gru_ode10_cuda"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cluster-ks", type=int, nargs="+", default=[8, 6, 5])
    args = parser.parse_args()

    x, y, *_ = load_data(args.root)
    meta = json.loads((args.pred_dir / "meta.json").read_text(encoding="utf-8"))
    folds = int(meta["folds"])
    seed = int(meta["seed"])
    gru = np.load(args.pred_dir / "oof_gru.npy")
    ode = np.load(args.pred_dir / "oof_ode.npy")
    h = np.load(args.pred_dir / "oof_h.npy") if (args.pred_dir / "oof_h.npy").exists() else None

    candidates: dict[str, np.ndarray] = {
        "gru": gru,
        "ode": ode,
        "gru_ode_050": 0.5 * gru + 0.5 * ode,
        "gru_015_ode_085": 0.15 * gru + 0.85 * ode,
        "gru_020_ode_080": 0.20 * gru + 0.80 * ode,
        "gru_030_ode_070": 0.30 * gru + 0.70 * ode,
    }
    if h is not None:
        candidates.update(
            {
                "h": h,
                "goh_lite_equal": (gru + ode + h) / 3.0,
                "g10_o05_h85": 0.10 * gru + 0.05 * ode + 0.85 * h,
                "g05_o20_h75": 0.05 * gru + 0.20 * ode + 0.75 * h,
                "g15_o05_h80": 0.15 * gru + 0.05 * ode + 0.80 * h,
                "g00_o25_h75": 0.00 * gru + 0.25 * ode + 0.75 * h,
                "g05_o10_h85": 0.05 * gru + 0.10 * ode + 0.85 * h,
                "g10_o10_h80": 0.10 * gru + 0.10 * ode + 0.80 * h,
                "g20_o60_h20": 0.20 * gru + 0.60 * ode + 0.20 * h,
                "g15_o65_h20": 0.15 * gru + 0.65 * ode + 0.20 * h,
                "g15_o70_h15": 0.15 * gru + 0.70 * ode + 0.15 * h,
            }
        )
    for k in args.cluster_ks:
        path = args.pred_dir / f"kmeans{k}_cluster_best_pred.npy"
        if path.exists():
            candidates[f"kmeans{k}_cluster_moe"] = np.load(path)

    split_indices = []
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold_id, (_, va) in enumerate(kf.split(x), start=1):
        split_indices.append((fold_id, va))
    split_indices.append(("all", np.arange(len(x))))

    rows = []
    for name, pred in candidates.items():
        for fold_id, idx in split_indices:
            rows.append(row(name, fold_id, pred, y, idx))

    df = pd.DataFrame(rows)
    out = args.out or (args.pred_dir / "foldwise_oof_report.csv")
    df.to_csv(out, index=False)

    pivot = df.pivot(index="candidate", columns="fold", values="hit").reset_index()
    if "all" in pivot.columns:
        pivot = pivot.sort_values("all", ascending=False)
    print(pivot.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
