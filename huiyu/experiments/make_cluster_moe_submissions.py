from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from goh30_oof_lite import hit_rate, load_data
from search_cluster_moe import search_cluster_weights, trajectory_features


def write_submission(path: Path, ids: list[str], pred: np.ndarray) -> None:
    sub = pd.DataFrame({"id": ids, "x": pred[:, 0], "y": pred[:, 1], "z": pred[:, 2]})
    sub.to_csv(path, index=False)
    print(f"wrote {path} {sub.shape}")


def apply_cluster_weights(
    pred_g: np.ndarray,
    pred_ode: np.ndarray,
    pred_h: np.ndarray,
    clusters: np.ndarray,
    cluster_w_ode: dict[int, float],
    h_weight: float,
) -> np.ndarray:
    out = np.zeros_like(pred_g)
    remain = 1.0 - h_weight
    for c, w_ode in cluster_w_ode.items():
        mask = clusters == c
        wg = remain * (1.0 - w_ode)
        wo = remain * w_ode
        out[mask] = wg * pred_g[mask] + wo * pred_ode[mask] + h_weight * pred_h[mask]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--oof-dir", type=Path, default=Path("outputs/goh30_oof_lite_gru_ode10_cuda"))
    parser.add_argument("--component-dir", type=Path, default=Path("outputs/goh30_component_submissions"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/goh30_cluster_moe_submissions"))
    parser.add_argument("--clusters", type=int, nargs="+", default=[8, 6, 5])
    parser.add_argument("--h-weights", type=float, nargs="+", default=[0.15, 0.20, 0.25])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    x, y, _, x_test, test_ids = load_data(args.root)
    oof_gru = np.load(args.oof_dir / "oof_gru.npy")
    oof_ode = np.load(args.oof_dir / "oof_ode.npy")
    pred_g = np.load(args.component_dir / "pred_gru.npy")
    pred_ode = np.load(args.component_dir / "pred_ode.npy")
    pred_h = np.load(args.component_dir / "pred_h.npy")

    train_feats, feat_names = trajectory_features(x)
    test_feats, _ = trajectory_features(x_test)
    scaler = StandardScaler().fit(train_feats)
    z_train = scaler.transform(train_feats)
    z_test = scaler.transform(test_feats)
    weight_grid = np.linspace(0.0, 1.0, 21)

    summary_rows = []
    detail_rows = []
    for k in args.clusters:
        km = KMeans(n_clusters=k, random_state=args.seed, n_init=20)
        train_clusters = km.fit_predict(z_train)
        test_clusters = km.predict(z_test)
        oof_pred, cluster_w_ode, details = search_cluster_weights(y, oof_gru, oof_ode, train_clusters, weight_grid)
        oof_hit = hit_rate(oof_pred, y)
        for row in details:
            row = dict(row)
            row["k"] = k
            row["test_n"] = int((test_clusters == row["cluster"]).sum())
            detail_rows.append(row)

        for h_weight in args.h_weights:
            pred = apply_cluster_weights(pred_g, pred_ode, pred_h, test_clusters, cluster_w_ode, h_weight)
            name = f"case_k{k}_cluster_moe_h{int(round(h_weight * 100)):02d}"
            write_submission(args.out_dir / f"{name}.csv", test_ids, pred)
            summary_rows.append(
                {
                    "candidate": name,
                    "k": k,
                    "h_weight": h_weight,
                    "oof_gru_ode_cluster_hit": oof_hit,
                    "cluster_w_ode": json.dumps(cluster_w_ode, sort_keys=True),
                }
            )

    # Also keep the already successful global ODE-heavy reference family in the same folder.
    references = {
        "ref_g20_o60_h20": (0.20, 0.60, 0.20),
        "ref_g15_o65_h20": (0.15, 0.65, 0.20),
        "ref_g15_o70_h15": (0.15, 0.70, 0.15),
    }
    for name, (wg, wo, wh) in references.items():
        write_submission(args.out_dir / f"{name}.csv", test_ids, wg * pred_g + wo * pred_ode + wh * pred_h)

    pd.DataFrame(summary_rows).to_csv(args.out_dir / "cluster_moe_submission_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(args.out_dir / "cluster_moe_weights.csv", index=False)
    pd.DataFrame({"feature": feat_names}).to_csv(args.out_dir / "cluster_features.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
