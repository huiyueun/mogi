from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_submission(path: Path) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path)
    return df["id"].tolist(), df[["x", "y", "z"]].to_numpy(np.float32)


def write_submission(path: Path, ids: list[str], pred: np.ndarray) -> None:
    pd.DataFrame({"id": ids, "x": pred[:, 0], "y": pred[:, 1], "z": pred[:, 2]}).to_csv(path, index=False)
    print(f"wrote {path}")


def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)


def shift_row(name: str, pred: np.ndarray, ref: np.ndarray) -> dict[str, float | str | int]:
    d = dist(pred, ref)
    return {
        "candidate": name,
        "mean_shift": float(d.mean()),
        "median_shift": float(np.median(d)),
        "p90_shift": float(np.quantile(d, 0.90)),
        "p95_shift": float(np.quantile(d, 0.95)),
        "p99_shift": float(np.quantile(d, 0.99)),
        "max_shift": float(d.max()),
        "n_shift_gt_1cm": int((d > 0.01).sum()),
        "n_shift_gt_5mm": int((d > 0.005).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed42-dir", type=Path, default=Path("outputs/goh30_full_moe_submissions"))
    parser.add_argument("--seed777-dir", type=Path, default=Path("outputs/goh30_full_moe_seed777_submissions"))
    parser.add_argument("--component-dir", type=Path, default=Path("outputs/goh30_component_submissions"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/goh30_stable_moe_submissions"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(args.component_dir / "ids.csv")["id"].tolist()
    equal = np.load(args.component_dir / "pred_equal.npy")

    files = {
        "seed42_k8": args.seed42_dir / "k8_cluster_goh_moe.csv",
        "seed42_k6": args.seed42_dir / "k6_cluster_goh_moe.csv",
        "seed42_k5": args.seed42_dir / "k5_cluster_goh_moe.csv",
        "seed777_k8": args.seed777_dir / "k8_cluster_goh_moe.csv",
        "seed777_k6": args.seed777_dir / "k6_cluster_goh_moe.csv",
        "seed777_k5": args.seed777_dir / "k5_cluster_goh_moe.csv",
    }
    preds = {}
    for name, path in files.items():
        file_ids, pred = read_submission(path)
        if file_ids != ids:
            raise ValueError(f"id mismatch for {path}")
        preds[name] = pred

    stable = {
        "avg_seed42_seed777_k8": 0.5 * preds["seed42_k8"] + 0.5 * preds["seed777_k8"],
        "avg_seed777_k5_k6_k8": (preds["seed777_k5"] + preds["seed777_k6"] + preds["seed777_k8"]) / 3.0,
        "avg_seed42_k5_k6_k8": (preds["seed42_k5"] + preds["seed42_k6"] + preds["seed42_k8"]) / 3.0,
        "avg_all_k5_k6_k8_both_seeds": sum(preds.values()) / len(preds),
        "blend_seed777_k8_90_equal_10": 0.9 * preds["seed777_k8"] + 0.1 * equal,
        "blend_seed777_k8_80_equal_20": 0.8 * preds["seed777_k8"] + 0.2 * equal,
        "blend_avg_k8_90_equal_10": 0.9 * (0.5 * preds["seed42_k8"] + 0.5 * preds["seed777_k8"]) + 0.1 * equal,
    }

    for name, pred in stable.items():
        write_submission(args.out_dir / f"{name}.csv", ids, pred)

    rows = []
    for name, pred in {**preds, **stable}.items():
        rows.append(shift_row(name, pred, equal))
    report = pd.DataFrame(rows).sort_values(["mean_shift", "p90_shift"])
    report.to_csv(args.out_dir / "candidate_shift_vs_equal.csv", index=False)

    pair_rows = [
        shift_row("seed42_k8_vs_seed777_k8", preds["seed42_k8"], preds["seed777_k8"]),
        shift_row("seed42_k6_vs_seed777_k6", preds["seed42_k6"], preds["seed777_k6"]),
        shift_row("seed42_k5_vs_seed777_k5", preds["seed42_k5"], preds["seed777_k5"]),
        shift_row("seed777_k8_vs_avg_k8", preds["seed777_k8"], stable["avg_seed42_seed777_k8"]),
    ]
    pd.DataFrame(pair_rows).to_csv(args.out_dir / "candidate_pairwise_shift.csv", index=False)

    print("Shift vs equal:")
    print(report.to_string(index=False))
    print("\nPairwise:")
    print(pd.DataFrame(pair_rows).to_string(index=False))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
