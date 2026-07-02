from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

R_HIT = 0.01


def safe_norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.linalg.norm(x, axis=axis)


def load_labels(root: Path) -> tuple[np.ndarray, list[str]]:
    data_dir = root / "open"
    label_path = data_dir / "train_labels.csv"
    train_dir = data_dir / "train"
    if not label_path.exists() or not train_dir.exists():
        raise FileNotFoundError(f"Expected {data_dir}/train and {label_path}")
    train_paths = sorted(train_dir.glob("TRAIN_*.csv"))
    ids = [p.stem for p in train_paths]
    labels = pd.read_csv(label_path).set_index("id")
    y = labels.loc[ids][["x", "y", "z"]].to_numpy(np.float32)
    return y, ids


def load_candidate(path: Path, n: int) -> np.ndarray:
    pred = np.load(path)
    if pred.shape != (n, 3):
        raise ValueError(f"{path} has shape {pred.shape}, expected {(n, 3)}")
    return pred.astype(np.float32)


def hit_at_k(preds: list[np.ndarray], y: np.ndarray, radius: float) -> tuple[float, float, float]:
    stacked = np.stack(preds, axis=1)
    distances = safe_norm(stacked - y[:, None, :], axis=2)
    best = distances.min(axis=1)
    return float((best <= radius).mean()), float(best.mean()), float(np.median(best))


def evaluate_prefixes(candidates: dict[str, np.ndarray], order: list[str], y: np.ndarray, radius: float) -> pd.DataFrame:
    rows = []
    chosen = []
    for k, name in enumerate(order, start=1):
        chosen.append(candidates[name])
        hit, mean_dist, median_dist = hit_at_k(chosen, y, radius)
        rows.append(
            {
                "k": k,
                "added": name,
                "laser_set": "+".join(order[:k]),
                "hit_at_k": hit,
                "mean_best_distance": mean_dist,
                "median_best_distance": median_dist,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate multi-laser Hit@K from OOF predictions. A sample is hit if any candidate is within radius."
    )
    parser.add_argument("--root", type=Path, default=Path("data"), help="Dataset root containing open/.")
    parser.add_argument("--oof-dir", type=Path, required=True, help="Directory with oof_*.npy files.")
    parser.add_argument("--radius", type=float, default=R_HIT, help="Hit radius in meters. Default is 0.01.")
    parser.add_argument(
        "--order",
        nargs="+",
        default=["oof_ode", "oof_gru", "oof_kalman", "oof_cv", "oof_h", "oof_tcn", "oof_transformer"],
        help="Candidate file stems in firing order. Example: oof_ode oof_gru oof_kalman.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path.")
    args = parser.parse_args()

    y, ids = load_labels(args.root)
    n = len(ids)
    candidates = {}
    missing = []
    for stem in args.order:
        path = args.oof_dir / f"{stem}.npy"
        if path.exists():
            candidates[stem] = load_candidate(path, n)
        else:
            missing.append(stem)

    order = [name for name in args.order if name in candidates]
    if not order:
        raise FileNotFoundError(f"No requested OOF files found in {args.oof_dir}")

    report = evaluate_prefixes(candidates, order, y, args.radius)
    out = args.out or (args.oof_dir / "multi_laser_hit_report.csv")
    report.to_csv(out, index=False)

    meta = {
        "root": str(args.root),
        "oof_dir": str(args.oof_dir),
        "radius": args.radius,
        "used_order": order,
        "missing": missing,
    }
    (out.with_suffix(".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(report.to_string(index=False))
    if missing:
        print(f"missing candidates skipped: {', '.join(missing)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
