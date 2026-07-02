from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic train fold assignments.")
    parser.add_argument("--root", type=Path, default=Path("data"), help="Dataset root containing open/train_labels.csv.")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds. 5 folds means 4:1 train/valid per run.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--out", type=Path, default=Path("jinyong/fold_assignments_5fold.csv"))
    args = parser.parse_args()

    label_path = args.root / "open" / "train_labels.csv"
    labels = pd.read_csv(label_path).sort_values("id").reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    shuffled_idx = rng.permutation(len(labels))
    fold_id = np.full(len(labels), -1, dtype=int)
    for fold, valid_idx in enumerate(np.array_split(shuffled_idx, args.folds)):
        fold_id[valid_idx] = fold

    out_df = labels[["id"]].copy()
    out_df["fold"] = fold_id
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    summary = out_df["fold"].value_counts().sort_index()
    print(f"wrote {args.out}")
    for fold, valid_count in summary.items():
        train_count = len(out_df) - int(valid_count)
        print(f"fold {fold}: train={train_count} valid={int(valid_count)}")


if __name__ == "__main__":
    main()
