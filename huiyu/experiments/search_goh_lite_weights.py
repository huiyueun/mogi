from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from goh30_oof_lite import hit_rate, load_data, safe_norm


def score(name: str, pred: np.ndarray, y: np.ndarray) -> dict[str, float | str]:
    err = safe_norm(pred - y)
    return {
        "candidate": name,
        "hit": hit_rate(pred, y),
        "mean_error": float(err.mean()),
        "median_error": float(np.median(err)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    _, y, *_ = load_data(args.root)
    g = np.load(args.pred_dir / "oof_gru.npy")
    o = np.load(args.pred_dir / "oof_ode.npy")
    h = np.load(args.pred_dir / "oof_h.npy")

    rows = [
        score("gru", g, y),
        score("ode", o, y),
        score("h", h, y),
        score("equal_goh_lite", (g + o + h) / 3.0, y),
    ]
    weights = np.arange(0.0, 1.0 + 1e-9, args.step)
    for wg in weights:
        for wo in weights:
            wh = 1.0 - wg - wo
            if wh < -1e-9:
                continue
            wh = round(float(wh), 10)
            pred = wg * g + wo * o + wh * h
            rows.append(score(f"g{wg:.2f}_o{wo:.2f}_h{wh:.2f}", pred, y))

    df = pd.DataFrame(rows).sort_values(["hit", "mean_error"], ascending=[False, True])
    out = args.out or (args.pred_dir / "goh_lite_weight_search.csv")
    df.to_csv(out, index=False)
    print(df.head(30).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
