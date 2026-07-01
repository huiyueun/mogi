from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from goh30_oof_lite import hit_rate, kalman_cv_predict, load_data, regime_masks, safe_norm


def masked_blend(base: np.ndarray, alt: np.ndarray, mask: np.ndarray, w: float) -> np.ndarray:
    out = base.copy()
    out[mask] = (1.0 - w) * base[mask] + w * alt[mask]
    return out


def score_row(name: str, pred: np.ndarray, y: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float | str]:
    err = safe_norm(pred - y)
    row: dict[str, float | str] = {
        "candidate": name,
        "hit": hit_rate(pred, y),
        "mean_error": float(err.mean()),
        "median_error": float(np.median(err)),
    }
    for regime, mask in masks.items():
        row[f"hit_{regime}"] = hit_rate(pred[mask], y[mask])
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    x, y, *_ = load_data(args.root)
    gru = np.load(args.pred_dir / "oof_gru.npy")
    ode = np.load(args.pred_dir / "oof_ode.npy")
    kalman = np.load(args.pred_dir / "oof_kalman.npy") if (args.pred_dir / "oof_kalman.npy").exists() else kalman_cv_predict(x)
    masks = regime_masks(x)

    rows = []
    for w_ode in np.linspace(0.0, 1.0, 21):
        pred = (1.0 - w_ode) * gru + w_ode * ode
        rows.append(score_row(f"gru_{1.0 - w_ode:.2f}_ode_{w_ode:.2f}", pred, y, masks))
        for gate_name, mask in [
            ("hs", masks["high_speed"]),
            ("hn", masks["high_noise"]),
            ("hs_or_hn", masks["high_speed"] | masks["high_noise"]),
        ]:
            for kw in [0.02, 0.05, 0.10]:
                rows.append(
                    score_row(
                        f"gru_{1.0 - w_ode:.2f}_ode_{w_ode:.2f}_{gate_name}_kalman_{kw:.2f}",
                        masked_blend(pred, kalman, mask, kw),
                        y,
                        masks,
                    )
                )

    df = pd.DataFrame(rows).sort_values(["hit", "mean_error"], ascending=[False, True])
    out = args.out or (args.pred_dir / "blend_search.csv")
    df.to_csv(out, index=False)
    print(df.head(25).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
