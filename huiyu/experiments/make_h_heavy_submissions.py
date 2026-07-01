from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def write_submission(path: Path, ids: list[str], pred: np.ndarray) -> None:
    sub = pd.DataFrame({"id": ids, "x": pred[:, 0], "y": pred[:, 1], "z": pred[:, 2]})
    sub.to_csv(path, index=False)
    print(f"wrote {path} {sub.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-dir", type=Path, default=Path("outputs/goh30_component_submissions"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/goh30_h_heavy_submissions"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(args.component_dir / "ids.csv")["id"].tolist()
    g = np.load(args.component_dir / "pred_gru.npy")
    o = np.load(args.component_dir / "pred_ode.npy")
    h = np.load(args.component_dir / "pred_h.npy")

    cases = {
        "case00_equal_goh30": (1 / 3, 1 / 3, 1 / 3),
        "case01_h_heavy_g10_o05_h85": (0.10, 0.05, 0.85),
        "case02_h_heavy_g05_o20_h75": (0.05, 0.20, 0.75),
        "case03_h_heavy_g15_o05_h80": (0.15, 0.05, 0.80),
        "case04_h_heavy_g00_o25_h75": (0.00, 0.25, 0.75),
        "case05_h_heavy_g05_o10_h85": (0.05, 0.10, 0.85),
        "case06_h_only": (0.00, 0.00, 1.00),
        "case07_h_heavy_g10_o10_h80": (0.10, 0.10, 0.80),
    }
    for name, (wg, wo, wh) in cases.items():
        write_submission(args.out_dir / f"{name}.csv", ids, wg * g + wo * o + wh * h)


if __name__ == "__main__":
    main()
