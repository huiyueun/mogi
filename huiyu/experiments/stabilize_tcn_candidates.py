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


def shift_row(name: str, pred: np.ndarray, ref: np.ndarray) -> dict[str, float | int | str]:
    d = np.linalg.norm(pred - ref, axis=1)
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
    parser.add_argument("--component-dir", type=Path, default=Path("outputs/goh30_component_submissions"))
    parser.add_argument("--tcn-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/goh30_tcn_stable_submissions"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(args.component_dir / "ids.csv")["id"].tolist()
    pred_g = np.load(args.component_dir / "pred_gru.npy")
    pred_o = np.load(args.component_dir / "pred_ode.npy")
    pred_h = np.load(args.component_dir / "pred_h.npy")
    pred_equal = np.load(args.component_dir / "pred_equal.npy")
    confirmed = 0.20 * pred_g + 0.60 * pred_o + 0.20 * pred_h
    g15_o70_h15 = 0.15 * pred_g + 0.70 * pred_o + 0.15 * pred_h

    tcn_preds = []
    for d in args.tcn_dirs:
        pred = np.load(d / "pred_tcn.npy")
        tcn_preds.append(pred)
    avg_tcn = sum(tcn_preds) / len(tcn_preds)
    np.save(args.out_dir / "pred_tcn_avg.npy", avg_tcn.astype(np.float32))

    cases = {
        "avg_tcn_only": avg_tcn,
        "confirmed_ode_heavy_975_tcn025": 0.975 * confirmed + 0.025 * avg_tcn,
        "confirmed_ode_heavy_950_tcn050": 0.950 * confirmed + 0.050 * avg_tcn,
        "confirmed_ode_heavy_900_tcn100": 0.900 * confirmed + 0.100 * avg_tcn,
        "confirmed_ode_heavy_850_tcn150": 0.850 * confirmed + 0.150 * avg_tcn,
        "g15_o70_h15_950_tcn050": 0.950 * g15_o70_h15 + 0.050 * avg_tcn,
        "g15_o70_h15_900_tcn100": 0.900 * g15_o70_h15 + 0.100 * avg_tcn,
        "equal_950_tcn050": 0.950 * pred_equal + 0.050 * avg_tcn,
    }
    for name, pred in cases.items():
        write_submission(args.out_dir / f"{name}.csv", ids, pred.astype(np.float32))

    rows = [shift_row(name + "_vs_confirmed", pred, confirmed) for name, pred in cases.items()]
    for i, pred in enumerate(tcn_preds):
        rows.append(shift_row(f"tcn_seedset_{i}_vs_avg_tcn", pred, avg_tcn))
    if len(tcn_preds) == 2:
        rows.append(shift_row("tcn_seedset_0_vs_1", tcn_preds[0], tcn_preds[1]))
    report = pd.DataFrame(rows)
    report.to_csv(args.out_dir / "tcn_shift_report.csv", index=False)
    print(report.to_string(index=False))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
