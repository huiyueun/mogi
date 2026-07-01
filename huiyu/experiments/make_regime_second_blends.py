from __future__ import annotations

import argparse
import base64
import json
import os
import re
import zlib
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd

from compare_top_experts import add_regimes, load_test, trajectory_features


REGIME_COLS = [
    "all",
    "hard_turn_regime",
    "high_noise",
    "vertical_change_regime",
    "recent_turn_regime",
    "high_acc",
    "high_speed",
    "low_straightness",
]


def extract_b64(notebook_path: Path, name: str) -> str:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    src = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))
    match = re.search(rf"{re.escape(name)}\s*=\s*\"([^\"]+)\"", src)
    if not match:
        raise ValueError(f"could not find {name} in {notebook_path}")
    return match.group(1)


def decode_pred(b64: str, shape: tuple[int, int] = (10000, 3)) -> np.ndarray:
    arr = np.frombuffer(zlib.decompress(base64.b64decode(b64)), dtype=np.float32)
    return arr.reshape(shape).astype(np.float32)


def write_submission(path: Path, ids: list[str], pred: np.ndarray) -> None:
    pd.DataFrame({"id": ids, "x": pred[:, 0], "y": pred[:, 1], "z": pred[:, 2]}).to_csv(path, index=False)
    print(f"wrote {path}")


def blend_with_weights(ode_heavy: np.ndarray, second: np.ndarray, second_weight: np.ndarray) -> np.ndarray:
    w = second_weight.astype(np.float32)[:, None]
    return (1.0 - w) * ode_heavy + w * second


def distance_report(name: str, pred: np.ndarray, ref: np.ndarray, regimes: pd.DataFrame) -> list[dict[str, float | int | str]]:
    dist = np.linalg.norm(pred - ref, axis=1)
    vec = pred - ref
    rows = []
    for regime in REGIME_COLS:
        mask = regimes[regime].to_numpy(bool)
        d = dist[mask]
        v = vec[mask]
        rows.append(
            {
                "candidate": name,
                "regime": regime,
                "count": int(mask.sum()),
                "mean_shift": float(d.mean()),
                "median_shift": float(np.median(d)),
                "p90_shift": float(np.quantile(d, 0.90)),
                "p95_shift": float(np.quantile(d, 0.95)),
                "p99_shift": float(np.quantile(d, 0.99)),
                "max_shift": float(d.max()),
                "n_shift_gt_1cm": int((d > 0.01).sum()),
                "n_shift_gt_5mm": int((d > 0.005).sum()),
                "mean_dx": float(v[:, 0].mean()),
                "mean_dy": float(v[:, 1].mean()),
                "mean_dz": float(v[:, 2].mean()),
                "mean_abs_dx": float(np.abs(v[:, 0]).mean()),
                "mean_abs_dy": float(np.abs(v[:, 1]).mean()),
                "mean_abs_dz": float(np.abs(v[:, 2]).mean()),
            }
        )
    return rows


def weight_summary(name: str, weight: np.ndarray, regimes: pd.DataFrame) -> list[dict[str, float | int | str]]:
    rows = []
    for regime in REGIME_COLS:
        mask = regimes[regime].to_numpy(bool)
        w = weight[mask]
        rows.append(
            {
                "candidate": name,
                "regime": regime,
                "count": int(mask.sum()),
                "mean_second_weight": float(w.mean()),
                "min_second_weight": float(w.min()),
                "max_second_weight": float(w.max()),
                "share_above_020": float((w > 0.20).mean()),
                "share_below_020": float((w < 0.20).mean()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--component-dir", type=Path, default=Path("outputs/goh30_component_submissions"))
    parser.add_argument("--notebook", type=Path, default=Path("best_solve/_[private 2nd] 코드 공유.ipynb"))
    parser.add_argument("--out-dir", type=Path, default=Path("huiyu/submissions/regime_second_blends"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ids, x_test = load_test(args.root)
    component_ids = pd.read_csv(args.component_dir / "ids.csv")["id"].tolist()
    if ids != component_ids:
        raise ValueError("test ids and component ids are not aligned")

    pred_g = np.load(args.component_dir / "pred_gru.npy")
    pred_o = np.load(args.component_dir / "pred_ode.npy")
    pred_h = np.load(args.component_dir / "pred_h.npy")
    ode_heavy = 0.20 * pred_g + 0.60 * pred_o + 0.20 * pred_h

    second_base = decode_pred(extract_b64(args.notebook, "_BASE_B64"))
    second_phys = decode_pred(extract_b64(args.notebook, "_PHYS_B64"))
    second_final = 0.60 * second_base + 0.40 * second_phys

    regimes = add_regimes(trajectory_features(x_test))
    risk_any = (
        regimes["hard_turn_regime"]
        | regimes["high_noise"]
        | regimes["vertical_change_regime"]
        | regimes["low_straightness"]
    ).to_numpy(bool)
    turn_noise = (regimes["hard_turn_regime"] | regimes["high_noise"]).to_numpy(bool)
    z_noise = (regimes["vertical_change_regime"] | regimes["high_noise"]).to_numpy(bool)
    recent_dynamic = (regimes["recent_turn_regime"] | regimes["high_acc"]).to_numpy(bool)

    base_weight = np.full(len(ids), 0.20, dtype=np.float32)

    cases: dict[str, np.ndarray] = {
        "case00_fixed_second020": base_weight,
        "case01_risk_second025": np.where(risk_any, 0.25, 0.20).astype(np.float32),
        "case02_risk_second022": np.where(risk_any, 0.22, 0.18).astype(np.float32),
        "case03_turn_noise_second025": np.where(turn_noise, 0.25, 0.20).astype(np.float32),
        "case04_z_noise_second025": np.where(z_noise, 0.25, 0.20).astype(np.float32),
        "case05_risk025_dynamic015": np.where(risk_any, 0.25, np.where(recent_dynamic, 0.15, 0.20)).astype(
            np.float32
        ),
        "case06_soft_score_018_026": (
            0.18
            + 0.02 * regimes["hard_turn_regime"].to_numpy(np.float32)
            + 0.02 * regimes["high_noise"].to_numpy(np.float32)
            + 0.02 * regimes["vertical_change_regime"].to_numpy(np.float32)
            + 0.02 * regimes["low_straightness"].to_numpy(np.float32)
        ).clip(0.18, 0.26),
    }

    best_fixed = blend_with_weights(ode_heavy, second_final, base_weight)
    report_rows = []
    weight_rows = []
    for name, weight in cases.items():
        pred = blend_with_weights(ode_heavy, second_final, weight).astype(np.float32)
        write_submission(args.out_dir / f"{name}.csv", ids, pred)
        report_rows.extend(distance_report(name + "_vs_fixed020", pred, best_fixed, regimes))
        report_rows.extend(distance_report(name + "_vs_ode_heavy", pred, ode_heavy, regimes))
        weight_rows.extend(weight_summary(name, weight, regimes))

    pd.DataFrame(report_rows).to_csv(args.out_dir / "regime_second_blend_shift_by_regime.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(args.out_dir / "regime_second_blend_weight_summary.csv", index=False)
    regimes.insert(0, "id", ids)
    regimes.to_csv(args.out_dir / "trajectory_regimes.csv", index=False)

    summary = pd.DataFrame(weight_rows)
    print(summary[summary["regime"].eq("all")].to_string(index=False))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
