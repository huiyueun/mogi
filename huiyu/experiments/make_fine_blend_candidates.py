from __future__ import annotations

import argparse
import base64
import json
import os
import re
import zlib
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_top_experts import add_regimes, angle_to_direction, load_pred, load_test, norm, safe_unit, trajectory_features


REGIME_COLS = [
    "all",
    "high_speed",
    "hard_turn_regime",
    "recent_turn_regime",
    "high_acc",
    "high_noise",
    "vertical_change_regime",
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


def build_candidates(ode: np.ndarray, base: np.ndarray, phys: np.ndarray) -> dict[str, tuple[np.ndarray, dict[str, float]]]:
    second_final = 0.60 * base + 0.40 * phys
    specs = {
        "fixed_0475_second_final": {"ode": 0.525, "base": 0.285, "phys": 0.190},
        "fixed_0500_second_final": {"ode": 0.500, "base": 0.300, "phys": 0.200},
        "fixed_0525_second_final": {"ode": 0.475, "base": 0.315, "phys": 0.210},
        "fixed_0550_second_final": {"ode": 0.450, "base": 0.330, "phys": 0.220},
        "ode50_base35_phys15": {"ode": 0.500, "base": 0.350, "phys": 0.150},
        "ode50_base25_phys25": {"ode": 0.500, "base": 0.250, "phys": 0.250},
        "ode45_base35_phys20": {"ode": 0.450, "base": 0.350, "phys": 0.200},
        "ode55_base30_phys15": {"ode": 0.550, "base": 0.300, "phys": 0.150},
        "ode45_base30_phys25": {"ode": 0.450, "base": 0.300, "phys": 0.250},
    }
    out: dict[str, tuple[np.ndarray, dict[str, float]]] = {}
    for name, w in specs.items():
        pred = w["ode"] * ode + w["base"] * base + w["phys"] * phys
        out[name] = (pred.astype(np.float32), w)
    out["second_final_only"] = (second_final.astype(np.float32), {"ode": 0.0, "base": 0.6, "phys": 0.4})
    return out


def distance_rows(candidates: dict[str, tuple[np.ndarray, dict[str, float]]], refs: dict[str, np.ndarray], regimes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (pred, weights) in candidates.items():
        for ref_name, ref in refs.items():
            vec = pred - ref
            dist = norm(vec)
            for regime in REGIME_COLS:
                mask = regimes[regime].to_numpy(bool)
                d = dist[mask]
                v = vec[mask]
                rows.append(
                    {
                        "candidate": name,
                        "reference": ref_name,
                        "regime": regime,
                        "count": int(mask.sum()),
                        "ode_w": weights["ode"],
                        "base_w": weights["base"],
                        "phys_w": weights["phys"],
                        "mean_shift": float(d.mean()),
                        "p95_shift": float(np.quantile(d, 0.95)),
                        "p99_shift": float(np.quantile(d, 0.99)),
                        "max_shift": float(d.max()),
                        "mean_dx": float(v[:, 0].mean()),
                        "mean_dy": float(v[:, 1].mean()),
                        "mean_dz": float(v[:, 2].mean()),
                    }
                )
    return pd.DataFrame(rows)


def behavior_rows(x: np.ndarray, candidates: dict[str, tuple[np.ndarray, dict[str, float]]], refs: dict[str, np.ndarray], regimes: pd.DataFrame) -> pd.DataFrame:
    last = x[:, -1].astype(np.float64)
    prev = x[:, -2].astype(np.float64)
    last_vel = last - prev
    last_dir = safe_unit(last_vel)
    cv = last + 2.0 * (last - prev)
    ode = refs["ode_heavy"].astype(np.float64)
    fixed50 = refs["fixed_050"].astype(np.float64)
    ode_vec = ode - last
    fixed50_vec = fixed50 - last
    ode_step = norm(ode_vec)
    fixed50_step = norm(fixed50_vec)
    ode_turn = angle_to_direction(ode_vec, last_vel)
    fixed50_turn = angle_to_direction(fixed50_vec, last_vel)
    ode_z_move = np.abs(ode[:, 2] - last[:, 2])
    fixed50_z_move = np.abs(fixed50[:, 2] - last[:, 2])
    ode_z_cv = np.abs(ode[:, 2] - cv[:, 2])
    fixed50_z_cv = np.abs(fixed50[:, 2] - cv[:, 2])

    rows = []
    for name, (pred, weights) in candidates.items():
        p = pred.astype(np.float64)
        vec = p - last
        step = norm(vec)
        forward = (vec * last_dir).sum(axis=1)
        turn = angle_to_direction(vec, last_vel)
        z_move = np.abs(p[:, 2] - last[:, 2])
        z_cv = np.abs(p[:, 2] - cv[:, 2])
        for regime in REGIME_COLS:
            mask = regimes[regime].to_numpy(bool)
            rows.append(
                {
                    "candidate": name,
                    "regime": regime,
                    "count": int(mask.sum()),
                    "ode_w": weights["ode"],
                    "base_w": weights["base"],
                    "phys_w": weights["phys"],
                    "mean_step_delta_vs_ode": float((step - ode_step)[mask].mean()),
                    "mean_step_delta_vs_fixed50": float((step - fixed50_step)[mask].mean()),
                    "mean_forward": float(forward[mask].mean()),
                    "mean_turn_delta_vs_ode_deg": float(np.degrees(turn - ode_turn)[mask].mean()),
                    "mean_turn_delta_vs_fixed50_deg": float(np.degrees(turn - fixed50_turn)[mask].mean()),
                    "mean_abs_z_move_delta_vs_ode": float((z_move - ode_z_move)[mask].mean()),
                    "mean_abs_z_move_delta_vs_fixed50": float((z_move - fixed50_z_move)[mask].mean()),
                    "closer_to_cv_z_rate_vs_ode": float((z_cv[mask] < ode_z_cv[mask]).mean()),
                    "closer_to_cv_z_rate_vs_fixed50": float((z_cv[mask] < fixed50_z_cv[mask]).mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_candidate_metric(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    rows = df[df["regime"].eq("all")].copy()
    rows = rows.sort_values(["ode_w", "base_w", "phys_w"], ascending=[False, False, False])
    plt.figure(figsize=(11, 5))
    plt.bar(rows["candidate"], rows[metric])
    plt.xticks(rotation=40, ha="right")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--ode-heavy", type=Path, default=Path("huiyu/submissions/goh30_component/case02_ode_heavy_g20_o60_h20.csv"))
    parser.add_argument("--notebook", type=Path, default=Path("best_solve/_[private 2nd] 코드 공유.ipynb"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/fine_blend_candidates"))
    args = parser.parse_args()

    ids, x_test = load_test(args.root)
    ode = load_pred(args.ode_heavy, ids)
    base = decode_pred(extract_b64(args.notebook, "_BASE_B64"))
    phys = decode_pred(extract_b64(args.notebook, "_PHYS_B64"))
    candidates = build_candidates(ode, base, phys)
    fixed50 = candidates["fixed_0500_second_final"][0]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.out_dir / "submissions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for name, (pred, _) in candidates.items():
        write_submission(pred_dir / f"{name}.csv", ids, pred)

    regimes = add_regimes(trajectory_features(x_test))
    regimes.insert(0, "id", ids)
    regimes.to_csv(args.out_dir / "trajectory_regimes.csv", index=False)

    refs = {"ode_heavy": ode, "fixed_050": fixed50}
    distance = distance_rows(candidates, refs, regimes)
    behavior = behavior_rows(x_test, candidates, refs, regimes)
    distance.to_csv(args.out_dir / "fine_blend_distance_by_regime.csv", index=False)
    behavior.to_csv(args.out_dir / "fine_blend_behavior_by_regime.csv", index=False)
    plot_candidate_metric(behavior, "mean_turn_delta_vs_fixed50_deg", args.out_dir / "candidate_turn_delta_vs_fixed50.png")
    plot_candidate_metric(behavior, "mean_abs_z_move_delta_vs_fixed50", args.out_dir / "candidate_z_delta_vs_fixed50.png")
    plot_candidate_metric(distance[distance["reference"].eq("fixed_050")], "mean_shift", args.out_dir / "candidate_shift_vs_fixed50.png")

    print("Distance vs fixed_050 (all):")
    print(
        distance[(distance["reference"].eq("fixed_050")) & (distance["regime"].eq("all"))]
        .sort_values("mean_shift")[["candidate", "ode_w", "base_w", "phys_w", "mean_shift", "p95_shift", "max_shift"]]
        .to_string(index=False)
    )
    print("\nBehavior vs fixed_050 (all):")
    print(
        behavior[behavior["regime"].eq("all")]
        .sort_values("candidate")[
            [
                "candidate",
                "ode_w",
                "base_w",
                "phys_w",
                "mean_turn_delta_vs_fixed50_deg",
                "mean_abs_z_move_delta_vs_fixed50",
                "mean_step_delta_vs_fixed50",
                "closer_to_cv_z_rate_vs_fixed50",
            ]
        ]
        .to_string(index=False)
    )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
