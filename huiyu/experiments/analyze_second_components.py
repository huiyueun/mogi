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

from compare_top_experts import (
    add_regimes,
    angle_to_direction,
    load_pred,
    load_test,
    norm,
    safe_unit,
    summarize_distance,
    trajectory_features,
)


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


def load_second_components(notebook: Path) -> dict[str, np.ndarray]:
    base = decode_pred(extract_b64(notebook, "_BASE_B64"))
    phys = decode_pred(extract_b64(notebook, "_PHYS_B64"))
    final = 0.60 * base + 0.40 * phys
    return {
        "second_base": base,
        "second_phys": phys,
        "second_final": final.astype(np.float32),
    }


def pairwise_component_summary(preds: dict[str, np.ndarray], regimes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = []
    for comp in ["second_base", "second_phys", "second_final"]:
        pairs.append((f"original_vs_{comp}", "original", comp))
        pairs.append((f"ode_heavy_vs_{comp}", "ode_heavy", comp))
    pairs += [
        ("second_base_vs_second_phys", "second_base", "second_phys"),
        ("second_base_vs_second_final", "second_base", "second_final"),
        ("second_phys_vs_second_final", "second_phys", "second_final"),
    ]

    summary_rows = []
    regime_rows = []
    for pair, left, right in pairs:
        vec = preds[right] - preds[left]
        dist = norm(vec)
        summary_rows.append(summarize_distance(pair, dist, vec))
        for reg in REGIME_COLS:
            mask = regimes[reg].to_numpy(bool)
            row = summarize_distance(pair, dist[mask], vec[mask])
            row["regime"] = reg
            row["regime_share"] = float(mask.mean())
            regime_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(regime_rows)


def component_behavior(x: np.ndarray, preds: dict[str, np.ndarray], regimes: pd.DataFrame) -> pd.DataFrame:
    last = x[:, -1].astype(np.float64)
    prev = x[:, -2].astype(np.float64)
    last_vel = last - prev
    last_dir = safe_unit(last_vel)
    cv = last + 2.0 * (last - prev)

    original_vec = preds["original"].astype(np.float64) - last
    ode_vec_from_original = preds["ode_heavy"].astype(np.float64) - preds["original"].astype(np.float64)
    original_step = norm(original_vec)
    original_forward = (original_vec * last_dir).sum(axis=1)
    original_turn_angle = angle_to_direction(original_vec, last_vel)
    original_z_move = np.abs(preds["original"][:, 2].astype(np.float64) - last[:, 2])
    original_z_cv_resid = np.abs(preds["original"][:, 2].astype(np.float64) - cv[:, 2])

    rows = []
    for comp in ["second_base", "second_phys", "second_final"]:
        comp_vec = preds[comp].astype(np.float64) - last
        comp_step = norm(comp_vec)
        comp_forward = (comp_vec * last_dir).sum(axis=1)
        comp_turn_angle = angle_to_direction(comp_vec, last_vel)
        comp_z_move = np.abs(preds[comp][:, 2].astype(np.float64) - last[:, 2])
        comp_z_cv_resid = np.abs(preds[comp][:, 2].astype(np.float64) - cv[:, 2])
        comp_vec_from_original = preds[comp].astype(np.float64) - preds["original"].astype(np.float64)
        align_denom = norm(ode_vec_from_original) * norm(comp_vec_from_original) + 1e-12
        ode_comp_cos = (ode_vec_from_original * comp_vec_from_original).sum(axis=1) / align_denom
        ode_comp_cos = np.clip(ode_comp_cos, -1.0, 1.0)

        for reg in REGIME_COLS:
            mask = regimes[reg].to_numpy(bool)
            rows.append(
                {
                    "component": comp,
                    "regime": reg,
                    "count": int(mask.sum()),
                    "component_longer_step_rate": float((comp_step[mask] > original_step[mask]).mean()),
                    "mean_step_original": float(original_step[mask].mean()),
                    "mean_step_component": float(comp_step[mask].mean()),
                    "mean_step_delta_component_minus_original": float((comp_step - original_step)[mask].mean()),
                    "component_more_forward_rate": float((comp_forward[mask] > original_forward[mask]).mean()),
                    "mean_forward_delta_component_minus_original": float((comp_forward - original_forward)[mask].mean()),
                    "component_more_turn_rate": float((comp_turn_angle[mask] > original_turn_angle[mask]).mean()),
                    "mean_turn_angle_original_deg": float(np.degrees(original_turn_angle[mask]).mean()),
                    "mean_turn_angle_component_deg": float(np.degrees(comp_turn_angle[mask]).mean()),
                    "mean_turn_angle_delta_deg": float(np.degrees(comp_turn_angle - original_turn_angle)[mask].mean()),
                    "component_smaller_z_move_rate": float((comp_z_move[mask] < original_z_move[mask]).mean()),
                    "mean_abs_z_move_original": float(original_z_move[mask].mean()),
                    "mean_abs_z_move_component": float(comp_z_move[mask].mean()),
                    "mean_abs_z_move_delta_component_minus_original": float((comp_z_move - original_z_move)[mask].mean()),
                    "component_closer_to_cv_z_rate": float((comp_z_cv_resid[mask] < original_z_cv_resid[mask]).mean()),
                    "mean_abs_z_cv_resid_original": float(original_z_cv_resid[mask].mean()),
                    "mean_abs_z_cv_resid_component": float(comp_z_cv_resid[mask].mean()),
                    "ode_component_same_direction_rate": float((ode_comp_cos[mask] > 0.0).mean()),
                    "ode_component_strong_same_direction_rate": float((ode_comp_cos[mask] > 0.5).mean()),
                    "mean_ode_component_cos": float(ode_comp_cos[mask].mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_metric_by_component(diag: pd.DataFrame, metric: str, ylabel: str, title: str, out_path: Path) -> None:
    rows = diag[diag["regime"].isin(["all", "hard_turn_regime", "recent_turn_regime", "high_acc", "high_noise", "vertical_change_regime"])].copy()
    regimes = rows["regime"].drop_duplicates().tolist()
    comps = ["second_base", "second_phys", "second_final"]
    x = np.arange(len(regimes))
    width = 0.24
    plt.figure(figsize=(11, 5))
    for j, comp in enumerate(comps):
        vals = [rows[(rows["component"].eq(comp)) & (rows["regime"].eq(reg))][metric].iloc[0] for reg in regimes]
        plt.bar(x + (j - 1) * width, vals, width, label=comp.replace("second_", ""))
    plt.axhline(0.0, color="black", lw=1)
    plt.xticks(x, regimes, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def write_component_submissions(out_dir: Path, ids: list[str], comps: dict[str, np.ndarray]) -> None:
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for name, pred in comps.items():
        pd.DataFrame({"id": ids, "x": pred[:, 0], "y": pred[:, 1], "z": pred[:, 2]}).to_csv(
            pred_dir / f"{name}.csv", index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--original", type=Path, default=Path("submission_GOH30.csv"))
    parser.add_argument("--ode-heavy", type=Path, default=Path("huiyu/submissions/goh30_component/case02_ode_heavy_g20_o60_h20.csv"))
    parser.add_argument("--notebook", type=Path, default=Path("best_solve/_[private 2nd] 코드 공유.ipynb"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/second_component_analysis"))
    args = parser.parse_args()

    ids, x_test = load_test(args.root)
    second_comps = load_second_components(args.notebook)
    preds = {
        "original": load_pred(args.original, ids),
        "ode_heavy": load_pred(args.ode_heavy, ids),
        **second_comps,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_component_submissions(args.out_dir, ids, second_comps)

    feat = add_regimes(trajectory_features(x_test))
    feat.insert(0, "id", ids)
    feat.to_csv(args.out_dir / "trajectory_regimes.csv", index=False)

    summary, by_regime = pairwise_component_summary(preds, feat)
    diag = component_behavior(x_test, preds, feat)
    summary.to_csv(args.out_dir / "second_component_pairwise_summary.csv", index=False)
    by_regime.to_csv(args.out_dir / "second_component_pairwise_by_regime.csv", index=False)
    diag.to_csv(args.out_dir / "second_component_behavior_by_regime.csv", index=False)

    plot_metric_by_component(
        diag,
        "mean_turn_angle_delta_deg",
        "Mean angle delta vs original (deg)",
        "Turn angle: component - original",
        args.out_dir / "component_turn_angle_delta_by_regime.png",
    )
    plot_metric_by_component(
        diag,
        "mean_step_delta_component_minus_original",
        "Mean step length delta vs original",
        "Step length: component - original",
        args.out_dir / "component_step_delta_by_regime.png",
    )
    plot_metric_by_component(
        diag,
        "mean_abs_z_move_delta_component_minus_original",
        "Mean abs z movement delta vs original",
        "Z movement: component - original",
        args.out_dir / "component_z_move_delta_by_regime.png",
    )
    plot_metric_by_component(
        diag,
        "mean_ode_component_cos",
        "Mean cosine",
        "Alignment of ODE-heavy shift and component shift",
        args.out_dir / "component_ode_alignment_by_regime.png",
    )

    print("wrote", args.out_dir)
    print(summary.to_string(index=False))
    print(diag[diag["regime"].eq("all")].to_string(index=False))


if __name__ == "__main__":
    main()
