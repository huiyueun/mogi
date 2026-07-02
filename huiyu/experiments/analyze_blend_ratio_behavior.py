from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_second_components import load_second_components
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


def build_blends(ode_heavy: np.ndarray, second_final: np.ndarray, ratios: list[float]) -> dict[str, np.ndarray]:
    return {
        f"second_{int(round(r * 100)):03d}": ((1.0 - r) * ode_heavy + r * second_final).astype(np.float32)
        for r in ratios
    }


def behavior_by_ratio(x: np.ndarray, ode_heavy: np.ndarray, second_final: np.ndarray, blends: dict[str, np.ndarray], regimes: pd.DataFrame) -> pd.DataFrame:
    last = x[:, -1].astype(np.float64)
    prev = x[:, -2].astype(np.float64)
    last_vel = last - prev
    last_dir = safe_unit(last_vel)
    cv = last + 2.0 * (last - prev)

    ode_vec = ode_heavy.astype(np.float64) - last
    second_vec = second_final.astype(np.float64) - last
    ode_step = norm(ode_vec)
    second_shift = second_final.astype(np.float64) - ode_heavy.astype(np.float64)
    second_from_ode_dist = norm(second_shift)
    second_turn_angle = angle_to_direction(second_vec, last_vel)
    ode_turn_angle = angle_to_direction(ode_vec, last_vel)
    ode_z_move = np.abs(ode_heavy[:, 2].astype(np.float64) - last[:, 2])
    ode_z_cv_resid = np.abs(ode_heavy[:, 2].astype(np.float64) - cv[:, 2])

    rows = []
    for name, pred in blends.items():
        ratio = int(name.rsplit("_", 1)[1]) / 100.0
        pred64 = pred.astype(np.float64)
        vec = pred64 - last
        blend_shift = pred64 - ode_heavy.astype(np.float64)
        step = norm(vec)
        forward = (vec * last_dir).sum(axis=1)
        turn_angle = angle_to_direction(vec, last_vel)
        z_move = np.abs(pred64[:, 2] - last[:, 2])
        z_cv_resid = np.abs(pred64[:, 2] - cv[:, 2])
        shift_dist = norm(blend_shift)
        shift_fraction = shift_dist / (second_from_ode_dist + 1e-12)

        for regime in REGIME_COLS:
            mask = regimes[regime].to_numpy(bool)
            rows.append(
                {
                    "candidate": name,
                    "second_ratio": ratio,
                    "regime": regime,
                    "count": int(mask.sum()),
                    "mean_shift_from_ode": float(shift_dist[mask].mean()),
                    "p95_shift_from_ode": float(np.quantile(shift_dist[mask], 0.95)),
                    "mean_shift_fraction_to_second": float(shift_fraction[mask].mean()),
                    "mean_step": float(step[mask].mean()),
                    "mean_step_delta_vs_ode": float((step - ode_step)[mask].mean()),
                    "mean_forward": float(forward[mask].mean()),
                    "mean_turn_angle_deg": float(np.degrees(turn_angle[mask]).mean()),
                    "mean_turn_delta_vs_ode_deg": float(np.degrees(turn_angle - ode_turn_angle)[mask].mean()),
                    "mean_turn_delta_vs_second_deg": float(np.degrees(turn_angle - second_turn_angle)[mask].mean()),
                    "mean_abs_z_move": float(z_move[mask].mean()),
                    "mean_abs_z_move_delta_vs_ode": float((z_move - ode_z_move)[mask].mean()),
                    "closer_to_cv_z_rate_vs_ode": float((z_cv_resid[mask] < ode_z_cv_resid[mask]).mean()),
                }
            )
    return pd.DataFrame(rows)


def pairwise_ratio_summary(blends: dict[str, np.ndarray], regimes: pd.DataFrame) -> pd.DataFrame:
    names = list(blends)
    rows = []
    for left, right in zip(names[:-1], names[1:]):
        vec = blends[right] - blends[left]
        dist = norm(vec)
        for regime in REGIME_COLS:
            mask = regimes[regime].to_numpy(bool)
            rows.append(
                {
                    "pair": f"{left}_to_{right}",
                    "regime": regime,
                    "count": int(mask.sum()),
                    "mean_delta": float(dist[mask].mean()),
                    "p95_delta": float(np.quantile(dist[mask], 0.95)),
                    "max_delta": float(dist[mask].max()),
                    "mean_dx": float(vec[mask, 0].mean()),
                    "mean_dy": float(vec[mask, 1].mean()),
                    "mean_dz": float(vec[mask, 2].mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_ratio_metric(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    rows = df[df["regime"].isin(["all", "hard_turn_regime", "high_noise", "vertical_change_regime", "low_straightness"])]
    plt.figure(figsize=(9, 5))
    for regime, grp in rows.groupby("regime"):
        grp = grp.sort_values("second_ratio")
        plt.plot(grp["second_ratio"], grp[metric], marker="o", label=regime)
    plt.axvline(0.50, color="black", lw=1, linestyle="--")
    plt.xlabel("second_final ratio")
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--ode-heavy", type=Path, default=Path("huiyu/submissions/goh30_component/case02_ode_heavy_g20_o60_h20.csv"))
    parser.add_argument("--notebook", type=Path, default=Path("best_solve/_[private 2nd] 코드 공유.ipynb"))
    parser.add_argument("--ratios", default="0.20,0.40,0.50,0.60,1.00")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/blend_ratio_behavior"))
    args = parser.parse_args()

    ratios = [float(x) for x in args.ratios.split(",")]
    ids, x_test = load_test(args.root)
    ode_heavy = load_pred(args.ode_heavy, ids)
    second_final = load_second_components(args.notebook)["second_final"]
    blends = build_blends(ode_heavy, second_final, ratios)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    regimes = add_regimes(trajectory_features(x_test))
    regimes.insert(0, "id", ids)
    regimes.to_csv(args.out_dir / "trajectory_regimes.csv", index=False)

    behavior = behavior_by_ratio(x_test, ode_heavy, second_final, blends, regimes)
    pairwise = pairwise_ratio_summary(blends, regimes)
    behavior.to_csv(args.out_dir / "blend_ratio_behavior_by_regime.csv", index=False)
    pairwise.to_csv(args.out_dir / "blend_ratio_pairwise_steps.csv", index=False)

    plot_ratio_metric(behavior, "mean_turn_delta_vs_ode_deg", "Mean turn angle delta vs ODE-heavy (deg)", args.out_dir / "ratio_turn_delta_vs_ode.png")
    plot_ratio_metric(behavior, "mean_abs_z_move_delta_vs_ode", "Mean abs z movement delta vs ODE-heavy", args.out_dir / "ratio_z_delta_vs_ode.png")
    plot_ratio_metric(behavior, "mean_shift_from_ode", "Mean prediction shift from ODE-heavy", args.out_dir / "ratio_shift_from_ode.png")
    plot_ratio_metric(behavior, "closer_to_cv_z_rate_vs_ode", "Closer to CV-z than ODE-heavy rate", args.out_dir / "ratio_cv_z_rate.png")

    print(behavior[behavior["regime"].eq("all")].to_string(index=False))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
