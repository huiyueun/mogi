from __future__ import annotations

import argparse
import math
import os
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DT = 0.04


def ensure_open_dir(root: Path) -> Path:
    open_dir = root / "open"
    if (open_dir / "test").exists():
        return open_dir
    zip_path = root / "data" / "open.zip"
    if not zip_path.exists():
        raise FileNotFoundError("Expected open/ or data/open.zip")
    open_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(open_dir)
    return open_dir


def load_xyz(path: Path) -> np.ndarray:
    return pd.read_csv(path).sort_values("timestep_ms")[["x", "y", "z"]].to_numpy(np.float32)


def load_test(root: Path) -> tuple[list[str], np.ndarray]:
    data_dir = ensure_open_dir(root)
    paths = sorted((data_dir / "test").glob("TEST_*.csv"))
    ids = [p.stem for p in paths]
    x = np.stack([load_xyz(p) for p in paths])
    return ids, x


def load_pred(path: Path, ids: list[str]) -> np.ndarray:
    df = pd.read_csv(path).set_index("id")
    missing = [i for i in ids if i not in df.index]
    if missing:
        raise ValueError(f"{path} is missing ids, first={missing[:3]}")
    return df.loc[ids][["x", "y", "z"]].to_numpy(np.float32)


def norm(a: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.linalg.norm(a, axis=axis)


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = norm(a) * norm(b) + 1e-12
    cos = np.clip((a * b).sum(axis=1) / denom, -1.0, 1.0)
    return np.arccos(cos)


def trajectory_features(x: np.ndarray) -> pd.DataFrame:
    vel = np.gradient(x.astype(np.float64), DT, axis=1)
    speed = norm(vel)
    acc = np.gradient(vel, DT, axis=1)
    acc_mag = norm(acc)
    steps = norm(np.diff(x, axis=1))
    path = steps.sum(axis=1)
    net = norm(x[:, -1] - x[:, 0])
    straightness = net / (path + 1e-12)

    v_prev = vel[:, -2]
    v_last = vel[:, -1]
    recent_turn = angle_between(v_prev, v_last)

    v_norm = vel / (norm(vel, axis=2)[..., None] + 1e-12)
    cos_all = (v_norm[:, :-1] * v_norm[:, 1:]).sum(axis=2)
    hard_turn = np.arccos(np.clip(cos_all, -1.0, 1.0)).max(axis=1)

    t = np.arange(x.shape[1], dtype=np.float64)
    noise = np.zeros(len(x), dtype=np.float64)
    for i, w in enumerate(x.astype(np.float64)):
        axis_noise = []
        for d in range(3):
            coef = np.polyfit(t, w[:, d], 2)
            fit = np.polyval(coef, t)
            axis_noise.append(np.std(w[:, d] - fit))
        noise[i] = float(np.mean(axis_noise))

    vertical_change = np.abs(x[:, -1, 2] - x[:, 0, 2])
    recent_vertical_change = np.abs(x[:, -1, 2] - x[:, -3, 2])

    return pd.DataFrame(
        {
            "last_speed": speed[:, -1],
            "mean_speed": speed.mean(axis=1),
            "max_speed": speed.max(axis=1),
            "last_acc": acc_mag[:, -1],
            "mean_acc": acc_mag.mean(axis=1),
            "max_acc": acc_mag.max(axis=1),
            "recent_turn": recent_turn,
            "hard_turn": hard_turn,
            "noise": noise,
            "path": path,
            "straightness": straightness,
            "vertical_change": vertical_change,
            "recent_vertical_change": recent_vertical_change,
        }
    )


def add_regimes(feat: pd.DataFrame) -> pd.DataFrame:
    out = feat.copy()
    specs = {
        "high_speed": ("last_speed", 0.80),
        "hard_turn_regime": ("hard_turn", 0.80),
        "recent_turn_regime": ("recent_turn", 0.80),
        "high_acc": ("last_acc", 0.80),
        "high_noise": ("noise", 0.80),
        "vertical_change_regime": ("vertical_change", 0.80),
        "low_straightness": ("straightness", 0.20),
    }
    for name, (col, q) in specs.items():
        thr = out[col].quantile(q)
        if q >= 0.5:
            out[name] = out[col] >= thr
        else:
            out[name] = out[col] <= thr
        out[f"{name}_threshold"] = float(thr)
    out["all"] = True
    return out


def summarize_distance(name: str, dist: np.ndarray, vec: np.ndarray) -> dict[str, float | str | int]:
    return {
        "pair": name,
        "count": int(len(dist)),
        "mean": float(dist.mean()),
        "median": float(np.median(dist)),
        "p90": float(np.quantile(dist, 0.90)),
        "p95": float(np.quantile(dist, 0.95)),
        "p99": float(np.quantile(dist, 0.99)),
        "max": float(dist.max()),
        "over_1cm": int((dist > 0.01).sum()),
        "mean_dx": float(vec[:, 0].mean()),
        "mean_dy": float(vec[:, 1].mean()),
        "mean_dz": float(vec[:, 2].mean()),
        "mean_abs_dx": float(np.abs(vec[:, 0]).mean()),
        "mean_abs_dy": float(np.abs(vec[:, 1]).mean()),
        "mean_abs_dz": float(np.abs(vec[:, 2]).mean()),
    }


def pairwise_tables(ids: list[str], preds: dict[str, np.ndarray], regimes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_defs = [
        ("original_vs_second", "original", "second"),
        ("ode_heavy_vs_second", "ode_heavy", "second"),
        ("original_vs_ode_heavy", "original", "ode_heavy"),
        ("original_vs_final_blend", "original", "final_blend"),
        ("ode_heavy_vs_final_blend", "ode_heavy", "final_blend"),
    ]
    sample = pd.DataFrame({"id": ids})
    summary_rows = []
    regime_rows = []
    regime_cols = [
        "all",
        "high_speed",
        "hard_turn_regime",
        "recent_turn_regime",
        "high_acc",
        "high_noise",
        "vertical_change_regime",
        "low_straightness",
    ]

    for pair, left, right in pair_defs:
        vec = preds[right] - preds[left]
        dist = norm(vec)
        sample[f"{pair}_dist"] = dist
        sample[f"{pair}_dx"] = vec[:, 0]
        sample[f"{pair}_dy"] = vec[:, 1]
        sample[f"{pair}_dz"] = vec[:, 2]
        summary_rows.append(summarize_distance(pair, dist, vec))

        for reg in regime_cols:
            mask = regimes[reg].to_numpy(bool)
            row = summarize_distance(pair, dist[mask], vec[mask])
            row["regime"] = reg
            row["regime_share"] = float(mask.mean())
            regime_rows.append(row)

    return sample, pd.DataFrame(summary_rows), pd.DataFrame(regime_rows)


def plot_hist(sample: pd.DataFrame, out_dir: Path) -> None:
    cols = [
        "original_vs_second_dist",
        "ode_heavy_vs_second_dist",
        "original_vs_ode_heavy_dist",
        "original_vs_final_blend_dist",
    ]
    plt.figure(figsize=(9, 5))
    for col in cols:
        plt.hist(sample[col], bins=60, alpha=0.45, label=col.replace("_dist", ""))
    plt.xlabel("Prediction distance")
    plt.ylabel("Count")
    plt.title("Prediction distance distributions")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_distance_hist.png", dpi=160)
    plt.close()


def plot_regime_bars(regime_summary: pd.DataFrame, out_dir: Path) -> None:
    rows = regime_summary[regime_summary["pair"].eq("original_vs_second")].copy()
    rows = rows[~rows["regime"].eq("all")]
    plt.figure(figsize=(10, 5))
    plt.bar(rows["regime"], rows["mean"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Mean prediction distance")
    plt.title("Original vs second: distance by regime")
    plt.tight_layout()
    plt.savefig(out_dir / "original_vs_second_regime_mean_distance.png", dpi=160)
    plt.close()


def plot_direction_bars(regime_summary: pd.DataFrame, out_dir: Path) -> None:
    rows = regime_summary[regime_summary["pair"].eq("original_vs_second")].copy()
    rows = rows[rows["regime"].isin(["all", "high_speed", "hard_turn_regime", "high_noise", "vertical_change_regime"])]
    x = np.arange(len(rows))
    width = 0.25
    plt.figure(figsize=(10, 5))
    plt.bar(x - width, rows["mean_dx"], width, label="dx")
    plt.bar(x, rows["mean_dy"], width, label="dy")
    plt.bar(x + width, rows["mean_dz"], width, label="dz")
    plt.xticks(x, rows["regime"], rotation=25, ha="right")
    plt.ylabel("Mean second - original")
    plt.title("Original vs second: mean direction shift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "original_vs_second_direction_by_regime.png", dpi=160)
    plt.close()


def safe_unit(v: np.ndarray) -> np.ndarray:
    return v / (norm(v)[..., None] + 1e-12)


def angle_to_direction(v: np.ndarray, direction: np.ndarray) -> np.ndarray:
    denom = norm(v) * norm(direction) + 1e-12
    cos = np.clip((v * direction).sum(axis=1) / denom, -1.0, 1.0)
    return np.arccos(cos)


def behavior_diagnostics(x: np.ndarray, preds: dict[str, np.ndarray], regimes: pd.DataFrame) -> pd.DataFrame:
    last = x[:, -1].astype(np.float64)
    prev = x[:, -2].astype(np.float64)
    last_vel = last - prev
    last_dir = safe_unit(last_vel)
    cv = last + 2.0 * (last - prev)

    original_vec = preds["original"].astype(np.float64) - last
    second_vec = preds["second"].astype(np.float64) - last
    ode_vec_from_original = preds["ode_heavy"].astype(np.float64) - preds["original"].astype(np.float64)
    second_vec_from_original = preds["second"].astype(np.float64) - preds["original"].astype(np.float64)

    original_step = norm(original_vec)
    second_step = norm(second_vec)
    original_forward = (original_vec * last_dir).sum(axis=1)
    second_forward = (second_vec * last_dir).sum(axis=1)
    original_turn_angle = angle_to_direction(original_vec, last_vel)
    second_turn_angle = angle_to_direction(second_vec, last_vel)

    original_z_move = np.abs(preds["original"][:, 2].astype(np.float64) - last[:, 2])
    second_z_move = np.abs(preds["second"][:, 2].astype(np.float64) - last[:, 2])
    original_z_cv_resid = np.abs(preds["original"][:, 2].astype(np.float64) - cv[:, 2])
    second_z_cv_resid = np.abs(preds["second"][:, 2].astype(np.float64) - cv[:, 2])

    align_denom = norm(ode_vec_from_original) * norm(second_vec_from_original) + 1e-12
    ode_second_cos = (ode_vec_from_original * second_vec_from_original).sum(axis=1) / align_denom
    ode_second_cos = np.clip(ode_second_cos, -1.0, 1.0)

    regime_cols = [
        "all",
        "high_speed",
        "hard_turn_regime",
        "recent_turn_regime",
        "high_acc",
        "high_noise",
        "vertical_change_regime",
        "low_straightness",
    ]
    rows = []
    for reg in regime_cols:
        mask = regimes[reg].to_numpy(bool)
        rows.append(
            {
                "regime": reg,
                "count": int(mask.sum()),
                "second_longer_step_rate": float((second_step[mask] > original_step[mask]).mean()),
                "mean_step_original": float(original_step[mask].mean()),
                "mean_step_second": float(second_step[mask].mean()),
                "mean_step_delta_second_minus_original": float((second_step - original_step)[mask].mean()),
                "second_more_forward_rate": float((second_forward[mask] > original_forward[mask]).mean()),
                "mean_forward_delta_second_minus_original": float((second_forward - original_forward)[mask].mean()),
                "second_more_turn_rate": float((second_turn_angle[mask] > original_turn_angle[mask]).mean()),
                "mean_turn_angle_original_deg": float(np.degrees(original_turn_angle[mask]).mean()),
                "mean_turn_angle_second_deg": float(np.degrees(second_turn_angle[mask]).mean()),
                "mean_turn_angle_delta_deg": float(np.degrees(second_turn_angle - original_turn_angle)[mask].mean()),
                "second_smaller_z_move_rate": float((second_z_move[mask] < original_z_move[mask]).mean()),
                "mean_abs_z_move_original": float(original_z_move[mask].mean()),
                "mean_abs_z_move_second": float(second_z_move[mask].mean()),
                "mean_abs_z_move_delta_second_minus_original": float((second_z_move - original_z_move)[mask].mean()),
                "second_closer_to_cv_z_rate": float((second_z_cv_resid[mask] < original_z_cv_resid[mask]).mean()),
                "mean_abs_z_cv_resid_original": float(original_z_cv_resid[mask].mean()),
                "mean_abs_z_cv_resid_second": float(second_z_cv_resid[mask].mean()),
                "ode_second_same_direction_rate": float((ode_second_cos[mask] > 0.0).mean()),
                "ode_second_strong_same_direction_rate": float((ode_second_cos[mask] > 0.5).mean()),
                "mean_ode_second_cos": float(ode_second_cos[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_behavior_diagnostics(diag: pd.DataFrame, out_dir: Path) -> None:
    rows = diag[~diag["regime"].eq("all")].copy()
    x = np.arange(len(rows))

    plt.figure(figsize=(10, 5))
    plt.bar(x, rows["mean_turn_angle_delta_deg"])
    plt.axhline(0.0, color="black", lw=1)
    plt.xticks(x, rows["regime"], rotation=35, ha="right")
    plt.ylabel("Mean angle delta, second - original (deg)")
    plt.title("Does second turn more or less than original?")
    plt.tight_layout()
    plt.savefig(out_dir / "behavior_turn_angle_delta_by_regime.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x, rows["mean_step_delta_second_minus_original"])
    plt.axhline(0.0, color="black", lw=1)
    plt.xticks(x, rows["regime"], rotation=35, ha="right")
    plt.ylabel("Mean step length delta, second - original")
    plt.title("Does second predict farther or shorter?")
    plt.tight_layout()
    plt.savefig(out_dir / "behavior_step_delta_by_regime.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x, rows["mean_abs_z_move_delta_second_minus_original"])
    plt.axhline(0.0, color="black", lw=1)
    plt.xticks(x, rows["regime"], rotation=35, ha="right")
    plt.ylabel("Mean abs z movement delta, second - original")
    plt.title("Does second stabilize z movement?")
    plt.tight_layout()
    plt.savefig(out_dir / "behavior_z_move_delta_by_regime.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x, rows["mean_ode_second_cos"])
    plt.axhline(0.0, color="black", lw=1)
    plt.xticks(x, rows["regime"], rotation=35, ha="right")
    plt.ylabel("Mean cosine")
    plt.title("Are ODE-heavy and second shifts aligned from original?")
    plt.tight_layout()
    plt.savefig(out_dir / "behavior_ode_second_alignment_by_regime.png", dpi=160)
    plt.close()


def plot_case(case_id: str, traj: np.ndarray, points: dict[str, np.ndarray], out_path: Path) -> None:
    colors = {
        "original": "#1f77b4",
        "ode_heavy": "#ff7f0e",
        "second": "#2ca02c",
        "final_blend": "#d62728",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(traj[:, 0], traj[:, 1], "-o", ms=3, color="black", label="history")
    ax.scatter(traj[-1, 0], traj[-1, 1], s=50, color="black", marker="x", label="last")
    for name, p in points.items():
        ax.scatter(p[0], p[1], s=45, color=colors[name], label=name)
        ax.plot([traj[-1, 0], p[0]], [traj[-1, 1], p[1]], "--", lw=1, color=colors[name], alpha=0.7)
    ax.set_title(f"{case_id}: xy")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.legend(fontsize=7)

    ax = axes[1]
    steps = np.arange(len(traj))
    ax.plot(steps, traj[:, 2], "-o", ms=3, color="black", label="history z")
    next_step = len(traj) + 1
    for name, p in points.items():
        ax.scatter(next_step, p[2], s=45, color=colors[name], label=name)
    ax.set_title(f"{case_id}: z")
    ax.set_xlabel("step")
    ax.set_ylabel("z")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_top_plots(
    ids: list[str],
    x_test: np.ndarray,
    preds: dict[str, np.ndarray],
    sample: pd.DataFrame,
    out_dir: Path,
    top_n: int,
) -> pd.DataFrame:
    top = sample.sort_values("original_vs_second_dist", ascending=False).head(top_n).copy()
    plot_dir = out_dir / "top_case_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    id_to_idx = {case_id: i for i, case_id in enumerate(ids)}
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        case_id = row.id
        idx = id_to_idx[case_id]
        points = {name: pred[idx] for name, pred in preds.items()}
        plot_case(case_id, x_test[idx], points, plot_dir / f"{rank:03d}_{case_id}.png")
    return top


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--original", type=Path, default=Path("submission_GOH30.csv"))
    parser.add_argument("--ode-heavy", type=Path, default=Path("huiyu/submissions/goh30_component/case02_ode_heavy_g20_o60_h20.csv"))
    parser.add_argument("--second", type=Path, default=Path("huiyu/submissions/second_place_blends/second_place_restored.csv"))
    parser.add_argument("--final-blend", type=Path, default=Path("huiyu/submissions/second_place_blends/confirmed_ode_heavy_800_second200.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/top_expert_comparison"))
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()

    ids, x_test = load_test(args.root)
    preds = {
        "original": load_pred(args.original, ids),
        "ode_heavy": load_pred(args.ode_heavy, ids),
        "second": load_pred(args.second, ids),
        "final_blend": load_pred(args.final_blend, ids),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    feat = add_regimes(trajectory_features(x_test))
    feat.insert(0, "id", ids)
    sample, summary, regime_summary = pairwise_tables(ids, preds, feat)
    sample = sample.merge(feat, on="id", how="left")

    sample.to_csv(args.out_dir / "per_sample_differences.csv", index=False)
    summary.to_csv(args.out_dir / "pairwise_summary.csv", index=False)
    regime_summary.to_csv(args.out_dir / "pairwise_by_regime.csv", index=False)
    diag = behavior_diagnostics(x_test, preds, feat)
    diag.to_csv(args.out_dir / "behavior_diagnostics_by_regime.csv", index=False)

    top = make_top_plots(ids, x_test, preds, sample, args.out_dir, args.top_n)
    top.to_csv(args.out_dir / f"top{args.top_n}_original_vs_second.csv", index=False)

    plot_hist(sample, args.out_dir)
    plot_regime_bars(regime_summary, args.out_dir)
    plot_direction_bars(regime_summary, args.out_dir)
    plot_behavior_diagnostics(diag, args.out_dir)

    print("wrote", args.out_dir)
    print(summary.to_string(index=False))
    print(diag.to_string(index=False))


if __name__ == "__main__":
    main()
