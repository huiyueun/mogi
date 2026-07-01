from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


DT = 0.04
PRED_DT = 0.08
R_HIT = 0.01
EPS = 1e-12


def ensure_open_dir(root: Path) -> Path:
    open_dir = root / "open"
    if (open_dir / "train").exists() and (open_dir / "test").exists():
        return open_dir

    zip_path = root / "data" / "open.zip"
    if not zip_path.exists():
        raise FileNotFoundError("Expected open/ or data/open.zip")

    open_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(open_dir)
    return open_dir


def load_xyz(path: Path) -> np.ndarray:
    return pd.read_csv(path).sort_values("timestep_ms")[["x", "y", "z"]].to_numpy(np.float64)


def load_data(root: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, list[str], pd.DataFrame]:
    data_dir = ensure_open_dir(root)
    train_files = sorted((data_dir / "train").glob("TRAIN_*.csv"))
    test_files = sorted((data_dir / "test").glob("TEST_*.csv"))
    labels = pd.read_csv(data_dir / "train_labels.csv").set_index("id")
    submission = pd.read_csv(data_dir / "sample_submission.csv")
    ids = [p.stem for p in train_files]
    test_ids = [p.stem for p in test_files]
    x = np.stack([load_xyz(p) for p in train_files], axis=0)
    x_test = np.stack([load_xyz(p) for p in test_files], axis=0)
    y = labels.loc[ids][["x", "y", "z"]].to_numpy(np.float64)
    return x, y, ids, x_test, test_ids, submission


def safe_norm(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    return np.linalg.norm(x, axis=axis, keepdims=keepdims)


def safe_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = safe_norm(a, axis=-1) * safe_norm(b, axis=-1) + EPS
    return np.clip(np.sum(a * b, axis=-1) / denom, -1.0, 1.0)


def xy_turn_cross(v_prev: np.ndarray, v_curr: np.ndarray) -> np.ndarray:
    return v_prev[..., 0] * v_curr[..., 1] - v_prev[..., 1] * v_curr[..., 0]


def cv_predict(x: np.ndarray) -> np.ndarray:
    return x[:, -1] + (PRED_DT / DT) * (x[:, -1] - x[:, -2])


def kalman_cv_predict(
    x: np.ndarray,
    sigma_obs: float = 0.30e-3,
    sigma_proc: float = 1.0,
    p0: float = 1.0,
) -> np.ndarray:
    n, t, _ = x.shape
    f = np.array([[1.0, DT], [0.0, 1.0]])
    f_pred = np.array([[1.0, PRED_DT], [0.0, 1.0]])
    q = sigma_proc**2 * np.array([[DT**4 / 4.0, DT**3 / 2.0], [DT**3 / 2.0, DT**2]])
    r = sigma_obs**2
    pred = np.zeros((n, 3), dtype=np.float64)

    for axis in range(3):
        z = x[:, :, axis]
        state = np.zeros((n, 2), dtype=np.float64)
        state[:, 0] = z[:, 0]
        cov = np.eye(2) * p0
        for step in range(1, t):
            state = state @ f.T
            cov = f @ cov @ f.T + q
            innovation = z[:, step] - state[:, 0]
            s = cov[0, 0] + r
            k = cov[:, 0] / s
            state = state + innovation[:, None] * k[None, :]
            cov = cov - np.outer(k, cov[0])
        pred[:, axis] = (state @ f_pred.T)[:, 0]
    return pred


def noise_features(x: np.ndarray) -> np.ndarray:
    t_obs = np.arange(x.shape[1], dtype=np.float64) * DT
    vand = np.vander(t_obs, 3, increasing=False)
    poly_noise = np.zeros(len(x), dtype=np.float64)
    for axis in range(3):
        coef = np.linalg.lstsq(vand, x[:, :, axis].T, rcond=None)[0]
        fit = (vand @ coef).T
        poly_noise += (x[:, :, axis] - fit).std(axis=1)
    poly_noise /= 3.0

    smooth = savgol_filter(x, window_length=5, polyorder=2, axis=1)
    savgol_noise = (x - smooth).std(axis=1).mean(axis=1)
    return np.column_stack([poly_noise, savgol_noise, np.log1p(poly_noise), np.log1p(savgol_noise)])


def base_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    d = np.diff(x, axis=1)
    v = d / DT
    a = np.diff(v, axis=1) / DT
    j = np.diff(a, axis=1) / DT

    speed = safe_norm(v)
    acc = safe_norm(a)
    jerk = safe_norm(j)
    v_last = v[:, -1]
    a_last = a[:, -1]
    a_recent = a[:, -3:].mean(axis=1)

    net_vec = x[:, -1] - x[:, 0]
    net_disp = safe_norm(net_vec)
    path_len = safe_norm(d).sum(axis=1)
    straightness = net_disp / (path_len + EPS)
    turn_cos = safe_cos(v_last, v[:, :-1].mean(axis=1))

    rel = x - x[:, -1:, :]
    rel_flat = rel.reshape(len(x), -1)

    cols = []
    parts = []

    def add(name: str, arr: np.ndarray) -> None:
        arr2 = np.asarray(arr)
        if arr2.ndim == 1:
            parts.append(arr2[:, None])
            cols.append(name)
        else:
            parts.append(arr2)
            cols.extend([f"{name}_{i}" for i in range(arr2.shape[1])])

    add("rel", rel_flat)
    add("mean_speed", speed.mean(axis=1))
    add("max_speed", speed.max(axis=1))
    add("speed_std", speed.std(axis=1))
    add("last_speed", safe_norm(v_last))
    add("mean_acc", acc.mean(axis=1))
    add("max_acc", acc.max(axis=1))
    add("last_acc", safe_norm(a_last))
    add("recent_acc", safe_norm(a_recent))
    add("max_jerk", jerk.max(axis=1))
    add("jerk_last", jerk[:, -1])
    add("jerk_recent", jerk[:, -3:].mean(axis=1))
    add("net_disp", net_disp)
    add("path_len", path_len)
    add("straightness", straightness)
    add("turn_cos", turn_cos)
    add("z_speed_last", v_last[:, 2])
    add("z_acc_last", a_last[:, 2])
    add("high_speed", (safe_norm(v_last) > 1.0).astype(float))
    add("high_acc", (acc.max(axis=1) > 15.0).astype(float))
    add("noise", noise_features(x))

    return np.hstack(parts), cols


def turn_phase_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    d = np.diff(x, axis=1)
    v = d / DT
    a = np.diff(v, axis=1) / DT
    j = np.diff(a, axis=1) / DT

    v_prev = v[:, :-1]
    v_next = v[:, 1:]
    cos_theta = safe_cos(v_next, v_prev)
    theta = np.arccos(cos_theta)
    theta_thr = 0.20
    turn_flags = theta > theta_thr
    any_turn = turn_flags.any(axis=1)
    first_turn = np.where(any_turn, np.argmax(turn_flags, axis=1), -1)
    last_turn_from_end = np.full(len(x), 99.0, dtype=np.float64)
    for i in np.where(any_turn)[0]:
        last_idx = np.where(turn_flags[i])[0][-1]
        last_turn_from_end[i] = (theta.shape[1] - 1) - last_idx

    v_last = v[:, -1]
    a_last = a[:, -1]
    j_last = j[:, -1]
    fwd = v_last / (safe_norm(v_last, keepdims=True) + EPS)

    a_parallel = np.sum(a_last * fwd, axis=1)
    a_perp_vec = a_last - a_parallel[:, None] * fwd
    a_perp = safe_norm(a_perp_vec)
    j_parallel = np.sum(j_last * fwd, axis=1)
    j_perp_vec = j_last - j_parallel[:, None] * fwd
    j_perp = safe_norm(j_perp_vec)

    cross_z_seq = xy_turn_cross(v_prev, v_next)
    cross_z_last = cross_z_seq[:, -1]
    cross_z_recent = cross_z_seq[:, -3:].mean(axis=1)
    turn_dir = np.sign(cross_z_last)
    lateral_acc_sign = np.sign(xy_turn_cross(v_last, a_last))
    acc_phase = np.sign(np.sum(v_last * a_last, axis=1))

    z_v = v[:, :, 2]
    z_a = a[:, :, 2]
    z_j = j[:, :, 2]
    z_flip_count = np.sum(np.diff(np.sign(z_v), axis=1) != 0, axis=1)

    feats = np.column_stack(
        [
            theta[:, -1],
            theta[:, -3:].mean(axis=1),
            theta[:, -3:].max(axis=1),
            theta.mean(axis=1),
            theta.std(axis=1),
            any_turn.astype(float),
            first_turn.astype(float),
            last_turn_from_end,
            cross_z_last,
            cross_z_recent,
            turn_dir,
            lateral_acc_sign,
            a_parallel,
            a_perp,
            j_parallel,
            j_perp,
            acc_phase,
            z_v[:, -1],
            z_a[:, -1],
            z_j[:, -1],
            z_flip_count.astype(float),
        ]
    )
    names = [
        "theta_last",
        "theta_recent_mean",
        "theta_recent_max",
        "theta_mean",
        "theta_std",
        "turn_any",
        "turn_first_idx",
        "turn_recentness",
        "cross_z_last",
        "cross_z_recent",
        "turn_dir",
        "lateral_acc_sign",
        "a_parallel",
        "a_perp",
        "j_parallel",
        "j_perp",
        "acc_phase",
        "z_v_last",
        "z_a_last",
        "z_j_last",
        "z_flip_count",
    ]
    return feats, names


def hit_rate(pred: np.ndarray, y: np.ndarray, radius: float = R_HIT) -> float:
    return float((safe_norm(pred - y) <= radius).mean())


def regime_masks(x: np.ndarray) -> dict[str, np.ndarray]:
    d = np.diff(x, axis=1)
    v = d / DT
    a = np.diff(v, axis=1) / DT
    speed = safe_norm(v)
    acc = safe_norm(a)
    theta = np.arccos(safe_cos(v[:, 1:], v[:, :-1]))
    noise = noise_features(x)
    return {
        "all": np.ones(len(x), dtype=bool),
        "hard_turn": theta[:, -3:].max(axis=1) > 0.20,
        "recent_turn": theta[:, -2:].max(axis=1) > 0.20,
        "high_acc": acc.max(axis=1) > 15.0,
        "minority_acc_last": safe_norm(a[:, -1]) >= 5.0,
        "high_speed": speed[:, -1] > 1.0,
        "vertical_change": np.abs(a[:, -1, 2]) > np.quantile(np.abs(a[:, -1, 2]), 0.75),
        "high_noise": noise[:, 1] > np.quantile(noise[:, 1], 0.75),
    }


def make_model(random_state: int) -> MultiOutputRegressor:
    base = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.035,
        max_iter=220,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        validation_fraction=0.12,
        n_iter_no_change=20,
        random_state=random_state,
    )
    return MultiOutputRegressor(base)


def evaluate_feature_set(
    x: np.ndarray,
    y: np.ndarray,
    features: np.ndarray,
    feature_name: str,
    base_pred: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    residual = y - base_pred
    oof_res = np.zeros_like(residual)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    for fold, (tr, va) in enumerate(kf.split(features)):
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(features[tr])
        x_va = scaler.transform(features[va])
        model = make_model(seed + fold)
        model.fit(x_tr, residual[tr])
        oof_res[va] = model.predict(x_va)

    pred = base_pred + oof_res
    errors = safe_norm(pred - y)
    metrics = {
        "feature_set": feature_name,
        "hit": hit_rate(pred, y),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "mean_error": float(errors.mean()),
        "median_error": float(np.median(errors)),
        "p90_error": float(np.quantile(errors, 0.90)),
    }
    return pred, metrics


def fit_predict_residual(
    train_features: np.ndarray,
    target_residual: np.ndarray,
    test_features: np.ndarray,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_features)
    x_test = scaler.transform(test_features)
    model = make_model(seed)
    model.fit(x_train, target_residual)
    return model.predict(x_test)


def regime_report(preds: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    masks = regime_masks(x)
    rows = []
    for regime, mask in masks.items():
        for name, pred in preds.items():
            if mask.sum() == 0:
                hit = np.nan
                mean_err = np.nan
            else:
                err = safe_norm(pred[mask] - y[mask])
                hit = float((err <= R_HIT).mean())
                mean_err = float(err.mean())
            rows.append(
                {
                    "regime": regime,
                    "n": int(mask.sum()),
                    "model": name,
                    "hit": hit,
                    "mean_error": mean_err,
                }
            )
    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    metrics: list[dict[str, object]],
    regimes: pd.DataFrame,
    base_cols: list[str],
    turn_cols: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    regimes.to_csv(out_dir / "regime_metrics.csv", index=False)
    with open(out_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump({"base": base_cols, "turn_phase": turn_cols}, f, indent=2)

    pivot = regimes.pivot(index=["regime", "n"], columns="model", values="hit").reset_index()

    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        out = df.copy()
        for col in out.columns:
            if pd.api.types.is_float_dtype(out[col]):
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.6f}")
        headers = [str(c) for c in out.columns]
        rows = [[str(v) for v in row] for row in out.to_numpy()]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    lines = [
        "# Turn / Phase Residual Experiment",
        "",
        "This experiment predicts the residual from a constant-velocity baseline.",
        "",
        "## Overall",
        "",
        markdown_table(metrics_df),
        "",
        "## Regime Hit@1cm",
        "",
        markdown_table(pivot),
        "",
        "## Feature Sets",
        "",
        f"- base feature count: {len(base_cols)}",
        f"- added turn/phase feature count: {len(turn_cols)}",
        "",
        "Outputs:",
        "",
        "- `metrics.csv`",
        "- `regime_metrics.csv`",
        "- `feature_columns.json`",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/turn_phase_experiment"))
    parser.add_argument("--make-submission", action="store_true")
    parser.add_argument("--submission-name", default="submission_turn_phase_kalman_gate.csv")
    args = parser.parse_args()

    x, y, _, x_test, test_ids, sample_submission = load_data(args.root)
    cv = cv_predict(x)
    kalman = kalman_cv_predict(x)

    base_x, base_cols = base_features(x)
    turn_x, turn_cols = turn_phase_features(x)
    enhanced_x = np.hstack([base_x, turn_x])

    preds = {
        "constant_velocity": cv,
        "kalman_cv": kalman,
    }
    metrics = [
        {
            "feature_set": "constant_velocity",
            "hit": hit_rate(cv, y),
            "rmse": float(math.sqrt(mean_squared_error(y, cv))),
            "mean_error": float(safe_norm(cv - y).mean()),
            "median_error": float(np.median(safe_norm(cv - y))),
            "p90_error": float(np.quantile(safe_norm(cv - y), 0.90)),
        },
        {
            "feature_set": "kalman_cv",
            "hit": hit_rate(kalman, y),
            "rmse": float(math.sqrt(mean_squared_error(y, kalman))),
            "mean_error": float(safe_norm(kalman - y).mean()),
            "median_error": float(np.median(safe_norm(kalman - y))),
            "p90_error": float(np.quantile(safe_norm(kalman - y), 0.90)),
        },
    ]

    base_pred, base_metrics = evaluate_feature_set(
        x, y, base_x, "cv_residual_base_features", cv, args.folds, args.seed
    )
    enhanced_pred, enhanced_metrics = evaluate_feature_set(
        x, y, enhanced_x, "cv_residual_base_plus_turn_phase", cv, args.folds, args.seed
    )
    kalman_enhanced_pred, kalman_enhanced_metrics = evaluate_feature_set(
        x,
        y,
        enhanced_x,
        "kalman_residual_base_plus_turn_phase",
        kalman,
        args.folds,
        args.seed,
    )
    preds["cv_residual_base_features"] = base_pred
    preds["cv_residual_base_plus_turn_phase"] = enhanced_pred
    preds["kalman_residual_base_plus_turn_phase"] = kalman_enhanced_pred
    metrics.extend([base_metrics, enhanced_metrics, kalman_enhanced_metrics])

    masks = regime_masks(x)
    gated_high_speed = enhanced_pred.copy()
    gated_high_speed[masks["high_speed"]] = kalman[masks["high_speed"]]
    gated_high_speed_noise = enhanced_pred.copy()
    gate_mask = masks["high_speed"] | masks["high_noise"]
    gated_high_speed_noise[gate_mask] = kalman[gate_mask]
    for name, pred in [
        ("turn_phase_else_kalman_high_speed", gated_high_speed),
        ("turn_phase_else_kalman_high_speed_or_noise", gated_high_speed_noise),
    ]:
        err = safe_norm(pred - y)
        preds[name] = pred
        metrics.append(
            {
                "feature_set": name,
                "hit": hit_rate(pred, y),
                "rmse": float(math.sqrt(mean_squared_error(y, pred))),
                "mean_error": float(err.mean()),
                "median_error": float(np.median(err)),
                "p90_error": float(np.quantile(err, 0.90)),
            }
        )

    blend_grid = np.linspace(0.0, 1.0, 21)
    best_blend = None
    for w in blend_grid:
        blended = w * enhanced_pred + (1.0 - w) * kalman_enhanced_pred
        score = hit_rate(blended, y)
        if best_blend is None or score > best_blend[0]:
            best_blend = (score, float(w), blended)
    assert best_blend is not None
    blend_score, blend_w, blend_pred = best_blend
    blend_err = safe_norm(blend_pred - y)
    blend_name = f"blend_cv_turn_kalman_turn_w{blend_w:.2f}"
    preds[blend_name] = blend_pred
    metrics.append(
        {
            "feature_set": blend_name,
            "hit": blend_score,
            "rmse": float(math.sqrt(mean_squared_error(y, blend_pred))),
            "mean_error": float(blend_err.mean()),
            "median_error": float(np.median(blend_err)),
            "p90_error": float(np.quantile(blend_err, 0.90)),
        }
    )
    blend_gated = blend_pred.copy()
    blend_gated[masks["high_speed"]] = kalman[masks["high_speed"]]
    blend_gated_err = safe_norm(blend_gated - y)
    blend_gated_name = f"{blend_name}_else_kalman_high_speed"
    preds[blend_gated_name] = blend_gated
    metrics.append(
        {
            "feature_set": blend_gated_name,
            "hit": hit_rate(blend_gated, y),
            "rmse": float(math.sqrt(mean_squared_error(y, blend_gated))),
            "mean_error": float(blend_gated_err.mean()),
            "median_error": float(np.median(blend_gated_err)),
            "p90_error": float(np.quantile(blend_gated_err, 0.90)),
        }
    )

    regimes = regime_report(preds, x, y)
    write_report(args.out_dir, metrics, regimes, base_cols, turn_cols)

    if args.make_submission:
        test_cv = cv_predict(x_test)
        test_kalman = kalman_cv_predict(x_test)
        test_base_x, _ = base_features(x_test)
        test_turn_x, _ = turn_phase_features(x_test)
        test_enhanced_x = np.hstack([test_base_x, test_turn_x])
        full_residual = fit_predict_residual(enhanced_x, y - cv, test_enhanced_x, args.seed)
        test_pred_cv_res = test_cv + full_residual
        test_kalman_residual = fit_predict_residual(
            enhanced_x, y - kalman, test_enhanced_x, args.seed + 100
        )
        test_pred_kalman_res = test_kalman + test_kalman_residual

        test_masks = regime_masks(x_test)
        metric_df = pd.DataFrame(metrics)
        candidate_names = [
            "turn_phase_else_kalman_high_speed",
            blend_gated_name,
            "kalman_residual_base_plus_turn_phase",
            "cv_residual_base_plus_turn_phase",
        ]
        best_name = str(
            metric_df[metric_df["feature_set"].isin(candidate_names)]
            .sort_values("hit", ascending=False)
            .iloc[0]["feature_set"]
        )
        if best_name == "turn_phase_else_kalman_high_speed":
            test_pred = test_pred_cv_res
            test_pred[test_masks["high_speed"]] = test_kalman[test_masks["high_speed"]]
        elif best_name == blend_gated_name:
            blend_w_for_test = float(blend_name.rsplit("w", 1)[1])
            test_pred = blend_w_for_test * test_pred_cv_res + (1.0 - blend_w_for_test) * test_pred_kalman_res
            test_pred[test_masks["high_speed"]] = test_kalman[test_masks["high_speed"]]
        elif best_name == "kalman_residual_base_plus_turn_phase":
            test_pred = test_pred_kalman_res
        else:
            test_pred = test_pred_cv_res
        print(f"Selected submission model by OOF: {best_name}")

        sub = sample_submission.copy()
        if list(sub["id"]) != test_ids:
            sub = sub.set_index("id").loc[test_ids].reset_index()
        sub[["x", "y", "z"]] = test_pred
        sub_path = args.out_dir / args.submission_name
        sub.to_csv(sub_path, index=False)
        print(f"Wrote submission {sub_path}")

    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"\nWrote {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
