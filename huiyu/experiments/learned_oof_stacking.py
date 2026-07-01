from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from goh30_oof_lite import hit_rate, load_data, safe_norm
from search_cluster_moe import trajectory_features


def mix(g: np.ndarray, o: np.ndarray, h: np.ndarray, w: np.ndarray) -> np.ndarray:
    return w[:, [0]] * g + w[:, [1]] * o + w[:, [2]] * h


def write_submission(path: Path, ids: list[str], pred: np.ndarray) -> None:
    pd.DataFrame({"id": ids, "x": pred[:, 0], "y": pred[:, 1], "z": pred[:, 2]}).to_csv(path, index=False)
    print(f"wrote {path}")


def read_submission(path: Path) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path)
    return df["id"].tolist(), df[["x", "y", "z"]].to_numpy(np.float32)


def score_row(name: str, pred: np.ndarray, y: np.ndarray) -> dict[str, float | str]:
    err = safe_norm(pred - y)
    return {
        "candidate": name,
        "hit": hit_rate(pred, y),
        "mean_error": float(err.mean()),
        "median_error": float(np.median(err)),
    }


def simplex(step: float) -> list[tuple[float, float, float]]:
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    out = []
    for wg in vals:
        for wo in vals:
            wh = 1.0 - wg - wo
            if wh >= -1e-9:
                out.append((round(float(wg), 10), round(float(wo), 10), round(float(wh), 10)))
    return out


def global_best_weights(g: np.ndarray, o: np.ndarray, h: np.ndarray, y: np.ndarray, step: float) -> tuple[float, float, float]:
    best_w = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    best_s = -1.0
    for w in simplex(step):
        ww = np.tile(np.array(w, dtype=np.float32), (len(y), 1))
        s = hit_rate(mix(g, o, h, ww), y)
        if s > best_s:
            best_s = s
            best_w = w
    return best_w


def build_meta_features(
    x: np.ndarray,
    g: np.ndarray,
    o: np.ndarray,
    h: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    traj, names = trajectory_features(x)
    last = x[:, -1]
    preds = [g, o, h]
    pred_names = ["g", "o", "h"]
    parts = [traj]
    feat_names = list(names)

    for pred, name in zip(preds, pred_names):
        delta = pred - last
        dist = safe_norm(delta)
        parts.extend([delta, dist[:, None]])
        feat_names.extend([f"{name}_dx", f"{name}_dy", f"{name}_dz", f"{name}_dist_last"])

    pair_defs = [("go", g, o), ("gh", g, h), ("oh", o, h)]
    for name, a, b in pair_defs:
        d = a - b
        parts.extend([d, safe_norm(d)[:, None]])
        feat_names.extend([f"{name}_dx", f"{name}_dy", f"{name}_dz", f"{name}_dist"])

    return np.column_stack(parts).astype(np.float32), feat_names


def expert_labels(g: np.ndarray, o: np.ndarray, h: np.ndarray, y: np.ndarray) -> np.ndarray:
    errs = np.column_stack([safe_norm(g - y), safe_norm(o - y), safe_norm(h - y)])
    return np.argmin(errs, axis=1).astype(np.int64)


def aligned_proba(model: object, x: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)
    classes = getattr(model, "classes_")
    out = np.zeros((len(x), 3), dtype=np.float32)
    for j, cls in enumerate(classes):
        out[:, int(cls)] = raw[:, j]
    row_sum = out.sum(axis=1, keepdims=True)
    missing = row_sum[:, 0] <= 0
    out[missing] = 1.0 / 3.0
    out[~missing] /= row_sum[~missing]
    return out


def fit_model(kind: str, x: np.ndarray, y: np.ndarray, seed: int) -> object:
    if kind == "logreg":
        model = LogisticRegression(
            C=0.7,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )
    elif kind == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.15,
            random_state=seed,
        )
    else:
        raise ValueError(kind)
    return model.fit(x, y)


def stack_oof_weights(
    features: np.ndarray,
    labels: np.ndarray,
    folds: int,
    seed: int,
    kind: str,
) -> np.ndarray:
    proba = np.zeros((len(features), 3), dtype=np.float32)
    split = KFold(n_splits=folds, shuffle=True, random_state=seed).split(features)
    for tr, va in split:
        scaler = StandardScaler().fit(features[tr])
        model = fit_model(kind, scaler.transform(features[tr]), labels[tr], seed)
        proba[va] = aligned_proba(model, scaler.transform(features[va]))
    return proba


def train_full_weights(features: np.ndarray, labels: np.ndarray, test_features: np.ndarray, seed: int, kind: str) -> np.ndarray:
    scaler = StandardScaler().fit(features)
    model = fit_model(kind, scaler.transform(features), labels, seed)
    return aligned_proba(model, scaler.transform(test_features))


def evaluate_one_seed(
    root: Path,
    oof_dir: Path,
    component_dir: Path,
    out_dir: Path,
    stack_folds: int,
    seed: int,
    global_step: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[str]]:
    x, y, _, x_test, test_ids = load_data(root)
    og = np.load(oof_dir / "oof_gru.npy")
    oo = np.load(oof_dir / "oof_ode.npy")
    oh = np.load(oof_dir / "oof_h.npy")
    tg = np.load(component_dir / "pred_gru.npy")
    to = np.load(component_dir / "pred_ode.npy")
    th = np.load(component_dir / "pred_h.npy")

    meta_x, feat_names = build_meta_features(x, og, oo, oh)
    meta_test, _ = build_meta_features(x_test, tg, to, th)
    labels = expert_labels(og, oo, oh, y)
    global_w = np.array(global_best_weights(og, oo, oh, y, global_step), dtype=np.float32)
    prior_defs = {
        "global": global_w,
        "h85": np.array([0.10, 0.05, 0.85], dtype=np.float32),
        "h75": np.array([0.05, 0.20, 0.75], dtype=np.float32),
    }
    alphas = [0.25, 0.50, 0.75, 1.00]

    candidates: dict[str, np.ndarray] = {
        "equal": mix(og, oo, oh, np.tile(np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32), (len(y), 1))),
        "h_only": oh,
        "global_best": mix(og, oo, oh, np.tile(global_w, (len(y), 1))),
    }
    test_candidates: dict[str, np.ndarray] = {
        "h_only": th,
        "global_best": mix(tg, to, th, np.tile(global_w, (len(x_test), 1))),
    }
    weight_summaries = []

    for kind in ["logreg", "hgb"]:
        learned_w = stack_oof_weights(meta_x, labels, stack_folds, seed, kind)
        learned_test_w = train_full_weights(meta_x, labels, meta_test, seed, kind)
        for prior_name, prior in prior_defs.items():
            prior_train = np.tile(prior, (len(y), 1))
            prior_test = np.tile(prior, (len(x_test), 1))
            for alpha in alphas:
                name = f"{kind}_{prior_name}_a{int(alpha * 100):03d}"
                w = (1.0 - alpha) * prior_train + alpha * learned_w
                wt = (1.0 - alpha) * prior_test + alpha * learned_test_w
                w /= w.sum(axis=1, keepdims=True)
                wt /= wt.sum(axis=1, keepdims=True)
                candidates[name] = mix(og, oo, oh, w)
                test_candidates[name] = mix(tg, to, th, wt)
                weight_summaries.append(
                    {
                        "candidate": name,
                        "mean_wg": float(w[:, 0].mean()),
                        "mean_wo": float(w[:, 1].mean()),
                        "mean_wh": float(w[:, 2].mean()),
                        "std_wg": float(w[:, 0].std()),
                        "std_wo": float(w[:, 1].std()),
                        "std_wh": float(w[:, 2].std()),
                    }
                )

    summary = pd.DataFrame([score_row(name, pred, y) for name, pred in candidates.items()])
    summary = summary.sort_values(["hit", "mean_error"], ascending=[False, True]).reset_index(drop=True)
    summary.insert(0, "oof_dir", str(oof_dir))
    summary.to_csv(out_dir / f"{oof_dir.name}_learned_stacking_summary.csv", index=False)
    pd.DataFrame(weight_summaries).to_csv(out_dir / f"{oof_dir.name}_weight_summary.csv", index=False)
    pd.DataFrame({"feature": feat_names}).to_csv(out_dir / "meta_features.csv", index=False)

    top_names = summary.head(6)["candidate"].tolist()
    for name in top_names:
        if name in test_candidates:
            write_submission(out_dir / f"{oof_dir.name}_{name}.csv", test_ids, test_candidates[name])

    print(f"\n{oof_dir}")
    print(summary.head(12).to_string(index=False))
    return summary, test_candidates, test_ids


def shift_row(name: str, pred: np.ndarray, ref: np.ndarray) -> dict[str, float | str | int]:
    d = safe_norm(pred - ref)
    return {
        "candidate": name,
        "mean_shift": float(d.mean()),
        "p90_shift": float(np.quantile(d, 0.90)),
        "p99_shift": float(np.quantile(d, 0.99)),
        "max_shift": float(d.max()),
        "n_shift_gt_1cm": int((d > 0.01).sum()),
        "n_shift_gt_5mm": int((d > 0.005).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--oof-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--component-dir", type=Path, default=Path("outputs/goh30_component_submissions"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/goh30_learned_stacking"))
    parser.add_argument("--stack-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--global-step", type=float, default=0.05)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    test_by_seed: dict[str, dict[str, np.ndarray]] = {}
    ids_ref: list[str] | None = None

    for i, oof_dir in enumerate(args.oof_dirs):
        summary, test_candidates, ids = evaluate_one_seed(
            args.root,
            oof_dir,
            args.component_dir,
            args.out_dir,
            args.stack_folds,
            args.seed + i * 100,
            args.global_step,
        )
        summary = summary.copy()
        summary.insert(1, "seed_index", i)
        all_rows.append(summary)
        test_by_seed[oof_dir.name] = test_candidates
        if ids_ref is None:
            ids_ref = ids
        elif ids_ref != ids:
            raise ValueError("test id mismatch")

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(args.out_dir / "learned_stacking_all_summary.csv", index=False)

    stability = (
        combined.groupby("candidate", as_index=False)
        .agg(hit_mean=("hit", "mean"), hit_std=("hit", "std"), hit_min=("hit", "min"), hit_max=("hit", "max"), mean_error=("mean_error", "mean"))
        .sort_values(["hit_mean", "hit_min", "mean_error"], ascending=[False, False, True])
    )
    stability.to_csv(args.out_dir / "learned_stacking_stability.csv", index=False)
    print("\nStability across OOF dirs:")
    print(stability.head(15).to_string(index=False))

    if ids_ref is not None and len(test_by_seed) >= 2:
        equal = np.load(args.component_dir / "pred_equal.npy")
        stable_names = stability.head(5)["candidate"].tolist()
        shift_rows = []
        for name in stable_names:
            preds = [cands[name] for cands in test_by_seed.values() if name in cands]
            if len(preds) != len(test_by_seed):
                continue
            avg = sum(preds) / len(preds)
            out_name = f"avg_seeds_{name}"
            write_submission(args.out_dir / f"{out_name}.csv", ids_ref, avg)
            shift_rows.append(shift_row(out_name, avg, equal))
            if len(preds) == 2:
                shift_rows.append(shift_row(f"{name}_seed_pair_shift", preds[0], preds[1]))
        if shift_rows:
            pd.DataFrame(shift_rows).to_csv(args.out_dir / "learned_stacking_shift_report.csv", index=False)
            print("\nShift report:")
            print(pd.DataFrame(shift_rows).to_string(index=False))

    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
