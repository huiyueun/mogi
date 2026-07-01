from __future__ import annotations

import argparse
import base64
import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from goh30_oof_lite import hit_rate, load_data, safe_norm
from search_cluster_moe import trajectory_features


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


def mix(base: np.ndarray, alt: np.ndarray, w_alt: np.ndarray) -> np.ndarray:
    w = w_alt.astype(np.float32)[:, None]
    return (1.0 - w) * base + w * alt


def score_row(name: str, pred: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float | str]:
    err = safe_norm(pred - y)
    row: dict[str, float | str] = {
        "candidate": name,
        "hit": hit_rate(pred, y),
        "mean_error": float(err.mean()),
        "median_error": float(np.median(err)),
    }
    if weights is not None:
        row.update(
            {
                "mean_w_alt": float(weights.mean()),
                "std_w_alt": float(weights.std()),
                "min_w_alt": float(weights.min()),
                "max_w_alt": float(weights.max()),
            }
        )
    return row


def build_features(x: np.ndarray, base: np.ndarray, alt: np.ndarray) -> tuple[np.ndarray, list[str]]:
    traj, names = trajectory_features(x)
    last = x[:, -1].astype(np.float32)
    base_delta = base - last
    alt_delta = alt - last
    diff = alt - base
    parts = [
        traj,
        base_delta,
        safe_norm(base_delta)[:, None],
        alt_delta,
        safe_norm(alt_delta)[:, None],
        diff,
        safe_norm(diff)[:, None],
        np.abs(diff)[:, [2]],
    ]
    feat_names = (
        names
        + ["base_dx", "base_dy", "base_dz", "base_dist_last"]
        + ["alt_dx", "alt_dy", "alt_dz", "alt_dist_last"]
        + ["alt_base_dx", "alt_base_dy", "alt_base_dz", "alt_base_dist", "alt_base_abs_dz"]
    )
    return np.column_stack(parts).astype(np.float32), feat_names


def best_weight_labels(base: np.ndarray, alt: np.ndarray, y: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    errs = []
    for w in grid:
        errs.append(safe_norm(((1.0 - w) * base + w * alt) - y))
    err_mat = np.column_stack(errs)
    idx = np.argmin(err_mat, axis=1)
    return idx.astype(np.int64), grid[idx].astype(np.float32)


def reg_model(kind: str, seed: int) -> object:
    if kind == "ridge":
        return Ridge(alpha=3.0)
    if kind == "hgb_reg":
        return HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.10,
            random_state=seed,
        )
    raise ValueError(kind)


def cls_model(kind: str, seed: int) -> object:
    if kind == "logreg":
        return LogisticRegression(C=0.6, class_weight="balanced", max_iter=1200, random_state=seed)
    if kind == "hgb_cls":
        return HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.10,
            random_state=seed,
        )
    raise ValueError(kind)


def aligned_grid_proba(model: object, features: np.ndarray, n_classes: int) -> np.ndarray:
    raw = model.predict_proba(features)
    classes = getattr(model, "classes_")
    out = np.zeros((len(features), n_classes), dtype=np.float32)
    for j, cls in enumerate(classes):
        out[:, int(cls)] = raw[:, j]
    row_sum = out.sum(axis=1, keepdims=True)
    missing = row_sum[:, 0] <= 0
    out[missing] = 1.0 / n_classes
    out[~missing] /= row_sum[~missing]
    return out


def oof_weight_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    target_w: np.ndarray,
    grid: np.ndarray,
    folds: int,
    seed: int,
) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {
        "ridge": np.zeros(len(features), dtype=np.float32),
        "hgb_reg": np.zeros(len(features), dtype=np.float32),
        "logreg": np.zeros(len(features), dtype=np.float32),
        "hgb_cls": np.zeros(len(features), dtype=np.float32),
    }
    split = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, va in split.split(features):
        scaler = StandardScaler().fit(features[tr])
        xtr = scaler.transform(features[tr])
        xva = scaler.transform(features[va])
        for kind in ["ridge", "hgb_reg"]:
            model = reg_model(kind, seed).fit(xtr, target_w[tr])
            preds[kind][va] = model.predict(xva).astype(np.float32)
        for kind in ["logreg", "hgb_cls"]:
            model = cls_model(kind, seed).fit(xtr, labels[tr])
            proba = aligned_grid_proba(model, xva, len(grid))
            preds[kind][va] = (proba * grid[None, :]).sum(axis=1).astype(np.float32)
    return {k: np.clip(v, float(grid.min()), float(grid.max())) for k, v in preds.items()}


def full_weight_predictions(
    features: np.ndarray,
    test_features: np.ndarray,
    labels: np.ndarray,
    target_w: np.ndarray,
    grid: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    scaler = StandardScaler().fit(features)
    xtr = scaler.transform(features)
    xte = scaler.transform(test_features)
    preds: dict[str, np.ndarray] = {}
    for kind in ["ridge", "hgb_reg"]:
        model = reg_model(kind, seed).fit(xtr, target_w)
        preds[kind] = model.predict(xte).astype(np.float32)
    for kind in ["logreg", "hgb_cls"]:
        model = cls_model(kind, seed).fit(xtr, labels)
        proba = aligned_grid_proba(model, xte, len(grid))
        preds[kind] = (proba * grid[None, :]).sum(axis=1).astype(np.float32)
    return {k: np.clip(v, float(grid.min()), float(grid.max())) for k, v in preds.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--oof-dir", type=Path, default=Path("outputs/goh30_oof_lite_gru_ode_h10_6_cuda"))
    parser.add_argument("--second-phys-dir", type=Path, required=True)
    parser.add_argument("--component-dir", type=Path, default=Path("outputs/goh30_component_submissions"))
    parser.add_argument("--notebook", type=Path, default=Path("best_solve/_[private 2nd] 코드 공유.ipynb"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/learned_second_gating"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--grid", default="0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40")
    parser.add_argument("--test-alt", choices=["second_final", "second_phys"], default="second_final")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test limit for train/test rows.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid = np.array([float(x) for x in args.grid.split(",")], dtype=np.float32)

    x, y, _, x_test, test_ids = load_data(args.root)
    g = np.load(args.oof_dir / "oof_gru.npy")
    o = np.load(args.oof_dir / "oof_ode.npy")
    h = np.load(args.oof_dir / "oof_h.npy")
    base_oof = 0.20 * g + 0.60 * o + 0.20 * h
    alt_oof = np.load(args.second_phys_dir / "oof_second_phys.npy")
    if args.limit is not None:
        x = x[: args.limit]
        y = y[: args.limit]
        x_test = x_test[: args.limit]
        test_ids = test_ids[: args.limit]
        base_oof = base_oof[: args.limit]
        alt_oof = alt_oof[: args.limit]
    if len(alt_oof) != len(y):
        raise ValueError(f"second phys OOF length mismatch: {len(alt_oof)} != {len(y)}")

    pred_g = np.load(args.component_dir / "pred_gru.npy")
    pred_o = np.load(args.component_dir / "pred_ode.npy")
    pred_h = np.load(args.component_dir / "pred_h.npy")
    base_test = 0.20 * pred_g + 0.60 * pred_o + 0.20 * pred_h
    second_base = decode_pred(extract_b64(args.notebook, "_BASE_B64"))
    second_phys = decode_pred(extract_b64(args.notebook, "_PHYS_B64"))
    second_final = 0.60 * second_base + 0.40 * second_phys
    alt_test = second_final if args.test_alt == "second_final" else second_phys
    if args.limit is not None:
        base_test = base_test[: args.limit]
        alt_test = alt_test[: args.limit]

    features, feat_names = build_features(x, base_oof, alt_oof)
    test_features, _ = build_features(x_test, base_test, alt_test)
    labels, target_w = best_weight_labels(base_oof, alt_oof, y, grid)

    oof_weights = oof_weight_predictions(features, labels, target_w, grid, args.folds, args.seed)
    test_weights = full_weight_predictions(features, test_features, labels, target_w, grid, args.seed)

    candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    test_candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for w in grid:
        ww = np.full(len(y), w, dtype=np.float32)
        wt = np.full(len(x_test), w, dtype=np.float32)
        candidates[f"fixed_{int(w * 100):03d}"] = (mix(base_oof, alt_oof, ww), ww)
        test_candidates[f"fixed_{int(w * 100):03d}"] = (mix(base_test, alt_test, wt), wt)
    oracle_pred = mix(base_oof, alt_oof, target_w)
    candidates["oracle_grid"] = (oracle_pred, target_w)

    for kind, w in oof_weights.items():
        for alpha in [0.25, 0.50, 1.00]:
            name = f"{kind}_a{int(alpha * 100):03d}"
            ww = np.clip((1.0 - alpha) * 0.20 + alpha * w, float(grid.min()), float(grid.max()))
            candidates[name] = (mix(base_oof, alt_oof, ww), ww)
            wt = np.clip((1.0 - alpha) * 0.20 + alpha * test_weights[kind], float(grid.min()), float(grid.max()))
            test_candidates[name] = (mix(base_test, alt_test, wt), wt)

    summary = pd.DataFrame([score_row(name, pred, y, w) for name, (pred, w) in candidates.items()])
    summary = summary.sort_values(["hit", "mean_error"], ascending=[False, True]).reset_index(drop=True)
    summary.to_csv(args.out_dir / "learned_second_gating_oof_summary.csv", index=False)
    pd.DataFrame({"feature": feat_names}).to_csv(args.out_dir / "meta_features.csv", index=False)
    pd.DataFrame({"best_weight": target_w, "best_label": labels}).to_csv(args.out_dir / "oracle_weight_labels.csv", index=False)

    top_names = [n for n in summary["candidate"].head(8).tolist() if n in test_candidates]
    if "fixed_020" in test_candidates and "fixed_020" not in top_names:
        top_names.append("fixed_020")
    for name in top_names:
        pred, wt = test_candidates[name]
        write_submission(args.out_dir / f"{name}_{args.test_alt}.csv", test_ids, pred.astype(np.float32))
        pd.DataFrame({"id": test_ids, "weight_alt": wt}).to_csv(args.out_dir / f"{name}_{args.test_alt}_weights.csv", index=False)

    print(summary.head(20).to_string(index=False))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
