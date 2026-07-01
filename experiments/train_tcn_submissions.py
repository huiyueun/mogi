from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from goh30_oof_lite import TCNModel, build_cache, build_eval_features, load_data, make_stats, predict_model, set_seed, train_model


def write_submission(path: Path, ids: list[str], pred: np.ndarray) -> None:
    pd.DataFrame({"id": ids, "x": pred[:, 0], "y": pred[:, 1], "z": pred[:, 2]}).to_csv(path, index=False)
    print(f"wrote {path}")


def load_component(component_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = pd.read_csv(component_dir / "ids.csv")["id"].tolist()
    pred_g = np.load(component_dir / "pred_gru.npy")
    pred_o = np.load(component_dir / "pred_ode.npy")
    pred_h = np.load(component_dir / "pred_h.npy")
    pred_equal = np.load(component_dir / "pred_equal.npy")
    return ids, pred_g, pred_o, pred_h, pred_equal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--component-dir", type=Path, default=Path("outputs/goh30_component_submissions"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/goh30_tcn_submissions"))
    parser.add_argument("--models-dir", type=Path, default=Path("models_tcn"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=3000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-interiors", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device={device} epochs={args.epochs} seeds={args.seeds}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    x, y, _, x_test, test_ids = load_data(args.root)
    component_ids, pred_g, pred_o, pred_h, pred_equal = load_component(args.component_dir)
    if component_ids != test_ids:
        raise ValueError("component ids do not match test ids")

    stats = make_stats(x, np.arange(len(x)))
    cache = build_cache(x, y, np.arange(len(x)), stats, use_interiors=not args.no_interiors)
    test_feats = build_eval_features(x_test, np.arange(len(x_test)), stats)
    scal_dim = cache["scal"].shape[1]

    preds = []
    for k in range(args.seeds):
        seed = args.seed_offset + k
        model_path = args.models_dir / f"tcn_full_{k}.pt"
        set_seed(seed)
        if model_path.exists():
            model = TCNModel(scal_dim=scal_dim).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)["model_state"])
            model.eval()
            print(f"loaded {model_path}", flush=True)
        else:
            print(f"train TCN {k}/{args.seeds - 1}: seed={seed}", flush=True)
            model = train_model(cache, lambda: TCNModel(scal_dim=scal_dim), args.epochs, seed, device)
            torch.save({"model_state": model.state_dict(), "seed": seed, "epochs": args.epochs}, model_path)
            print(f"saved {model_path}", flush=True)
        preds.append(predict_model(model, test_feats, device))

    pred_tcn = np.mean(preds, axis=0).astype(np.float32)
    np.save(args.out_dir / "pred_tcn.npy", pred_tcn)
    write_submission(args.out_dir / "case00_tcn_only.csv", test_ids, pred_tcn)

    ode_heavy_g20_o60_h20 = 0.20 * pred_g + 0.60 * pred_o + 0.20 * pred_h
    ode_heavy_g15_o70_h15 = 0.15 * pred_g + 0.70 * pred_o + 0.15 * pred_h

    cases = {
        "case01_confirmed_ode_heavy_95_tcn05": 0.95 * ode_heavy_g20_o60_h20 + 0.05 * pred_tcn,
        "case02_confirmed_ode_heavy_90_tcn10": 0.90 * ode_heavy_g20_o60_h20 + 0.10 * pred_tcn,
        "case03_g15_o70_h15_95_tcn05": 0.95 * ode_heavy_g15_o70_h15 + 0.05 * pred_tcn,
        "case04_g15_o70_h15_90_tcn10": 0.90 * ode_heavy_g15_o70_h15 + 0.10 * pred_tcn,
        "case05_ode_95_tcn05": 0.95 * pred_o + 0.05 * pred_tcn,
        "case06_ode_90_tcn10": 0.90 * pred_o + 0.10 * pred_tcn,
        "case07_equal_95_tcn05": 0.95 * pred_equal + 0.05 * pred_tcn,
    }
    for name, pred in cases.items():
        write_submission(args.out_dir / f"{name}.csv", test_ids, pred.astype(np.float32))

    meta = {
        "epochs": args.epochs,
        "seeds": args.seeds,
        "seed_offset": args.seed_offset,
        "device": str(device),
        "no_interiors": args.no_interiors,
        "note": "TCN expert trained on full train and blended conservatively with existing GOH30 components.",
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
