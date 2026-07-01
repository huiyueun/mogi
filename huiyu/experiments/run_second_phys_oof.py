from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


NOTEBOOK_CELLS = [2, 3, 5, 6, 15, 19, 20, 21]
SPECS = [
    ("xy2", 42, "dirnet"),
    ("xy2s1", 1, "dirnet"),
    ("xy2h3", 42, "3step"),
]


def exec_notebook_cells(notebook: Path, root: Path) -> dict[str, object]:
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    ns: dict[str, object] = {
        "__name__": "__second_phys_oof__",
        "tqdm": lambda iterable=None, *args, **kwargs: iterable if iterable is not None else [],
    }
    for idx in NOTEBOOK_CELLS:
        src = "".join(nb["cells"][idx].get("source", []))
        src = src.replace("from tqdm.auto import tqdm\n", "")
        exec(compile(src, f"{notebook}:cell{idx}", "exec"), ns)

    # The shared notebook searches for data/ by default, but this repo keeps the
    # competition files under open/. Patch the paths after setup execution.
    ns["ROOT"] = root
    ns["DATA_DIR"] = root / "open"
    ns["CACHE_DIR"] = root / "data" / "cache"
    ns["CACHE_DIR"].mkdir(parents=True, exist_ok=True)
    return ns


def hit_rate(pred: np.ndarray, y: np.ndarray) -> float:
    return float((np.linalg.norm(pred - y, axis=1) <= 0.01).mean())


def train_phys_oof(
    ns: dict[str, object],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    folds: int,
    epochs: int,
    min_win: int,
    aug: str,
    device: str,
    spec_limit: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    seed_everything = ns["seed_everything"]
    train_fold = ns["train_fold"]
    predict_full = ns["predict_full"]
    torch = ns["torch"]

    specs = SPECS[: spec_limit or len(SPECS)]
    targets_ext = [4, 5, 6, 7, 8, 9, 10, 12] if aug == "full" else [6, 7, 8, 9, 10, 12]
    dev = torch.device(device)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    spec_oofs = []
    spec_tests = []
    rows = []
    for spec_name, spec_seed, heading in specs:
        seed_everything(spec_seed)
        oof = np.zeros((len(x_train), 3), dtype=np.float32)
        test_preds = []
        for fold, (tr, va) in enumerate(kf.split(np.arange(len(x_train))), start=1):
            print(
                f"spec={spec_name} seed={spec_seed} heading={heading} "
                f"fold={fold}/{folds} train={len(tr)} valid={len(va)}",
                flush=True,
            )
            model = train_fold(
                x_train[tr],
                y_train[tr],
                epochs,
                min_win,
                targets_ext,
                dev,
                use_dirnet=(heading == "dirnet"),
            ).eval()
            pred_va = predict_full(model, x_train[va], dev).astype(np.float32)
            pred_te = predict_full(model, x_test, dev).astype(np.float32)
            oof[va] = pred_va
            test_preds.append(pred_te)
            rows.append(
                {
                    "spec": spec_name,
                    "seed": spec_seed,
                    "heading": heading,
                    "fold": fold,
                    "hit": hit_rate(pred_va, y_train[va]),
                }
            )
        test_mean = np.mean(test_preds, axis=0).astype(np.float32)
        rows.append(
            {
                "spec": spec_name,
                "seed": spec_seed,
                "heading": heading,
                "fold": "all",
                "hit": hit_rate(oof, y_train),
            }
        )
        spec_oofs.append(oof)
        spec_tests.append(test_mean)

    oof_ens = np.mean(spec_oofs, axis=0).astype(np.float32)
    test_ens = np.mean(spec_tests, axis=0).astype(np.float32)
    rows.append({"spec": "ensemble", "seed": -1, "heading": "avg", "fold": "all", "hit": hit_rate(oof_ens, y_train)})
    return oof_ens, test_ens, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--notebook", type=Path, default=Path("best_solve/_[private 2nd] 코드 공유.ipynb"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/second_phys_oof"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--min-win", type=int, default=5)
    parser.add_argument("--aug", choices=["reduced", "full"], default="reduced")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--spec-limit", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test limit for train/test rows.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ns = exec_notebook_cells(args.notebook, args.root.resolve())
    x_train, x_test, y_train, sub = ns["load_data"]()
    test_ids = sub["id"].tolist()
    train_ids = [f"TRAIN_{i:05d}" for i in range(1, len(x_train) + 1)]

    if args.limit is not None:
        x_train = x_train[: args.limit]
        y_train = y_train[: args.limit]
        train_ids = train_ids[: args.limit]
        x_test = x_test[: args.limit]
        test_ids = test_ids[: args.limit]

    oof, test_pred, metrics = train_phys_oof(
        ns,
        x_train,
        y_train,
        x_test,
        args.folds,
        args.epochs,
        args.min_win,
        args.aug,
        args.device,
        args.spec_limit,
        args.seed,
    )
    np.save(args.out_dir / "oof_second_phys.npy", oof)
    np.save(args.out_dir / "pred_second_phys.npy", test_pred)
    pd.DataFrame({"id": train_ids}).to_csv(args.out_dir / "train_ids.csv", index=False)
    pd.DataFrame({"id": test_ids}).to_csv(args.out_dir / "test_ids.csv", index=False)
    metrics.to_csv(args.out_dir / "second_phys_oof_metrics.csv", index=False)
    print(metrics.to_string(index=False))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
