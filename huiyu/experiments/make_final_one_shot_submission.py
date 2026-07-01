from __future__ import annotations

import argparse
import base64
import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


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
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "x": pred[:, 0], "y": pred[:, 1], "z": pred[:, 2]}).to_csv(path, index=False)
    print(f"wrote {path}")


def distance_row(name: str, pred: np.ndarray, ref: np.ndarray) -> dict[str, float | int | str]:
    d = np.linalg.norm(pred - ref, axis=1)
    return {
        "candidate": name,
        "mean_shift": float(d.mean()),
        "median_shift": float(np.median(d)),
        "p90_shift": float(np.quantile(d, 0.90)),
        "p95_shift": float(np.quantile(d, 0.95)),
        "p99_shift": float(np.quantile(d, 0.99)),
        "max_shift": float(d.max()),
        "n_shift_gt_1cm": int((d > 0.01).sum()),
        "n_shift_gt_5mm": int((d > 0.005).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-dir", type=Path, default=Path("outputs/goh30_component_submissions"))
    parser.add_argument("--notebook", type=Path, default=Path("best_solve/_[private 2nd] 코드 공유.ipynb"))
    parser.add_argument("--out-dir", type=Path, default=Path("huiyu/submissions/final_one_shot"))
    parser.add_argument("--filename", default="submission_ode80_second20_final.csv")
    args = parser.parse_args()

    ids = pd.read_csv(args.component_dir / "ids.csv")["id"].tolist()
    pred_g = np.load(args.component_dir / "pred_gru.npy")
    pred_o = np.load(args.component_dir / "pred_ode.npy")
    pred_h = np.load(args.component_dir / "pred_h.npy")

    ode_heavy = 0.20 * pred_g + 0.60 * pred_o + 0.20 * pred_h

    second_base = decode_pred(extract_b64(args.notebook, "_BASE_B64"))
    second_phys = decode_pred(extract_b64(args.notebook, "_PHYS_B64"))
    second_final = 0.60 * second_base + 0.40 * second_phys

    # Best confirmed submission so far:
    # public 0.7034 / private 0.7042
    final_pred = 0.80 * ode_heavy + 0.20 * second_final

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_submission(args.out_dir / args.filename, ids, final_pred.astype(np.float32))

    report = pd.DataFrame(
        [
            distance_row("second_final_vs_ode_heavy", second_final, ode_heavy),
            distance_row("final_vs_ode_heavy", final_pred, ode_heavy),
            distance_row("final_vs_second_final", final_pred, second_final),
        ]
    )
    report.to_csv(args.out_dir / "final_one_shot_shift_report.csv", index=False)
    print(report.to_string(index=False))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
