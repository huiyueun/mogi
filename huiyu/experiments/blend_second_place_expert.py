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
    pd.DataFrame({"id": ids, "x": pred[:, 0], "y": pred[:, 1], "z": pred[:, 2]}).to_csv(path, index=False)
    print(f"wrote {path}")


def shift_row(name: str, pred: np.ndarray, ref: np.ndarray) -> dict[str, float | int | str]:
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
    parser.add_argument("--notebook", type=Path, default=Path("best_solve/_[private 2nd] 코드 공유.ipynb"))
    parser.add_argument("--component-dir", type=Path, default=Path("outputs/goh30_component_submissions"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/second_place_expert_blends"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(args.component_dir / "ids.csv")["id"].tolist()
    pred_g = np.load(args.component_dir / "pred_gru.npy")
    pred_o = np.load(args.component_dir / "pred_ode.npy")
    pred_h = np.load(args.component_dir / "pred_h.npy")
    pred_equal = np.load(args.component_dir / "pred_equal.npy")
    confirmed = 0.20 * pred_g + 0.60 * pred_o + 0.20 * pred_h

    base = decode_pred(extract_b64(args.notebook, "_BASE_B64"))
    phys = decode_pred(extract_b64(args.notebook, "_PHYS_B64"))
    second = 0.60 * base + 0.40 * phys

    # The 2nd-place notebook writes ids as TEST_00001..TEST_10000, matching sample submission order.
    write_submission(args.out_dir / "second_place_restored.csv", ids, second)

    cases = {
        "confirmed_ode_heavy_975_second025": 0.975 * confirmed + 0.025 * second,
        "confirmed_ode_heavy_950_second050": 0.950 * confirmed + 0.050 * second,
        "confirmed_ode_heavy_900_second100": 0.900 * confirmed + 0.100 * second,
        "confirmed_ode_heavy_850_second150": 0.850 * confirmed + 0.150 * second,
        "confirmed_ode_heavy_800_second200": 0.800 * confirmed + 0.200 * second,
        "confirmed_ode_heavy_750_second250": 0.750 * confirmed + 0.250 * second,
        "confirmed_ode_heavy_700_second300": 0.700 * confirmed + 0.300 * second,
        "confirmed_ode_heavy_600_second400": 0.600 * confirmed + 0.400 * second,
        "confirmed_ode_heavy_500_second500": 0.500 * confirmed + 0.500 * second,
        "equal_950_second050": 0.950 * pred_equal + 0.050 * second,
        "equal_900_second100": 0.900 * pred_equal + 0.100 * second,
    }
    for name, pred in cases.items():
        write_submission(args.out_dir / f"{name}.csv", ids, pred.astype(np.float32))

    rows = [shift_row("second_place_vs_confirmed", second, confirmed)]
    rows += [shift_row(name + "_vs_confirmed", pred, confirmed) for name, pred in cases.items()]
    report = pd.DataFrame(rows)
    report.to_csv(args.out_dir / "second_place_shift_report.csv", index=False)
    print(report.to_string(index=False))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
