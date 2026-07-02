from __future__ import annotations

import argparse
import base64
import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from goh30_oof_lite import hit_rate, load_data, safe_norm
from learned_second_gating import write_submission


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


def load_second_components(notebook_path: Path) -> dict[str, np.ndarray]:
    base = decode_pred(extract_b64(notebook_path, "_BASE_B64"))
    phys = decode_pred(extract_b64(notebook_path, "_PHYS_B64"))
    final = 0.60 * base + 0.40 * phys
    return {
        "second_base": base,
        "second_phys": phys,
        "second_final": final.astype(np.float32),
    }


def pairwise_dist(experts: np.ndarray) -> np.ndarray:
    diff = experts[:, :, None, :] - experts[:, None, :, :]
    return np.linalg.norm(diff, axis=-1)


def consensus_weights(
    experts: np.ndarray,
    prior: np.ndarray,
    radius: float,
    gamma: float,
    kernel: str,
) -> tuple[np.ndarray, np.ndarray]:
    dist = pairwise_dist(experts)
    if kernel == "linear":
        affinity = np.clip(1.0 - dist / radius, 0.0, 1.0)
    elif kernel == "gaussian":
        affinity = np.exp(-0.5 * (dist / radius) ** 2)
    else:
        raise ValueError(kernel)
    density = (affinity * prior[None, None, :]).sum(axis=2)
    raw = prior[None, :] * np.power(np.maximum(density, 1e-8), gamma)
    weights = raw / raw.sum(axis=1, keepdims=True)
    return weights.astype(np.float32), density.astype(np.float32)


def blend(experts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (experts * weights[:, :, None]).sum(axis=1).astype(np.float32)


def mix_pred(base: np.ndarray, alt: np.ndarray, beta: float) -> np.ndarray:
    return ((1.0 - beta) * base + beta * alt).astype(np.float32)


def score(name: str, pred: np.ndarray, y: np.ndarray, weights: np.ndarray) -> dict[str, float | str]:
    err = safe_norm(pred - y)
    return {
        "candidate": name,
        "hit": hit_rate(pred, y),
        "mean_error": float(err.mean()),
        "median_error": float(np.median(err)),
        "mean_max_weight": float(weights.max(axis=1).mean()),
        "p95_max_weight": float(np.quantile(weights.max(axis=1), 0.95)),
        "mean_weight_entropy": float((-(weights * np.log(weights + 1e-12)).sum(axis=1)).mean()),
    }


def weight_summary(names: list[str], weights: np.ndarray) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(names):
        w = weights[:, j]
        rows.append(
            {
                "expert": name,
                "mean_weight": float(w.mean()),
                "std_weight": float(w.std()),
                "min_weight": float(w.min()),
                "p05_weight": float(np.quantile(w, 0.05)),
                "p50_weight": float(np.quantile(w, 0.50)),
                "p95_weight": float(np.quantile(w, 0.95)),
                "max_weight": float(w.max()),
            }
        )
    return pd.DataFrame(rows)


def build_oof_candidates(args: argparse.Namespace) -> pd.DataFrame:
    x, y, *_ = load_data(args.root)
    g = np.load(args.oof_dir / "oof_gru.npy")
    o = np.load(args.oof_dir / "oof_ode.npy")
    h = np.load(args.oof_dir / "oof_h.npy")
    sp = np.load(args.second_phys_dir / "oof_second_phys.npy")
    experts = np.stack([g, o, h, sp], axis=1).astype(np.float32)
    names = ["gru", "ode", "h", "second_phys"]

    # second_base OOF가 없으므로 최종 5-expert prior에서 base 30%를 제외하고 재정규화한다.
    prior = np.array([0.10, 0.30, 0.10, 0.20], dtype=np.float32)
    prior = prior / prior.sum()
    base_weights = np.tile(prior[None, :], (len(y), 1)).astype(np.float32)
    base_pred = blend(experts, base_weights)

    rows = [score("oof_proxy_fixed_prior", base_pred, y, base_weights)]
    for kernel in args.kernels:
        for radius in args.radii:
            for gamma in args.gammas:
                cw, _ = consensus_weights(experts, prior, radius, gamma, kernel)
                consensus_pred = blend(experts, cw)
                for beta in args.betas:
                    name = f"oof_{kernel}_r{int(radius * 1000):03d}_g{gamma:g}_b{int(beta * 100):03d}"
                    pred = mix_pred(base_pred, consensus_pred, beta)
                    used_w = (1.0 - beta) * base_weights + beta * cw
                    rows.append(score(name, pred, y, used_w))

    out = pd.DataFrame(rows).sort_values(["hit", "mean_error"], ascending=[False, True]).reset_index(drop=True)
    out.to_csv(args.out_dir / "consensus_distance_oof_proxy_summary.csv", index=False)
    return out


def build_test_candidates(args: argparse.Namespace) -> None:
    _, _, _, _, test_ids = load_data(args.root)
    pred_g = np.load(args.component_dir / "pred_gru.npy")
    pred_o = np.load(args.component_dir / "pred_ode.npy")
    pred_h = np.load(args.component_dir / "pred_h.npy")
    second = load_second_components(args.notebook)
    experts = np.stack(
        [pred_g, pred_o, pred_h, second["second_base"], second["second_phys"]],
        axis=1,
    ).astype(np.float32)
    names = ["gru", "ode", "h", "second_base", "second_phys"]
    prior = np.array([0.10, 0.30, 0.10, 0.30, 0.20], dtype=np.float32)
    base_weights = np.tile(prior[None, :], (len(test_ids), 1)).astype(np.float32)
    base_pred = blend(experts, base_weights)

    sub_dir = args.out_dir / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    write_submission(sub_dir / "fixed50_reference.csv", test_ids, base_pred)
    weight_summary(names, base_weights).to_csv(args.out_dir / "fixed50_reference_weight_summary.csv", index=False)

    rows = []
    all_weight_rows = []
    for kernel in args.kernels:
        for radius in args.radii:
            for gamma in args.gammas:
                cw, density = consensus_weights(experts, prior, radius, gamma, kernel)
                consensus_pred = blend(experts, cw)
                for beta in args.betas:
                    tag = f"{kernel}_r{int(radius * 1000):03d}_g{gamma:g}_b{int(beta * 100):03d}"
                    pred = mix_pred(base_pred, consensus_pred, beta)
                    used_w = (1.0 - beta) * base_weights + beta * cw
                    write_submission(sub_dir / f"consensus_{tag}.csv", test_ids, pred)
                    dist = safe_norm(pred - base_pred)
                    rows.append(
                        {
                            "candidate": f"consensus_{tag}",
                            "kernel": kernel,
                            "radius": radius,
                            "gamma": gamma,
                            "beta": beta,
                            "mean_shift_vs_fixed50": float(dist.mean()),
                            "p95_shift_vs_fixed50": float(np.quantile(dist, 0.95)),
                            "p99_shift_vs_fixed50": float(np.quantile(dist, 0.99)),
                            "max_shift_vs_fixed50": float(dist.max()),
                            "over_1cm_vs_fixed50": int((dist > 0.01).sum()),
                            "mean_max_weight": float(used_w.max(axis=1).mean()),
                            "p95_max_weight": float(np.quantile(used_w.max(axis=1), 0.95)),
                        }
                    )
                    ws = weight_summary(names, used_w)
                    ws.insert(0, "candidate", f"consensus_{tag}")
                    all_weight_rows.append(ws)
                np.save(args.out_dir / f"density_{kernel}_r{int(radius * 1000):03d}_g{gamma:g}.npy", density)

    pd.DataFrame(rows).sort_values("mean_shift_vs_fixed50").to_csv(
        args.out_dir / "consensus_distance_test_diagnostics.csv", index=False
    )
    pd.concat(all_weight_rows, ignore_index=True).to_csv(args.out_dir / "consensus_distance_weight_summary.csv", index=False)


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x]


def parse_str_list(text: str) -> list[str]:
    return [x for x in text.split(",") if x]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--oof-dir", type=Path, default=Path("outputs/goh30_oof_lite_gru_ode_h10_6_cuda"))
    parser.add_argument("--second-phys-dir", type=Path, default=Path("outputs/second_phys_oof_full"))
    parser.add_argument("--component-dir", type=Path, default=Path("outputs/goh30_component_submissions"))
    parser.add_argument("--notebook", type=Path, default=Path("best_solve/_[private 2nd] 코드 공유.ipynb"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/consensus_distance_blends"))
    parser.add_argument("--radii", type=parse_float_list, default=parse_float_list("0.005,0.0075,0.010,0.0125"))
    parser.add_argument("--gammas", type=parse_float_list, default=parse_float_list("0.5,1.0,1.5,2.0"))
    parser.add_argument("--betas", type=parse_float_list, default=parse_float_list("0.25,0.50,0.75,1.0"))
    parser.add_argument("--kernels", type=parse_str_list, default=parse_str_list("linear,gaussian"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    oof = build_oof_candidates(args)
    build_test_candidates(args)

    print("OOF proxy top:")
    print(oof.head(20).to_string(index=False))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
