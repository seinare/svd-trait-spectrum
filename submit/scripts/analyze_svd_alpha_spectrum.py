#!/usr/bin/env python3
"""Compute singular-value alpha-transform statistics for MLP up/down weights.

This script reads safetensors shards directly and does not save perturbed
weights. It computes base singular values once per selected matrix, then
analytically applies the Matthew alpha transform to the singular values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


ALPHAS = [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_label", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--alphas", default=",".join(str(x) for x in ALPHAS))
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def resolve_model_dir(path: Path) -> Path:
    if (path / "model.safetensors.index.json").exists() or list(path.glob("*.safetensors")):
        return path
    snapshots = path / "snapshots"
    if snapshots.exists():
        dirs = sorted([p for p in snapshots.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        for d in dirs:
            if (d / "model.safetensors.index.json").exists() or list(d.glob("*.safetensors")):
                return d
    raise FileNotFoundError(f"No safetensors model found under {path}")


def tensor_files(model_dir: Path) -> dict[str, Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text())
        return {name: model_dir / fname for name, fname in data["weight_map"].items()}
    out = {}
    for sf in sorted(model_dir.glob("*.safetensors")):
        with safe_open(sf, framework="pt", device="cpu") as f:
            for key in f.keys():
                out[key] = sf
    return out


DENSE_RE = re.compile(r"(?:model\.)?layers\.(\d+)\.mlp\.(up_proj|down_proj)\.weight$")
EXPERT_RE = re.compile(r"(?:model\.)?layers\.(\d+)\.mlp\.experts\.(\d+)\.(up_proj|down_proj)\.weight$")
SHARED_RE = re.compile(r"(?:model\.)?layers\.(\d+)\.mlp\.shared_expert\.(up_proj|down_proj)\.weight$")


def selected_weights(weight_map: dict[str, Path]):
    rows = []
    for name, path in weight_map.items():
        m = EXPERT_RE.search(name)
        if m:
            rows.append(
                {
                    "name": name,
                    "path": path,
                    "layer": int(m.group(1)),
                    "projection": m.group(3),
                    "module_kind": "moe_expert",
                    "expert": int(m.group(2)),
                }
            )
            continue
        m = SHARED_RE.search(name)
        if m:
            rows.append(
                {
                    "name": name,
                    "path": path,
                    "layer": int(m.group(1)),
                    "projection": m.group(2),
                    "module_kind": "moe_shared",
                    "expert": "",
                }
            )
            continue
        m = DENSE_RE.search(name)
        if m:
            rows.append(
                {
                    "name": name,
                    "path": path,
                    "layer": int(m.group(1)),
                    "projection": m.group(2),
                    "module_kind": "dense",
                    "expert": "",
                }
            )
    return sorted(rows, key=lambda r: (r["layer"], r["projection"], str(r["expert"]), r["name"]))


def load_tensor(path: Path, name: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def gini(values: torch.Tensor) -> float:
    x = values.detach().float().flatten()
    n = x.numel()
    if n == 0:
        return float("nan")
    x, _ = torch.sort(x)
    total = torch.sum(x)
    if float(total) <= 0.0:
        return 0.0
    idx = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    return float(torch.sum((2 * idx - n - 1) * x) / (n * total))


def svd_values(weight: torch.Tensor, device: str) -> torch.Tensor:
    w = weight.float()
    if device == "cuda" and torch.cuda.is_available():
        w = w.cuda()
    try:
        s = torch.linalg.svdvals(w)
    except RuntimeError:
        s = torch.linalg.svdvals(w.cpu())
    return s.detach().cpu().float()


def alpha_stats(s: torch.Tensor, alpha: float) -> dict[str, float]:
    s = torch.clamp(s.float(), min=1e-12)
    log_s = torch.log(s)
    mean_log = torch.mean(log_s)
    centered = log_s - mean_log
    s_alpha = torch.exp(log_s + alpha * centered)
    top_base = float(torch.max(s))
    top_alpha = float(torch.max(s_alpha))
    geom = float(torch.exp(mean_log))
    return {
        "gini_alpha": gini(s_alpha),
        "delta_gini": gini(s_alpha) - gini(s),
        "top_sv_alpha": top_alpha,
        "top_sv_rel": top_alpha / top_base if top_base else float("nan"),
        "top_sv_abs_delta": top_alpha - top_base,
        "top_to_geomean_base": top_base / geom if geom else float("nan"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def maxf(xs: list[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return max(xs) if xs else float("nan")


def aggregate(rows: list[dict], model_label: str) -> tuple[list[dict], list[dict]]:
    by_layer = defaultdict(list)
    for r in rows:
        by_layer[(r["layer"], r["projection"], r["alpha"], r["module_kind"])].append(r)

    layer_rows = []
    for (layer, proj, alpha, kind), group in sorted(by_layer.items()):
        if kind in ("dense", "moe_shared"):
            g = group[0]
            layer_rows.append(
                {
                    "model": model_label,
                    "layer": layer,
                    "projection": proj,
                    "alpha": alpha,
                    "aggregate": kind,
                    "n_matrices": len(group),
                    "gini_base_mean": g["gini_base"],
                    "gini_alpha_mean": g["gini_alpha"],
                    "delta_gini_mean": g["delta_gini"],
                    "delta_gini_max": g["delta_gini"],
                    "top_sv_rel_mean": g["top_sv_rel"],
                    "top_sv_rel_max": g["top_sv_rel"],
                    "top_sv_abs_delta_mean": g["top_sv_abs_delta"],
                    "top_sv_abs_delta_max": g["top_sv_abs_delta"],
                    "top_to_geomean_base_mean": g["top_to_geomean_base"],
                    "top_to_geomean_base_max": g["top_to_geomean_base"],
                }
            )
        else:
            layer_rows.append(
                {
                    "model": model_label,
                    "layer": layer,
                    "projection": proj,
                    "alpha": alpha,
                    "aggregate": "moe_expert_mean_max",
                    "n_matrices": len(group),
                    "gini_base_mean": mean([g["gini_base"] for g in group]),
                    "gini_alpha_mean": mean([g["gini_alpha"] for g in group]),
                    "delta_gini_mean": mean([g["delta_gini"] for g in group]),
                    "delta_gini_max": maxf([g["delta_gini"] for g in group]),
                    "top_sv_rel_mean": mean([g["top_sv_rel"] for g in group]),
                    "top_sv_rel_max": maxf([g["top_sv_rel"] for g in group]),
                    "top_sv_abs_delta_mean": mean([g["top_sv_abs_delta"] for g in group]),
                    "top_sv_abs_delta_max": maxf([g["top_sv_abs_delta"] for g in group]),
                    "top_to_geomean_base_mean": mean([g["top_to_geomean_base"] for g in group]),
                    "top_to_geomean_base_max": maxf([g["top_to_geomean_base"] for g in group]),
                }
            )

    depth_rows = []
    layers = sorted({int(r["layer"]) for r in layer_rows})
    if layers:
        n = len(layers)
        cut1, cut2 = n / 3, 2 * n / 3
        layer_bucket = {}
        for i, layer in enumerate(layers):
            layer_bucket[layer] = "early" if i < cut1 else "middle" if i < cut2 else "late"
        buckets = defaultdict(list)
        for r in layer_rows:
            buckets[(r["projection"], r["alpha"], r["aggregate"], layer_bucket[int(r["layer"])])].append(r)
        for (proj, alpha, agg, bucket), group in sorted(buckets.items()):
            depth_rows.append(
                {
                    "model": model_label,
                    "projection": proj,
                    "alpha": alpha,
                    "aggregate": agg,
                    "depth_bucket": bucket,
                    "n_layers": len(group),
                    "gini_base_mean": mean([g["gini_base_mean"] for g in group]),
                    "gini_alpha_mean": mean([g["gini_alpha_mean"] for g in group]),
                    "delta_gini_mean": mean([g["delta_gini_mean"] for g in group]),
                    "top_sv_rel_mean": mean([g["top_sv_rel_mean"] for g in group]),
                    "top_sv_abs_delta_mean": mean([g["top_sv_abs_delta_mean"] for g in group]),
                    "top_to_geomean_base_mean": mean([g["top_to_geomean_base_mean"] for g in group]),
                }
            )
    return layer_rows, depth_rows


def main() -> None:
    args = parse_args()
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    out_dir = Path(args.output_dir)
    model_dir = resolve_model_dir(Path(args.model_dir))
    weights = selected_weights(tensor_files(model_dir))
    if args.limit:
        weights = weights[: args.limit]
    if not weights:
        raise RuntimeError(f"No up/down weights found in {model_dir}")

    matrix_rows = []
    for idx, item in enumerate(weights, 1):
        print(f"[{idx}/{len(weights)}] {item['name']}", flush=True)
        weight = load_tensor(item["path"], item["name"])
        s = svd_values(weight, args.device)
        s = torch.clamp(s, min=1e-12)
        g_base = gini(s)
        log_s = torch.log(s)
        geom = float(torch.exp(torch.mean(log_s)))
        top_base = float(torch.max(s))
        for alpha in alphas:
            st = alpha_stats(s, alpha)
            matrix_rows.append(
                {
                    "model": args.model_label,
                    "layer": item["layer"],
                    "projection": item["projection"],
                    "module_kind": item["module_kind"],
                    "expert": item["expert"],
                    "alpha": alpha,
                    "matrix_shape": "x".join(str(x) for x in weight.shape),
                    "rank": int(s.numel()),
                    "gini_base": g_base,
                    "gini_alpha": st["gini_alpha"],
                    "delta_gini": st["delta_gini"],
                    "top_sv_base": top_base,
                    "top_sv_alpha": st["top_sv_alpha"],
                    "top_sv_rel": st["top_sv_rel"],
                    "top_sv_abs_delta": st["top_sv_abs_delta"],
                    "geomean_sv_base": geom,
                    "top_to_geomean_base": st["top_to_geomean_base"],
                    "weight_name": item["name"],
                }
            )
        del weight, s
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    layer_rows, depth_rows = aggregate(matrix_rows, args.model_label)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model_label).strip("_").lower()
    write_csv(out_dir / f"{safe}_matrix_alpha_svd_stats.csv", matrix_rows)
    write_csv(out_dir / f"{safe}_layer_alpha_svd_summary.csv", layer_rows)
    write_csv(out_dir / f"{safe}_depth_alpha_svd_summary.csv", depth_rows)
    print(out_dir / f"{safe}_layer_alpha_svd_summary.csv")


if __name__ == "__main__":
    main()
