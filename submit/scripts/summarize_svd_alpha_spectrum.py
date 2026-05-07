#!/usr/bin/env python3
"""Summarize alpha singular-value statistics into CSV tables and an md report."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path("docs/results/svd_alpha_spectrum")
RAW = ROOT / "raw"
REPORT = ROOT / "svd_alpha_spectrum_report.md"

ALPHA_SHOW = [-0.2, -0.1, 0.1, 0.2]
MODEL_NAMES = {
    "llama32_1b_instruct": "Llama 3.2 1B Instruct",
    "llama32_3b_instruct": "Llama 3.2 3B Instruct",
    "llama31_8b_instruct": "Llama 3.1 8B Instruct",
    "qwen3_8b": "Qwen3 8B",
    "qwen3_30b_a3b_moe": "Qwen3 30B-A3B MoE",
}


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def f(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return str(x)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(out)


def model_key_from_path(path: Path) -> str:
    return path.name.replace("_depth_alpha_svd_summary.csv", "")


def gather():
    depth, layer, matrix = [], [], []
    for p in sorted(RAW.glob("**/*_depth_alpha_svd_summary.csv")):
        depth.extend(read_csv(p))
    for p in sorted(RAW.glob("**/*_layer_alpha_svd_summary.csv")):
        layer.extend(read_csv(p))
    for p in sorted(RAW.glob("**/*_matrix_alpha_svd_stats.csv")):
        matrix.extend(read_csv(p))
    return depth, layer, matrix


def dense_depth_rows(depth):
    rows = []
    for r in depth:
        alpha = float(r["alpha"])
        if alpha not in ALPHA_SHOW:
            continue
        if r["aggregate"] not in ("dense", "moe_expert_mean_max"):
            continue
        rows.append(
            {
                "model": MODEL_NAMES.get(r["model"], r["model"]),
                "projection": r["projection"],
                "aggregate": r["aggregate"],
                "depth_bucket": r["depth_bucket"],
                "alpha": alpha,
                "gini_base_mean": f(r["gini_base_mean"]),
                "gini_alpha_mean": f(r["gini_alpha_mean"]),
                "delta_gini_mean": f(r["delta_gini_mean"]),
                "top_sv_rel_mean": f(r["top_sv_rel_mean"]),
                "top_sv_abs_delta_mean": f(r["top_sv_abs_delta_mean"]),
                "top_to_geomean_base_mean": f(r["top_to_geomean_base_mean"]),
            }
        )
    return rows


def top_layer_rows(layer, alpha: float, metric: str, largest=True, n=12):
    rows = [r for r in layer if float(r["alpha"]) == alpha and r["aggregate"] in ("dense", "moe_expert_mean_max")]
    rows.sort(key=lambda r: float(r[metric]), reverse=largest)
    out = []
    for r in rows[:n]:
        out.append(
            {
                "model": MODEL_NAMES.get(r["model"], r["model"]),
                "layer": r["layer"],
                "projection": r["projection"],
                "aggregate": r["aggregate"],
                "alpha": alpha,
                "gini_base_mean": f(r["gini_base_mean"]),
                "delta_gini_mean": f(r["delta_gini_mean"]),
                "top_sv_rel_mean": f(r["top_sv_rel_mean"]),
                "top_sv_abs_delta_mean": f(r["top_sv_abs_delta_mean"]),
                "top_to_geomean_base_mean": f(r["top_to_geomean_base_mean"]),
            }
        )
    return out


def top_moe_max_rows(layer, alpha: float, metric: str, largest=True, n=12):
    rows = [
        r
        for r in layer
        if float(r["alpha"]) == alpha and r["aggregate"] == "moe_expert_mean_max" and r["model"] == "qwen3_30b_a3b_moe"
    ]
    rows.sort(key=lambda r: float(r[metric]), reverse=largest)
    out = []
    for r in rows[:n]:
        out.append(
            {
                "model": MODEL_NAMES.get(r["model"], r["model"]),
                "layer": r["layer"],
                "projection": r["projection"],
                "alpha": alpha,
                "n_matrices": r["n_matrices"],
                "gini_base_mean": f(r["gini_base_mean"]),
                "delta_gini_mean": f(r["delta_gini_mean"]),
                "delta_gini_max": f(r["delta_gini_max"]),
                "top_sv_rel_mean": f(r["top_sv_rel_mean"]),
                "top_sv_rel_max": f(r["top_sv_rel_max"]),
                "top_sv_abs_delta_mean": f(r["top_sv_abs_delta_mean"]),
                "top_sv_abs_delta_max": f(r["top_sv_abs_delta_max"]),
                "top_to_geomean_base_mean": f(r["top_to_geomean_base_mean"]),
                "top_to_geomean_base_max": f(r["top_to_geomean_base_max"]),
            }
        )
    return out


def late_minus_early(depth):
    by_key = defaultdict(dict)
    for r in depth:
        if float(r["alpha"]) != 0.0:
            continue
        if r["depth_bucket"] not in ("early", "late"):
            continue
        if r["aggregate"] not in ("dense", "moe_expert_mean_max"):
            continue
        by_key[(r["model"], r["projection"], r["aggregate"])][r["depth_bucket"]] = r
    rows = []
    for (model, proj, agg), vals in sorted(by_key.items()):
        if "early" not in vals or "late" not in vals:
            continue
        e, l = vals["early"], vals["late"]
        rows.append(
            {
                "model": MODEL_NAMES.get(model, model),
                "projection": proj,
                "aggregate": agg,
                "early_gini": f(e["gini_base_mean"]),
                "late_gini": f(l["gini_base_mean"]),
                "late_minus_early_gini": f(float(l["gini_base_mean"]) - float(e["gini_base_mean"])),
                "early_top_to_geomean": f(e["top_to_geomean_base_mean"]),
                "late_top_to_geomean": f(l["top_to_geomean_base_mean"]),
                "late_minus_early_top_to_geomean": f(float(l["top_to_geomean_base_mean"]) - float(e["top_to_geomean_base_mean"])),
            }
        )
    return rows


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    depth, layer, matrix = gather()
    write_csv(ROOT / "combined_depth_alpha_svd_summary.csv", depth)
    write_csv(ROOT / "combined_layer_alpha_svd_summary.csv", layer)
    write_csv(ROOT / "combined_matrix_alpha_svd_stats.csv", matrix)

    depth_table = dense_depth_rows(depth)
    write_csv(ROOT / "table_depth_alpha_svd_summary_selected.csv", depth_table)
    top_spike = top_layer_rows(layer, 0.2, "delta_gini_mean", True)
    top_smooth = top_layer_rows(layer, -0.2, "delta_gini_mean", False)
    top_toprel = top_layer_rows(layer, 0.2, "top_sv_rel_mean", True)
    moe_max_gini = top_moe_max_rows(layer, 0.2, "delta_gini_max", True)
    moe_max_toprel = top_moe_max_rows(layer, 0.2, "top_sv_rel_max", True)
    late_table = late_minus_early(depth)
    write_csv(ROOT / "table_top_delta_gini_alpha_pos02.csv", top_spike)
    write_csv(ROOT / "table_top_delta_gini_alpha_neg02.csv", top_smooth)
    write_csv(ROOT / "table_top_sv_rel_alpha_pos02.csv", top_toprel)
    write_csv(ROOT / "table_moe_expert_max_delta_gini_alpha_pos02.csv", moe_max_gini)
    write_csv(ROOT / "table_moe_expert_max_top_sv_rel_alpha_pos02.csv", moe_max_toprel)
    write_csv(ROOT / "table_late_minus_early_base_spectrum.csv", late_table)

    models = sorted({r["model"] for r in depth})
    lines = [
        "# Alpha SVD Spectrum Statistics",
        "",
        "This report is data-first. It summarizes singular-value distribution changes induced by the implemented alpha transform on MLP `up_proj` and `down_proj` weights.",
        "",
        "Transform:",
        "",
        "```text",
        "s_i' = G * (s_i / G)^(1 + alpha)",
        "G = exp(mean_i log s_i)",
        "```",
        "",
        "The transform preserves the geometric mean of singular values. Positive alpha increases spectral inequality; negative alpha smooths it.",
        "",
        "## Available Models",
        "",
        md_table(["model"], [[MODEL_NAMES.get(m, m)] for m in models]),
        "",
        "## Output Tables",
        "",
        f"- Matrix-level full table: `{ROOT / 'combined_matrix_alpha_svd_stats.csv'}`",
        f"- Layer-level summary: `{ROOT / 'combined_layer_alpha_svd_summary.csv'}`",
        f"- Depth-bucket summary: `{ROOT / 'combined_depth_alpha_svd_summary.csv'}`",
        f"- Selected depth table: `{ROOT / 'table_depth_alpha_svd_summary_selected.csv'}`",
        f"- Top positive-alpha Gini increases: `{ROOT / 'table_top_delta_gini_alpha_pos02.csv'}`",
        f"- Top negative-alpha Gini decreases: `{ROOT / 'table_top_delta_gini_alpha_neg02.csv'}`",
        f"- Top max-singular-value relative increases: `{ROOT / 'table_top_sv_rel_alpha_pos02.csv'}`",
        f"- MoE expert-max Gini increases: `{ROOT / 'table_moe_expert_max_delta_gini_alpha_pos02.csv'}`",
        f"- MoE expert-max top singular value relative increases: `{ROOT / 'table_moe_expert_max_top_sv_rel_alpha_pos02.csv'}`",
        f"- Late-minus-early base spectrum table: `{ROOT / 'table_late_minus_early_base_spectrum.csv'}`",
        "",
        "## Late vs Early Base Spectrum",
        "",
        md_table(
            [
                "model",
                "proj",
                "agg",
                "early Gini",
                "late Gini",
                "late-early Gini",
                "early top/G",
                "late top/G",
                "late-early top/G",
            ],
            [
                [
                    r["model"],
                    r["projection"],
                    r["aggregate"],
                    r["early_gini"],
                    r["late_gini"],
                    r["late_minus_early_gini"],
                    r["early_top_to_geomean"],
                    r["late_top_to_geomean"],
                    r["late_minus_early_top_to_geomean"],
                ]
                for r in late_table
            ],
        ),
        "",
        "## Largest Gini Increase at alpha=+0.2",
        "",
        md_table(
            ["model", "layer", "proj", "agg", "base Gini", "delta Gini", "top rel", "top/G"],
            [[r["model"], r["layer"], r["projection"], r["aggregate"], r["gini_base_mean"], r["delta_gini_mean"], r["top_sv_rel_mean"], r["top_to_geomean_base_mean"]] for r in top_spike],
        ),
        "",
        "## Largest Gini Decrease at alpha=-0.2",
        "",
        md_table(
            ["model", "layer", "proj", "agg", "base Gini", "delta Gini", "top rel", "top/G"],
            [[r["model"], r["layer"], r["projection"], r["aggregate"], r["gini_base_mean"], r["delta_gini_mean"], r["top_sv_rel_mean"], r["top_to_geomean_base_mean"]] for r in top_smooth],
        ),
        "",
        "## Largest Top Singular Value Relative Increase at alpha=+0.2",
        "",
        md_table(
            ["model", "layer", "proj", "agg", "base Gini", "delta Gini", "top rel", "top abs delta", "top/G"],
            [[r["model"], r["layer"], r["projection"], r["aggregate"], r["gini_base_mean"], r["delta_gini_mean"], r["top_sv_rel_mean"], r["top_sv_abs_delta_mean"], r["top_to_geomean_base_mean"]] for r in top_toprel],
        ),
        "",
        "## MoE Expert-Max Gini Increase at alpha=+0.2",
        "",
        md_table(
            ["model", "layer", "proj", "n experts", "mean base Gini", "mean delta Gini", "max delta Gini", "mean top rel", "max top rel"],
            [[r["model"], r["layer"], r["projection"], r["n_matrices"], r["gini_base_mean"], r["delta_gini_mean"], r["delta_gini_max"], r["top_sv_rel_mean"], r["top_sv_rel_max"]] for r in moe_max_gini],
        ),
        "",
        "## MoE Expert-Max Top Singular Value Relative Increase at alpha=+0.2",
        "",
        md_table(
            ["model", "layer", "proj", "n experts", "mean delta Gini", "max delta Gini", "mean top rel", "max top rel", "max top/G"],
            [[r["model"], r["layer"], r["projection"], r["n_matrices"], r["delta_gini_mean"], r["delta_gini_max"], r["top_sv_rel_mean"], r["top_sv_rel_max"], r["top_to_geomean_base_max"]] for r in moe_max_toprel],
        ),
        "",
        "## Interpretation",
        "",
        "The tables show that alpha has a stable sign effect but a non-uniform layer effect. Positive alpha consistently raises Gini and top singular values; negative alpha consistently lowers them. However, the amount of movement depends on each layer's original spectrum, especially `top_sv / geometric_mean_sv`.",
        "",
        "The late-vs-early table should therefore be read per model and projection. In the current dense results, Llama up-projections often show higher late-layer Gini than early-layer Gini, while Qwen3-8B has stronger early-layer spectral concentration. This means a global alpha is not a uniform perturbation across depth: it is filtered through each layer's existing spectral imbalance.",
        "",
        "For MoE models, `moe_expert_mean_max` rows aggregate routed experts by layer/projection. Mean captures typical expert behavior; max captures the most spectrally concentrated expert in that layer. Shared experts are retained separately when available.",
        "",
    ]
    REPORT.write_text("\n".join(lines))
    print(REPORT)


if __name__ == "__main__":
    main()
