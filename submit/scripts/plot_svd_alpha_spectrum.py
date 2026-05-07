#!/usr/bin/env python3
"""Plot alpha-SVD spectrum figures from computed CSV statistics."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


ROOT = Path("docs/results/svd_alpha_spectrum")
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    ("llama32_1b_instruct", "Llama 3.2 1B"),
    ("llama32_3b_instruct", "Llama 3.2 3B"),
    ("llama31_8b_instruct", "Llama 3.1 8B"),
    ("qwen3_8b", "Qwen3 8B"),
    ("qwen3_30b_a3b_moe", "Qwen3 30B-A3B MoE"),
]
PROJ_ORDER = [("up_proj", "up_proj"), ("down_proj", "down_proj")]
ALPHA_COLORS = {-0.2: "#2563eb", 0.0: "#111827", 0.2: "#dc2626"}
ALPHA_LABELS = {-0.2: "alpha=-0.2", 0.0: "alpha=0", 0.2: "alpha=+0.2"}


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def plot_gini_panel():
    rows = read_csv(ROOT / "combined_layer_alpha_svd_summary.csv")
    rows = [r for r in rows if float(r["alpha"]) == 0.0]
    matrix_rows = read_csv(ROOT / "combined_matrix_alpha_svd_stats.csv")

    fig, axes = plt.subplots(2, 5, figsize=(22, 7.8), sharex=False, sharey=True)
    for col, (model, label) in enumerate(MODEL_ORDER):
        for row_idx, (proj, proj_label) in enumerate(PROJ_ORDER):
            ax = axes[row_idx][col]
            sub = [r for r in rows if r["model"] == model and r["projection"] == proj]
            dense = [r for r in sub if r["aggregate"] in ("dense", "moe_shared")]
            moe = [r for r in sub if r["aggregate"] == "moe_expert_mean_max"]
            if dense:
                dense = sorted(dense, key=lambda r: int(r["layer"]))
                ax.plot(
                    [int(r["layer"]) for r in dense],
                    [float(r["gini_base_mean"]) for r in dense],
                    color="#2563eb",
                    linewidth=2.2,
                    marker="o",
                    markersize=3,
                    label="base Gini",
                )
            if moe:
                moe = sorted(moe, key=lambda r: int(r["layer"]))
                ax.plot(
                    [int(r["layer"]) for r in moe],
                    [float(r["gini_base_mean"]) for r in moe],
                    color="#2563eb",
                    linewidth=2.2,
                    marker="o",
                    markersize=2.4,
                    label="expert mean",
                )
                ax.plot(
                    [int(r["layer"]) for r in moe],
                    [float(r["top_to_geomean_base_max"]) * 0 for r in moe],
                    alpha=0,
                )
                ax.plot(
                    [int(r["layer"]) for r in moe],
                    [float(r["gini_base_mean"]) + (float(r["delta_gini_max"]) - float(r["delta_gini_mean"])) * 0 for r in moe],
                    alpha=0,
                )
                mm = {}
                for r in matrix_rows:
                    if r["model"] == model and r["projection"] == proj and float(r["alpha"]) == 0.0 and r["module_kind"] == "moe_expert":
                        layer = int(r["layer"])
                        mm[layer] = max(mm.get(layer, 0.0), float(r["gini_base"]))
                ax.plot(
                    sorted(mm),
                    [mm[k] for k in sorted(mm)],
                    color="#dc2626",
                    linestyle="--",
                    linewidth=1.8,
                    label="expert max",
                )
            ax.set_title(f"{label}\n{proj_label}", fontsize=12)
            ax.grid(True, color="#e5e7eb", linewidth=0.8)
            ax.set_xlabel("layer")
            if col == 0:
                ax.set_ylabel("Gini")
            ax.set_ylim(0.12, 0.50)
            if col == 4 and row_idx == 0:
                ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Base singular-value Gini by layer for MLP up/down projections", fontsize=16, y=1.02)
    fig.tight_layout()
    out = FIG / "svd_gini_by_layer_2x5.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out)


def plot_max_singular_value_panels(metric: str, ylabel: str, title: str, outfile: str):
    rows = read_csv(ROOT / "combined_layer_alpha_svd_summary.csv")
    matrix_rows = read_csv(ROOT / "combined_matrix_alpha_svd_stats.csv") if metric == "top_sv_alpha" else []
    alphas = [-0.2, 0.0, 0.2]

    def top_sv_alpha_series(model: str, proj: str, alpha: float, max_value: bool) -> dict[int, float]:
        grouped = {}
        for r in matrix_rows:
            if r["model"] != model or r["projection"] != proj or abs(float(r["alpha"]) - alpha) > 1e-12:
                continue
            layer = int(r["layer"])
            grouped.setdefault(layer, []).append(float(r["top_sv_alpha"]))
        if max_value:
            return {layer: max(vals) for layer, vals in grouped.items()}
        return {layer: sum(vals) / len(vals) for layer, vals in grouped.items()}

    fig, axes = plt.subplots(2, 5, figsize=(22, 7.8), sharex=False, sharey=False)
    for col, (model, label) in enumerate(MODEL_ORDER):
        for row_idx, (proj, proj_label) in enumerate(PROJ_ORDER):
            ax = axes[row_idx][col]
            for alpha in alphas:
                sub = [
                    r
                    for r in rows
                    if r["model"] == model
                    and r["projection"] == proj
                    and abs(float(r["alpha"]) - alpha) < 1e-12
                ]
                dense = sorted([r for r in sub if r["aggregate"] in ("dense", "moe_shared")], key=lambda r: int(r["layer"]))
                moe = sorted([r for r in sub if r["aggregate"] == "moe_expert_mean_max"], key=lambda r: int(r["layer"]))
                color = ALPHA_COLORS[alpha]
                label_text = ALPHA_LABELS[alpha]
                if dense:
                    if metric == "top_sv_alpha":
                        series = top_sv_alpha_series(model, proj, alpha, max_value=False)
                        xs = sorted(series)
                        ys = [series[x] for x in xs]
                    else:
                        xs = [int(r["layer"]) for r in dense]
                        ys = [float(r[f"{metric}_mean"]) for r in dense]
                    ax.plot(
                        xs,
                        ys,
                        color=color,
                        linewidth=2.0,
                        marker="o",
                        markersize=2.5,
                        label=label_text,
                    )
                if moe:
                    if metric == "top_sv_alpha":
                        mean_series = top_sv_alpha_series(model, proj, alpha, max_value=False)
                        max_series = top_sv_alpha_series(model, proj, alpha, max_value=True)
                        mean_xs = sorted(mean_series)
                        mean_ys = [mean_series[x] for x in mean_xs]
                        max_xs = sorted(max_series)
                        max_ys = [max_series[x] for x in max_xs]
                    else:
                        mean_xs = [int(r["layer"]) for r in moe]
                        mean_ys = [float(r[f"{metric}_mean"]) for r in moe]
                        max_xs = [int(r["layer"]) for r in moe]
                        max_ys = [float(r[f"{metric}_max"]) for r in moe]
                    ax.plot(
                        mean_xs,
                        mean_ys,
                        color=color,
                        linewidth=2.0,
                        marker="o",
                        markersize=2.2,
                        label=label_text if alpha != 0.0 else f"{label_text} mean",
                    )
                    ax.plot(
                        max_xs,
                        max_ys,
                        color=color,
                        linewidth=1.7,
                        linestyle="--",
                        label=f"{label_text} max" if alpha == 0.2 else None,
                    )
            ax.set_title(f"{label}\n{proj_label}", fontsize=12)
            ax.grid(True, color="#e5e7eb", linewidth=0.8)
            ax.set_xlabel("layer")
            if col == 0:
                ax.set_ylabel(ylabel)
            if metric == "top_sv_rel":
                ax.axhline(1.0, color="#9ca3af", linewidth=0.9, linestyle=":")
            if metric in ("top_sv_abs_delta", "top_sv_alpha"):
                ax.axhline(0.0, color="#9ca3af", linewidth=0.9, linestyle=":")
            if col == 4 and row_idx == 0:
                ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()
    out = FIG / outfile
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out)


def plot_llama8b_tail_head_barplane3d():
    values_path = ROOT / "llama31_8b_selected_singular_values.csv"
    vals = read_csv(values_path)
    fig = plt.figure(figsize=(19, 7.8))
    for proj in ("up_proj", "down_proj"):
        sub = [r for r in vals if r["projection"] == proj]
        layers = sorted({int(r["layer"]) for r in sub})
        layer_pos = {layer: i for i, layer in enumerate(layers)}
        ax = fig.add_subplot(1, 2, 1 if proj == "up_proj" else 2, projection="3d")
        max_z = 0.0
        for layer in layers:
            y = layer_pos[layer] * 4.0
            for segment in ("head50", "tail50"):
                rows = sorted(
                    [r for r in sub if int(r["layer"]) == layer and r["segment"] == segment],
                    key=lambda r: int(r["within_segment_rank"]),
                )
                xs = [int(r["within_segment_rank"]) if segment == "head50" else 60 + int(r["within_segment_rank"]) for r in rows]
                heights = [float(r["singular_value"]) for r in rows]
                max_z = max(max_z, max(heights) if heights else 0.0)
                ax.bar(
                    xs,
                    heights,
                    zs=y,
                    zdir="y",
                    width=0.72,
                    color="#60a5fa",
                    edgecolor="#3b82f6",
                    linewidth=0.15,
                    alpha=0.78,
                    label=segment if layer == layers[0] else None,
                )
            ax.text(55, y, 0, "...", color="#111827", fontsize=13, ha="center", va="center")
        ax.set_title(f"Llama 3.1 8B {proj}: sorted-index singular values")
        ax.set_xlabel("sorted singular-value index")
        ax.set_ylabel("selected layer")
        ax.set_zlabel("singular value")
        ax.set_yticks([layer_pos[l] * 4.0 for l in layers])
        ax.set_yticklabels([str(l) for l in layers])
        rank = max(int(r["matrix_rank"]) for r in sub)
        ax.set_xticks([1, 25, 50, 55, 61, 85, 110])
        ax.set_xticklabels(["1", "25", "50", "...", f"{rank-49}", f"{rank-25}", f"{rank}"])
        ax.set_xlim(0, 113)
        ax.set_ylim(-1.0, max(layer_pos.values()) * 4.0 + 1.0)
        ax.set_zlim(0, max_z * 1.08)
        ax.view_init(elev=25, azim=-62)
        ax.legend(frameon=False)
    fig.suptitle("Llama 3.1 8B selected layers: first/last 50 sorted singular values", fontsize=16)
    fig.tight_layout()
    out = FIG / "llama31_8b_up_down_head_tail50_barplane3d_2panel.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out)


def main():
    plot_gini_panel()
    plot_max_singular_value_panels(
        "top_sv_rel",
        "max singular value relative ratio",
        "Max singular value relative change by layer",
        "svd_top_sv_relative_by_layer_2x5.png",
    )
    plot_max_singular_value_panels(
        "top_sv_alpha",
        "max singular value",
        "Max singular value by layer",
        "svd_top_sv_absolute_value_by_layer_2x5.png",
    )
    if (ROOT / "llama31_8b_selected_singular_values.csv").exists():
        plot_llama8b_tail_head_barplane3d()


if __name__ == "__main__":
    main()
