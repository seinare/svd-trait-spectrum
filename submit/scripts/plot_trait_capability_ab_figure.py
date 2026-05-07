#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plot_trait_capability_main_figure import (
    ALPHAS,
    CAP_MODEL,
    CAP_SHORT,
    CAPS,
    COL_NEG,
    COL_POS,
    MODELS,
    OUT_DIR,
    ROOT,
    TRAITS,
    alpha_label,
    read_capabilities,
    read_trait_scores,
)
from plot_trait_capability_panel_c_mpl import TRAIT_SHORT


SEGMENTS = [
    (-0.2, -0.1, "[-.2,-.1]", "#2166ac"),
    (-0.1, 0.0, "[-.1,0]", "#67a9cf"),
    (0.0, 0.1, "[0,.1]", "#ef8a62"),
    (0.1, 0.2, "[.1,.2]", "#b2182b"),
]


def slope(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    if denom == 0:
        return 0.0
    return float((xc * (y - y.mean())).sum() / denom)


def segment_slopes(points: dict[float, float]) -> list[float]:
    out = []
    for lo, hi, _label, _color in SEGMENTS:
        xs = [a for a in ALPHAS if lo - 1e-9 <= a <= hi + 1e-9 and a in points]
        out.append(slope(xs, [points[a] for a in xs]))
    return out


def sem(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    arr = np.asarray(vals, dtype=float)
    return float(arr.std(ddof=1) / np.sqrt(len(arr)))


def read_segment_response() -> tuple[list[str], dict[str, list[list[float]]], dict[str, str]]:
    caps = read_capabilities()
    traits = read_trait_scores()
    labels = [CAP_SHORT[c] for c in CAPS]
    kind = {CAP_SHORT[c]: "Capability" for c in CAPS}
    values: dict[str, list[list[float]]] = {}

    for cap in CAPS:
        per_model = []
        for trait_model, cap_model in CAP_MODEL.items():
            per_model.append(segment_slopes(caps[(cap_model, cap)]))
        values[CAP_SHORT[cap]] = [[m[i] for m in per_model] for i in range(len(SEGMENTS))]

    trait_order = sorted(
        TRAITS,
        key=lambda t: sum(
            abs(np.mean([segment_slopes({a: traits[(m, t)][a] - traits[(m, t)][0.0] for a in ALPHAS})[i] for m in MODELS]))
            for i in range(len(SEGMENTS))
        ),
        reverse=True,
    )
    for trait in trait_order:
        label = TRAIT_SHORT[trait]
        labels.append(label)
        kind[label] = "TRAIT"
        per_model = []
        for model in MODELS:
            base = traits[(model, trait)][0.0]
            per_model.append(segment_slopes({a: traits[(model, trait)][a] - base for a in ALPHAS}))
        values[label] = [[m[i] for m in per_model] for i in range(len(SEGMENTS))]
    return labels, values, kind


def read_pearson() -> dict[tuple[str, str], float]:
    out = {}
    path = ROOT / "docs/results/trait_capability_correlation/trait_capability_pearson.csv"
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[(row["dimension"], row["trait"])] = float(row["pearson_r"])
    return out


def blend(color: str, strength: float) -> str:
    color = color.lstrip("#")
    r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
    rr = round(255 - (255 - r) * strength)
    gg = round(255 - (255 - g) * strength)
    bb = round(255 - (255 - b) * strength)
    return f"#{rr:02x}{gg:02x}{bb:02x}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels, values, kind = read_segment_response()
    pearson = read_pearson()

    fig = plt.figure(figsize=(16.6, 6.2), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 0.92], left=0.045, right=0.975, top=0.90, bottom=0.31, wspace=0.24)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    x = np.arange(len(labels))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(SEGMENTS))
    for i, (_lo, _hi, seg_label, color) in enumerate(SEGMENTS):
        means = [float(np.mean(values[label][i])) for label in labels]
        errs = [sem(values[label][i]) for label in labels]
        ax_a.bar(x + offsets[i], means, width=width, color=color, edgecolor="none", alpha=0.92, label=seg_label)
        ax_a.errorbar(x + offsets[i], means, yerr=errs, fmt="none", ecolor="#222", elinewidth=0.6, capsize=1.5, alpha=0.75)

    ax_a.axhline(0, color="#333", lw=0.8)
    ax_a.set_ylabel("response slope, d(delta) / d alpha", fontsize=10.5, weight="bold")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, rotation=48, ha="right", fontsize=8.8)
    ax_a.tick_params(axis="y", labelsize=9)
    ax_a.grid(axis="y", color="#e7e7e7", lw=0.7)
    ax_a.set_axisbelow(True)
    cap_count = len(CAPS)
    ax_a.axvline(cap_count - 0.5, color="#bdbdbd", lw=0.8, ls="--")
    group_label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2}
    ax_a.text(
        (cap_count - 1) / 2,
        0.95,
        "capabilities",
        transform=ax_a.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=9.5,
        color="#555",
        weight="bold",
        bbox=group_label_box,
    )
    ax_a.text(
        (cap_count + len(labels) - 1) / 2,
        0.95,
        "TRAIT",
        transform=ax_a.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=9.5,
        color="#555",
        weight="bold",
        bbox=group_label_box,
    )
    ax_a.legend(
        frameon=False,
        ncol=4,
        fontsize=8.5,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.91),
        title="alpha segment",
        title_fontsize=8.5,
    )
    ax_a.spines[["top", "right"]].set_visible(False)

    mat = np.array([[pearson[(cap, trait)] for trait in TRAITS] for cap in CAPS])
    vmax = max(0.5, float(np.max(np.abs(mat))))
    for i, cap in enumerate(CAPS):
        for j, trait in enumerate(TRAITS):
            r = pearson[(cap, trait)]
            strength = min(abs(r) / vmax, 1.0)
            face = blend(COL_POS if r > 0 else COL_NEG, 0.12 + 0.78 * strength)
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face, edgecolor="#111" if abs(r) >= 0.30 else "#d0d0d0", lw=1.0 if abs(r) >= 0.30 else 0.55)
            ax_b.add_patch(rect)
            ax_b.text(j, i, f"{r:+.2f}".replace("+", ""), ha="center", va="center", fontsize=9.2, weight="bold" if abs(r) >= 0.30 else "normal")
    ax_b.set_xlim(-0.5, len(TRAITS) - 0.5)
    ax_b.set_ylim(len(CAPS) - 0.5, -0.5)
    ax_b.set_xticks(np.arange(len(TRAITS)))
    ax_b.set_xticklabels([TRAIT_SHORT[t] for t in TRAITS], rotation=45, ha="right", fontsize=8.7)
    ax_b.set_yticks(np.arange(len(CAPS)))
    ax_b.set_yticklabels([CAP_SHORT[c] for c in CAPS], fontsize=9.2, weight="bold")
    ax_b.tick_params(length=0)
    ax_b.set_aspect(0.76)
    ax_b.set_anchor("C")
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    ax_b.text(
        0,
        -0.72,
        "Cells show simple Pearson r across nonzero-alpha model points; dark borders mark |r| >= 0.30.",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#555",
    )
    ax_b.text(1.02, 0.98, "red: positive r\nblue: negative r", transform=ax_b.transAxes, ha="left", va="top", fontsize=8.5, color="#333")

    out_png = OUT_DIR / "trait_capability_ab_horizontal.png"
    out_svg = OUT_DIR / "trait_capability_ab_horizontal.svg"
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    fig_a, panel_a = plt.subplots(figsize=(10.8, 5.6))
    for i, (_lo, _hi, seg_label, color) in enumerate(SEGMENTS):
        means = [float(np.mean(values[label][i])) for label in labels]
        errs = [sem(values[label][i]) for label in labels]
        panel_a.bar(x + offsets[i], means, width=width, color=color, edgecolor="none", alpha=0.92, label=seg_label)
        panel_a.errorbar(x + offsets[i], means, yerr=errs, fmt="none", ecolor="#222", elinewidth=0.6, capsize=1.5, alpha=0.75)
    panel_a.axhline(0, color="#333", lw=0.8)
    panel_a.set_ylabel("response slope, d(delta) / d alpha", fontsize=10.5, weight="bold")
    panel_a.set_xticks(x)
    panel_a.set_xticklabels(labels, rotation=48, ha="right", fontsize=8.8)
    panel_a.tick_params(axis="y", labelsize=9)
    panel_a.grid(axis="y", color="#e7e7e7", lw=0.7)
    panel_a.set_axisbelow(True)
    panel_a.axvline(cap_count - 0.5, color="#bdbdbd", lw=0.8, ls="--")
    group_label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2}
    panel_a.text(
        (cap_count - 1) / 2,
        0.95,
        "capabilities",
        transform=panel_a.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=9.5,
        color="#555",
        weight="bold",
        bbox=group_label_box,
    )
    panel_a.text(
        (cap_count + len(labels) - 1) / 2,
        0.95,
        "TRAIT",
        transform=panel_a.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=9.5,
        color="#555",
        weight="bold",
        bbox=group_label_box,
    )
    panel_a.legend(
        frameon=False,
        ncol=4,
        fontsize=8.5,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.91),
        title="alpha segment",
        title_fontsize=8.5,
    )
    panel_a.spines[["top", "right"]].set_visible(False)
    panel_a_png = OUT_DIR / "trait_capability_panel_a_alpha_response.png"
    panel_a_svg = OUT_DIR / "trait_capability_panel_a_alpha_response.svg"
    fig_a.savefig(panel_a_png, dpi=220, bbox_inches="tight", pad_inches=0.08)
    fig_a.savefig(panel_a_svg, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig_a)

    fig_b, panel_b = plt.subplots(figsize=(6.8, 4.1))
    for i, cap in enumerate(CAPS):
        for j, trait in enumerate(TRAITS):
            r = pearson[(cap, trait)]
            strength = min(abs(r) / vmax, 1.0)
            face = blend(COL_POS if r > 0 else COL_NEG, 0.12 + 0.78 * strength)
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face, edgecolor="#111" if abs(r) >= 0.30 else "#d0d0d0", lw=1.0 if abs(r) >= 0.30 else 0.55)
            panel_b.add_patch(rect)
            panel_b.text(j, i, f"{r:+.2f}".replace("+", ""), ha="center", va="center", fontsize=9.2, weight="bold" if abs(r) >= 0.30 else "normal")
    panel_b.set_xlim(-0.5, len(TRAITS) - 0.5)
    panel_b.set_ylim(len(CAPS) - 0.5, -0.5)
    panel_b.set_xticks(np.arange(len(TRAITS)))
    panel_b.set_xticklabels([TRAIT_SHORT[t] for t in TRAITS], rotation=45, ha="right", fontsize=8.7)
    panel_b.set_yticks(np.arange(len(CAPS)))
    panel_b.set_yticklabels([CAP_SHORT[c] for c in CAPS], fontsize=9.2, weight="bold")
    panel_b.tick_params(length=0)
    panel_b.set_aspect(0.76)
    for spine in panel_b.spines.values():
        spine.set_visible(False)
    panel_b.text(
        0,
        -0.72,
        "Cells show simple Pearson r across nonzero-alpha model points; dark borders mark |r| >= 0.30.",
        transform=panel_b.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#555",
    )
    panel_b.text(1.02, 0.98, "red: positive r\nblue: negative r", transform=panel_b.transAxes, ha="left", va="top", fontsize=8.5, color="#333")
    panel_b_png = OUT_DIR / "trait_capability_panel_b_pearson.png"
    panel_b_svg = OUT_DIR / "trait_capability_panel_b_pearson.svg"
    fig_b.savefig(panel_b_png, dpi=220, bbox_inches="tight", pad_inches=0.08)
    fig_b.savefig(panel_b_svg, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig_b)

    with (OUT_DIR / "panel_a_alpha_segment_slopes.csv").open("w", newline="") as f:
        fields = ["group", "item", "segment", "mean_slope", "sem_slope"] + list(MODELS)
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label in labels:
            for i, (_lo, _hi, seg_label, _color) in enumerate(SEGMENTS):
                vals = values[label][i]
                writer.writerow(
                    {
                        "group": kind[label],
                        "item": label,
                        "segment": seg_label,
                        "mean_slope": float(np.mean(vals)),
                        "sem_slope": sem(vals),
                        **{model: vals[j] for j, model in enumerate(MODELS)},
                    }
                )
    print(out_svg)
    print(out_png)
    print(panel_a_svg)
    print(panel_a_png)
    print(panel_b_svg)
    print(panel_b_png)


if __name__ == "__main__":
    main()
