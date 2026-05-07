#!/usr/bin/env python3
"""Plot 3 capability dimensions by 5 eval6 scopes in one panel figure."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "docs/results/eval6_capability_5panel_independent/capability_independent_points_long.csv"
OUT = ROOT / "docs/results/eval6_capability_5panel_independent/eval6_capability_3x5_independent_errorbars.png"

DIMENSIONS = ["Factual Knowledge", "Language Understanding", "Deductive Reasoning"]
SCOPES = ["mmlu_pro", "mmlu_redux", "agieval", "bbh", "all"]
SCOPE_LABELS = {
    "mmlu_pro": "MMLU-Pro",
    "mmlu_redux": "MMLU-Redux",
    "agieval": "AGIEval",
    "bbh": "BBH",
    "all": "All",
}
MODEL_LABELS = {
    "llama1b": "Llama 1B",
    "llama3b": "Llama 3B",
    "llama8b": "Llama 8B",
    "qwen3_8b": "Qwen3 8B",
    "qwen3_30b_a3b": "Qwen3 MoE",
}
MODEL_COLORS = {
    "llama1b": "#2563eb",
    "llama3b": "#16a34a",
    "llama8b": "#7e22ce",
    "qwen3_8b": "#b91c1c",
    "qwen3_30b_a3b": "#c2410c",
}
MODEL_MARKERS = {
    "llama1b": "o",
    "llama3b": "D",
    "llama8b": "s",
    "qwen3_8b": "^",
    "qwen3_30b_a3b": "X",
}


def main() -> None:
    df = pd.read_csv(IN)
    df["alpha"] = df["alpha"].astype(float)
    models = list(MODEL_LABELS)

    fig, axes = plt.subplots(
        len(DIMENSIONS),
        len(SCOPES),
        figsize=(24, 11.8),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )

    for r, dim in enumerate(DIMENSIONS):
        dim_df = df[df["dimension"] == dim]
        lo = (dim_df["mean_mle"] - dim_df["sigma_mle"]).quantile(0.01)
        hi = (dim_df["mean_mle"] + dim_df["sigma_mle"]).quantile(0.99)
        pad = max((hi - lo) * 0.12, 0.02)
        y_min, y_max = lo - pad, hi + pad
        for c, scope in enumerate(SCOPES):
            ax = axes[r, c]
            sub = df[(df["dimension"] == dim) & (df["scope"] == scope)]
            for model in models:
                m = sub[sub["model"] == model].sort_values("alpha")
                if m.empty:
                    continue
                ax.errorbar(
                    m["alpha"],
                    m["mean_mle"],
                    yerr=m["sigma_mle"],
                    color=MODEL_COLORS[model],
                    marker=MODEL_MARKERS[model],
                    markersize=4.8,
                    linewidth=1.35,
                    elinewidth=0.8,
                    capsize=2.0,
                    alpha=0.88,
                    label=MODEL_LABELS[model],
                )
            ax.axhline(0, color="#111827", linewidth=0.8, alpha=0.55)
            ax.axvline(0, color="#64748b", linewidth=0.6, alpha=0.45)
            ax.set_xlim(-0.215, 0.215)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, color="#e5e7eb", linewidth=0.7)
            ax.yaxis.set_minor_locator(MultipleLocator(0.01))
            ax.tick_params(axis="both", labelsize=8)
            if r == 0:
                ax.set_title(SCOPE_LABELS[scope], fontsize=13, pad=8)
            if c == 0:
                ax.set_ylabel(dim, fontsize=12)
            if r == len(DIMENSIONS) - 1:
                ax.set_xlabel("alpha", fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=10)
    fig.suptitle("Independent Capability Fits Across Benchmark Modules", fontsize=16, y=1.055)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
