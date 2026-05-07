#!/usr/bin/env python3
"""Generate radar-chart assets from TRAIT softmax likelihood outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRAITS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
    "Machiavellianism",
    "Narcissism",
    "Psychopathy",
]

MODEL_LABELS = {
    "llama1b": "Llama-3.2-1B",
    "llama3b": "Llama-3.2-3B",
    "qwen3_8b": "Qwen3-8B",
}


def parse_alpha(path: Path, payload: dict) -> float:
    if "config" in payload and "alpha" in payload["config"]:
        return float(payload["config"]["alpha"])
    match = re.search(r"alpha(-?\d+(?:\.\d+)?)", path.name)
    if not match:
        raise ValueError(f"Cannot parse alpha from {path}")
    return float(match.group(1))


def load_rows(input_root: Path) -> list[dict]:
    rows: list[dict] = []
    for model_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        model = model_dir.name
        for path in sorted(model_dir.glob("*.json")):
            payload = json.loads(path.read_text())
            alpha = parse_alpha(path, payload)
            scores = payload.get("scores", {})
            for trait in TRAITS:
                if trait not in scores:
                    raise ValueError(f"{path} missing trait {trait}")
                item = scores[trait]
                rows.append(
                    {
                        "model": model,
                        "alpha": alpha,
                        "trait": trait,
                        "total": item.get("total", ""),
                        "trait_score": float(item["trait_score"]),
                        "high1_prob": float(item.get("high1_prob", math.nan)),
                        "high2_prob": float(item.get("high2_prob", math.nan)),
                        "low1_prob": float(item.get("low1_prob", math.nan)),
                        "low2_prob": float(item.get("low2_prob", math.nan)),
                    }
                )
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "alpha",
        "trait",
        "total",
        "trait_score",
        "high1_prob",
        "high2_prob",
        "low1_prob",
        "low2_prob",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["model"], r["alpha"], TRAITS.index(r["trait"]))):
            writer.writerow(row)


def plot_model(rows: list[dict], model: str, output_dir: Path) -> Path:
    by_alpha: dict[float, dict[str, float]] = {}
    for row in rows:
        if row["model"] == model:
            by_alpha.setdefault(row["alpha"], {})[row["trait"]] = row["trait_score"]
    if not by_alpha:
        raise ValueError(f"No rows for model {model}")

    alphas = sorted(by_alpha)
    angles = [n / float(len(TRAITS)) * 2 * np.pi for n in range(len(TRAITS))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    colors = ["#08519c", "#3182bd", "#9ecae1", "#333333", "#fc9272", "#de2d26", "#a50f15"]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], TRAITS, size=11, fontweight="bold")
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7"], color="grey", size=9)
    plt.ylim(0.1, 0.75)

    for idx, alpha in enumerate(alphas):
        values = [by_alpha[alpha][trait] for trait in TRAITS]
        values += values[:1]
        linewidth = 3.0 if alpha == 0 else 1.5
        if idx in [0, len(alphas) - 1]:
            linewidth = max(linewidth, 2.0)
        label = f"alpha = {alpha:g}" + (" (baseline)" if alpha == 0 else "")
        ax.plot(angles, values, color=colors[idx % len(colors)], linewidth=linewidth, alpha=1.0 if alpha == 0 else 0.82, label=label)
        if alpha == 0:
            ax.fill(angles, values, color=colors[idx % len(colors)], alpha=0.1)

    for tick in ax.get_xticklabels():
        if tick.get_text() in {"Machiavellianism", "Narcissism", "Psychopathy"}:
            tick.set_color("darkred")
        elif tick.get_text() == "Conscientiousness":
            tick.set_color("darkblue")

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), title="Alpha", title_fontsize="12")
    ax.set_title(
        f"{MODEL_LABELS.get(model, model)} Matthew Perturbation\nTRAIT Personality Evolution",
        size=14,
        y=1.1,
        fontweight="bold",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / f"{model}_trait_softmax_radar.png"
    output_svg = output_dir / f"{model}_trait_softmax_radar.svg"
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)
    return output_png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=Path, default=Path("docs/results/raw/trait_softmax"))
    parser.add_argument("--csv", type=Path, default=Path("docs/results/trait_softmax_scores_long.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("assets/trait_radar"))
    args = parser.parse_args()

    rows = load_rows(args.input_root)
    write_csv(rows, args.csv)
    for model in sorted({row["model"] for row in rows}):
        print(plot_model(rows, model, args.output_dir))


if __name__ == "__main__":
    main()
