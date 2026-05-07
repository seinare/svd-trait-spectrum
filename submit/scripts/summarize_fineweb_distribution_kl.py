#!/usr/bin/env python3
"""Summarize FineWeb distribution KL runs into one markdown table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODEL_ORDER = [
    "llama32_1b_instruct",
    "llama32_3b_instruct",
    "llama31_8b_instruct",
    "qwen3_8b",
    "qwen3_30b_a3b_moe",
]


def fmt(x: float | int | str) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, str):
        return x
    return f"{float(x):.6g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("docs/results/fineweb_distribution_kl"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for item in args.inputs:
        path = Path(item)
        if path.exists() and path.stat().st_size > 0:
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("No non-empty CSV inputs.")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["model", "alpha"], keep="last")
    df["model_order"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(99)
    df = df.sort_values(["model_order", "alpha"]).drop(columns=["model_order"])

    long_path = args.output_dir / "fineweb_distribution_kl_128k_long.csv"
    df.to_csv(long_path, index=False)

    pivot = df.pivot_table(
        index="model",
        columns="alpha",
        values="kl_alpha_to_base_mean",
        aggfunc="first",
    ).reset_index()
    for alpha in sorted(df["alpha"].unique()):
        pivot = pivot.rename(columns={alpha: f"KL(P_alpha||P_0) alpha={alpha:g}"})
    baseline_cols = [
        "model",
        "baseline_random_pair_kl_mean",
        "baseline_random_pair_kl_std",
        "baseline_random_pair_kl_pairs",
    ]
    baseline = df[baseline_cols].drop_duplicates("model", keep="last")
    summary = baseline.merge(pivot, on="model", how="left")
    summary["model_order"] = summary["model"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(99)
    summary = summary.sort_values("model_order").drop(columns=["model_order"])
    summary_path = args.output_dir / "fineweb_distribution_kl_128k_summary.csv"
    summary.to_csv(summary_path, index=False)

    md = [
        "# FineWeb 128k Output Distribution KL",
        "",
        "KL is computed token-wise on a fixed 128k-token FineWeb10B validation sample.",
        "`KL(P_alpha || P_0)` compares the Matthew-perturbed model output distribution against the alpha=0 output distribution at the same token positions.",
        "`baseline_random_pair_kl_mean` is computed under alpha=0 by randomly pairing token positions in the same sample and averaging `KL(P_i || P_j)`.",
        "",
        "| model | baseline random-token KL mean | baseline random-token KL std | pairs | alpha=-0.2 KL | alpha=+0.2 KL |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        md.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    fmt(row.get("baseline_random_pair_kl_mean")),
                    fmt(row.get("baseline_random_pair_kl_std")),
                    fmt(row.get("baseline_random_pair_kl_pairs")),
                    fmt(row.get("KL(P_alpha||P_0) alpha=-0.2")),
                    fmt(row.get("KL(P_alpha||P_0) alpha=0.2")),
                ]
            )
            + " |"
        )
    md.extend(
        [
            "",
            "Raw long table:",
            "",
            f"- `{long_path}`",
            f"- `{summary_path}`",
        ]
    )
    report_path = args.output_dir / "fineweb_distribution_kl_128k_report.md"
    report_path.write_text("\n".join(md) + "\n")
    print(report_path)
    print(summary_path)


if __name__ == "__main__":
    main()
