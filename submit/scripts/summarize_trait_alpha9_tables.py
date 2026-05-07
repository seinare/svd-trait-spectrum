#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "results" / "trait_alpha9_tables"
ALPHAS = [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]
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


MODEL_DIRS = {
    "llama32_1b_instruct": [
        ROOT / "docs/results/raw/trait_softmax/llama1b",
        ROOT / "docs/results/raw/trait_alpha9/llama32_1b_instruct",
    ],
    "llama32_3b_instruct": [
        ROOT / "docs/results/raw/trait_softmax/llama3b",
        ROOT / "docs/results/raw/trait_alpha9/llama32_3b_instruct",
    ],
    "llama31_8b_instruct": [
        ROOT / "docs/results/raw/trait_alpha9/llama31_8b_instruct",
    ],
    "qwen3_8b": [
        ROOT / "docs/results/raw/trait_softmax/qwen3_8b",
        ROOT / "docs/results/raw/trait_alpha9/qwen3_8b",
    ],
    "qwen3_30b_a3b": [
        ROOT / "docs/results/raw/trait_alpha9/qwen3_30b_a3b",
    ],
    "llama32_1b_base": [
        ROOT / "docs/results/raw/trait_softmax_base/llama32_1b_base",
        ROOT / "docs/results/raw/trait_alpha9/llama32_1b_base",
    ],
    "llama32_3b_base": [
        ROOT / "docs/results/raw/trait_softmax_base/llama32_3b_base",
        ROOT / "docs/results/raw/trait_alpha9/llama32_3b_base",
    ],
    "llama31_8b_base": [
        ROOT / "docs/results/raw/trait_softmax_base/llama31_8b_base",
        ROOT / "docs/results/raw/trait_alpha9/llama31_8b_base",
    ],
}


ALPHA_RE = re.compile(r"res_alpha(-?\d+(?:\.\d+)?)\.json$")


def alpha_label(alpha: float) -> str:
    if alpha == 0:
        return "0"
    text = f"{alpha:.2f}".rstrip("0").rstrip(".")
    return text


def read_scores(path: Path) -> dict[str, dict[str, float]]:
    data = json.loads(path.read_text())
    return data.get("scores", {})


def collect_model(model: str, dirs: list[Path]) -> dict[float, dict[str, dict[str, float]]]:
    found: dict[float, tuple[Path, dict[str, dict[str, float]]]] = {}
    for directory in dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            match = ALPHA_RE.search(path.name)
            if not match:
                continue
            alpha = round(float(match.group(1)), 2)
            if alpha not in {round(a, 2) for a in ALPHAS}:
                continue
            scores = read_scores(path)
            if not scores:
                continue
            found[alpha] = (path, scores)
    return {alpha: scores for alpha, (_path, scores) in found.items()}


def write_model_table(model: str, scores_by_alpha: dict[float, dict[str, dict[str, float]]]) -> tuple[list[dict[str, str]], list[float]]:
    rows: list[dict[str, str]] = []
    missing: list[float] = []
    for trait in TRAITS:
        row = {"trait": trait}
        for alpha in ALPHAS:
            score = scores_by_alpha.get(round(alpha, 2), {}).get(trait, {}).get("trait_score")
            row[alpha_label(alpha)] = "" if score is None else f"{float(score):.6f}"
        rows.append(row)
    for alpha in ALPHAS:
        per_alpha = scores_by_alpha.get(round(alpha, 2), {})
        if any(trait not in per_alpha for trait in TRAITS):
            missing.append(alpha)

    out_csv = OUT_DIR / f"{model}_trait_alpha9_table.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trait"] + [alpha_label(a) for a in ALPHAS])
        writer.writeheader()
        writer.writerows(rows)
    return rows, missing


def markdown_table(rows: list[dict[str, str]]) -> str:
    headers = ["trait"] + [alpha_label(a) for a in ALPHAS]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row.get(h, "") for h in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = [
        "# TRAIT Alpha-9 Tables",
        "",
        "Scoring uses softmax over log likelihoods of `response_high1`, `response_high2`, `response_low1`, and `response_low2`; `trait_score = P(high1) + P(high2)`.",
        "",
    ]
    coverage_rows = []
    for model, dirs in MODEL_DIRS.items():
        scores_by_alpha = collect_model(model, dirs)
        rows, missing = write_model_table(model, scores_by_alpha)
        coverage_rows.append(
            {
                "model": model,
                "available_alpha_count": str(len(ALPHAS) - len(missing)),
                "missing_alphas": " ".join(alpha_label(a) for a in missing),
            }
        )
        report.extend(
            [
                f"## {model}",
                "",
                f"- Missing alpha: {', '.join(alpha_label(a) for a in missing) if missing else 'none'}",
                "",
                markdown_table(rows),
                "",
            ]
        )

    with (OUT_DIR / "trait_alpha9_coverage.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "available_alpha_count", "missing_alphas"])
        writer.writeheader()
        writer.writerows(coverage_rows)
    (OUT_DIR / "trait_alpha9_tables.md").write_text("\n".join(report))
    print(OUT_DIR / "trait_alpha9_tables.md")
    print(OUT_DIR / "trait_alpha9_coverage.csv")


if __name__ == "__main__":
    main()
