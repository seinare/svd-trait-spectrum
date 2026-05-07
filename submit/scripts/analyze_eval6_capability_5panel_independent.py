#!/usr/bin/env python3
"""Five-panel eval6 capability fit tables.

For each model and each scope (mmlu_pro, mmlu_redux, agieval, bbh, all), fit
three capability dimensions independently at every alpha point. Both the
dimension means and dimension sigmas are fitted independently per alpha.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_eval6_all_models_alpha9 import (  # noqa: E402
    ALPHAS,
    DIMENSIONS,
    MODELS,
    MODEL_LABELS,
    MODULES,
    build_values,
    extract_scores,
    fit_alpha_point,
    read_caps,
)


OUT = ROOT / "docs/results/eval6_capability_5panel_independent"
REPORT = OUT / "eval6_capability_5panel_independent.md"
SCOPES = [*MODULES, "all"]


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def complete_tasks_for_scope(values, caps, model: str, scope: str) -> list[tuple[str, str]]:
    modules = MODULES if scope == "all" else [scope]
    tasks = []
    for module in modules:
        for task, vals in values[(model, module)].items():
            if task in caps and set(ALPHAS).issubset(vals):
                tasks.append((module, task))
    return sorted(tasks)


def fit_scope(values, caps, model: str, scope: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tasks = complete_tasks_for_scope(values, caps, model, scope)
    point_rows = []
    pred_rows = []
    if not tasks:
        return point_rows, pred_rows
    for alpha in ALPHAS:
        obs = []
        obs_meta = []
        for module, task in tasks:
            vals = values[(model, module)][task]
            y = vals[alpha] - vals[0.0]
            weights = caps[task]
            obs.append((weights, y))
            obs_meta.append((module, task, weights, y))
        means, sigmas, nll = fit_alpha_point(obs)
        for dim, mean, sigma in zip(DIMENSIONS, means, sigmas, strict=True):
            point_rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "alpha": alpha,
                    "dimension": dim,
                    "mean_mle": mean,
                    "sigma_mle": sigma,
                    "nll": nll,
                    "complete_tasks": len(tasks),
                }
            )
        for module, task, weights, y in obs_meta:
            pred = sum(w * m for w, m in zip(weights, means, strict=True))
            pred_rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "alpha": alpha,
                    "module": module,
                    "task": task,
                    "observed_delta": y,
                    "predicted_delta": pred,
                    "residual": y - pred,
                    "complete_tasks": len(tasks),
                    "nll": nll,
                }
            )
    return point_rows, pred_rows


def fmt_mean_sigma(mean: float, sigma: float, tasks: int) -> str:
    return f"{mean:+.4f} +/- {sigma:.4f} ({tasks})"


def markdown_table(fields: list[str], rows: list[dict[str, object]]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        vals = []
        for field in fields:
            val = row[field]
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def make_five_panel_rows(point_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {}
    for row in point_rows:
        key = (row["model"], row["alpha"], row["dimension"])
        by_key.setdefault(key, {})[row["scope"]] = row
    rows = []
    for model in MODELS:
        for alpha in ALPHAS:
            for dim in DIMENSIONS:
                key = (model, alpha, dim)
                panels = by_key.get(key, {})
                out = {
                    "model": model,
                    "alpha": alpha,
                    "dimension": dim,
                }
                for scope in SCOPES:
                    panel = panels.get(scope)
                    out[scope] = (
                        fmt_mean_sigma(float(panel["mean_mle"]), float(panel["sigma_mle"]), int(panel["complete_tasks"]))
                        if panel
                        else ""
                    )
                rows.append(out)
    return rows


def make_nll_rows(point_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen = {}
    for row in point_rows:
        key = (row["model"], row["alpha"], row["scope"])
        seen[key] = {
            "model": row["model"],
            "alpha": row["alpha"],
            "scope": row["scope"],
            "nll": row["nll"],
            "complete_tasks": row["complete_tasks"],
        }
    return [seen[k] for k in sorted(seen, key=lambda x: (x[0], x[1], SCOPES.index(x[2])))]


def write_report(point_rows: list[dict[str, object]], five_rows: list[dict[str, object]], nll_rows: list[dict[str, object]]) -> None:
    coverage = defaultdict(dict)
    for row in point_rows:
        coverage[row["model"]][row["scope"]] = row["complete_tasks"]

    report = [
        "# Eval6 Capability Fits by Module",
        "",
        "This table fits the three retained capability dimensions separately for `mmlu_pro`, `mmlu_redux`, `agieval`, `bbh`, and the union of all complete subtasks. For every model and every alpha point, capability means and sigmas are fitted independently by maximum likelihood over subtask deltas relative to alpha 0.",
        "",
        "Cell format: `mean +/- sigma (n_tasks)`. Means are capability-level score deltas. Sigmas are the independently fitted per-alpha uncertainty scales for that capability dimension.",
        "",
        "## Coverage",
        "",
    ]
    cov_rows = []
    for model in MODELS:
        row = {"model": model}
        for scope in SCOPES:
            row[scope] = coverage.get(model, {}).get(scope, 0)
        cov_rows.append(row)
    report.append(markdown_table(["model", *SCOPES], cov_rows))

    report += [
        "",
        "## Five-Panel Capability Table",
        "",
        markdown_table(["model", "alpha", "dimension", *SCOPES], five_rows),
        "",
        "## Scope-Level NLL",
        "",
        markdown_table(["model", "alpha", "scope", "nll", "complete_tasks"], nll_rows),
        "",
        "## Output Files",
        "",
        "- `capability_5panel_mean_sigma.csv`",
        "- `capability_independent_points_long.csv`",
        "- `capability_independent_task_predictions_long.csv`",
        "- `capability_independent_nll.csv`",
    ]
    REPORT.write_text("\n".join(report))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    score_rows = extract_scores()
    caps = read_caps()
    values, _ = build_values(score_rows)

    point_rows = []
    pred_rows = []
    for model in MODELS:
        for scope in SCOPES:
            points, preds = fit_scope(values, caps, model, scope)
            point_rows.extend(points)
            pred_rows.extend(preds)

    five_rows = make_five_panel_rows(point_rows)
    nll_rows = make_nll_rows(point_rows)

    write_csv(
        OUT / "capability_independent_points_long.csv",
        point_rows,
        ["scope", "model", "model_label", "alpha", "dimension", "mean_mle", "sigma_mle", "nll", "complete_tasks"],
    )
    write_csv(
        OUT / "capability_independent_task_predictions_long.csv",
        pred_rows,
        ["scope", "model", "alpha", "module", "task", "observed_delta", "predicted_delta", "residual", "complete_tasks", "nll"],
    )
    write_csv(OUT / "capability_5panel_mean_sigma.csv", five_rows, ["model", "alpha", "dimension", *SCOPES])
    write_csv(OUT / "capability_independent_nll.csv", nll_rows, ["model", "alpha", "scope", "nll", "complete_tasks"])
    write_report(point_rows, five_rows, nll_rows)


if __name__ == "__main__":
    main()
