#!/usr/bin/env python3
"""Summarize synced Llama/Qwen experiment artifacts into CSV/Markdown tables."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "docs/results/raw/remote_sync"
OUT = ROOT / "docs/results"

ALPHAS = {
    "llama1b": [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
    "llama3b": [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
    "llama8b": [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
    "qwen3_8b": [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
}

SUMMARY_TASKS = [
    ("standard", "GSM8K", "correct_rate", "GSM8K"),
    ("standard", "MATH", "correct_rate", "MATH"),
    ("standard", "ARC-Challenge", "correct_rate", "ARC"),
    ("standard", "DROP", "correct_rate", "DROP"),
    ("bfcl", "BFCL_v3_simple", "Retry_enabled_success", "BFCL_retry"),
    ("judge", "TruthfulQA", "score", "TruthfulQA"),
    ("judge", "HaluEval", "score", "HaluEval"),
    ("judge", "AdvBench", "score", "AdvBench"),
    ("standard_cot", "GSM8K", "correct_rate", "GSM8K_cot"),
    ("standard_cot", "MATH", "correct_rate", "MATH_cot"),
    ("lm_eval", "hellaswag", "value", "HellaSwag"),
    ("lm_eval", "ifeval", "value", "IFEval"),
]


def model_from_path(path: Path) -> str | None:
    s = str(path)
    if "Llama-3_2-1B" in s or "Llama-3.2-1B" in s:
        return "llama1b"
    if "Llama-3_2-3B" in s or "Llama-3.2-3B" in s:
        return "llama3b"
    if "Llama-3_1-8B" in s or "Llama-3.1-8B" in s:
        return "llama8b"
    if "Qwen3-8B" in s:
        return "qwen3_8b"
    return None


def alpha_from_path(path: Path, data: dict | None = None) -> float | None:
    if data:
        config = data.get("config") or {}
        if "alpha" in config:
            return round(float(config["alpha"]), 1)
    text = str(path)
    patterns = [
        r"res_alpha(-?\d+(?:\.\d+)?)",
        r"_alpha(-?\d+(?:\.\d+)?)_vllm",
        r"-alpha(-?\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return round(float(m.group(1)), 1)
    return None


def module_from_path(path: Path) -> str | None:
    parts = set(path.parts)
    for module in ["standard_cot", "standard", "bfcl", "judge", "lm_eval"]:
        if module in parts:
            return module
    return None


def fmt_alpha(alpha: float) -> str:
    if alpha == 0:
        return "0"
    return f"{alpha:.1f}"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def parse_module_rows() -> list[dict]:
    rows = []
    seen = set()
    for path in sorted(RAW.rglob("*.json")):
        module = module_from_path(path)
        if module not in {"standard", "standard_cot", "bfcl", "judge"}:
            continue
        model = model_from_path(path)
        data = read_json(path)
        alpha = alpha_from_path(path, data)
        if model is None or data is None or alpha is None:
            continue

        def add(task: str, metric: str, value, detail: dict | None = None) -> None:
            key = (model, alpha, module, task, metric)
            if key in seen:
                return
            seen.add(key)
            row = {
                "model": model,
                "alpha": alpha,
                "module": module,
                "task": task,
                "metric": metric,
                "value": value,
                "source": str(path.relative_to(ROOT)),
            }
            if detail:
                row.update(detail)
            rows.append(row)

        if module in {"standard", "standard_cot"}:
            score_block = data.get("scores_no_cot") if module == "standard" else data.get("scores_cot")
            if not isinstance(score_block, dict):
                continue
            for task, stats in score_block.items():
                if not isinstance(stats, dict):
                    continue
                task_name = "ARC-Challenge" if task in {"ARC", "ARC-Challenge"} else task
                add(
                    task_name,
                    "correct_rate",
                    stats.get("Correct_Rate"),
                    {
                        "format_error_rate": stats.get("Format_Error_Rate", ""),
                        "wrong_answer_rate": stats.get("Wrong_Answer_Rate", ""),
                        "length_finish_rate": stats.get("Length_Finish_Rate", ""),
                    },
                )
        elif module == "bfcl":
            metrics = data.get("metrics") or {}
            for metric, value in metrics.items():
                add("BFCL_v3_simple", metric, value)
        elif module == "judge":
            scores = data.get("scores") or {}
            for task, value in scores.items():
                add(task, "score", value)
    return sorted(rows, key=lambda r: (r["model"], float(r["alpha"]), r["module"], r["task"], r["metric"]))


def parse_lm_eval_rows() -> list[dict]:
    rows = []
    seen = set()
    for path in sorted(RAW.rglob("results_*.json")):
        model = model_from_path(path)
        data = read_json(path)
        alpha = alpha_from_path(path, data)
        if model is None or data is None or alpha is None:
            continue
        results = data.get("results") or {}
        for task, metrics in results.items():
            if task.startswith("mmlu_"):
                group, metric = "mmlu", "acc,none"
            elif task.startswith("gpqa_"):
                group, metric = "gpqa", "acc_norm,none"
            elif task.startswith("agieval_"):
                group, metric = "agieval", "acc_norm,none"
            elif task == "hellaswag":
                group, metric = "hellaswag", "acc_norm,none"
            elif task == "ifeval":
                group, metric = "ifeval", "prompt_level_strict_acc,none"
            else:
                continue
            value = metrics.get(metric)
            if value is None:
                continue
            key = (model, alpha, group, task, metric)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "model": model,
                    "alpha": alpha,
                    "group": group,
                    "task": task,
                    "metric": metric,
                    "value": value,
                    "source": str(path.relative_to(ROOT)),
                }
            )
    return sorted(rows, key=lambda r: (r["model"], float(r["alpha"]), r["group"], r["task"]))


def make_summary_wide(module_rows: list[dict], lm_rows: list[dict]) -> list[dict]:
    values = {}
    for row in module_rows:
        values[(row["model"], float(row["alpha"]), row["module"], row["task"], row["metric"])] = row["value"]
    for row in lm_rows:
        if row["task"] in {"hellaswag", "ifeval"}:
            values[(row["model"], float(row["alpha"]), "lm_eval", row["task"], "value")] = row["value"]

    rows = []
    for model, alphas in ALPHAS.items():
        for alpha in alphas:
            out = {"model": model, "alpha": alpha}
            for module, task, metric, col in SUMMARY_TASKS:
                out[col] = values.get((model, alpha, module, task, metric), "")
            rows.append(out)
    return rows


def make_lm_wide(lm_rows: list[dict], groups: set[str]) -> list[dict]:
    by_key = defaultdict(dict)
    meta = {}
    for row in lm_rows:
        if row["group"] not in groups:
            continue
        key = (row["model"], row["group"], row["task"], row["metric"])
        by_key[key][float(row["alpha"])] = float(row["value"])
        meta[key] = row
    rows = []
    for key in sorted(by_key):
        model, group, task, metric = key
        vals = by_key[key]
        alphas = ALPHAS[model]
        base = vals.get(0.0)
        best_alpha, best_value = max(vals.items(), key=lambda kv: kv[1])
        out = {"model": model, "group": group, "task": task, "metric": metric}
        for alpha in alphas:
            out[f"alpha_{fmt_alpha(alpha)}"] = vals.get(alpha, "")
        out["best_alpha"] = best_alpha
        out["best_value"] = best_value
        out["delta_best_vs_0"] = "" if base is None else best_value - base
        rows.append(out)
    return rows


def markdown_table(rows: list[dict], fields: list[str], max_rows: int | None = None) -> str:
    rows = rows[:max_rows] if max_rows else rows
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        vals = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    module_rows = parse_module_rows()
    lm_rows = parse_lm_eval_rows()
    summary_rows = make_summary_wide(module_rows, lm_rows)
    lm_wide_all = make_lm_wide(lm_rows, {"mmlu", "gpqa", "agieval"})
    mmlu_wide = make_lm_wide(lm_rows, {"mmlu"})
    gpqa_wide = make_lm_wide(lm_rows, {"gpqa"})
    agieval_wide = make_lm_wide(lm_rows, {"agieval"})

    module_fields = [
        "model",
        "alpha",
        "module",
        "task",
        "metric",
        "value",
        "format_error_rate",
        "wrong_answer_rate",
        "length_finish_rate",
        "source",
    ]
    lm_fields = ["model", "alpha", "group", "task", "metric", "value", "source"]
    summary_fields = ["model", "alpha"] + [col for *_, col in SUMMARY_TASKS]
    alpha_fields = ["model", "group", "task", "metric"]
    for alpha in [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]:
        alpha_fields.append(f"alpha_{fmt_alpha(alpha)}")
    alpha_fields += ["best_alpha", "best_value", "delta_best_vs_0"]

    write_csv(OUT / "all_module_metrics_long.csv", module_rows, module_fields)
    write_csv(OUT / "all_lm_eval_subtasks_long.csv", lm_rows, lm_fields)
    write_csv(OUT / "all_module_summary_wide.csv", summary_rows, summary_fields)
    write_csv(OUT / "all_lm_eval_alpha_subtasks.csv", lm_wide_all, alpha_fields)
    write_csv(OUT / "all_mmlu_subtasks_alpha.csv", mmlu_wide, alpha_fields)
    write_csv(OUT / "all_gpqa_subtasks_alpha.csv", gpqa_wide, alpha_fields)
    write_csv(OUT / "all_agieval_subtasks_alpha.csv", agieval_wide, alpha_fields)

    coverage = []
    modules = ["standard", "bfcl", "judge", "standard_cot", "lm_eval"]
    present = {(r["model"], float(r["alpha"]), r["module"]) for r in module_rows}
    present |= {(r["model"], float(r["alpha"]), "lm_eval") for r in lm_rows}
    for model, alphas in ALPHAS.items():
        for alpha in alphas:
            for module in modules:
                coverage.append({"model": model, "alpha": alpha, "module": module, "present": (model, alpha, module) in present})
    write_csv(OUT / "all_coverage_matrix.csv", coverage, ["model", "alpha", "module", "present"])

    missing = [r for r in coverage if not r["present"]]
    md = [
        "# Llama/Qwen Result Summary",
        "",
        "Generated from synced raw artifacts under `docs/results/raw/remote_sync/`.",
        "",
        "Primary CSV outputs:",
        "- `all_module_summary_wide.csv`: one row per model/alpha with standard, BFCL, judge, CoT, HellaSwag, and IFEval metrics.",
        "- `all_module_metrics_long.csv`: long-form standard/BFCL/judge/standard-cot metrics.",
        "- `all_lm_eval_subtasks_long.csv`: long-form MMLU, GPQA, AGIEval, HellaSwag, and IFEval metrics.",
        "- `all_mmlu_subtasks_alpha.csv`, `all_gpqa_subtasks_alpha.csv`, `all_agieval_subtasks_alpha.csv`: requested subtask alpha-gradient tables.",
        "",
        "## Coverage",
        "",
    ]
    if missing:
        md.append(markdown_table(missing, ["model", "alpha", "module", "present"]))
    else:
        md.append("All expected model/alpha/module combinations are present.")
    md += [
        "",
        "## Main Module Summary",
        "",
        markdown_table(summary_rows, summary_fields),
        "",
        "## MMLU Subtask Alpha Table",
        "",
        markdown_table(mmlu_wide, alpha_fields),
        "",
        "## GPQA Subtask Alpha Table",
        "",
        markdown_table(gpqa_wide, alpha_fields),
        "",
        "## AGIEval Subtask Alpha Table",
        "",
        markdown_table(agieval_wide, alpha_fields),
        "",
    ]
    (OUT / "llama_qwen_results_summary.md").write_text("\n".join(md))

    manifest = {
        "module_rows": len(module_rows),
        "lm_eval_rows": len(lm_rows),
        "summary_rows": len(summary_rows),
        "mmlu_rows": len(mmlu_wide),
        "gpqa_rows": len(gpqa_wide),
        "agieval_rows": len(agieval_wide),
        "missing": missing,
    }
    (OUT / "all_result_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
