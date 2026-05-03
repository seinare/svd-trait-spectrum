#!/usr/bin/env python3
"""Label benchmark subtasks with capability-dimension weights using a judge LLM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from datasets import get_dataset_config_names, load_dataset
from openai import OpenAI
from tqdm import tqdm


DIMENSIONS = [
    "Factual Knowledge",
    "Language Understanding",
    "Inductive Reasoning",
    "Deductive Reasoning",
    "Mathematical Computation",
    "Structural Analysis",
    "Ethical & Safety Judgment",
]

DIMENSION_DESCRIPTIONS = {
    "Factual Knowledge": "The ability to retrieve and recall world knowledge, concepts, terminology, and established facts. It represents crystallized intelligence and memory for specific information.",
    "Language Understanding": "The ability to precisely parse complex texts, interpret domain-specific jargon, resolve syntactic ambiguity, and grasp nuanced semantics. It reflects deep reading comprehension.",
    "Inductive Reasoning": "The ability to derive general rules or patterns from specific examples, categorize, perform analogical reasoning, and make data-driven predictions.",
    "Deductive Reasoning": "The ability to derive necessarily true conclusions from given premises through rigorous logical steps, applying general rules to specific cases.",
    "Mathematical Computation": "The ability to execute precise, multi-step arithmetic, algebraic, or symbolic manipulation. It focuses on formal operational accuracy.",
    "Structural Analysis": "The ability to decompose complex systems into constituent parts, understand relationships and causal links, and synthesize system-level behavior.",
    "Ethical & Safety Judgment": "The ability to identify moral dilemmas, assess safety implications, recognize social norms and legal boundaries, and distinguish compliant from harmful behavior.",
}

AGIEVAL_CONFIGS = {
    "agieval_aqua_rat": "aqua-rat",
    "agieval_logiqa_en": "logiqa-en",
    "agieval_lsat_ar": "lsat-ar",
    "agieval_lsat_lr": "lsat-lr",
    "agieval_lsat_rc": "lsat-rc",
    "agieval_sat_en": "sat-en",
    "agieval_sat_en_without_passage": "sat-en-without-passage",
    "agieval_sat_math": "sat-math",
}

GPQA_CONFIGS = {
    "gpqa_main_zeroshot": "gpqa_main",
    "gpqa_extended_zeroshot": "gpqa_extended",
    "gpqa_diamond_zeroshot": "gpqa_diamond",
}

SCRIPT_MMLU_SUBJECTS = [
    "abstract_algebra",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "elementary_mathematics",
    "formal_logic",
    "high_school_mathematics",
    "high_school_physics",
    "international_law",
    "logical_fallacies",
    "machine_learning",
    "miscellaneous",
    "moral_disputes",
    "philosophy",
    "professional_law",
    "professional_medicine",
]


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def all_mmlu_subjects() -> list[str]:
    last_error = None
    for attempt in range(1, 6):
        try:
            configs = get_dataset_config_names("cais/mmlu")
            break
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            wait_s = min(10 * attempt, 60)
            print(
                json.dumps(
                    {
                        "event": "dataset_config_retry",
                        "dataset": "cais/mmlu",
                        "attempt": attempt,
                        "wait_s": wait_s,
                        "error": last_error,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            time.sleep(wait_s)
    else:
        raise RuntimeError(f"failed to list cais/mmlu configs: {last_error}")
    return sorted(c for c in configs if c not in {"all", "auxiliary_train"})


def build_task_specs(scope: str) -> list[dict[str, str]]:
    if scope == "script":
        mmlu_subjects = SCRIPT_MMLU_SUBJECTS
    else:
        mmlu_subjects = all_mmlu_subjects()

    specs = []
    for subject in mmlu_subjects:
        specs.append({"group": "mmlu", "task": f"mmlu_{subject}", "dataset": "cais/mmlu", "config": subject})
    for task, config in AGIEVAL_CONFIGS.items():
        specs.append({"group": "agieval", "task": task, "dataset": "RUCAIBox/agieval", "config": config})
    for task, config in GPQA_CONFIGS.items():
        specs.append({"group": "gpqa", "task": task, "dataset": "Idavidrein/gpqa", "config": config})
    return specs


def choice_label(index: int) -> str:
    return chr(ord("A") + index)


def normalize_options(options: Any) -> list[str]:
    if options is None:
        return []
    if isinstance(options, list):
        return [str(x) for x in options]
    if isinstance(options, dict):
        return [f"{k}. {v}" for k, v in options.items()]
    return [str(options)]


def format_mmlu(config: str, row: dict[str, Any], index: int) -> dict[str, Any]:
    answer_idx = int(row["answer"])
    answer = f"{choice_label(answer_idx)}. {row['choices'][answer_idx]}"
    return {
        "item_id": stable_id(f"mmlu:{config}:{index}:{row['question']}"),
        "question": row["question"],
        "options": [f"{choice_label(i)}. {choice}" for i, choice in enumerate(row["choices"])],
        "answer": answer,
        "explanation": "",
    }


def format_agieval(config: str, row: dict[str, Any], index: int) -> dict[str, Any]:
    parts = []
    if row.get("passage"):
        parts.append(f"Passage:\n{row['passage']}")
    parts.append(f"Question:\n{row.get('question', '')}")
    answer_label = str(row.get("label", "")).strip()
    options = normalize_options(row.get("options"))
    answer_text = answer_label
    for opt in options:
        if opt.strip().startswith(f"({answer_label})") or opt.strip().startswith(f"{answer_label}."):
            answer_text = opt
            break
    other = row.get("other") or {}
    explanation = row.get("explanation") or other.get("solution") or ""
    return {
        "item_id": stable_id(f"agieval:{config}:{index}:{parts[-1]}"),
        "question": "\n\n".join(parts),
        "options": options,
        "answer": answer_text,
        "explanation": explanation,
    }


def format_gpqa(config: str, row: dict[str, Any], index: int) -> dict[str, Any]:
    correct = str(row.get("Correct Answer", "")).strip()
    incorrect = [
        str(row.get("Incorrect Answer 1", "")).strip(),
        str(row.get("Incorrect Answer 2", "")).strip(),
        str(row.get("Incorrect Answer 3", "")).strip(),
    ]
    options = [correct] + [x for x in incorrect if x]
    return {
        "item_id": stable_id(f"gpqa:{config}:{index}:{row.get('Question', '')}"),
        "question": str(row.get("Question", "")),
        "options": options,
        "answer": correct,
        "explanation": str(row.get("Explanation", "")),
    }


def load_dataset_with_retry(dataset: str, config: str, split: str, attempts: int = 5) -> Any:
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return load_dataset(dataset, config, split=split)
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            wait_s = min(10 * attempt, 60)
            print(
                json.dumps(
                    {
                        "event": "dataset_load_retry",
                        "dataset": dataset,
                        "config": config,
                        "split": split,
                        "attempt": attempt,
                        "wait_s": wait_s,
                        "error": last_error,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            time.sleep(wait_s)
    raise RuntimeError(f"failed to load {dataset}/{config}/{split}: {last_error}")


def load_items(spec: dict[str, str]) -> list[dict[str, Any]]:
    if spec["group"] == "mmlu":
        ds = load_dataset_with_retry(spec["dataset"], spec["config"], "test")
        return [format_mmlu(spec["config"], row, idx) for idx, row in enumerate(ds)]
    if spec["group"] == "agieval":
        ds = load_dataset_with_retry(spec["dataset"], spec["config"], "test")
        return [format_agieval(spec["config"], row, idx) for idx, row in enumerate(ds)]
    if spec["group"] == "gpqa":
        ds = load_dataset_with_retry(spec["dataset"], spec["config"], "train")
        return [format_gpqa(spec["config"], row, idx) for idx, row in enumerate(ds)]
    raise ValueError(spec["group"])


def build_prompt(task: str, item: dict[str, Any]) -> str:
    dims = "\n".join(f"- {name}: {DIMENSION_DESCRIPTIONS[name]}" for name in DIMENSIONS)
    options = "\n".join(item["options"]) if item["options"] else "(No explicit options)"
    explanation = item.get("explanation") or "(No explanation provided)"
    return f"""You are analyzing what capabilities are required to solve a benchmark item, not solving it.

Assign capability weights across the seven dimensions below. The weights must be non-negative numbers and sum to exactly 1.0. Use the provided gold answer and explanation when judging the required capabilities.

Capability dimensions:
{dims}

Benchmark subtask: {task}

Question:
{item['question']}

Options:
{options}

Gold answer:
{item['answer']}

Gold explanation:
{explanation}

Return only valid JSON with this exact schema:
{{
  "weights": {{
    "Factual Knowledge": 0.0,
    "Language Understanding": 0.0,
    "Inductive Reasoning": 0.0,
    "Deductive Reasoning": 0.0,
    "Mathematical Computation": 0.0,
    "Structural Analysis": 0.0,
    "Ethical & Safety Judgment": 0.0
  }}
}}
"""


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_response(payload: dict[str, Any]) -> dict[str, float]:
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("missing weights object")
    cleaned = {}
    for dim in DIMENSIONS:
        if dim not in weights:
            raise ValueError(f"missing dimension {dim}")
        value = float(weights[dim])
        if value < -1e-9:
            raise ValueError(f"negative weight {dim}={value}")
        cleaned[dim] = max(0.0, value)
    total = sum(cleaned.values())
    if not (0.995 <= total <= 1.005):
        raise ValueError(f"weights sum {total}, expected 1")
    return {dim: cleaned[dim] / total for dim in DIMENSIONS}


def judge_item(
    client: OpenAI,
    model: str,
    task: str,
    item: dict[str, Any],
    max_retries: int,
    request_timeout: int,
    max_tokens: int,
) -> dict[str, Any]:
    prompt = build_prompt(task, item)
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a careful benchmark capability taxonomist. Output only JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
                timeout=request_timeout,
            )
            text = response.choices[0].message.content or ""
            payload = parse_json(text)
            weights = validate_response(payload)
            return {
                "ok": True,
                "attempt": attempt,
                "weights": weights,
                "raw": text,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            time.sleep(min(2 * attempt, 10))
    return {"ok": False, "error": last_error}


def read_existing(path: Path) -> set[tuple[str, str]]:
    done = set()
    if not path.exists():
        return done
    with path.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ok"):
                done.add((row["task"], row["item_id"]))
    return done


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def summarize(jsonl_path: Path, summary_csv: Path) -> None:
    buckets: dict[str, dict[str, Any]] = {}
    seen: set[tuple[str, str]] = set()
    if not jsonl_path.exists():
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with summary_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["group", "task", "n", *DIMENSIONS])
            writer.writeheader()
        return
    with jsonl_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("ok"):
                continue
            key = (row["task"], row["item_id"])
            if key in seen:
                continue
            seen.add(key)
            task = row["task"]
            bucket = buckets.setdefault(
                task,
                {
                    "group": row["group"],
                    "task": task,
                    "n": 0,
                    **{dim: 0.0 for dim in DIMENSIONS},
                },
            )
            bucket["n"] += 1
            for dim in DIMENSIONS:
                bucket[dim] += float(row["weights"][dim])

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="") as handle:
        fields = ["group", "task", "n", *DIMENSIONS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in sorted(buckets):
            row = buckets[task]
            out = {"group": row["group"], "task": task, "n": row["n"]}
            for dim in DIMENSIONS:
                out[dim] = row[dim] / row["n"] if row["n"] else ""
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["all", "script"], default="all")
    parser.add_argument("--samples_per_task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base_url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
    parser.add_argument("--api_key_env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--request_timeout", type=int, default=300)
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--output_jsonl", type=Path, default=Path("docs/results/capability_dimension_labels.jsonl"))
    parser.add_argument("--summary_csv", type=Path, default=Path("docs/results/capability_dimension_summary.csv"))
    parser.add_argument("--task", action="append", default=None, help="Optional exact task names to run.")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    task_filter = set(args.task or [])
    specs = build_task_specs(args.scope)
    if task_filter:
        specs = [spec for spec in specs if spec["task"] in task_filter]
    done = read_existing(args.output_jsonl)

    rng = random.Random(args.seed)
    for spec in specs:
        items = load_items(spec)
        rng.shuffle(items)
        valid_for_task = sum(1 for task, _ in done if task == spec["task"])
        pbar = tqdm(total=args.samples_per_task, initial=min(valid_for_task, args.samples_per_task), desc=spec["task"])
        for item in items:
            if valid_for_task >= args.samples_per_task:
                break
            key = (spec["task"], item["item_id"])
            if key in done:
                continue
            judged = judge_item(
                client,
                args.model,
                spec["task"],
                item,
                args.max_retries,
                args.request_timeout,
                args.max_tokens,
            )
            row = {
                "group": spec["group"],
                "task": spec["task"],
                "dataset": spec["dataset"],
                "config": spec["config"],
                "item_id": item["item_id"],
                "question": item["question"],
                "options": item["options"],
                "answer": item["answer"],
                "ok": judged["ok"],
            }
            row.update(judged)
            append_jsonl(args.output_jsonl, row)
            print(
                json.dumps(
                    {
                        "event": "judged",
                        "task": spec["task"],
                        "item_id": item["item_id"],
                        "ok": judged["ok"],
                        "attempt": judged.get("attempt"),
                        "error": judged.get("error"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if judged["ok"]:
                done.add(key)
                valid_for_task += 1
                pbar.update(1)
        pbar.close()
        summarize(args.output_jsonl, args.summary_csv)

    summarize(args.output_jsonl, args.summary_csv)
    print(f"Wrote {args.output_jsonl}")
    print(f"Wrote {args.summary_csv}")


if __name__ == "__main__":
    main()
