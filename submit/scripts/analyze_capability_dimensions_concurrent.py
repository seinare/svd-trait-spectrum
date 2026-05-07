#!/usr/bin/env python3
"""Concurrent batched capability-dimension labeling for selected lm-eval subtasks.

Each judge request contains 10 questions with gold answers from one subtask and
asks for one capability mixture for that subtask batch. Each subtask is judged
for N valid batches, then averaged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from eval_script6_lm_eval_full_subtasks import (
        AGIEVAL_SUBTASKS,
        BBH_ZEROSHOT_SUBTASKS,
        MMLU_PRO_SUBTASKS,
        MMLU_REDUX_SUBTASKS,
    )
    from analyze_capability_dimensions import (
        AGIEVAL_CONFIGS,
        DIMENSIONS,
        DIMENSION_DESCRIPTIONS,
        SCRIPT_MMLU_SUBJECTS,
        load_items,
        parse_json,
    )
except ModuleNotFoundError:
    from scripts.eval_script6_lm_eval_full_subtasks import (
        AGIEVAL_SUBTASKS,
        BBH_ZEROSHOT_SUBTASKS,
        MMLU_PRO_SUBTASKS,
        MMLU_REDUX_SUBTASKS,
    )
    from scripts.analyze_capability_dimensions import (
        AGIEVAL_CONFIGS,
        DIMENSIONS,
        DIMENSION_DESCRIPTIONS,
        SCRIPT_MMLU_SUBJECTS,
        load_items,
        parse_json,
    )


ALL_DIMENSIONS = list(DIMENSIONS)
CORE3_DIMENSIONS = [
    "Factual Knowledge",
    "Language Understanding",
    "Deductive Reasoning",
]
AGIEVAL_ENGLISH_TASKS = [
    "agieval_aqua_rat",
    "agieval_logiqa_en",
    "agieval_lsat_ar",
    "agieval_lsat_lr",
    "agieval_lsat_rc",
    "agieval_sat_en",
    "agieval_sat_en_without_passage",
    "agieval_sat_math",
]

THREAD_LOCAL = threading.local()
JSONL_LOCK = threading.Lock()
LAST_REQUEST_LOCK = threading.Lock()
LAST_REQUEST_TIME = 0.0
TASK_MANAGER_LOCK = threading.Lock()
TASK_MANAGER = None


def set_dimension_profile(profile: str) -> None:
    global DIMENSIONS
    if profile == "full":
        DIMENSIONS = list(ALL_DIMENSIONS)
    elif profile == "core3":
        DIMENSIONS = list(CORE3_DIMENSIONS)
    else:
        raise ValueError(f"unknown dimension profile: {profile}")


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


def parse_judge_json(text: str) -> dict[str, Any]:
    """Parse judge JSON while tolerating markdown fences and surrounding text."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty judge response")
    try:
        return parse_json(raw)
    except Exception as first_error:  # noqa: BLE001
        candidates = []
        fenced = re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.S | re.I)
        candidates.extend(fenced)
        weight_start = raw.find('"weights"')
        if weight_start >= 0:
            left = raw.rfind("{", 0, weight_start)
            right = raw.find("}", weight_start)
            if left >= 0 and right >= 0:
                # Include the outer object if present; otherwise wrap a weights-only object.
                outer_right = raw.find("}", right + 1)
                if outer_right >= 0:
                    candidates.append(raw[left : outer_right + 1])
                candidates.append("{" + raw[left : right + 1].lstrip("{"))
        for match in re.finditer(r"\{", raw):
            start = match.start()
            depth = 0
            in_string = False
            escape = False
            for idx, ch in enumerate(raw[start:], start=start):
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(raw[start : idx + 1])
                        break
        for cand in candidates:
            cand = cand.strip()
            if not cand:
                continue
            try:
                payload = json.loads(cand)
            except Exception:
                continue
            if isinstance(payload, dict) and "weights" in payload:
                return payload
        raise ValueError(f"failed to parse judge JSON: {first_error}; raw_prefix={raw[:300]!r}")


def script_task_specs() -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for subject in SCRIPT_MMLU_SUBJECTS:
        specs.append(
            {
                "group": "mmlu",
                "task": f"mmlu_{subject}",
                "dataset": "cais/mmlu",
                "config": subject,
            }
        )
    for task in AGIEVAL_ENGLISH_TASKS:
        specs.append(
            {
                "group": "agieval",
                "task": task,
                "dataset": "RUCAIBox/agieval",
                "config": AGIEVAL_CONFIGS[task],
            }
        )
    return specs


def eval6_task_specs() -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for group, tasks in [
        ("mmlu_pro", MMLU_PRO_SUBTASKS),
        ("mmlu_redux", MMLU_REDUX_SUBTASKS),
        ("agieval", AGIEVAL_SUBTASKS),
        ("bbh", BBH_ZEROSHOT_SUBTASKS),
    ]:
        for task in tasks:
            specs.append(
                {
                    "group": group,
                    "task": task,
                    "dataset": "lm_eval",
                    "config": task,
                    "loader": "lm_eval",
                }
            )
    return specs


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def get_client(api_key: str, base_url: str) -> OpenAI:
    client = getattr(THREAD_LOCAL, "client", None)
    if client is None:
        client = OpenAI(api_key=api_key, base_url=base_url)
        THREAD_LOCAL.client = client
    return client


def get_task_manager():
    global TASK_MANAGER
    with TASK_MANAGER_LOCK:
        if TASK_MANAGER is None:
            from lm_eval.tasks import TaskManager

            TASK_MANAGER = TaskManager()
        return TASK_MANAGER


def first_available_docs(task) -> list[dict[str, Any]]:
    for method in ("test_docs", "validation_docs", "training_docs"):
        if not hasattr(task, method):
            continue
        docs = getattr(task, method)()
        if docs:
            return list(docs)
    raise ValueError(f"no docs available for task {getattr(task, 'config', None)}")


def safe_call(task, method: str, doc: dict[str, Any]) -> Any:
    if not hasattr(task, method):
        return None
    try:
        return getattr(task, method)(doc)
    except Exception:
        return None


def normalize_target(target: Any, choices: list[str]) -> str:
    if isinstance(target, list):
        return ", ".join(normalize_target(item, choices) for item in target)
    if isinstance(target, int):
        if 0 <= target < len(choices):
            return f"{chr(ord('A') + target)}. {choices[target]}"
        return str(target)
    if isinstance(target, str):
        text = target.strip()
        if len(text) == 1 and "A" <= text.upper() <= "Z" and choices:
            idx = ord(text.upper()) - ord("A")
            if 0 <= idx < len(choices):
                return f"{text.upper()}. {choices[idx]}"
        return text
    return str(target)


def fallback_choices(doc: dict[str, Any]) -> list[str]:
    for key in ("choices", "options"):
        value = doc.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, dict):
            return [f"{key}. {val}" for key, val in value.items()]
    return []


def load_lm_eval_items(spec: dict[str, str]) -> list[dict[str, Any]]:
    task = get_task_manager().load_task_or_group([spec["task"]])[spec["task"]]
    docs = first_available_docs(task)
    items = []
    for idx, doc in enumerate(docs):
        text = safe_call(task, "doc_to_text", doc)
        if not text:
            text = doc.get("question") or doc.get("query") or doc.get("input") or json.dumps(doc, ensure_ascii=False)
        choices = safe_call(task, "doc_to_choice", doc)
        if not isinstance(choices, list):
            choices = fallback_choices(doc)
        choices = [str(choice) for choice in choices]
        target = safe_call(task, "doc_to_target", doc)
        if target is None:
            target = doc.get("answer", doc.get("gold", doc.get("target", "")))
        items.append(
            {
                "item_id": stable_id(f"{spec['task']}:{idx}:{json.dumps(doc, ensure_ascii=False, sort_keys=True, default=str)}"),
                "question": str(text),
                "options": choices,
                "answer": normalize_target(target, choices),
                "explanation": str(doc.get("explanation") or doc.get("cot_content") or doc.get("potential_reason") or ""),
            }
        )
    if not items:
        raise ValueError(f"no formatted items for {spec['task']}")
    return items


def read_done(path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not path.exists():
        return done
    with path.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ok") and "repeat" in row:
                done.add((row["task"], int(row["repeat"])))
    return done


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with JSONL_LOCK:
        with path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


def summarize(jsonl_path: Path, summary_csv: Path, coverage_csv: Path) -> None:
    buckets: dict[str, dict[str, Any]] = {}
    seen: set[tuple[str, int]] = set()
    if jsonl_path.exists():
        with jsonl_path.open() as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not row.get("ok"):
                    continue
                key = (row["task"], int(row["repeat"]))
                if key in seen:
                    continue
                seen.add(key)
                task = row["task"]
                bucket = buckets.setdefault(
                    task,
                    {
                        "group": row["group"],
                        "task": task,
                        "valid_batches": 0,
                        "questions": 0,
                        **{dim: 0.0 for dim in DIMENSIONS},
                    },
                )
                bucket["valid_batches"] += 1
                bucket["questions"] += int(row.get("batch_size") or 0)
                for dim in DIMENSIONS:
                    bucket[dim] += float(row["weights"][dim])

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="") as handle:
        fields = ["group", "task", "valid_batches", "questions", *DIMENSIONS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in sorted(buckets):
            row = buckets[task]
            out = {
                "group": row["group"],
                "task": task,
                "valid_batches": row["valid_batches"],
                "questions": row["questions"],
            }
            for dim in DIMENSIONS:
                out[dim] = row[dim] / row["valid_batches"] if row["valid_batches"] else ""
            writer.writerow(out)

    coverage: dict[str, dict[str, Any]] = {}
    for task, repeat in seen:
        row = coverage.setdefault(task, {"task": task, "valid_batches": 0})
        row["valid_batches"] += 1
    with coverage_csv.open("w", newline="") as handle:
        fields = ["task", "valid_batches"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in sorted(coverage):
            writer.writerow(coverage[task])


def throttle(min_interval_s: float) -> None:
    if min_interval_s <= 0:
        return
    global LAST_REQUEST_TIME
    with LAST_REQUEST_LOCK:
        now = time.monotonic()
        wait_s = LAST_REQUEST_TIME + min_interval_s - now
        if wait_s > 0:
            time.sleep(wait_s)
        LAST_REQUEST_TIME = time.monotonic()


def build_batch_prompt(task: str, items: list[dict[str, Any]]) -> str:
    dims = "\n".join(f"- {name}: {DIMENSION_DESCRIPTIONS[name]}" for name in DIMENSIONS)
    schema = ",\n    ".join(f'"{name}": 0.0' for name in DIMENSIONS)
    blocks = []
    for idx, item in enumerate(items, start=1):
        options = "\n".join(item["options"]) if item["options"] else "(No explicit options)"
        explanation = item.get("explanation") or "(No explanation provided)"
        blocks.append(
            f"""Item {idx}
Question:
{item['question']}

Options:
{options}

Gold answer:
{item['answer']}

Gold explanation:
{explanation}"""
        )
    item_text = "\n\n---\n\n".join(blocks)
    return f"""You are analyzing what capabilities are required by a benchmark subtask, not solving the items.

You will receive 10 representative items from the same benchmark subtask, including gold answers. Judge the overall capability mixture required by this subtask batch. Do not assign weights per item; return one aggregate weight vector for the whole batch.

Assign capability weights across the {len(DIMENSIONS)} dimensions below. The weights must be non-negative numbers and sum to exactly 1.0.

Critical output rules:
- Return exactly one JSON object and nothing else.
- Do not wrap the JSON in markdown fences.
- Do not include explanations, comments, analysis, or extra keys.
- Use exactly the dimension names shown in the schema.
- The numeric weights must sum to 1.0.

Capability dimensions:
{dims}

Benchmark subtask: {task}

Representative items:
{item_text}

Return only valid JSON with this exact schema:
{{
  "weights": {{
    {schema}
  }}
}}
"""


def judge_batch(
    *,
    api_key: str,
    base_url: str,
    model: str,
    spec: dict[str, str],
    items: list[dict[str, Any]],
    repeat: int,
    max_retries: int,
    request_timeout: int,
    max_tokens: int,
    min_interval_s: float,
) -> dict[str, Any]:
    prompt = build_batch_prompt(spec["task"], items)
    last_error = None
    last_raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            throttle(min_interval_s)
            client = get_client(api_key, base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful benchmark capability taxonomist. Return only a strict JSON object. No markdown, no prose.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
                timeout=request_timeout,
            )
            text = response.choices[0].message.content or ""
            last_raw = text
            payload = parse_judge_json(text)
            weights = validate_response(payload)
            return {
                "group": spec["group"],
                "task": spec["task"],
                "dataset": spec["dataset"],
                "config": spec["config"],
                "repeat": repeat,
                "batch_size": len(items),
                "item_ids": [item["item_id"] for item in items],
                "questions": [
                    {
                        "item_id": item["item_id"],
                        "question": item["question"],
                        "options": item["options"],
                        "answer": item["answer"],
                    }
                    for item in items
                ],
                "ok": True,
                "attempt": attempt,
                "weights": weights,
                "raw": text,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            time.sleep(min(2 * attempt, 10))
    return {
        "group": spec["group"],
        "task": spec["task"],
        "dataset": spec["dataset"],
        "config": spec["config"],
        "repeat": repeat,
        "batch_size": len(items),
        "item_ids": [item["item_id"] for item in items],
        "ok": False,
        "error": last_error,
        "raw": last_raw,
    }


def select_batches(
    spec: dict[str, str],
    samples_per_batch: int,
    repeats_per_task: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    if spec.get("loader") == "lm_eval":
        items = load_lm_eval_items(spec)
    else:
        items = load_items(spec)
    rng = random.Random(f"{seed}:{spec['task']}")
    rng.shuffle(items)
    total_needed = samples_per_batch * repeats_per_task
    if len(items) >= total_needed:
        selected = items[:total_needed]
    else:
        selected = list(items)
        while len(selected) < total_needed:
            extra = list(items)
            rng.shuffle(extra)
            selected.extend(extra)
    return [
        selected[i * samples_per_batch : (i + 1) * samples_per_batch]
        for i in range(repeats_per_task)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=["legacy", "eval6"],
        default="legacy",
        help="legacy uses MMLU20 + AGIEval English; eval6 uses the full eval_script6 subtask set.",
    )
    parser.add_argument("--samples_per_batch", type=int, default=10)
    parser.add_argument("--repeats_per_task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base_url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
    parser.add_argument("--api_key_env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max_workers", type=int, default=3)
    parser.add_argument("--max_inflight", type=int, default=6)
    parser.add_argument("--min_interval_s", type=float, default=0.75)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--request_timeout", type=int, default=180)
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument(
        "--dimension_profile",
        choices=["full", "core3"],
        default="full",
        help="full uses the original seven dimensions; core3 uses Factual/Language/Deductive only.",
    )
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_csv", type=Path, required=True)
    parser.add_argument("--coverage_csv", type=Path, required=True)
    parser.add_argument("--task", action="append", default=None, help="Optional exact task filter.")
    args = parser.parse_args()
    set_dimension_profile(args.dimension_profile)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    specs = eval6_task_specs() if args.suite == "eval6" else script_task_specs()
    if args.task:
        wanted = set(args.task)
        specs = [spec for spec in specs if spec["task"] in wanted]

    done = read_done(args.output_jsonl)
    total_expected = len(specs) * args.repeats_per_task
    print(
        json.dumps(
            {
                "event": "start",
                "tasks": len(specs),
                "samples_per_batch": args.samples_per_batch,
                "repeats_per_task": args.repeats_per_task,
                "expected_judgments": total_expected,
                "already_done": len(done),
                "max_workers": args.max_workers,
                "max_inflight": args.max_inflight,
                "min_interval_s": args.min_interval_s,
                "dimension_profile": args.dimension_profile,
                "dimensions": DIMENSIONS,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {}
        progress = tqdm(total=total_expected, initial=min(len(done), total_expected), desc="capability_batch_judge")
        for spec in specs:
            print(json.dumps({"event": "load_task", "task": spec["task"]}, ensure_ascii=False), flush=True)
            batches = select_batches(spec, args.samples_per_batch, args.repeats_per_task, args.seed)
            pending = [(repeat, batch) for repeat, batch in enumerate(batches) if (spec["task"], repeat) not in done]
            print(
                json.dumps(
                    {
                        "event": "task_ready",
                        "task": spec["task"],
                        "batches": len(batches),
                        "pending_batches": len(pending),
                        "questions_per_batch": args.samples_per_batch,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            pending_iter = iter(pending)
            while True:
                while len(futures) < args.max_inflight:
                    try:
                        repeat, batch = next(pending_iter)
                    except StopIteration:
                        break
                    fut = pool.submit(
                        judge_batch,
                        api_key=api_key,
                        base_url=args.base_url,
                        model=args.model,
                        spec=spec,
                        items=batch,
                        repeat=repeat,
                        max_retries=args.max_retries,
                        request_timeout=args.request_timeout,
                        max_tokens=args.max_tokens,
                        min_interval_s=args.min_interval_s,
                    )
                    futures[fut] = (spec, repeat, batch)
                if not futures:
                    break
                done_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done_futures:
                    spec_for_row, repeat, batch = futures.pop(fut)
                    try:
                        row = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        row = {
                            "group": spec_for_row["group"],
                            "task": spec_for_row["task"],
                            "dataset": spec_for_row["dataset"],
                            "config": spec_for_row["config"],
                            "repeat": repeat,
                            "batch_size": len(batch),
                            "item_ids": [item["item_id"] for item in batch],
                            "ok": False,
                            "error": repr(exc),
                        }
                    append_jsonl(args.output_jsonl, row)
                    if row.get("ok"):
                        done.add((row["task"], int(row["repeat"])))
                    progress.update(1)
                    print(
                        json.dumps(
                            {
                                "event": "judged",
                                "task": row.get("task"),
                                "repeat": row.get("repeat"),
                                "batch_size": row.get("batch_size"),
                                "ok": row.get("ok"),
                                "attempt": row.get("attempt"),
                                "error": row.get("error"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            summarize(args.output_jsonl, args.summary_csv, args.coverage_csv)
        while futures:
            done_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done_futures:
                spec_for_row, repeat, batch = futures.pop(fut)
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "group": spec_for_row["group"],
                        "task": spec_for_row["task"],
                        "dataset": spec_for_row["dataset"],
                        "config": spec_for_row["config"],
                        "repeat": repeat,
                        "batch_size": len(batch),
                        "item_ids": [item["item_id"] for item in batch],
                        "ok": False,
                        "error": repr(exc),
                    }
                append_jsonl(args.output_jsonl, row)
                if row.get("ok"):
                    done.add((row["task"], int(row["repeat"])))
                progress.update(1)
        progress.close()

    summarize(args.output_jsonl, args.summary_csv, args.coverage_csv)
    print(f"Wrote {args.output_jsonl}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.coverage_csv}")


if __name__ == "__main__":
    main()
