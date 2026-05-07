#!/usr/bin/env python3
"""Concurrent capability-dimension labeling for selected lm-eval subtasks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from analyze_capability_dimensions import (
    AGIEVAL_CONFIGS,
    DIMENSIONS,
    SCRIPT_MMLU_SUBJECTS,
    build_prompt,
    judge_item,
    load_items,
)


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


def get_client(api_key: str, base_url: str) -> OpenAI:
    client = getattr(THREAD_LOCAL, "client", None)
    if client is None:
        client = OpenAI(api_key=api_key, base_url=base_url)
        THREAD_LOCAL.client = client
    return client


def read_done(path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not path.exists():
        return done
    with path.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ok"):
                done.add((row["task"], row["item_id"], int(row["repeat"])))
    return done


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with JSONL_LOCK:
        with path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


def summarize(jsonl_path: Path, summary_csv: Path, coverage_csv: Path) -> None:
    buckets: dict[str, dict[str, Any]] = {}
    seen: set[tuple[str, str, int]] = set()
    if jsonl_path.exists():
        with jsonl_path.open() as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not row.get("ok"):
                    continue
                key = (row["task"], row["item_id"], int(row["repeat"]))
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

    coverage: dict[str, dict[str, Any]] = {}
    for task, item_id, repeat in seen:
        row = coverage.setdefault(task, {"task": task, "unique_items": set(), "ok_repeats": 0})
        row["unique_items"].add(item_id)
        row["ok_repeats"] += 1
    with coverage_csv.open("w", newline="") as handle:
        fields = ["task", "unique_items", "ok_repeats"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in sorted(coverage):
            row = coverage[task]
            writer.writerow(
                {
                    "task": task,
                    "unique_items": len(row["unique_items"]),
                    "ok_repeats": row["ok_repeats"],
                }
            )


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


def judge_repeat(
    *,
    api_key: str,
    base_url: str,
    model: str,
    spec: dict[str, str],
    item: dict[str, Any],
    repeat: int,
    max_retries: int,
    request_timeout: int,
    max_tokens: int,
    min_interval_s: float,
) -> dict[str, Any]:
    throttle(min_interval_s)
    client = get_client(api_key, base_url)
    judged = judge_item(
        client=client,
        model=model,
        task=spec["task"],
        item=item,
        max_retries=max_retries,
        request_timeout=request_timeout,
        max_tokens=max_tokens,
    )
    row = {
        "group": spec["group"],
        "task": spec["task"],
        "dataset": spec["dataset"],
        "config": spec["config"],
        "item_id": item["item_id"],
        "repeat": repeat,
        "question": item["question"],
        "options": item["options"],
        "answer": item["answer"],
        "ok": judged["ok"],
    }
    row.update(judged)
    return row


def select_items(spec: dict[str, str], samples_per_task: int, seed: int) -> list[dict[str, Any]]:
    items = load_items(spec)
    rng = random.Random(f"{seed}:{spec['task']}")
    rng.shuffle(items)
    return items[:samples_per_task]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_per_task", type=int, default=10)
    parser.add_argument("--repeats_per_item", type=int, default=10)
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
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_csv", type=Path, required=True)
    parser.add_argument("--coverage_csv", type=Path, required=True)
    parser.add_argument("--task", action="append", default=None, help="Optional exact task filter.")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    specs = script_task_specs()
    if args.task:
        wanted = set(args.task)
        specs = [spec for spec in specs if spec["task"] in wanted]

    done = read_done(args.output_jsonl)
    total_expected = len(specs) * args.samples_per_task * args.repeats_per_item
    print(
        json.dumps(
            {
                "event": "start",
                "tasks": len(specs),
                "samples_per_task": args.samples_per_task,
                "repeats_per_item": args.repeats_per_item,
                "expected_judgments": total_expected,
                "already_done": len(done),
                "max_workers": args.max_workers,
                "max_inflight": args.max_inflight,
                "min_interval_s": args.min_interval_s,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {}
        progress = tqdm(total=total_expected, initial=min(len(done), total_expected), desc="capability_judge")
        for spec in specs:
            print(json.dumps({"event": "load_task", "task": spec["task"]}, ensure_ascii=False), flush=True)
            items = select_items(spec, args.samples_per_task, args.seed)
            pending = []
            for item in items:
                for repeat in range(args.repeats_per_item):
                    key = (spec["task"], item["item_id"], repeat)
                    if key in done:
                        continue
                    pending.append((item, repeat))
            print(
                json.dumps(
                    {
                        "event": "task_ready",
                        "task": spec["task"],
                        "items": len(items),
                        "pending_repeats": len(pending),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            pending_iter = iter(pending)
            while True:
                while len(futures) < args.max_inflight:
                    try:
                        item, repeat = next(pending_iter)
                    except StopIteration:
                        break
                    fut = pool.submit(
                        judge_repeat,
                        api_key=api_key,
                        base_url=args.base_url,
                        model=args.model,
                        spec=spec,
                        item=item,
                        repeat=repeat,
                        max_retries=args.max_retries,
                        request_timeout=args.request_timeout,
                        max_tokens=args.max_tokens,
                        min_interval_s=args.min_interval_s,
                    )
                    futures[fut] = (spec["task"], item["item_id"], repeat)
                if not futures:
                    break
                done_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done_futures:
                    key = futures.pop(fut)
                    try:
                        row = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        task, item_id, repeat = key
                        row = {
                            "task": task,
                            "item_id": item_id,
                            "repeat": repeat,
                            "ok": False,
                            "error": repr(exc),
                        }
                    append_jsonl(args.output_jsonl, row)
                    if row.get("ok"):
                        done.add((row["task"], row["item_id"], int(row["repeat"])))
                    progress.update(1)
                    print(
                        json.dumps(
                            {
                                "event": "judged",
                                "task": row.get("task"),
                                "item_id": row.get("item_id"),
                                "repeat": row.get("repeat"),
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
                key = futures.pop(fut)
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001
                    task, item_id, repeat = key
                    row = {"task": task, "item_id": item_id, "repeat": repeat, "ok": False, "error": repr(exc)}
                append_jsonl(args.output_jsonl, row)
                if row.get("ok"):
                    done.add((row["task"], row["item_id"], int(row["repeat"])))
                progress.update(1)
        progress.close()

    summarize(args.output_jsonl, args.summary_csv, args.coverage_csv)
    print(f"Wrote {args.output_jsonl}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.coverage_csv}")


if __name__ == "__main__":
    main()
