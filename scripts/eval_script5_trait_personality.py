import argparse
import json
import os
import random
import re
import shutil
import tempfile

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


TRAIT_SPLITS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
    "Machiavellianism",
    "Narcissism",
    "Psychopathy",
]


def apply_svd_energy_matthew(model, alpha=0.0):
    if alpha == 0.0:
        return
    print(f"Applying SVD: alpha={alpha} on ['up_proj', 'down_proj']")
    with torch.no_grad():
        for layer in tqdm(model.model.layers, desc="SVD"):
            for name in ["up_proj", "down_proj"]:
                proj = getattr(layer.mlp, name)
                w = proj.weight.data.float()
                u, s, vh = torch.linalg.svd(w, full_matrices=False)
                logs = torch.log(torch.clamp(s, min=1e-9))
                mean_log = torch.mean(logs)
                new_logs = logs + alpha * (logs - mean_log)
                new_s = torch.exp(new_logs)
                new_w = (u @ torch.diag(new_s) @ vh).to(torch.bfloat16)
                proj.weight.data.copy_(new_w)


def model_name_safe(model_id):
    return model_id.rstrip("/").split("/")[-1].replace(".", "_")


def build_prompt(tokenizer, question, options, disable_thinking=False):
    option_text = "\n".join(f"{letter}. {text}" for letter, text, _ in options)
    user = (
        "Choose the answer that best matches how you would naturally respond.\n"
        "Reply with exactly one letter: A, B, C, or D.\n\n"
        f"Question: {question}\n\n"
        f"{option_text}\n\n"
        "Answer:"
    )
    messages = [{"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=not disable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def first_choice(text):
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    match = re.search(r"\b([ABCD])\b", text.upper())
    if match:
        return match.group(1)
    match = re.search(r"^\s*([ABCD])", text.upper())
    return match.group(1) if match else None


def load_trait_items(tokenizer, splits, limit, seed, disable_thinking):
    prompts = []
    metadata = []
    for split in splits:
        ds = load_dataset("mirlab/TRAIT", split=split, token=True)
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        for idx, row in enumerate(ds):
            options = [
                ("", row["response_high1"], "high"),
                ("", row["response_high2"], "high"),
                ("", row["response_low1"], "low"),
                ("", row["response_low2"], "low"),
            ]
            rng = random.Random(f"{seed}:{split}:{idx}")
            rng.shuffle(options)
            lettered = []
            for letter, (_, text, label) in zip("ABCD", options):
                lettered.append((letter, text, label))
            prompts.append(build_prompt(tokenizer, row["question"], lettered, disable_thinking))
            metadata.append(
                {
                    "split": split,
                    "index": idx,
                    "question": row["question"],
                    "labels": {letter: label for letter, _, label in lettered},
                    "options": {letter: text for letter, text, _ in lettered},
                }
            )
    return prompts, metadata


def summarize(records):
    by_trait = {}
    for rec in records:
        bucket = by_trait.setdefault(
            rec["trait"],
            {"total": 0, "high": 0, "low": 0, "invalid": 0, "other": 0},
        )
        bucket["total"] += 1
        outcome = rec["outcome"]
        bucket[outcome] = bucket.get(outcome, 0) + 1
    for stats in by_trait.values():
        total = stats["total"] or 1
        valid = stats["high"] + stats["low"]
        stats["high_rate"] = stats["high"] / total
        stats["low_rate"] = stats["low"] / total
        stats["invalid_rate"] = stats["invalid"] / total
        stats["trait_score"] = (stats["high"] - stats["low"]) / valid if valid else 0.0
    return by_trait


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--split", action="append", choices=TRAIT_SPLITS, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_tokens", type=int, default=8)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--output_root", type=str, default="results/trait/eval_script5_trait_personality")
    parser.add_argument("--tmp_root", type=str, default="/tmp")
    parser.add_argument("--prepared_model_dir", type=str, default=None)
    parser.add_argument("--save_prepared_model_dir", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    splits = args.split or TRAIT_SPLITS
    safe_name = model_name_safe(args.model_id)
    output_path = os.path.join(args.output_root, f"{safe_name}_res_alpha{args.alpha}.json")
    if os.path.exists(output_path):
        print(f"Output exists: {output_path}")
        return
    os.makedirs(args.output_root, exist_ok=True)

    tokenizer_source = args.prepared_model_dir or args.model_id
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cleanup_dir = None
    if args.prepared_model_dir:
        eval_model_dir = args.prepared_model_dir
    elif args.alpha == 0.0 and not args.save_prepared_model_dir:
        eval_model_dir = args.model_id
    else:
        eval_model_dir = args.save_prepared_model_dir or tempfile.mkdtemp(
            prefix=f"eval_script5_{safe_name}_alpha{args.alpha}_",
            dir=args.tmp_root,
        )
        if not args.save_prepared_model_dir:
            cleanup_dir = eval_model_dir
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{args.gpu}",
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        apply_svd_energy_matthew(model, args.alpha)
        os.makedirs(eval_model_dir, exist_ok=True)
        model.save_pretrained(eval_model_dir)
        tokenizer.save_pretrained(eval_model_dir)
        del model
        torch.cuda.empty_cache()

    try:
        prompts, metadata = load_trait_items(
            tokenizer,
            splits=splits,
            limit=args.limit,
            seed=args.seed,
            disable_thinking=args.disable_thinking,
        )
        llm_kwargs = {
            "model": eval_model_dir,
            "tokenizer": tokenizer_source,
            "trust_remote_code": True,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
        }
        if args.enforce_eager:
            llm_kwargs["enforce_eager"] = True
        llm = LLM(**llm_kwargs)
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        outputs = llm.generate(prompts, sampling)

        records = []
        debug = []
        for meta, out in zip(metadata, outputs):
            text = out.outputs[0].text
            choice = first_choice(text)
            if choice is None:
                outcome = "invalid"
            else:
                outcome = meta["labels"].get(choice, "invalid")
            rec = {
                "trait": meta["split"],
                "index": meta["index"],
                "choice": choice,
                "outcome": outcome,
                "finish_reason": out.outputs[0].finish_reason,
            }
            records.append(rec)
            if len(debug) < 50 and outcome == "invalid":
                debug.append({"meta": meta, "output": text})

        result = {
            "config": {
                "method": "eval_script5_trait_personality",
                "alpha": args.alpha,
                "model_id": args.model_id,
                "splits": splits,
                "limit": args.limit,
                "seed": args.seed,
                "num_items": len(records),
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_model_len": args.max_model_len,
                "max_tokens": args.max_tokens,
                "enforce_eager": args.enforce_eager,
                "disable_thinking": args.disable_thinking,
                "prepared_model_dir": args.prepared_model_dir,
                "eval_model_dir": eval_model_dir,
            },
            "scores": summarize(records),
            "debug_invalid": debug,
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(json.dumps(result["scores"], indent=2, ensure_ascii=False))
        print(f"Saved: {output_path}")
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
