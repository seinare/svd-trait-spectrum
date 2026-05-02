import argparse
import json
import os
import random
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


def load_trait_items(tokenizer, splits, limit, seed):
    prompt_token_ids = []
    metadata = []
    for split in splits:
        ds = load_dataset("mirlab/TRAIT", split=split, token=True)
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        for idx, row in enumerate(ds):
            responses = [
                ("high1", row["response_high1"], "high"),
                ("high2", row["response_high2"], "high"),
                ("low1", row["response_low1"], "low"),
                ("low2", row["response_low2"], "low"),
            ]
            rng = random.Random(f"{seed}:{split}:{idx}")
            rng.shuffle(responses)
            prompt = f"Question: {row['question']}\nResponse: "
            base_ids = tokenizer.encode(prompt, add_special_tokens=True)
            options = []
            for name, text, label in responses:
                answer_ids = tokenizer.encode(text, add_special_tokens=False)
                if not answer_ids:
                    continue
                prompt_token_ids.append(base_ids + answer_ids)
                options.append(
                    {
                        "name": name,
                        "label": label,
                        "text": text,
                        "answer_token_count": len(answer_ids),
                    }
                )
            metadata.append(
                {
                    "split": split,
                    "index": idx,
                    "question": row["question"],
                    "base_token_count": len(base_ids),
                    "options": options,
                }
            )
    return prompt_token_ids, metadata


def token_logprob(entry, token_id):
    if entry is None:
        return None
    value = entry.get(token_id)
    if value is None:
        return None
    if hasattr(value, "logprob"):
        return value.logprob
    if isinstance(value, dict):
        return value.get("logprob")
    return value


def answer_log_likelihood(output, token_ids, base_token_count):
    total = 0.0
    used = 0
    prompt_logprobs = output.prompt_logprobs or []
    for pos in range(base_token_count, len(token_ids)):
        lp = token_logprob(prompt_logprobs[pos], token_ids[pos])
        if lp is None:
            continue
        total += float(lp)
        used += 1
    return total, used


def softmax(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    probs = torch.softmax(tensor, dim=0)
    return [float(x) for x in probs.tolist()]


def summarize(records):
    by_trait = {}
    for rec in records:
        bucket = by_trait.setdefault(
            rec["trait"],
            {
                "total": 0,
                "trait_score_sum": 0.0,
                "high1_prob_sum": 0.0,
                "high2_prob_sum": 0.0,
                "low1_prob_sum": 0.0,
                "low2_prob_sum": 0.0,
            },
        )
        bucket["total"] += 1
        bucket["trait_score_sum"] += rec["trait_score"]
        for name, prob in rec["probabilities"].items():
            bucket[f"{name}_prob_sum"] += prob
    for stats in by_trait.values():
        total = stats["total"] or 1
        stats["trait_score"] = stats.pop("trait_score_sum") / total
        for name in ["high1", "high2", "low1", "low2"]:
            stats[f"{name}_prob"] = stats.pop(f"{name}_prob_sum") / total
        stats["high_prob"] = stats["high1_prob"] + stats["high2_prob"]
        stats["low_prob"] = stats["low1_prob"] + stats["low2_prob"]
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
    parser.add_argument("--max_tokens", type=int, default=1)
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
        prompt_token_ids, metadata = load_trait_items(
            tokenizer,
            splits=splits,
            limit=args.limit,
            seed=args.seed,
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
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            prompt_logprobs=1,
        )
        outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling)

        records = []
        debug = []
        cursor = 0
        for meta in metadata:
            log_likelihoods = {}
            used_tokens = {}
            for option in meta["options"]:
                token_ids = prompt_token_ids[cursor]
                out = outputs[cursor]
                log_likelihood, used = answer_log_likelihood(
                    out,
                    token_ids=token_ids,
                    base_token_count=meta["base_token_count"],
                )
                log_likelihoods[option["name"]] = log_likelihood
                used_tokens[option["name"]] = used
                cursor += 1
            probs_order = [option["name"] for option in meta["options"]]
            probs = dict(zip(probs_order, softmax([log_likelihoods[name] for name in probs_order])))
            trait_score = sum(probs[name] for name in probs if name.startswith("high"))
            rec = {
                "trait": meta["split"],
                "index": meta["index"],
                "trait_score": trait_score,
                "probabilities": probs,
                "log_likelihoods": log_likelihoods,
                "used_tokens": used_tokens,
            }
            records.append(rec)
            if len(debug) < 10:
                debug.append({"meta": meta, **rec})

        result = {
            "config": {
                "method": "eval_script5_trait_personality",
                "scoring": "softmax_log_likelihood_high_probability",
                "prompt_template": "Question: {q}\\nResponse: ",
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
            "records": records,
            "debug_examples": debug,
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
