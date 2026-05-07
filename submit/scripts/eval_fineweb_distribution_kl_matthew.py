#!/usr/bin/env python3
"""Measure output-distribution KL shifts induced by Matthew SVD perturbations.

The script uses a fixed FineWeb10B validation sample, caches alpha=0 logits on
disk, then streams perturbed logits against that cache. It does not save
perturbed model weights.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


PROJECTIONS = ("up_proj", "down_proj")


def read_pg_tokens(path: Path, limit_sp_tokens: int | None = None) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    count = int(header[2])
    if limit_sp_tokens is not None:
        count = min(count, int(limit_sp_tokens))
    return np.fromfile(path, dtype="<u2", count=count, offset=header_bytes)


def tokenizer_fingerprint(tokenizer) -> str:
    payload = f"{tokenizer.name_or_path}|{len(tokenizer)}|{tokenizer.bos_token_id}|{tokenizer.eos_token_id}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def build_or_load_token_cache(
    tokenizer,
    sp_model_path: Path,
    val_bin_path: Path,
    cache_dir: Path,
    sample_tokens: int,
    limit_sp_tokens: int,
    decode_chunk_tokens: int,
) -> tuple[np.ndarray, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = tokenizer_fingerprint(tokenizer)
    cache_path = cache_dir / f"fineweb_val_ids_{fp}_sp{limit_sp_tokens}.npy"
    meta_path = cache_dir / f"fineweb_val_ids_{fp}_sp{limit_sp_tokens}.json"
    if cache_path.exists() and meta_path.exists():
        ids = np.load(cache_path, mmap_mode="r")
        if len(ids) >= sample_tokens + 1:
            return ids, cache_path

    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    sp_ids = read_pg_tokens(val_bin_path, limit_sp_tokens)
    chunks: list[np.ndarray] = []
    for start in tqdm(range(0, len(sp_ids), decode_chunk_tokens), desc="tokenize_fineweb"):
        text = sp.decode(sp_ids[start : start + decode_chunk_tokens].tolist())
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            chunks.append(np.asarray(ids, dtype=np.int32))
    if not chunks:
        raise RuntimeError("Tokenization produced no ids.")
    model_ids = np.concatenate(chunks)
    if len(model_ids) < sample_tokens + 1:
        raise RuntimeError(
            f"Need at least {sample_tokens + 1} model tokens, got {len(model_ids)} from {limit_sp_tokens} SP tokens."
        )
    np.save(cache_path, model_ids)
    meta_path.write_text(
        json.dumps(
            {
                "tokenizer": tokenizer.name_or_path,
                "tokenizer_fingerprint": fp,
                "val_bin": str(val_bin_path),
                "sp_model": str(sp_model_path),
                "limit_sp_tokens": limit_sp_tokens,
                "model_tokens": int(len(model_ids)),
            },
            indent=2,
        )
    )
    return np.load(cache_path, mmap_mode="r"), cache_path


def iter_matthew_linears(model):
    layers = getattr(getattr(model, "model", None), "layers", [])
    for layer_idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        for proj_name in PROJECTIONS:
            proj = getattr(mlp, proj_name, None)
            if proj is not None:
                yield f"layers.{layer_idx}.mlp.{proj_name}", proj
        experts = getattr(mlp, "experts", None)
        if experts is not None:
            for expert_idx, expert in enumerate(experts):
                for proj_name in PROJECTIONS:
                    proj = getattr(expert, proj_name, None)
                    if proj is not None:
                        yield f"layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}", proj
        shared_expert = getattr(mlp, "shared_expert", None)
        if shared_expert is not None:
            for proj_name in PROJECTIONS:
                proj = getattr(shared_expert, proj_name, None)
                if proj is not None:
                    yield f"layers.{layer_idx}.mlp.shared_expert.{proj_name}", proj


def apply_energy_conserving_matthew(model, alpha: float, svd_device: str = "cuda") -> dict:
    if alpha == 0.0:
        return {"alpha": alpha, "perturbed_modules": 0, "svd_seconds": 0.0}
    t0 = time.time()
    modules = list(iter_matthew_linears(model))
    if not modules:
        raise RuntimeError("No up/down projection modules found for Matthew perturbation.")
    meta = {"alpha": alpha, "perturbed_modules": len(modules), "svd_device": svd_device, "cuda_gesvd_fallbacks": 0}
    with torch.no_grad():
        for name, proj in tqdm(modules, desc=f"svd alpha={alpha}"):
            original_dtype = proj.weight.data.dtype
            original_device = proj.weight.data.device
            weight = proj.weight.data.float()
            if svd_device == "cpu":
                weight = weight.cpu()
            try:
                u, s, vh = torch.linalg.svd(weight, full_matrices=False)
            except RuntimeError as exc:
                if weight.is_cuda:
                    meta["cuda_gesvd_fallbacks"] += 1
                    print(f"CUDA SVD fallback for {name}: {exc}", flush=True)
                    u, s, vh = torch.linalg.svd(weight, full_matrices=False, driver="gesvd")
                else:
                    raise
            log_s = torch.log(torch.clamp(s, min=1e-9))
            s_new = torch.exp(log_s + alpha * (log_s - torch.mean(log_s)))
            new_weight = (u * s_new.unsqueeze(0)) @ vh
            proj.weight.data.copy_(new_weight.to(dtype=original_dtype, device=original_device))
            del weight, u, s, vh, log_s, s_new, new_weight
            if original_device.type == "cuda":
                torch.cuda.empty_cache()
    meta["svd_seconds"] = time.time() - t0
    return meta


def model_load_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "local_files_only": args.local_files_only,
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    if args.device_map == "single":
        kwargs["device_map"] = f"cuda:{args.gpu}"
    elif args.device_map == "auto":
        if args.max_memory_per_gpu:
            kwargs["max_memory"] = {idx: args.max_memory_per_gpu for idx in range(torch.cuda.device_count())}
        kwargs["device_map"] = "auto"
    else:
        raise ValueError(f"Unsupported device_map={args.device_map}")
    return kwargs


def first_device(model) -> torch.device:
    return next(model.parameters()).device


def forward_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return model(input_ids=input_ids).logits.detach()


def make_baseline_logits(
    model,
    ids: np.ndarray,
    logits_path: Path,
    sample_tokens: int,
    seq_len: int,
    batch_size: int,
) -> np.memmap:
    vocab_size = int(getattr(model.config, "vocab_size"))
    logits = np.memmap(logits_path, dtype=np.float16, mode="w+", shape=(sample_tokens, vocab_size))
    device = first_device(model)
    n_seq = sample_tokens // seq_len
    written = 0
    for start_seq in tqdm(range(0, n_seq, batch_size), desc="cache_baseline_logits"):
        end_seq = min(start_seq + batch_size, n_seq)
        raw_start = start_seq * seq_len
        raw_end = end_seq * seq_len
        x_np = np.asarray(ids[raw_start:raw_end], dtype=np.int64).reshape(end_seq - start_seq, seq_len)
        x = torch.as_tensor(x_np, device=device)
        batch_logits = forward_logits(model, x).float().cpu().numpy().astype(np.float16)
        flat = batch_logits.reshape(-1, vocab_size)
        logits[written : written + flat.shape[0]] = flat
        written += flat.shape[0]
        del x, batch_logits, flat
        torch.cuda.empty_cache()
    logits.flush()
    if written != sample_tokens:
        raise RuntimeError(f"Baseline wrote {written} rows, expected {sample_tokens}")
    return logits


def load_or_make_baseline_logits(
    model,
    ids: np.ndarray,
    logits_path: Path,
    meta_path: Path,
    sample_tokens: int,
    seq_len: int,
    batch_size: int,
) -> np.memmap:
    vocab_size = int(getattr(model.config, "vocab_size"))
    if logits_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if int(meta["sample_tokens"]) == sample_tokens and int(meta["vocab_size"]) == vocab_size:
            return np.memmap(logits_path, dtype=np.float16, mode="r", shape=(sample_tokens, vocab_size))
    logits = make_baseline_logits(model, ids, logits_path, sample_tokens, seq_len, batch_size)
    meta_path.write_text(json.dumps({"sample_tokens": sample_tokens, "vocab_size": vocab_size}, indent=2))
    return logits


def load_existing_baseline_logits(
    logits_path: Path,
    meta_path: Path,
    baseline_pair_path: Path,
    sample_tokens: int,
) -> tuple[np.memmap, int, dict] | None:
    if not (logits_path.exists() and meta_path.exists() and baseline_pair_path.exists()):
        return None
    meta = json.loads(meta_path.read_text())
    if int(meta["sample_tokens"]) != sample_tokens:
        return None
    vocab_size = int(meta["vocab_size"])
    logits = np.memmap(logits_path, dtype=np.float16, mode="r", shape=(sample_tokens, vocab_size))
    return logits, vocab_size, json.loads(baseline_pair_path.read_text())


def compute_random_pair_kl(
    baseline_logits: np.memmap,
    sample_tokens: int,
    vocab_size: int,
    n_pairs: int,
    chunk_pairs: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    left = rng.integers(0, sample_tokens, size=n_pairs, endpoint=False)
    right = rng.integers(0, sample_tokens, size=n_pairs, endpoint=False)
    same = left == right
    while np.any(same):
        right[same] = rng.integers(0, sample_tokens, size=int(np.sum(same)), endpoint=False)
        same = left == right
    total = 0.0
    total_sq = 0.0
    count = 0
    for start in tqdm(range(0, n_pairs, chunk_pairs), desc="baseline_random_pair_kl"):
        l_np = np.asarray(baseline_logits[left[start : start + chunk_pairs]], dtype=np.float32)
        r_np = np.asarray(baseline_logits[right[start : start + chunk_pairs]], dtype=np.float32)
        l = torch.from_numpy(l_np)
        r = torch.from_numpy(r_np)
        lp_l = torch.log_softmax(l, dim=-1)
        lp_r = torch.log_softmax(r, dim=-1)
        p_l = torch.exp(lp_l)
        kl = torch.sum(p_l * (lp_l - lp_r), dim=-1).numpy()
        total += float(np.sum(kl))
        total_sq += float(np.sum(kl * kl))
        count += int(kl.size)
        del l_np, r_np, l, r, lp_l, lp_r, p_l, kl
    mean = total / count
    var = max(total_sq / count - mean * mean, 0.0)
    return {
        "baseline_random_pair_kl_mean": mean,
        "baseline_random_pair_kl_std": math.sqrt(var),
        "baseline_random_pair_kl_pairs": count,
    }


def compute_perturbation_kl(
    model,
    ids: np.ndarray,
    baseline_logits: np.memmap,
    sample_tokens: int,
    seq_len: int,
    batch_size: int,
) -> dict:
    vocab_size = int(getattr(model.config, "vocab_size"))
    device = first_device(model)
    n_seq = sample_tokens // seq_len
    total_alpha_to_base = 0.0
    total_base_to_alpha = 0.0
    total_js = 0.0
    total_rows = 0
    for start_seq in tqdm(range(0, n_seq, batch_size), desc="kl_eval"):
        end_seq = min(start_seq + batch_size, n_seq)
        raw_start = start_seq * seq_len
        raw_end = end_seq * seq_len
        x_np = np.asarray(ids[raw_start:raw_end], dtype=np.int64).reshape(end_seq - start_seq, seq_len)
        x = torch.as_tensor(x_np, device=device)
        logits_alpha = forward_logits(model, x).float().reshape(-1, vocab_size).cpu()
        logits_base = torch.from_numpy(np.asarray(baseline_logits[raw_start:raw_end], dtype=np.float32))
        lp_alpha = torch.log_softmax(logits_alpha, dim=-1)
        lp_base = torch.log_softmax(logits_base, dim=-1)
        p_alpha = torch.exp(lp_alpha)
        p_base = torch.exp(lp_base)
        kl_ab = torch.sum(p_alpha * (lp_alpha - lp_base), dim=-1)
        kl_ba = torch.sum(p_base * (lp_base - lp_alpha), dim=-1)
        m = 0.5 * (p_alpha + p_base)
        log_m = torch.log(torch.clamp(m, min=1e-45))
        js = 0.5 * torch.sum(p_alpha * (lp_alpha - log_m), dim=-1) + 0.5 * torch.sum(p_base * (lp_base - log_m), dim=-1)
        total_alpha_to_base += float(torch.sum(kl_ab).item())
        total_base_to_alpha += float(torch.sum(kl_ba).item())
        total_js += float(torch.sum(js).item())
        total_rows += int(kl_ab.numel())
        del x, logits_alpha, logits_base, lp_alpha, lp_base, p_alpha, p_base, kl_ab, kl_ba, m, log_m, js
        torch.cuda.empty_cache()
    return {
        "kl_alpha_to_base_mean": total_alpha_to_base / total_rows,
        "kl_base_to_alpha_mean": total_base_to_alpha / total_rows,
        "js_mean": total_js / total_rows,
        "tokens": total_rows,
    }


def delete_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def safe_alpha(alpha: float) -> str:
    text = f"{alpha:g}".replace("-", "m").replace(".", "p")
    return text


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--alphas", default="-0.2,-0.1,0.1,0.2")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--val_bin", type=Path, default=Path("/data1/xjh/code/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin"))
    parser.add_argument("--sp_model", type=Path, default=Path("/data1/xjh/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"))
    parser.add_argument("--sample_tokens", type=int, default=131072)
    parser.add_argument("--limit_sp_tokens", type=int, default=300000)
    parser.add_argument("--decode_chunk_tokens", type=int, default=100000)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device_map", choices=["single", "auto"], default="single")
    parser.add_argument("--max_memory_per_gpu", default=None)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--svd_device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--random_pair_kl_pairs", type=int, default=2048)
    parser.add_argument("--random_pair_chunk_pairs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--keep_baseline_logits", action="store_true")
    parser.add_argument(
        "--reuse_existing_baseline",
        action="store_true",
        help="If baseline logits and random-pair KL exist, skip loading the alpha=0 model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    sample_tokens = (args.sample_tokens // args.seq_len) * args.seq_len
    if sample_tokens != args.sample_tokens:
        print(f"Rounded sample_tokens from {args.sample_tokens} to {sample_tokens} for seq_len={args.seq_len}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=args.local_files_only, trust_remote_code=True)
    ids, token_cache = build_or_load_token_cache(
        tokenizer=tokenizer,
        sp_model_path=args.sp_model,
        val_bin_path=args.val_bin,
        cache_dir=args.cache_dir,
        sample_tokens=sample_tokens,
        limit_sp_tokens=args.limit_sp_tokens,
        decode_chunk_tokens=args.decode_chunk_tokens,
    )
    ids = ids[: sample_tokens + 1]
    model_tag = args.model_name.replace("/", "_").replace(" ", "_")
    baseline_logits_path = args.output_dir / f"{model_tag}.baseline_logits.float16.memmap"
    baseline_meta_path = args.output_dir / f"{model_tag}.baseline_logits.json"
    summary_csv = args.output_dir / "fineweb_distribution_kl_summary.csv"

    baseline_pair_json = args.output_dir / f"{model_tag}.baseline_random_pair_kl.json"
    existing = (
        load_existing_baseline_logits(baseline_logits_path, baseline_meta_path, baseline_pair_json, sample_tokens)
        if args.reuse_existing_baseline
        else None
    )
    if existing is not None:
        baseline_logits, vocab_size, baseline_pair = existing
    else:
        print(f"Loading baseline model {args.model_name}", flush=True)
        model0 = AutoModelForCausalLM.from_pretrained(args.model_path, **model_load_kwargs(args))
        model0.eval()
        baseline_logits = load_or_make_baseline_logits(
            model=model0,
            ids=ids,
            logits_path=baseline_logits_path,
            meta_path=baseline_meta_path,
            sample_tokens=sample_tokens,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
        )
        vocab_size = int(getattr(model0.config, "vocab_size"))
        if baseline_pair_json.exists():
            baseline_pair = json.loads(baseline_pair_json.read_text())
        else:
            baseline_pair = compute_random_pair_kl(
                baseline_logits=baseline_logits,
                sample_tokens=sample_tokens,
                vocab_size=vocab_size,
                n_pairs=args.random_pair_kl_pairs,
                chunk_pairs=args.random_pair_chunk_pairs,
                seed=args.seed,
            )
            baseline_pair_json.write_text(json.dumps(baseline_pair, indent=2))
        delete_model(model0)

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    for alpha in alphas:
        result_json = args.output_dir / f"{model_tag}.alpha_{safe_alpha(alpha)}.kl.json"
        if result_json.exists():
            print(f"Skip existing result: {result_json}", flush=True)
            continue
        print(f"Loading perturbed model {args.model_name} alpha={alpha}", flush=True)
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_load_kwargs(args))
        model.eval()
        svd_meta = apply_energy_conserving_matthew(model, alpha=alpha, svd_device=args.svd_device)
        kl_meta = compute_perturbation_kl(
            model=model,
            ids=ids,
            baseline_logits=baseline_logits,
            sample_tokens=sample_tokens,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
        )
        row = {
            "model": args.model_name,
            "model_path": args.model_path,
            "alpha": alpha,
            "sample_tokens": sample_tokens,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "token_cache": str(token_cache),
            "kl_alpha_to_base_mean": kl_meta["kl_alpha_to_base_mean"],
            "kl_base_to_alpha_mean": kl_meta["kl_base_to_alpha_mean"],
            "js_mean": kl_meta["js_mean"],
            **baseline_pair,
            "perturbed_modules": svd_meta["perturbed_modules"],
            "svd_seconds": svd_meta["svd_seconds"],
            "total_seconds": time.time() - t0,
        }
        result_json.write_text(json.dumps(row, indent=2, ensure_ascii=False))
        append_csv(summary_csv, row)
        print(json.dumps(row, indent=2), flush=True)
        delete_model(model)

    if not args.keep_baseline_logits:
        try:
            baseline_logits._mmap.close()
        except Exception:
            pass
        for path in (baseline_logits_path, baseline_meta_path):
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    main()
