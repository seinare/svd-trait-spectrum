#!/usr/bin/env python3
"""Evaluate Matthew-perturbed Llama base models on Parameter Golf FineWeb BPB."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_svd_energy_matthew(model, alpha: float) -> None:
    if alpha == 0.0:
        return
    print(f"Applying Energy-Conserving Matthew SVD alpha={alpha} on ['up_proj', 'down_proj']", flush=True)
    with torch.no_grad():
        for layer in tqdm(model.model.layers, desc="svd_layers"):
            for name in ["up_proj", "down_proj"]:
                proj = getattr(layer.mlp, name)
                w = proj.weight.data.float()
                u, s, vh = torch.linalg.svd(w, full_matrices=False)
                logs = torch.log(torch.clamp(s, min=1e-9))
                new_s = torch.exp(logs + alpha * (logs - torch.mean(logs)))
                proj.weight.data.copy_((u @ torch.diag(new_s) @ vh).to(proj.weight.dtype))


def read_pg_tokens(path: Path, limit_sp_tokens: int | None = None) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    count = int(header[2])
    if limit_sp_tokens is not None:
        count = min(count, int(limit_sp_tokens))
    return np.fromfile(path, dtype="<u2", count=count, offset=header_bytes)


def sentencepiece_byte_count(sp: spm.SentencePieceProcessor, ids: np.ndarray) -> int:
    vocab_size = sp.vocab_size()
    base = np.zeros(vocab_size, dtype=np.int16)
    leading = np.zeros(vocab_size, dtype=np.bool_)
    boundary = np.ones(vocab_size, dtype=np.bool_)
    for token_id in range(vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        boundary[token_id] = False
        if sp.is_byte(token_id):
            base[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            leading[token_id] = True
            piece = piece[1:]
        base[token_id] = len(piece.encode("utf-8"))
    prev = ids[:-1]
    tgt = ids[1:]
    byte_counts = base[tgt].astype(np.int64)
    byte_counts += (leading[tgt] & ~boundary[prev]).astype(np.int64)
    return int(byte_counts.sum())


def tokenizer_fingerprint(tokenizer) -> str:
    payload = f"{tokenizer.name_or_path}|{len(tokenizer)}|{tokenizer.bos_token_id}|{tokenizer.eos_token_id}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def build_or_load_token_cache(
    tokenizer,
    sp_model_path: Path,
    val_bin_path: Path,
    cache_dir: Path,
    limit_sp_tokens: int | None,
    decode_chunk_tokens: int,
) -> tuple[np.ndarray, int, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    limit_tag = "full" if limit_sp_tokens is None else f"limit{limit_sp_tokens}"
    fp = tokenizer_fingerprint(tokenizer)
    cache_path = cache_dir / f"llama_val_ids_{fp}_{limit_tag}.npy"
    meta_path = cache_dir / f"llama_val_ids_{fp}_{limit_tag}.json"
    if cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return np.load(cache_path, mmap_mode="r"), int(meta["byte_count"]), cache_path

    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    sp_ids = read_pg_tokens(val_bin_path, limit_sp_tokens)
    byte_count = sentencepiece_byte_count(sp, sp_ids)
    chunks: list[np.ndarray] = []
    for start in tqdm(range(0, len(sp_ids), decode_chunk_tokens), desc="tokenize_val"):
        chunk_ids = sp_ids[start : start + decode_chunk_tokens].tolist()
        text = sp.decode(chunk_ids)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            chunks.append(np.asarray(ids, dtype=np.int32))
    llama_ids = np.concatenate(chunks) if chunks else np.asarray([], dtype=np.int32)
    np.save(cache_path, llama_ids)
    meta_path.write_text(
        json.dumps(
            {
                "tokenizer": tokenizer.name_or_path,
                "tokenizer_fingerprint": fp,
                "val_bin": str(val_bin_path),
                "sp_model": str(sp_model_path),
                "limit_sp_tokens": limit_sp_tokens,
                "sp_tokens": int(len(sp_ids)),
                "llama_tokens": int(len(llama_ids)),
                "byte_count": byte_count,
            },
            indent=2,
        )
    )
    return np.load(cache_path, mmap_mode="r"), byte_count, cache_path


def evaluate_nll(model, ids: np.ndarray, byte_count: int, seq_len: int, batch_size: int, device: str) -> dict:
    usable = ((len(ids) - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Not enough tokens for seq_len={seq_len}: {len(ids)}")
    ids = ids[: usable + 1]
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()
    model.eval()
    with torch.inference_mode():
        for start_seq in tqdm(range(0, usable // seq_len, batch_size), desc="eval"):
            end_seq = min(start_seq + batch_size, usable // seq_len)
            raw_start = start_seq * seq_len
            raw_end = end_seq * seq_len + 1
            local = torch.as_tensor(ids[raw_start:raw_end].astype(np.int64), device=device)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=x).logits
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1),
                reduction="mean",
            )
            token_count = int(y.numel())
            total_loss += float(loss.detach().item()) * token_count
            total_tokens += token_count
    elapsed = time.time() - t0
    val_loss = total_loss / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / byte_count)
    return {
        "val_loss": val_loss,
        "val_bpb": val_bpb,
        "bits_per_token": bits_per_token,
        "tokens": total_tokens,
        "bytes": byte_count,
        "tokens_per_byte": total_tokens / byte_count,
        "eval_seconds": elapsed,
        "tok_per_s": total_tokens / elapsed if elapsed > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--val_bin", type=Path, default=Path("/data1/xjh/code/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin"))
    parser.add_argument("--sp_model", type=Path, default=Path("/data1/xjh/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"))
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit_sp_tokens", type=int, default=None)
    parser.add_argument("--decode_chunk_tokens", type=int, default=100000)
    parser.add_argument("--cache_only", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        print(f"Output exists: {args.output}", flush=True)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu}"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    ids, byte_count, cache_path = build_or_load_token_cache(
        tokenizer=tokenizer,
        sp_model_path=args.sp_model,
        val_bin_path=args.val_bin,
        cache_dir=args.cache_dir,
        limit_sp_tokens=args.limit_sp_tokens,
        decode_chunk_tokens=args.decode_chunk_tokens,
    )
    if args.cache_only:
        print(
            json.dumps(
                {
                    "cache_path": str(cache_path),
                    "tokens": int(len(ids)),
                    "bytes": int(byte_count),
                    "limit_sp_tokens": args.limit_sp_tokens,
                },
                indent=2,
            ),
            flush=True,
        )
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    apply_svd_energy_matthew(model, args.alpha)
    metrics = evaluate_nll(model, ids, byte_count, args.seq_len, args.batch_size, device)
    result = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "alpha": args.alpha,
        "method": "energy_conserving_matthew_svd_up_down",
        "val_bin": str(args.val_bin),
        "sp_model": str(args.sp_model),
        "token_cache": str(cache_path),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "limit_sp_tokens": args.limit_sp_tokens,
        **metrics,
    }
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    tmp.replace(args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
