#!/usr/bin/env python3
"""Prepare a Matthew-perturbed Qwen3 MoE model directory.

The regular eval scripts perturb dense Llama-style MLPs on one CUDA device.
Qwen3-30B-A3B is a MoE model, so this helper prepares one temporary model
directory per alpha with device_map="auto", then the eval scripts can consume
it through --prepared_model_dir.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def iter_matthew_linears(model):
    """Yield up/down projection Linear modules for dense and Qwen3-MoE MLPs."""
    layers = getattr(getattr(model, "model", None), "layers", [])
    for layer_idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        if hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
            yield f"layers.{layer_idx}.mlp.up_proj", mlp.up_proj
            yield f"layers.{layer_idx}.mlp.down_proj", mlp.down_proj

        experts = getattr(mlp, "experts", None)
        if experts is not None:
            for expert_idx, expert in enumerate(experts):
                if hasattr(expert, "up_proj") and hasattr(expert, "down_proj"):
                    yield f"layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj", expert.up_proj
                    yield f"layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj", expert.down_proj

        shared_expert = getattr(mlp, "shared_expert", None)
        if shared_expert is not None:
            if hasattr(shared_expert, "up_proj") and hasattr(shared_expert, "down_proj"):
                yield f"layers.{layer_idx}.mlp.shared_expert.up_proj", shared_expert.up_proj
                yield f"layers.{layer_idx}.mlp.shared_expert.down_proj", shared_expert.down_proj


def apply_energy_conserving_matthew(model, alpha: float, svd_device: str) -> dict:
    if alpha == 0.0:
        return {"alpha": alpha, "perturbed_modules": 0}

    modules = list(iter_matthew_linears(model))
    if not modules:
        raise RuntimeError("No dense or MoE up/down projection modules found.")

    counts = {
        "alpha": alpha,
        "perturbed_modules": len(modules),
        "svd_device": svd_device,
        "cuda_gesvd_fallbacks": 0,
        "cpu_svd_fallbacks": 0,
    }
    with torch.no_grad():
        for name, proj in tqdm(modules, desc="Matthew SVD"):
            original_dtype = proj.weight.data.dtype
            original_device = proj.weight.data.device
            weight = proj.weight.data.float()
            if svd_device == "cpu":
                weight_for_svd = weight.cpu()
                u, s, vh = torch.linalg.svd(weight_for_svd, full_matrices=False)
            else:
                try:
                    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
                except RuntimeError as first_error:
                    counts["cuda_gesvd_fallbacks"] += 1
                    print(f"CUDA gesvd fallback for {name}: {first_error}", flush=True)
                    u, s, vh = torch.linalg.svd(weight, full_matrices=False, driver="gesvd")
            log_s = torch.log(torch.clamp(s, min=1e-9))
            centered = log_s - torch.mean(log_s)
            s_new = torch.exp(log_s + alpha * centered)
            weight_new = (u * s_new.unsqueeze(0)) @ vh
            proj.weight.data.copy_(weight_new.to(dtype=original_dtype, device=original_device))
            del weight, u, s, vh, log_s, centered, s_new, weight_new
            if svd_device == "cpu":
                del weight_for_svd
            if original_device.type == "cuda":
                torch.cuda.empty_cache()
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--svd_device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max_memory_per_gpu", default="42GiB")
    parser.add_argument("--max_shard_size", default="4GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    done = out / ".matthew_done.json"
    if done.exists() and not args.force:
        print(f"Prepared model already exists: {out}")
        return
    if out.exists():
        if not args.force:
            raise FileExistsError(f"{out} exists but is not marked done; pass --force to replace it")
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cuda_count = torch.cuda.device_count()
    max_memory = {idx: args.max_memory_per_gpu for idx in range(cuda_count)} if cuda_count else None
    print(f"Loading {args.model_id} with device_map=auto, cuda_count={cuda_count}, max_memory={max_memory}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    meta = apply_energy_conserving_matthew(model, args.alpha, args.svd_device)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(out)
    meta.update({"model_id": args.model_id, "output_dir": str(out)})
    done.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
