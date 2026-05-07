#!/usr/bin/env python3
"""Prepare a magnitude-SVD perturbed model directory.

For every transformer layer MLP up/down projection, decompose W = U S Vh and
write W' = U (beta * S) Vh. Negative beta values therefore flip the reconstructed
projection sign while scaling its singular-value magnitudes.
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


def iter_up_down_projections(model):
    layers = getattr(getattr(model, "model", None), "layers", [])
    for layer_idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        for name in ("up_proj", "down_proj"):
            if hasattr(mlp, name):
                yield f"model.layers.{layer_idx}.mlp.{name}", getattr(mlp, name)


def apply_magnitude_svd(model, beta: float, svd_device: str) -> dict:
    modules = list(iter_up_down_projections(model))
    if not modules:
        raise RuntimeError("No MLP up/down projection modules found.")

    meta = {
        "method": "magnitude_svd",
        "beta": beta,
        "svd_device": svd_device,
        "perturbed_modules": len(modules),
    }
    with torch.no_grad():
        for _, proj in tqdm(modules, desc="Magnitude SVD"):
            original_dtype = proj.weight.data.dtype
            original_device = proj.weight.data.device
            weight = proj.weight.data.float()
            weight_for_svd = weight.cpu() if svd_device == "cpu" else weight
            u, s, vh = torch.linalg.svd(weight_for_svd, full_matrices=False)
            weight_new = (u * (s * beta).unsqueeze(0)) @ vh
            proj.weight.data.copy_(weight_new.to(dtype=original_dtype, device=original_device))
            del weight, weight_for_svd, u, s, vh, weight_new
            if original_device.type == "cuda":
                torch.cuda.empty_cache()
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--svd_device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max_shard_size", default="4GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    done = out / ".magnitude_done.json"
    if done.exists() and not args.force:
        print(f"Prepared model already exists: {out}")
        return
    if out.exists():
        if not args.force:
            raise FileExistsError(f"{out} exists but is not marked done; pass --force")
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=args.local_files_only,
    )
    meta = apply_magnitude_svd(model, args.beta, args.svd_device)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(out)
    meta.update({"model_id": args.model_id, "output_dir": str(out)})
    done.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
