#!/bin/bash
#SBATCH --job-name=qwen3-smoke
#SBATCH --nodes=1
#SBATCH --nodelist=ailab01
#SBATCH --partition=L40
#SBATCH --gres=gpu:rtx5880ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH --output=/data1/cse12111103/runs/svd-trait-spectrum/%x-%j.out
#SBATCH --error=/data1/cse12111103/runs/svd-trait-spectrum/%x-%j.err

set -euo pipefail

cd /data1/cse12111103/code/svd-trait-spectrum

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/data1/cse12111103/.cache/uv
export HF_HOME=/data1/cse12111103/.cache/huggingface
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

uv run python scripts/eval_script4_lm_eval_tasks.py \
  --model_id Qwen/Qwen3-8B \
  --alpha 0.0 \
  --preset hellaswag \
  --smoke \
  --backend vllm \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.75 \
  --max_model_len 2048 \
  --enforce_eager \
  --batch_size auto \
  --output_root /data1/cse12111103/runs/svd-trait-spectrum/lm_eval \
  --force
