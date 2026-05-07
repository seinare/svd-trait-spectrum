#!/usr/bin/env bash
# Submit with:
#   HF_TOKEN=... bash scripts/slurm_ailab03_lm_eval_smoke.sh
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/data1/cse12111103/runs/svd-trait-spectrum/lm_eval_smoke_ailab03/$(date +%Y%m%d_%H%M%S)}
CODE_DIR=${CODE_DIR:-/data1/cse12111103/code/svd-trait-spectrum}

mkdir -p "$RUN_ROOT"

cat > "$RUN_ROOT/job.sbatch" <<'EOF'
#!/usr/bin/env bash
#SBATCH -J svd-lm-smoke
#SBATCH -p RTXPRO5000
#SBATCH --nodelist=ailab03
#SBATCH --gres=gpu:rtxpro5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH -o /tmp/svd-lm-smoke-%j.out
#SBATCH -e /tmp/svd-lm-smoke-%j.err
set -uo pipefail

export HF_HOME=/data1/cse12111103/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data1/cse12111103/.cache/huggingface/hub
export UV_CACHE_DIR=/data1/cse12111103/.cache/uv
export TMPDIR=/data1/cse12111103/tmp
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NO_USAGE_STATS=1

RUN_ROOT=__RUN_ROOT__
CODE_DIR=__CODE_DIR__
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/monitor" "$RUN_ROOT/tmp" "$TMPDIR"

monitor() {
  while true; do
    {
      echo "===== $(date -Is) ====="
      squeue -j "${SLURM_JOB_ID:-0}" -o "%i %j %t %M %N %R" || true
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
      df -h /dev/shm || true
    } >> "$RUN_ROOT/monitor/gpu_${SLURM_JOB_ID:-unknown}.log" 2>&1
    sleep 60
  done
}
monitor &
MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null || true' EXIT

cd "$CODE_DIR"
start=$(date +%s)
echo "[${start}] START lm_eval_all smoke"
uv run python scripts/eval_script4_lm_eval_tasks.py \
  --alpha 0 \
  --model_id Qwen/Qwen3-8B \
  --preset all \
  --backend vllm \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.70 \
  --max_model_len 2048 \
  --enforce_eager \
  --batch_size auto \
  --limit 1 \
  --local_files_only \
  --force \
  --tmp_root "$RUN_ROOT/tmp" \
  --output_root "$RUN_ROOT/lm_eval" \
  > "$RUN_ROOT/logs/lm_eval_all_smoke.out" \
  2> "$RUN_ROOT/logs/lm_eval_all_smoke.err"
rc=$?
end=$(date +%s)
echo -e "lm_eval_all_smoke\t$start\t$end\t$((end-start))\t$rc" > "$RUN_ROOT/timing.tsv"
echo "[${end}] END lm_eval_all smoke rc=$rc elapsed=$((end-start))"
exit "$rc"
EOF

sed -i "s#__RUN_ROOT__#$RUN_ROOT#g; s#__CODE_DIR__#$CODE_DIR#g" "$RUN_ROOT/job.sbatch"
sbatch --export=ALL,HF_TOKEN="${HF_TOKEN:-}",HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" "$RUN_ROOT/job.sbatch"
echo "$RUN_ROOT"
