#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


STAMP = time.strftime("%Y%m%d_%H%M%S")
BASE = f"/data1/xjh/runs/pruning_codex/trait_qwen3_30b_a3b_alpha9_batched_{STAMP}"
CODE = "/data1/xjh/code/pruning_codex"
MODEL_ID = "/data1/xjh/code/pruning/models/Qwen3-30B-A3B"
ALPHAS = ["-0.2", "-0.15", "-0.1", "-0.05", "0", "0.05", "0.1", "0.15", "0.2"]
LANES = {"1,2": [], "3,4": [], "5,6": []}
for idx, alpha in enumerate(ALPHAS):
    list(LANES.values())[idx % len(LANES)].append(alpha)


SCRIPT_TEMPLATE = r'''#!/usr/bin/env bash
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/data1/xjh/.cache/uv
export HF_HOME=/data1/xjh/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data1/xjh/.cache/huggingface/hub
export HF_DATASETS_CACHE=/data1/huggingface/datasets
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NLTK_DATA=/home/xjh/nltk_data
export TMPDIR=/data1/xjh/tmp
export CUDA_VISIBLE_DEVICES={gpu}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_NO_USAGE_STATS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

RUN_ROOT={run_root}
CODE={code}
MODEL_ID={model_id}
TASKS_FILE="$RUN_ROOT/tasks.tsv"
MAX_RETRIES=${{MAX_RETRIES:-2}}
BATCH_SIZE=${{BATCH_SIZE:-256}}

mkdir -p "$RUN_ROOT"/{{logs,state,tmp,monitor,outputs/qwen3_30b_a3b}} "$TMPDIR"
TIMING="$RUN_ROOT/timing.tsv"
if [ ! -s "$TIMING" ]; then
  echo -e "alpha\tattempt\tstart_epoch\tend_epoch\telapsed_sec\texit_code" > "$TIMING"
fi

monitor() {{
  while true; do
    {{
      echo "===== $(date -Is) ====="
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
      ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E "eval_script5|prepare_qwen3_moe|vllm|python" | grep -v grep || true
    }} >> "$RUN_ROOT/monitor/gpu_{gpu_safe}.log" 2>&1
    sleep 60
  done
}}
monitor &
MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null || true' EXIT

run_once() {{
  local alpha="$1" attempt="$2"
  local out_root log tmp_dir prepared
  out_root="$RUN_ROOT/outputs/qwen3_30b_a3b"
  log="$RUN_ROOT/logs/qwen3_30b_a3b.alpha_${{alpha}}.attempt${{attempt}}"
  tmp_dir="$RUN_ROOT/tmp/qwen3_30b_a3b_${{alpha}}"
  mkdir -p "$tmp_dir" "$out_root"
  cd "$CODE"

  if [ "$alpha" = "0" ] || [ "$alpha" = "0.0" ]; then
    uv run python scripts/eval_script5_trait_personality.py \
      --alpha "$alpha" --model_id "$MODEL_ID" --local_files_only \
      --tensor_parallel_size 2 --gpu_memory_utilization 0.78 --max_model_len 2048 --max_tokens 1 --batch_size "$BATCH_SIZE" --enforce_eager \
      --tmp_root "$tmp_dir" --output_root "$out_root" --disable_thinking \
      > "${{log}}.out" 2> "${{log}}.err"
    return "$?"
  fi

  prepared="$tmp_dir/prepared_model"
  uv run python scripts/prepare_qwen3_moe_matthew.py \
    --model_id "$MODEL_ID" --alpha "$alpha" --output_dir "$prepared" --local_files_only \
    --svd_device cpu --max_memory_per_gpu 42GiB --force \
    > "${{log}}.prepare.out" 2> "${{log}}.prepare.err"
  local prep_rc=$?
  if [ "$prep_rc" -ne 0 ]; then
    rm -rf "$prepared"
    return "$prep_rc"
  fi

  uv run python scripts/eval_script5_trait_personality.py \
    --alpha "$alpha" --model_id "$MODEL_ID" --local_files_only --prepared_model_dir "$prepared" \
    --tensor_parallel_size 2 --gpu_memory_utilization 0.78 --max_model_len 2048 --max_tokens 1 --batch_size "$BATCH_SIZE" --enforce_eager \
    --tmp_root "$tmp_dir" --output_root "$out_root" --disable_thinking \
    > "${{log}}.out" 2> "${{log}}.err"
  local eval_rc=$?
  rm -rf "$prepared"
  return "$eval_rc"
}}

run_task() {{
  local alpha="$1"
  local done="$RUN_ROOT/state/qwen3_30b_a3b.alpha_${{alpha}}.done"
  local fail="$RUN_ROOT/state/qwen3_30b_a3b.alpha_${{alpha}}.fail"
  if [ -f "$done" ]; then
    echo "SKIP alpha=$alpha"
    return 0
  fi
  local attempt rc start end
  for attempt in $(seq 1 "$MAX_RETRIES"); do
    start=$(date +%s)
    echo "[$(date -Is)] START qwen3_30b_a3b alpha=$alpha attempt=$attempt batch=$BATCH_SIZE"
    run_once "$alpha" "$attempt"
    rc=$?
    end=$(date +%s)
    echo -e "$alpha\t$attempt\t$start\t$end\t$((end-start))\t$rc" >> "$TIMING"
    echo "[$(date -Is)] END qwen3_30b_a3b alpha=$alpha attempt=$attempt rc=$rc elapsed=$((end-start))"
    if [ "$rc" -eq 0 ]; then
      rm -f "$fail"
      date -Is > "$done"
      return 0
    fi
    echo "$(date -Is) rc=$rc attempt=$attempt" > "$fail"
    sleep 30
  done
  return "$rc"
}}

while IFS=$'\t' read -r alpha; do
  [ -z "$alpha" ] && continue
  run_task "$alpha" || exit "$?"
done < "$TASKS_FILE"

{{ echo "run_root: $RUN_ROOT"; echo "generated_at: $(date -Is)"; cat "$TIMING"; }} > "$RUN_ROOT/time_summary.txt"
'''


def main() -> None:
    Path(BASE).mkdir(parents=True, exist_ok=True)
    rows = []
    for gpu, alphas in LANES.items():
        run_root = f"{BASE}/gpu{gpu.replace(',', '_')}"
        Path(run_root).mkdir(parents=True, exist_ok=True)
        tasks_file = f"{run_root}/tasks.tsv"
        with open(tasks_file, "w") as f:
            for alpha in alphas:
                f.write(f"{alpha}\n")
        script = f"{run_root}/run.sh"
        with open(script, "w") as f:
            f.write(
                SCRIPT_TEMPLATE.format(
                    gpu=gpu,
                    gpu_safe=gpu.replace(",", "_"),
                    run_root=run_root,
                    code=CODE,
                    model_id=MODEL_ID,
                )
            )
        os.chmod(script, 0o755)
        log = f"{run_root}/launcher.log"
        with open(log, "ab") as out:
            proc = subprocess.Popen(["nohup", "bash", script], stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        rows.append((gpu, proc.pid, run_root, tasks_file, len(alphas)))
    print(f"RUN_BASE={BASE}")
    for row in rows:
        print(f"gpu={row[0]}\tpid={row[1]}\ttasks={row[4]}\trun_root={row[2]}\ttasks_file={row[3]}")


if __name__ == "__main__":
    main()
