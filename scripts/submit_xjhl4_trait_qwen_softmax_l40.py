#!/usr/bin/env python3
import os
import subprocess
import time


STAMP = time.strftime("%Y%m%d_%H%M%S")
BASE = f"/data1/xjh/runs/pruning_codex/trait_qwen_softmax_l40_{STAMP}"
CODE = "/data1/xjh/code/pruning_codex"
GPU = "4"

ALPHAS = ["-0.3", "-0.2", "-0.1", "0", "0.1", "0.2", "0.3"]
MODEL_ID = "/data1/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
MODEL_ROOT = "/data1/xjh/models/pruning_codex/qwen3_8b"
PREPARED_PREFIX = "Qwen3-8B-alpha"

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
TASKS_FILE="$RUN_ROOT/tasks.tsv"
MAX_RETRIES=${{MAX_RETRIES:-2}}

mkdir -p "$RUN_ROOT"/{{logs,state,tmp,monitor,outputs/qwen3_8b}} "$TMPDIR"
TIMING="$RUN_ROOT/timing.tsv"
if [ ! -s "$TIMING" ]; then
  echo -e "alpha\tattempt\tstart_epoch\tend_epoch\telapsed_sec\texit_code" > "$TIMING"
fi

monitor() {{
  while true; do
    {{
      echo "===== $(date -Is) ====="
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
      ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E "eval_script5|vllm|python" | grep -v grep || true
    }} >> "$RUN_ROOT/monitor/gpu_{gpu}.log" 2>&1
    sleep 60
  done
}}
monitor &
MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null || true' EXIT

run_once() {{
  local alpha="$1" attempt="$2"
  local prep out_root log
  out_root="$RUN_ROOT/outputs/qwen3_8b"
  log="$RUN_ROOT/logs/qwen3_8b.alpha_${{alpha}}.attempt${{attempt}}"
  mkdir -p "$RUN_ROOT/tmp/qwen3_8b_${{alpha}}" "$out_root"
  cd "$CODE"

  if [ "$alpha" = "0" ] || [ "$alpha" = "0.0" ]; then
    uv run python scripts/eval_script5_trait_personality.py \
      --alpha "$alpha" --model_id "{model_id}" --local_files_only \
      --tensor_parallel_size 1 --gpu_memory_utilization 0.58 --max_model_len 2048 --max_tokens 1 --enforce_eager \
      --tmp_root "$RUN_ROOT/tmp/qwen3_8b_${{alpha}}" --output_root "$out_root" --disable_thinking \
      > "${{log}}.out" 2> "${{log}}.err"
  else
    prep="{model_root}/{prepared_prefix}${{alpha}}"
    if [ -d "$prep" ]; then
      uv run python scripts/eval_script5_trait_personality.py \
        --alpha "$alpha" --model_id "{model_id}" --local_files_only --prepared_model_dir "$prep" \
        --tensor_parallel_size 1 --gpu_memory_utilization 0.58 --max_model_len 2048 --max_tokens 1 --enforce_eager \
        --tmp_root "$RUN_ROOT/tmp/qwen3_8b_${{alpha}}" --output_root "$out_root" --disable_thinking \
        > "${{log}}.out" 2> "${{log}}.err"
    else
      uv run python scripts/eval_script5_trait_personality.py \
        --alpha "$alpha" --model_id "{model_id}" --local_files_only --save_prepared_model_dir "$prep" \
        --tensor_parallel_size 1 --gpu_memory_utilization 0.58 --max_model_len 2048 --max_tokens 1 --enforce_eager \
        --tmp_root "$RUN_ROOT/tmp/qwen3_8b_${{alpha}}" --output_root "$out_root" --disable_thinking \
        > "${{log}}.out" 2> "${{log}}.err"
    fi
  fi
}}

run_task() {{
  local alpha="$1"
  local done="$RUN_ROOT/state/qwen3_8b.alpha_${{alpha}}.done"
  local fail="$RUN_ROOT/state/qwen3_8b.alpha_${{alpha}}.fail"
  if [ -f "$done" ]; then
    echo "SKIP alpha=$alpha"
    return 0
  fi
  local attempt rc start end
  for attempt in $(seq 1 "$MAX_RETRIES"); do
    start=$(date +%s)
    echo "[$(date -Is)] START qwen3_8b alpha=$alpha attempt=$attempt"
    run_once "$alpha" "$attempt"
    rc=$?
    end=$(date +%s)
    echo -e "$alpha\t$attempt\t$start\t$end\t$((end-start))\t$rc" >> "$TIMING"
    echo "[$(date -Is)] END qwen3_8b alpha=$alpha attempt=$attempt rc=$rc elapsed=$((end-start))"
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
    run_root = f"{BASE}/gpu{GPU}"
    os.makedirs(run_root, exist_ok=True)
    tasks_file = f"{run_root}/tasks.tsv"
    with open(tasks_file, "w") as f:
        for alpha in ALPHAS:
            f.write(f"{alpha}\n")
    script = f"{run_root}/run.sh"
    with open(script, "w") as f:
        f.write(
            SCRIPT_TEMPLATE.format(
                gpu=GPU,
                run_root=run_root,
                code=CODE,
                model_id=MODEL_ID,
                model_root=MODEL_ROOT,
                prepared_prefix=PREPARED_PREFIX,
            )
        )
    os.chmod(script, 0o755)
    log = f"{run_root}/launcher.log"
    with open(log, "ab") as out:
        proc = subprocess.Popen(["nohup", "bash", script], stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"RUN_BASE={BASE}")
    print(f"gpu={GPU}\tpid={proc.pid}\trun_root={run_root}\ttasks={tasks_file}")


if __name__ == "__main__":
    main()
