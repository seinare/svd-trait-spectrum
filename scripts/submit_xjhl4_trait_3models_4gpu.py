#!/usr/bin/env python3
import os
import subprocess
import time


STAMP = time.strftime("%Y%m%d_%H%M%S")
BASE = f"/data1/xjh/runs/pruning_codex/trait_3models_4gpu_{STAMP}"
CODE = "/data1/xjh/code/pruning_codex"

MODELS = {
    "llama1b": {
        "model_id": "/data1/xjh/code/pruning/models/Llama-3.2-1B-Instruct",
        "model_root": "/data1/xjh/models/pruning_codex/llama1b",
        "prepared_prefix": "Llama-3.2-1B-Instruct-alpha",
        "alphas": ["-0.3", "-0.2", "-0.1", "0", "0.1", "0.2", "0.3"],
        "gpu_memory": "0.45",
    },
    "llama3b": {
        "model_id": "/data1/xjh/code/pruning/models/Llama-3.2-3B-Instruct",
        "model_root": "/data1/xjh/models/pruning_codex/llama3b",
        "prepared_prefix": "Llama-3.2-3B-Instruct-alpha",
        "alphas": ["-0.3", "-0.2", "-0.1", "0", "0.1", "0.2", "0.3"],
        "gpu_memory": "0.55",
    },
    "qwen3_8b": {
        "model_id": "/data1/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
        "model_root": "/data1/xjh/models/pruning_codex/qwen3_8b",
        "prepared_prefix": "Qwen3-8B-alpha",
        "alphas": ["-0.4", "-0.3", "-0.2", "-0.1", "0", "0.1", "0.2", "0.3", "0.4"],
        "gpu_memory": "0.62",
        "disable_thinking": True,
    },
}

LANES = {
    "1": [],
    "2": [],
    "3": [],
    "4": [],
}

all_tasks = []
for model_name, spec in MODELS.items():
    for alpha in spec["alphas"]:
        all_tasks.append((model_name, alpha))
for idx, task in enumerate(all_tasks):
    gpu = list(LANES)[idx % len(LANES)]
    LANES[gpu].append(task)

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

RUN_ROOT={run_root}
CODE={code}
TASKS_FILE="$RUN_ROOT/tasks.tsv"
MAX_RETRIES=${{MAX_RETRIES:-2}}

mkdir -p "$RUN_ROOT"/{{logs,state,tmp,monitor,outputs}} "$TMPDIR"
TIMING="$RUN_ROOT/timing.tsv"
if [ ! -s "$TIMING" ]; then
  echo -e "model\talpha\tattempt\tstart_epoch\tend_epoch\telapsed_sec\texit_code" > "$TIMING"
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

model_id() {{
  case "$1" in
    llama1b) echo "/data1/xjh/code/pruning/models/Llama-3.2-1B-Instruct" ;;
    llama3b) echo "/data1/xjh/code/pruning/models/Llama-3.2-3B-Instruct" ;;
    qwen3_8b) echo "/data1/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218" ;;
  esac
}}

model_root() {{
  case "$1" in
    llama1b) echo "/data1/xjh/models/pruning_codex/llama1b" ;;
    llama3b) echo "/data1/xjh/models/pruning_codex/llama3b" ;;
    qwen3_8b) echo "/data1/xjh/models/pruning_codex/qwen3_8b" ;;
  esac
}}

prepared_prefix() {{
  case "$1" in
    llama1b) echo "Llama-3.2-1B-Instruct-alpha" ;;
    llama3b) echo "Llama-3.2-3B-Instruct-alpha" ;;
    qwen3_8b) echo "Qwen3-8B-alpha" ;;
  esac
}}

gpu_memory() {{
  case "$1" in
    llama1b) echo "0.45" ;;
    llama3b) echo "0.55" ;;
    qwen3_8b) echo "0.62" ;;
  esac
}}

run_once() {{
  local model="$1" alpha="$2" attempt="$3"
  local mid root prefix prep out_root log mem extra
  mid="$(model_id "$model")"
  root="$(model_root "$model")"
  prefix="$(prepared_prefix "$model")"
  mem="$(gpu_memory "$model")"
  out_root="$RUN_ROOT/outputs/$model"
  log="$RUN_ROOT/logs/${{model}}.alpha_${{alpha}}.attempt${{attempt}}"
  mkdir -p "$root" "$out_root" "$RUN_ROOT/tmp/${{model}}_${{alpha}}"

  extra=()
  if [ "$model" = "qwen3_8b" ]; then
    extra+=(--disable_thinking)
  fi

  cd "$CODE"
  if [ "$alpha" = "0" ] || [ "$alpha" = "0.0" ]; then
    uv run python scripts/eval_script5_trait_personality.py \
      --alpha "$alpha" --model_id "$mid" --local_files_only \
      --tensor_parallel_size 1 --gpu_memory_utilization "$mem" --max_model_len 2048 --max_tokens 8 --enforce_eager \
      --tmp_root "$RUN_ROOT/tmp/${{model}}_${{alpha}}" --output_root "$out_root" "${{extra[@]}}" \
      > "${{log}}.out" 2> "${{log}}.err"
  else
    prep="$root/${{prefix}}${{alpha}}"
    if [ -d "$prep" ]; then
      uv run python scripts/eval_script5_trait_personality.py \
        --alpha "$alpha" --model_id "$mid" --local_files_only --prepared_model_dir "$prep" \
        --tensor_parallel_size 1 --gpu_memory_utilization "$mem" --max_model_len 2048 --max_tokens 8 --enforce_eager \
        --tmp_root "$RUN_ROOT/tmp/${{model}}_${{alpha}}" --output_root "$out_root" "${{extra[@]}}" \
        > "${{log}}.out" 2> "${{log}}.err"
    else
      uv run python scripts/eval_script5_trait_personality.py \
        --alpha "$alpha" --model_id "$mid" --local_files_only --save_prepared_model_dir "$prep" \
        --tensor_parallel_size 1 --gpu_memory_utilization "$mem" --max_model_len 2048 --max_tokens 8 --enforce_eager \
        --tmp_root "$RUN_ROOT/tmp/${{model}}_${{alpha}}" --output_root "$out_root" "${{extra[@]}}" \
        > "${{log}}.out" 2> "${{log}}.err"
    fi
  fi
}}

run_task() {{
  local model="$1" alpha="$2"
  local done="$RUN_ROOT/state/${{model}}.alpha_${{alpha}}.done"
  local fail="$RUN_ROOT/state/${{model}}.alpha_${{alpha}}.fail"
  if [ -f "$done" ]; then
    echo "SKIP model=$model alpha=$alpha"
    return 0
  fi
  local attempt rc start end
  for attempt in $(seq 1 "$MAX_RETRIES"); do
    start=$(date +%s)
    echo "[$(date -Is)] START model=$model alpha=$alpha attempt=$attempt"
    run_once "$model" "$alpha" "$attempt"
    rc=$?
    end=$(date +%s)
    echo -e "$model\t$alpha\t$attempt\t$start\t$end\t$((end-start))\t$rc" >> "$TIMING"
    echo "[$(date -Is)] END model=$model alpha=$alpha attempt=$attempt rc=$rc elapsed=$((end-start))"
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

while IFS=$'\t' read -r model alpha; do
  [ -z "$model" ] && continue
  run_task "$model" "$alpha" || exit "$?"
done < "$TASKS_FILE"

{{ echo "run_root: $RUN_ROOT"; echo "generated_at: $(date -Is)"; cat "$TIMING"; }} > "$RUN_ROOT/time_summary.txt"
'''


def main() -> None:
    os.makedirs(BASE, exist_ok=True)
    rows = []
    for gpu, tasks in LANES.items():
        run_root = f"{BASE}/gpu{gpu}"
        os.makedirs(run_root, exist_ok=True)
        tasks_file = f"{run_root}/tasks.tsv"
        with open(tasks_file, "w") as f:
            for model, alpha in tasks:
                f.write(f"{model}\t{alpha}\n")
        script = f"{run_root}/run.sh"
        with open(script, "w") as f:
            f.write(SCRIPT_TEMPLATE.format(gpu=gpu, run_root=run_root, code=CODE))
        os.chmod(script, 0o755)
        log = f"{run_root}/launcher.log"
        with open(log, "ab") as out:
            proc = subprocess.Popen(["nohup", "bash", script], stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        rows.append((gpu, proc.pid, run_root, tasks_file))

    print(f"RUN_BASE={BASE}")
    for gpu, pid, run_root, tasks_file in rows:
        print(f"gpu={gpu}\tpid={pid}\trun_root={run_root}\ttasks={tasks_file}")


if __name__ == "__main__":
    main()
