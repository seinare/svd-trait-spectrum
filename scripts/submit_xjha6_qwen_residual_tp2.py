#!/usr/bin/env python3
import os
import subprocess
import time


STAMP = time.strftime("%Y%m%d_%H%M%S")
BASE = f"/data1/xjh/runs/svd-trait-spectrum/xjha6_qwen_residual_tp2_{STAMP}"
CODE = "/data1/xjh/code/svd-trait-spectrum"
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_ROOT = "/data1/xjh/models/svd-trait-spectrum"

LANES = {
    "0,1": [
        ("judge", "-0.4"),
        ("standard_cot", "-0.4"),
        ("standard_cot", "0"),
        ("standard_cot", "0.3"),
        ("lm_eval", "-0.4"),
        ("lm_eval", "-0.3"),
        ("lm_eval", "-0.2"),
        ("lm_eval", "-0.1"),
        ("lm_eval", "0"),
    ],
    "2,3": [
        ("lm_eval", "0.1"),
        ("lm_eval", "0.2"),
        ("lm_eval", "0.3"),
        ("lm_eval", "0.4"),
    ],
}

SCRIPT_TEMPLATE = r'''#!/usr/bin/env bash
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/data1/xjh/.cache/uv
export HF_HOME=/data1/xjh/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data1/xjh/.cache/huggingface/hub
export HF_DATASETS_CACHE=/data1/xjh/.cache/huggingface/datasets
export HF_ENDPOINT=${{HF_ENDPOINT:-https://hf-mirror.com}}
export HF_HUB_ENABLE_HF_TRANSFER=1
export NLTK_DATA=/home/xjh/nltk_data
export TMPDIR=/data1/xjh/tmp
export CUDA_VISIBLE_DEVICES={gpus}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_NO_USAGE_STATS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -z "${{HF_TOKEN:-}}" ] && [ -f /data1/xjh/.config/svd/hf_token ]; then
  export HF_TOKEN=$(tr -d '\n' < /data1/xjh/.config/svd/hf_token)
fi
if [ -z "${{DEEPSEEK_API_KEY:-}}" ] && [ -f /data1/xjh/.config/svd/deepseek_api_key ]; then
  export DEEPSEEK_API_KEY=$(tr -d '\n' < /data1/xjh/.config/svd/deepseek_api_key)
fi
export OPENAI_API_KEY=${{OPENAI_API_KEY:-${{DEEPSEEK_API_KEY:-}}}}

RUN_ROOT={run_root}
CODE={code}
MODEL_ID={model_id}
MODEL_ROOT={model_root}
TASKS_FILE="$RUN_ROOT/tasks.tsv"
MAX_RETRIES=${{MAX_RETRIES:-3}}

mkdir -p "$RUN_ROOT"/{{logs,state,tmp,monitor,judge,lm_eval,standard_cot}} "$TMPDIR"
TIMING="$RUN_ROOT/timing.tsv"
if [ ! -s "$TIMING" ]; then
  echo -e "alpha\tstage\tattempt\tstart_epoch\tend_epoch\telapsed_sec\texit_code" > "$TIMING"
fi

monitor() {{
  while true; do
    {{
      echo "===== $(date -Is) ====="
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
      ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E "eval_script|lm_eval|vllm|python" | grep -v grep || true
    }} >> "$RUN_ROOT/monitor/gpu_{safe_gpus}.log" 2>&1
    sleep 60
  done
}}
monitor &
MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null || true' EXIT

prepared_dir() {{
  local alpha="$1"
  if [ "$alpha" = "0" ] || [ "$alpha" = "0.0" ]; then
    echo ""
  else
    echo "$MODEL_ROOT/Qwen3-8B-alpha${{alpha}}"
  fi
}}

run_once() {{
  local stage="$1" alpha="$2" attempt="$3"
  local prep log
  prep="$(prepared_dir "$alpha")"
  log="$RUN_ROOT/logs/alpha_${{alpha}}.${{stage}}.attempt${{attempt}}"
  mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/tmp/alpha_${{alpha}}"

  cd "$CODE"
  case "$stage" in
    judge)
      uv run python scripts/eval_script3_judge.py \
        --alpha "$alpha" --model_id "$MODEL_ID" --tensor_parallel_size 2 \
        --gpu_memory_utilization 0.60 --enforce_eager --tmp_root "$RUN_ROOT/tmp/alpha_${{alpha}}" \
        --prepared_model_dir "$prep" --max_model_len 4096 --judge_model v4-flash --judge_workers 5 \
        --output_root "$RUN_ROOT/judge" > "${{log}}.out" 2> "${{log}}.err"
      ;;
    standard_cot)
      local common=(--alpha "$alpha" --model_id "$MODEL_ID" --tensor_parallel_size 2 --gpu_memory_utilization 0.60 --enforce_eager --tmp_root "$RUN_ROOT/tmp/alpha_${{alpha}}" --max_model_len 8192 --max_tokens 8192 --output_root "$RUN_ROOT/standard_cot")
      if [ -n "$prep" ]; then common+=(--prepared_model_dir "$prep"); fi
      uv run python scripts/eval_script1_standard_cot.py "${{common[@]}}" > "${{log}}.out" 2> "${{log}}.err"
      ;;
    lm_eval)
      local common=(--alpha "$alpha" --model_id "$MODEL_ID" --tensor_parallel_size 2 --gpu_memory_utilization 0.58 --enforce_eager --tmp_root "$RUN_ROOT/tmp/alpha_${{alpha}}" --max_model_len 4096 --backend vllm --preset requested --batch_size 1 --force --output_root "$RUN_ROOT/lm_eval")
      if [ -n "$prep" ]; then common+=(--prepared_model_dir "$prep"); fi
      uv run python scripts/eval_script4_lm_eval_tasks.py "${{common[@]}}" > "${{log}}.out" 2> "${{log}}.err"
      ;;
    *)
      echo "unknown stage $stage" >&2
      return 2
      ;;
  esac
}}

run_stage() {{
  local stage="$1" alpha="$2"
  local done="$RUN_ROOT/state/alpha_${{alpha}}.${{stage}}.done"
  local fail="$RUN_ROOT/state/alpha_${{alpha}}.${{stage}}.fail"
  if [ -f "$done" ]; then
    echo "SKIP alpha=$alpha stage=$stage"
    return 0
  fi
  local attempt rc start end
  for attempt in $(seq 1 "$MAX_RETRIES"); do
    start=$(date +%s)
    echo "[$(date -Is)] START alpha=$alpha stage=$stage attempt=$attempt"
    run_once "$stage" "$alpha" "$attempt"
    rc=$?
    end=$(date +%s)
    echo -e "$alpha\t$stage\t$attempt\t$start\t$end\t$((end-start))\t$rc" >> "$TIMING"
    echo "[$(date -Is)] END alpha=$alpha stage=$stage attempt=$attempt rc=$rc elapsed=$((end-start))"
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

while IFS=$'\t' read -r stage alpha; do
  [ -z "$stage" ] && continue
  run_stage "$stage" "$alpha" || exit "$?"
done < "$TASKS_FILE"

{{ echo "run_root: $RUN_ROOT"; echo "generated_at: $(date -Is)"; cat "$TIMING"; }} > "$RUN_ROOT/time_summary.txt"
'''


def main() -> None:
    os.makedirs(BASE, exist_ok=True)
    rows = []
    for gpus, tasks in LANES.items():
        safe = gpus.replace(",", "")
        run_root = f"{BASE}/gpu{safe}"
        os.makedirs(run_root, exist_ok=True)
        tasks_file = f"{run_root}/tasks.tsv"
        with open(tasks_file, "w") as f:
            for stage, alpha in tasks:
                f.write(f"{stage}\t{alpha}\n")
        script = f"{run_root}/run.sh"
        with open(script, "w") as f:
            f.write(
                SCRIPT_TEMPLATE.format(
                    gpus=gpus,
                    safe_gpus=safe,
                    run_root=run_root,
                    code=CODE,
                    model_id=MODEL_ID,
                    model_root=MODEL_ROOT,
                )
            )
        os.chmod(script, 0o755)
        log = f"{run_root}/launcher.log"
        with open(log, "ab") as out:
            proc = subprocess.Popen(["nohup", "bash", script], stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        rows.append((gpus, proc.pid, run_root, tasks_file))

    print(f"RUN_BASE={BASE}")
    for gpus, pid, run_root, tasks_file in rows:
        print(f"gpus={gpus}\tpid={pid}\trun_root={run_root}\ttasks={tasks_file}")


if __name__ == "__main__":
    main()
