#!/usr/bin/env python3
import os
import subprocess
import time


STAMP = time.strftime("%Y%m%d_%H%M%S")
BASE = f"/data1/xjh/runs/pruning_codex/llama3b_7alpha_5modules_{STAMP}"
CODE = "/data1/xjh/code/pruning_codex"
MODEL_ID = "/data1/xjh/code/pruning/models/Llama-3.2-3B-Instruct"
MODEL_ROOT = "/data1/xjh/models/pruning_codex/llama3b"

LANES = {
    "1": ["-0.3"],
    "2": ["-0.2"],
    "3": ["-0.1"],
    "4": ["0"],
    "5": ["0.1"],
    "6": ["0.2"],
    "7": ["0.3"],
}

STAGES = ("standard", "bfcl", "judge", "lm_eval", "standard_cot")

SCRIPT_TEMPLATE = r'''#!/usr/bin/env bash
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/data1/xjh/.cache/uv
export HF_HOME=/data1/xjh/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data1/xjh/.cache/huggingface/hub
export HF_DATASETS_CACHE=/data1/xjh/.cache/huggingface/datasets
export HF_ENDPOINT=${{HF_ENDPOINT:-https://hf-mirror.com}}
export HF_HUB_ENABLE_HF_TRANSFER=1
export TMPDIR=/data1/xjh/tmp
export CUDA_VISIBLE_DEVICES={gpu}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_NO_USAGE_STATS=1

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
ALPHAS=({alphas})
STAGES=({stages})
MAX_RETRIES=${{MAX_RETRIES:-2}}

mkdir -p "$RUN_ROOT"/{{logs,state,tmp,monitor,standard,bfcl,judge,lm_eval,standard_cot}} "$MODEL_ROOT" "$TMPDIR"
TIMING="$RUN_ROOT/timing.tsv"
if [ ! -s "$TIMING" ]; then
  echo -e "alpha\tstage\tattempt\tstart_epoch\tend_epoch\telapsed_sec\texit_code" > "$TIMING"
fi

monitor() {{
  while true; do
    {{
      echo "===== $(date -Is) ====="
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
      df -h /data1 "$TMPDIR" || true
      ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E "eval_script|lm_eval|vllm|python" | grep -v grep || true
    }} >> "$RUN_ROOT/monitor/gpu_{gpu}.log" 2>&1
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
    echo "$MODEL_ROOT/Llama-3.2-3B-Instruct-alpha${{alpha}}"
  fi
}}

run_once() {{
  local alpha="$1" stage="$2" attempt="$3"
  local prep log common
  prep="$(prepared_dir "$alpha")"
  log="$RUN_ROOT/logs/alpha_${{alpha}}.${{stage}}.attempt${{attempt}}"
  mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/tmp/alpha_${{alpha}}"

  common=(--alpha "$alpha" --model_id "$MODEL_ID" --local_files_only --tensor_parallel_size 1 --gpu_memory_utilization 0.60 --enforce_eager --tmp_root "$RUN_ROOT/tmp/alpha_${{alpha}}")
  if [ -n "$prep" ] && [ "$stage" != "standard" ]; then
    common+=(--prepared_model_dir "$prep")
  fi

  cd "$CODE"
  case "$stage" in
    standard)
      if [ -n "$prep" ] && [ -d "$prep" ]; then
        uv run python scripts/eval_script1_standard.py "${{common[@]}}" --max_model_len 4096 --max_tokens 256 --prepared_model_dir "$prep" --output_root "$RUN_ROOT/standard" > "${{log}}.out" 2> "${{log}}.err"
      elif [ -n "$prep" ]; then
        uv run python scripts/eval_script1_standard.py "${{common[@]}}" --max_model_len 4096 --max_tokens 256 --save_prepared_model_dir "$prep" --output_root "$RUN_ROOT/standard" > "${{log}}.out" 2> "${{log}}.err"
      else
        uv run python scripts/eval_script1_standard.py "${{common[@]}}" --max_model_len 4096 --max_tokens 256 --output_root "$RUN_ROOT/standard" > "${{log}}.out" 2> "${{log}}.err"
      fi
      ;;
    bfcl)
      uv run python scripts/eval_script2_bfcl.py "${{common[@]}}" --max_model_len 4096 --output_root "$RUN_ROOT/bfcl" > "${{log}}.out" 2> "${{log}}.err"
      ;;
    judge)
      uv run python scripts/eval_script3_judge.py "${{common[@]}}" --max_model_len 4096 --judge_model v4-flash --judge_workers 20 --output_root "$RUN_ROOT/judge" > "${{log}}.out" 2> "${{log}}.err"
      ;;
    lm_eval)
      uv run python scripts/eval_script4_lm_eval_tasks.py "${{common[@]}}" --max_model_len 4096 --backend vllm --preset requested --batch_size auto --output_root "$RUN_ROOT/lm_eval" > "${{log}}.out" 2> "${{log}}.err"
      ;;
    standard_cot)
      uv run python scripts/eval_script1_standard_cot.py "${{common[@]}}" --max_model_len 8192 --max_tokens 8192 --output_root "$RUN_ROOT/standard_cot" > "${{log}}.out" 2> "${{log}}.err"
      ;;
    *)
      echo "unknown stage $stage" >&2
      return 2
      ;;
  esac
}}

run_stage() {{
  local alpha="$1" stage="$2"
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
    run_once "$alpha" "$stage" "$attempt"
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

for alpha in "${{ALPHAS[@]}}"; do
  for stage in "${{STAGES[@]}}"; do
    run_stage "$alpha" "$stage" || exit "$?"
  done
done

{{ echo "run_root: $RUN_ROOT"; echo "generated_at: $(date -Is)"; cat "$TIMING"; }} > "$RUN_ROOT/time_summary.txt"
'''


def main() -> None:
    os.makedirs(BASE, exist_ok=True)
    rows = []
    for gpu, alphas in LANES.items():
        run_root = f"{BASE}/gpu{gpu}"
        os.makedirs(run_root, exist_ok=True)
        script = f"{run_root}/run.sh"
        with open(script, "w") as f:
            f.write(
                SCRIPT_TEMPLATE.format(
                    gpu=gpu,
                    run_root=run_root,
                    code=CODE,
                    model_id=MODEL_ID,
                    model_root=MODEL_ROOT,
                    alphas=" ".join(alphas),
                    stages=" ".join(STAGES),
                )
            )
        os.chmod(script, 0o755)
        log = f"{run_root}/launcher.log"
        with open(log, "ab") as out:
            proc = subprocess.Popen(["nohup", "bash", script], stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        rows.append((gpu, ",".join(alphas), proc.pid, run_root))

    print(f"RUN_BASE={BASE}")
    for gpu, alphas, pid, run_root in rows:
        print(f"gpu={gpu}\talphas={alphas}\tpid={pid}\trun_root={run_root}")


if __name__ == "__main__":
    main()
