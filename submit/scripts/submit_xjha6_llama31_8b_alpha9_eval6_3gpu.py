#!/usr/bin/env python3
"""Launch Llama-3.1-8B-Instruct alpha -0.2..0.2 eval_script6 sweep on three xjha6 GPUs.

Perturbed weights are only written to eval_script6 temporary directories and
deleted after each preset; no persistent prepared model directory is used.
"""

from __future__ import annotations

import os
import subprocess
import time


STAMP = time.strftime("%Y%m%d_%H%M%S")
BASE = f"/data1/xjh/runs/svd-trait-spectrum/llama31_8b_alpha9_eval6_3gpu_{STAMP}"
CODE = "/data1/xjh/code/svd-trait-spectrum"
MODEL_ID = "/data1/xjh/code/pruning/models/Llama-3.1-8B-Instruct"
GPU_ALPHAS = {
    "0": ("-0.2", "-0.05", "0.1"),
    "1": ("-0.15", "0", "0.15"),
    "2": ("-0.1", "0.05", "0.2"),
}
PRESETS = ("mmlu_pro", "mmlu_redux", "agieval", "bbh")


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

RUN_ROOT={run_root}
CODE={code}
MODEL_ID={model_id}
ALPHAS=({alphas})
PRESETS=({presets})
MAX_RETRIES=${{MAX_RETRIES:-2}}

mkdir -p "$RUN_ROOT"/{{logs,state,tmp,monitor,mmlu_pro,mmlu_redux,agieval,bbh}} "$TMPDIR"
TIMING="$RUN_ROOT/timing.tsv"
if [ ! -s "$TIMING" ]; then
  echo -e "alpha\tpreset\tattempt\tstart_epoch\tend_epoch\telapsed_sec\texit_code" > "$TIMING"
fi

monitor() {{
  while true; do
    {{
      echo "===== $(date -Is) ====="
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
      df -h /data1 "$TMPDIR" || true
      ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E "eval_script6|lm_eval|vllm|python" | grep -v grep || true
    }} >> "$RUN_ROOT/monitor/gpu_{gpu}.log" 2>&1
    sleep 60
  done
}}
monitor &
MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null || true' EXIT

run_once() {{
  local alpha="$1" preset="$2" attempt="$3"
  local log output_root
  log="$RUN_ROOT/logs/alpha_${{alpha}}.${{preset}}.attempt${{attempt}}"
  output_root="$RUN_ROOT/$preset"
  mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/tmp/alpha_${{alpha}}" "$output_root"

  cd "$CODE"
  uv run python scripts/eval_script6_lm_eval_full_subtasks.py \
    --alpha "$alpha" \
    --model_id "$MODEL_ID" \
    --local_files_only \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.55 \
    --enforce_eager \
    --max_model_len 4096 \
    --backend vllm \
    --preset "$preset" \
    --batch_size auto \
    --tmp_root "$RUN_ROOT/tmp/alpha_${{alpha}}" \
    --output_root "$output_root" \
    > "${{log}}.out" 2> "${{log}}.err"
}}

run_preset() {{
  local alpha="$1" preset="$2"
  local done="$RUN_ROOT/state/alpha_${{alpha}}.${{preset}}.done"
  local fail="$RUN_ROOT/state/alpha_${{alpha}}.${{preset}}.fail"
  if [ -f "$done" ]; then
    echo "SKIP alpha=$alpha preset=$preset"
    return 0
  fi
  local attempt rc start end
  for attempt in $(seq 1 "$MAX_RETRIES"); do
    start=$(date +%s)
    echo "[$(date -Is)] START alpha=$alpha preset=$preset attempt=$attempt"
    run_once "$alpha" "$preset" "$attempt"
    rc=$?
    end=$(date +%s)
    echo -e "$alpha\t$preset\t$attempt\t$start\t$end\t$((end-start))\t$rc" >> "$TIMING"
    echo "[$(date -Is)] END alpha=$alpha preset=$preset attempt=$attempt rc=$rc elapsed=$((end-start))"
    if [ "$rc" -eq 0 ]; then
      rm -f "$fail"
      date -Is > "$done"
      rm -rf "$RUN_ROOT/tmp/alpha_${{alpha}}"/eval_script6_* 2>/dev/null || true
      return 0
    fi
    echo "$(date -Is) rc=$rc attempt=$attempt" > "$fail"
    rm -rf "$RUN_ROOT/tmp/alpha_${{alpha}}"/eval_script6_* 2>/dev/null || true
    sleep 30
  done
  return "$rc"
}}

for alpha in "${{ALPHAS[@]}}"; do
  for preset in "${{PRESETS[@]}}"; do
    run_preset "$alpha" "$preset" || exit "$?"
  done
done

{{ echo "run_root: $RUN_ROOT"; echo "generated_at: $(date -Is)"; cat "$TIMING"; }} > "$RUN_ROOT/time_summary.txt"
'''


def main() -> None:
    os.makedirs(BASE, exist_ok=True)
    print(f"RUN_BASE={BASE}")
    for gpu, alphas in GPU_ALPHAS.items():
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
                    alphas=" ".join(alphas),
                    presets=" ".join(PRESETS),
                )
            )
        os.chmod(script, 0o755)
        log = f"{run_root}/launcher.log"
        with open(log, "ab") as out:
            proc = subprocess.Popen(["nohup", "bash", script], stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"gpu={gpu}\talphas={','.join(alphas)}\tpresets={','.join(PRESETS)}\tpid={proc.pid}\trun_root={run_root}")


if __name__ == "__main__":
    main()
