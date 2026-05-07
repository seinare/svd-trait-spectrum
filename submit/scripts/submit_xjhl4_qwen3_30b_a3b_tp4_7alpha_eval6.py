#!/usr/bin/env python3
"""Launch Qwen3-30B-A3B seven-alpha eval_script6 suite on xjhl4 with TP=4."""

from __future__ import annotations

import os
import subprocess
import time


STAMP = time.strftime("%Y%m%d_%H%M%S")
RUN_ROOT = f"/data1/xjh/runs/pruning_codex/qwen3_30b_a3b_tp4_7alpha_eval6_{STAMP}"
CODE = "/data1/xjh/code/pruning_codex"
MODEL_ID = "/data1/xjh/code/pruning/models/Qwen3-30B-A3B"
PREP_ROOT = f"/data1/xjh/tmp/qwen3_30b_a3b_eval6_prepared_{STAMP}"
ALPHAS = ("-0.3", "-0.2", "-0.1", "0", "0.1", "0.2", "0.3")
PRESETS = ("mmlu_pro", "mmlu_redux", "agieval", "bbh")


SCRIPT = r'''#!/usr/bin/env bash
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/data1/xjh/.cache/uv
export HF_HOME=/data1/xjh/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data1/xjh/.cache/huggingface/hub
export HF_DATASETS_CACHE=/data1/xjh/.cache/huggingface/datasets
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_ENABLE_HF_TRANSFER=1
export TMPDIR=/data1/xjh/tmp
export CUDA_VISIBLE_DEVICES=1,2,3,4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_NO_USAGE_STATS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

if [ -z "${HF_TOKEN:-}" ] && [ -f /data1/xjh/.config/svd/hf_token ]; then
  export HF_TOKEN=$(tr -d '\n' < /data1/xjh/.config/svd/hf_token)
fi

RUN_ROOT={run_root}
CODE={code}
MODEL_ID={model_id}
PREP_ROOT={prep_root}
ALPHAS=({alphas})
PRESETS=({presets})
MAX_RETRIES=${MAX_RETRIES:-1}

mkdir -p "$RUN_ROOT"/{logs,state,tmp,monitor,mmlu_pro,mmlu_redux,agieval,bbh} "$PREP_ROOT" "$TMPDIR"
TIMING="$RUN_ROOT/timing.tsv"
if [ ! -s "$TIMING" ]; then
  echo -e "alpha\tpreset\tattempt\tstart_epoch\tend_epoch\telapsed_sec\texit_code" > "$TIMING"
fi

monitor() {
  while true; do
    {
      echo "===== $(date -Is) ====="
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
      df -h /data1 "$TMPDIR" || true
      ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E "prepare_qwen3|eval_script6|lm_eval|vllm|python" | grep -v grep || true
    } >> "$RUN_ROOT/monitor/gpu_tp4.log" 2>&1
    sleep 60
  done
}
monitor &
MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null || true' EXIT

alpha_tag() {
  echo "$1" | sed 's/-/m/g; s/\./p/g'
}

prepared_dir() {
  local alpha="$1"
  if [ "$alpha" = "0" ] || [ "$alpha" = "0.0" ]; then
    echo ""
  else
    echo "$PREP_ROOT/alpha_$(alpha_tag "$alpha")"
  fi
}

prepare_alpha() {
  local alpha="$1" prep
  prep="$(prepared_dir "$alpha")"
  [ -z "$prep" ] && return 0
  [ -f "$prep/.matthew_done.json" ] && return 0
  local log="$RUN_ROOT/logs/alpha_${alpha}.prepare"
  echo "[$(date -Is)] PREPARE alpha=$alpha -> $prep"
  cd "$CODE"
  uv run python scripts/prepare_qwen3_moe_matthew.py \
    --model_id "$MODEL_ID" --alpha "$alpha" --output_dir "$prep" \
    --local_files_only --max_memory_per_gpu 42GiB --max_shard_size 4GB --force \
    > "${log}.out" 2> "${log}.err"
}

cleanup_alpha() {
  local alpha="$1" prep
  prep="$(prepared_dir "$alpha")"
  if [ -n "$prep" ] && [ -d "$prep" ]; then
    echo "[$(date -Is)] CLEANUP alpha=$alpha $prep"
    rm -rf "$prep"
  fi
}

run_once() {
  local alpha="$1" preset="$2" attempt="$3"
  local prep log output_root common
  prep="$(prepared_dir "$alpha")"
  log="$RUN_ROOT/logs/alpha_${alpha}.${preset}.attempt${attempt}"
  output_root="$RUN_ROOT/$preset"
  mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/tmp/alpha_${alpha}" "$output_root"

  common=(--alpha "$alpha" --model_id "$MODEL_ID" --local_files_only --tensor_parallel_size 4 --gpu_memory_utilization 0.82 --enforce_eager --max_model_len 4096 --backend vllm --preset "$preset" --batch_size auto --tmp_root "$RUN_ROOT/tmp/alpha_${alpha}" --output_root "$output_root")
  if [ -n "$prep" ]; then
    common+=(--prepared_model_dir "$prep")
  fi

  cd "$CODE"
  uv run python scripts/eval_script6_lm_eval_full_subtasks.py "${common[@]}" > "${log}.out" 2> "${log}.err"
}

run_preset() {
  local alpha="$1" preset="$2"
  local done="$RUN_ROOT/state/alpha_${alpha}.${preset}.done"
  local fail="$RUN_ROOT/state/alpha_${alpha}.${preset}.fail"
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
      return 0
    fi
    echo "$(date -Is) rc=$rc attempt=$attempt" > "$fail"
    sleep 30
  done
  return "$rc"
}

for alpha in "${ALPHAS[@]}"; do
  prepare_alpha "$alpha" || exit "$?"
  for preset in "${PRESETS[@]}"; do
    run_preset "$alpha" "$preset" || exit "$?"
  done
  cleanup_alpha "$alpha"
done

{ echo "run_root: $RUN_ROOT"; echo "generated_at: $(date -Is)"; cat "$TIMING"; } > "$RUN_ROOT/time_summary.txt"
'''


def main() -> None:
    os.makedirs(RUN_ROOT, exist_ok=True)
    script = os.path.join(RUN_ROOT, "run.sh")
    text = SCRIPT
    for old, new in {
        "{run_root}": RUN_ROOT,
        "{code}": CODE,
        "{model_id}": MODEL_ID,
        "{prep_root}": PREP_ROOT,
        "{alphas}": " ".join(ALPHAS),
        "{presets}": " ".join(PRESETS),
    }.items():
        text = text.replace(old, new)
    with open(script, "w") as f:
        f.write(text)
    os.chmod(script, 0o755)
    log = os.path.join(RUN_ROOT, "launcher.log")
    with open(log, "ab") as out:
        proc = subprocess.Popen(["nohup", "bash", script], stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"RUN_ROOT={RUN_ROOT}")
    print(f"PID={proc.pid}")
    print(f"SCRIPT={script}")


if __name__ == "__main__":
    main()
