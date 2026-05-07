#!/usr/bin/env python3
"""Launch Qwen3-8B eval6 nine-alpha sweep across xjha6 and xjhl4.

The sweep uses two GPUs on xjha6 and one GPU on xjhl4. Perturbed models are
stored only under each run's tmp directory, reused across the four eval6
presets for one alpha, then deleted.
"""

from __future__ import annotations

import os
import subprocess
import time


STAMP = time.strftime("%Y%m%d_%H%M%S")
CODE = "/data1/xjh/code/svd-trait-spectrum"
BASE_PREFIX = f"/data1/xjh/runs/svd-trait-spectrum/qwen3_8b_eval6_9alpha_distributed_{STAMP}"
PRESETS = ("mmlu_pro", "mmlu_redux", "agieval", "bbh")

HOSTS = {
    "xjha6": {
        "model_id": "/data1/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
        "lanes": {
            "0": ("-0.2", "-0.05", "0.1"),
            "2": ("-0.15", "0", "0.15"),
        },
    },
    "xjhl4": {
        "model_id": "/data1/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
        "lanes": {
            "1": ("-0.1", "0.05", "0.2"),
        },
    },
}


SCRIPT_TEMPLATE = r'''#!/usr/bin/env bash
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/data1/xjh/.cache/uv
export UV_LINK_MODE=copy
export HF_HOME=/data1/xjh/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data1/xjh/.cache/huggingface/hub
export HF_DATASETS_CACHE=/data1/xjh/.cache/huggingface/datasets
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_ENABLE_HF_TRANSFER=1
export TMPDIR=/data1/xjh/tmp
export CUDA_VISIBLE_DEVICES=__GPU__
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false

if [ -z "${HF_TOKEN:-}" ] && [ -f /data1/xjh/.config/svd/hf_token ]; then
  export HF_TOKEN=$(tr -d '\n' < /data1/xjh/.config/svd/hf_token)
fi

RUN_ROOT=__RUN_ROOT__
CODE=__CODE__
MODEL_ID=__MODEL_ID__
ALPHAS=(__ALPHAS__)
PRESETS=(__PRESETS__)
MAX_RETRIES=${MAX_RETRIES:-2}
BATCH_SIZE=${BATCH_SIZE:-auto}
AGIEVAL_BATCH_SIZE=${AGIEVAL_BATCH_SIZE:-1}

mkdir -p "$RUN_ROOT"/{logs,state,tmp,monitor,mmlu_pro,mmlu_redux,agieval,bbh} "$TMPDIR"
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
      ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E "eval_script6|lm_eval|vllm|python" | grep -v grep || true
    } >> "$RUN_ROOT/monitor/gpu___GPU__.log" 2>&1
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
    echo "$RUN_ROOT/tmp/prepared_qwen3_8b_alpha_$(alpha_tag "$alpha")"
  fi
}

run_once() {
  local alpha="$1" preset="$2" attempt="$3"
  local prep log output_root common preset_batch
  prep="$(prepared_dir "$alpha")"
  log="$RUN_ROOT/logs/qwen3_8b.alpha_${alpha}.${preset}.attempt${attempt}"
  output_root="$RUN_ROOT/$preset"
  mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/tmp/alpha_${alpha}" "$output_root"

  preset_batch="$BATCH_SIZE"
  if [ "$preset" = "agieval" ]; then
    preset_batch="$AGIEVAL_BATCH_SIZE"
  fi
  common=(--alpha "$alpha" --model_id "$MODEL_ID" --local_files_only --tensor_parallel_size 1 --gpu_memory_utilization 0.62 --enforce_eager --max_model_len 4096 --backend vllm --preset "$preset" --batch_size "$preset_batch" --tmp_root "$RUN_ROOT/tmp/alpha_${alpha}" --output_root "$output_root" --force)

  if [ -n "$prep" ]; then
    if [ -f "$prep/config.json" ]; then
      common+=(--prepared_model_dir "$prep")
    else
      common+=(--save_prepared_model_dir "$prep")
    fi
  fi

  cd "$CODE"
  uv run python scripts/eval_script6_lm_eval_full_subtasks.py "${common[@]}" > "${log}.out" 2> "${log}.err"
}

run_preset() {
  local alpha="$1" preset="$2"
  local done="$RUN_ROOT/state/qwen3_8b.alpha_${alpha}.${preset}.done"
  local fail="$RUN_ROOT/state/qwen3_8b.alpha_${alpha}.${preset}.fail"
  if [ -f "$done" ]; then
    echo "SKIP alpha=$alpha preset=$preset"
    return 0
  fi
  local attempt rc start end
  for attempt in $(seq 1 "$MAX_RETRIES"); do
    start=$(date +%s)
    echo "[$(date -Is)] START model=qwen3_8b alpha=$alpha preset=$preset attempt=$attempt"
    run_once "$alpha" "$preset" "$attempt"
    rc=$?
    end=$(date +%s)
    echo -e "$alpha\t$preset\t$attempt\t$start\t$end\t$((end-start))\t$rc" >> "$TIMING"
    echo "[$(date -Is)] END model=qwen3_8b alpha=$alpha preset=$preset attempt=$attempt rc=$rc elapsed=$((end-start))"
    if [ "$rc" -eq 0 ]; then
      rm -f "$fail"
      date -Is > "$done"
      rm -rf "$RUN_ROOT/tmp/alpha_${alpha}"/eval_script6_* 2>/dev/null || true
      return 0
    fi
    echo "$(date -Is) rc=$rc attempt=$attempt" > "$fail"
    rm -rf "$RUN_ROOT/tmp/alpha_${alpha}"/eval_script6_* 2>/dev/null || true
    sleep 30
  done
  return "$rc"
}

for alpha in "${ALPHAS[@]}"; do
  for preset in "${PRESETS[@]}"; do
    run_preset "$alpha" "$preset" || exit "$?"
  done
  prep="$(prepared_dir "$alpha")"
  if [ -n "$prep" ]; then
    echo "[$(date -Is)] CLEANUP model=qwen3_8b alpha=$alpha $prep"
    rm -rf "$prep"
  fi
done

{ echo "run_root: $RUN_ROOT"; echo "generated_at: $(date -Is)"; cat "$TIMING"; } > "$RUN_ROOT/time_summary.txt"
'''


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def run(host: str, cmd: str) -> None:
    subprocess.run(["ssh", host, cmd], check=True)


def main() -> None:
    print(f"RUN_BASE_PREFIX={BASE_PREFIX}")
    for host, cfg in HOSTS.items():
        run(host, f"mkdir -p {shell_quote(BASE_PREFIX)} {shell_quote(CODE)}/scripts")
        for gpu, alphas in cfg["lanes"].items():
            run_root = f"{BASE_PREFIX}/{host}_gpu{gpu}"
            script = f"{run_root}/run.sh"
            text = (
                SCRIPT_TEMPLATE.replace("__GPU__", gpu)
                .replace("__RUN_ROOT__", run_root)
                .replace("__CODE__", CODE)
                .replace("__MODEL_ID__", cfg["model_id"])
                .replace("__ALPHAS__", " ".join(alphas))
                .replace("__PRESETS__", " ".join(PRESETS))
            )
            tmp_local = f"/tmp/qwen3_8b_eval6_{host}_gpu{gpu}_{STAMP}.sh"
            with open(tmp_local, "w") as handle:
                handle.write(text)
            run(host, f"mkdir -p {shell_quote(run_root)}")
            subprocess.run(["scp", tmp_local, f"{host}:{script}"], check=True)
            run(host, f"chmod +x {shell_quote(script)}")
            log = f"{run_root}/launcher.log"
            launch = f"nohup bash {shell_quote(script)} >> {shell_quote(log)} 2>&1 & echo $!"
            proc = subprocess.check_output(["ssh", host, launch], text=True).strip()
            print(f"host={host}\tgpu={gpu}\tpid={proc}\trun_root={run_root}\talphas={','.join(alphas)}")


if __name__ == "__main__":
    main()
