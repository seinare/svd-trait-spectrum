#!/usr/bin/env bash
# Submit with exported credentials or private credential files under
# /data1/cse12111103/.config/svd/.
# Re-submit with the same RUN_ROOT to continue from existing state/*.done markers.
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/data1/cse12111103/runs/svd-trait-spectrum/full_sbj_ailab03_2gpu/$(date +%Y%m%d_%H%M%S)}
CODE_DIR=${CODE_DIR:-/data1/cse12111103/code/svd-trait-spectrum}
MODEL_ROOT=${MODEL_ROOT:-/data1/cse12111103/models/svd-trait-spectrum}

mkdir -p "$RUN_ROOT"

cat > "$RUN_ROOT/job.sbatch" <<'EOF'
#!/usr/bin/env bash
#SBATCH -J svd-sbj5-full
#SBATCH -p RTXPRO5000
#SBATCH --nodelist=ailab03
#SBATCH --gres=gpu:rtxpro5000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=2-00:00:00
#SBATCH -o /tmp/svd-sbj5-full-%j.out
#SBATCH -e /tmp/svd-sbj5-full-%j.err
set -uo pipefail

export HF_HOME=/data1/cse12111103/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data1/cse12111103/.cache/huggingface/hub
export UV_CACHE_DIR=/data1/cse12111103/.cache/uv
export TMPDIR=/data1/cse12111103/tmp
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NO_USAGE_STATS=1
if [ -z "${HF_TOKEN:-}" ] && [ -f /data1/cse12111103/.config/svd/hf_token ]; then
  export HF_TOKEN=$(tr -d '\n' < /data1/cse12111103/.config/svd/hf_token)
fi
if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ -f /data1/cse12111103/.config/svd/deepseek_api_key ]; then
  export DEEPSEEK_API_KEY=$(tr -d '\n' < /data1/cse12111103/.config/svd/deepseek_api_key)
fi
export OPENAI_API_KEY=${OPENAI_API_KEY:-${DEEPSEEK_API_KEY:-}}

RUN_ROOT=__RUN_ROOT__
CODE_DIR=__CODE_DIR__
MODEL_ROOT=__MODEL_ROOT__
MODEL_ID=Qwen/Qwen3-8B
ALPHAS=(-0.2 -0.1 0 0.1 0.2)
STAGES=(standard bfcl judge)
MAX_RETRIES=${MAX_RETRIES:-2}

mkdir -p "$RUN_ROOT"/{logs,state,tmp,monitor} "$TMPDIR"
touch "$RUN_ROOT/timing.tsv"
if ! grep -q '^alpha' "$RUN_ROOT/timing.tsv"; then
  echo -e "alpha\tstage\tattempt\tstart_epoch\tend_epoch\telapsed_sec\texit_code" > "$RUN_ROOT/timing.tsv"
fi

monitor() {
  while true; do
    {
      echo "===== $(date -Is) ====="
      squeue -j "${SLURM_JOB_ID:-0}" -o "%i %j %t %M %N %R" || true
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
      df -h /dev/shm || true
      ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E "eval_script|vllm|lm_eval" | grep -v grep || true
    } >> "$RUN_ROOT/monitor/gpu_${SLURM_JOB_ID:-unknown}.log" 2>&1
    sleep 60
  done
}
monitor &
MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null || true' EXIT

prepared_dir_for_alpha() {
  local alpha="$1"
  if [ "$alpha" = "0" ] || [ "$alpha" = "0.0" ]; then
    echo ""
  else
    echo "$MODEL_ROOT/Qwen3-8B-alpha${alpha}"
  fi
}

run_stage_once() {
  local alpha="$1" stage="$2" attempt="$3"
  local alpha_safe="${alpha//- /m}"
  alpha_safe="${alpha_safe//-/m}"
  alpha_safe="${alpha_safe//./_}"
  local log_prefix="$RUN_ROOT/logs/alpha_${alpha}/${stage}.attempt${attempt}"
  mkdir -p "$RUN_ROOT/logs/alpha_${alpha}" "$RUN_ROOT/tmp/alpha_${alpha}"

  local prepared
  prepared="$(prepared_dir_for_alpha "$alpha")"
  local common=(--alpha "$alpha" --model_id "$MODEL_ID" --tensor_parallel_size 2 --gpu_memory_utilization 0.75 --max_model_len 2048 --enforce_eager --local_files_only --tmp_root "$RUN_ROOT/tmp/alpha_${alpha}")

  if [ -n "$prepared" ] && [ "$stage" != "standard" ]; then
    common+=(--prepared_model_dir "$prepared")
  fi

  cd "$CODE_DIR"
  case "$stage" in
    standard)
      if [ -n "$prepared" ] && [ -d "$prepared" ]; then
        uv run python scripts/eval_script1_standard.py "${common[@]}" --prepared_model_dir "$prepared" --output_root "$RUN_ROOT/standard" > "${log_prefix}.out" 2> "${log_prefix}.err"
      elif [ -n "$prepared" ]; then
        uv run python scripts/eval_script1_standard.py "${common[@]}" --save_prepared_model_dir "$prepared" --output_root "$RUN_ROOT/standard" > "${log_prefix}.out" 2> "${log_prefix}.err"
      else
        uv run python scripts/eval_script1_standard.py "${common[@]}" --output_root "$RUN_ROOT/standard" > "${log_prefix}.out" 2> "${log_prefix}.err"
      fi
      ;;
    bfcl)
      uv run python scripts/eval_script2_bfcl.py "${common[@]}" --output_root "$RUN_ROOT/bfcl" > "${log_prefix}.out" 2> "${log_prefix}.err"
      ;;
    judge)
      uv run python scripts/eval_script3_judge.py "${common[@]}" --judge_model v4-flash --output_root "$RUN_ROOT/judge" > "${log_prefix}.out" 2> "${log_prefix}.err"
      ;;
    *)
      echo "unknown stage $stage" >&2
      return 2
      ;;
  esac
}

run_stage_with_resume() {
  local alpha="$1" stage="$2"
  local done_file="$RUN_ROOT/state/alpha_${alpha}.${stage}.done"
  local fail_file="$RUN_ROOT/state/alpha_${alpha}.${stage}.fail"
  if [ -f "$done_file" ]; then
    echo "SKIP alpha=$alpha stage=$stage already done"
    return 0
  fi

  local attempt rc start end
  for attempt in $(seq 1 "$MAX_RETRIES"); do
    start=$(date +%s)
    echo "[$(date -Is)] START alpha=$alpha stage=$stage attempt=$attempt"
    run_stage_once "$alpha" "$stage" "$attempt"
    rc=$?
    end=$(date +%s)
    echo -e "$alpha\t$stage\t$attempt\t$start\t$end\t$((end-start))\t$rc" >> "$RUN_ROOT/timing.tsv"
    echo "[$(date -Is)] END alpha=$alpha stage=$stage attempt=$attempt rc=$rc elapsed=$((end-start))"
    if [ "$rc" -eq 0 ]; then
      rm -f "$fail_file"
      echo "$(date -Is)" > "$done_file"
      return 0
    fi
    echo "$(date -Is) rc=$rc attempt=$attempt" > "$fail_file"
    sleep 30
  done
  return "$rc"
}

overall_rc=0
for alpha in "${ALPHAS[@]}"; do
  for stage in "${STAGES[@]}"; do
    run_stage_with_resume "$alpha" "$stage" || {
      overall_rc=$?
      echo "FAILED alpha=$alpha stage=$stage rc=$overall_rc"
      exit "$overall_rc"
    }
  done
done

{
  echo "run_root: $RUN_ROOT"
  echo "generated_at: $(date -Is)"
  echo
  cat "$RUN_ROOT/timing.tsv"
} > "$RUN_ROOT/time_summary.txt"
exit 0
EOF

sed -i "s#__RUN_ROOT__#$RUN_ROOT#g; s#__CODE_DIR__#$CODE_DIR#g; s#__MODEL_ROOT__#$MODEL_ROOT#g" "$RUN_ROOT/job.sbatch"
sbatch --export=ALL,HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" "$RUN_ROOT/job.sbatch"
echo "$RUN_ROOT"
