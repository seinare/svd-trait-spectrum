#!/usr/bin/env python3
import os
import subprocess
import time


STAMP = time.strftime("20260501_%H%M%S")
BASE = f"/data1/cse12111103/runs/svd-trait-spectrum/parallel_sbj5_{STAMP}"
CODE = "/data1/cse12111103/code/svd-trait-spectrum"
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_ROOT = "/data1/cse12111103/models/svd-trait-spectrum"

JOB_SPECS = [
    ("-0.2", "ailab01", "L40", "gpu:l40:1"),
    ("-0.1", "ailab01", "L40", "gpu:l40:1"),
    ("0", "ailab03", "RTXPRO5000", "gpu:rtxpro5000:1"),
    ("0.1", "ailab01", "L40", "gpu:l40:1"),
    ("0.2", "ailab03", "RTXPRO5000", "gpu:rtxpro5000:1"),
]

SBATCH_TEMPLATE = r'''#!/usr/bin/env bash
#SBATCH -J {job_name}
#SBATCH -p {part}
#SBATCH --nodelist={node}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH -o /tmp/svd-parallel-{safe}-%j.out
#SBATCH -e /tmp/svd-parallel-{safe}-%j.err
set -uo pipefail
export HF_HOME=/data1/cse12111103/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data1/cse12111103/.cache/huggingface/hub
export UV_CACHE_DIR=/data1/cse12111103/.cache/uv
export TMPDIR=/data1/cse12111103/tmp
export HF_ENDPOINT=${{HF_ENDPOINT:-https://hf-mirror.com}}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NO_USAGE_STATS=1
RUN_ROOT={run_root}
CODE={code}
MODEL_ID={model_id}
MODEL_ROOT={model_root}
ALPHA={alpha}
MAX_RETRIES=2
mkdir -p "$RUN_ROOT"/logs "$RUN_ROOT"/state "$RUN_ROOT"/tmp "$RUN_ROOT"/monitor "$RUN_ROOT"/standard "$RUN_ROOT"/bfcl "$RUN_ROOT"/judge "$TMPDIR"
TIMING="$RUN_ROOT/timing.tsv"
if [ ! -s "$TIMING" ]; then echo -e "alpha\tstage\tattempt\tstart_epoch\tend_epoch\telapsed_sec\texit_code" > "$TIMING"; fi
monitor() {{
  while true; do
    {{
      echo "===== $(date -Is) ====="
      squeue -j "${{SLURM_JOB_ID:-0}}" -o "%i %j %t %M %N %R" || true
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
      df -h /dev/shm || true
      ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E "eval_script|vllm" | grep -v grep || true
    }} >> "$RUN_ROOT/monitor/gpu_${{SLURM_JOB_ID:-unknown}}.log" 2>&1
    sleep 60
  done
}}
monitor & MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null || true' EXIT
prepared_dir() {{
  if [ "$ALPHA" = "0" ] || [ "$ALPHA" = "0.0" ]; then echo ""; else echo "$MODEL_ROOT/Qwen3-8B-alpha${{ALPHA}}"; fi
}}
run_once() {{
  local stage="$1" attempt="$2"
  local prep; prep=$(prepared_dir)
  local log="$RUN_ROOT/logs/${{stage}}.attempt${{attempt}}"
  local common=(--alpha "$ALPHA" --model_id "$MODEL_ID" --tensor_parallel_size 1 --gpu_memory_utilization 0.72 --max_model_len 2048 --enforce_eager --local_files_only --tmp_root "$RUN_ROOT/tmp")
  if [ -n "$prep" ] && [ "$stage" != "standard" ]; then common+=(--prepared_model_dir "$prep"); fi
  cd "$CODE"
  case "$stage" in
    standard)
      if [ -n "$prep" ] && [ -d "$prep" ]; then
        uv run python scripts/eval_script1_standard.py "${{common[@]}}" --prepared_model_dir "$prep" --output_root "$RUN_ROOT/standard" > "${{log}}.out" 2> "${{log}}.err"
      elif [ -n "$prep" ]; then
        uv run python scripts/eval_script1_standard.py "${{common[@]}}" --save_prepared_model_dir "$prep" --output_root "$RUN_ROOT/standard" > "${{log}}.out" 2> "${{log}}.err"
      else
        uv run python scripts/eval_script1_standard.py "${{common[@]}}" --output_root "$RUN_ROOT/standard" > "${{log}}.out" 2> "${{log}}.err"
      fi
      ;;
    bfcl)
      uv run python scripts/eval_script2_bfcl.py "${{common[@]}}" --output_root "$RUN_ROOT/bfcl" > "${{log}}.out" 2> "${{log}}.err"
      ;;
    judge)
      uv run python scripts/eval_script3_judge.py "${{common[@]}}" --judge_model v4-flash --output_root "$RUN_ROOT/judge" > "${{log}}.out" 2> "${{log}}.err"
      ;;
  esac
}}
run_stage() {{
  local stage="$1" done="$RUN_ROOT/state/${{stage}}.done" fail="$RUN_ROOT/state/${{stage}}.fail"
  if [ -f "$done" ]; then echo "SKIP alpha=$ALPHA stage=$stage"; return 0; fi
  local attempt rc start end
  for attempt in $(seq 1 "$MAX_RETRIES"); do
    start=$(date +%s); echo "[$(date -Is)] START alpha=$ALPHA stage=$stage attempt=$attempt"
    run_once "$stage" "$attempt"; rc=$?
    end=$(date +%s); echo -e "$ALPHA\t$stage\t$attempt\t$start\t$end\t$((end-start))\t$rc" >> "$TIMING"
    echo "[$(date -Is)] END alpha=$ALPHA stage=$stage attempt=$attempt rc=$rc elapsed=$((end-start))"
    if [ "$rc" -eq 0 ]; then rm -f "$fail"; date -Is > "$done"; return 0; fi
    echo "$(date -Is) rc=$rc attempt=$attempt" > "$fail"; sleep 30
  done
  return "$rc"
}}
for stage in standard bfcl judge; do run_stage "$stage" || exit $?; done
{{ echo "run_root: $RUN_ROOT"; echo "generated_at: $(date -Is)"; cat "$TIMING"; }} > "$RUN_ROOT/time_summary.txt"
'''


def main():
    os.makedirs(BASE, exist_ok=True)
    rows = []
    for alpha, node, part, gres in JOB_SPECS:
        safe = alpha.replace("-", "m").replace(".", "_")
        run_root = f"{BASE}/{node}/alpha_{alpha}"
        script = f"/tmp/svd-parallel-{safe}-{node}-{STAMP}.sbatch"
        text = SBATCH_TEMPLATE.format(
            job_name=f"svd-a{safe}",
            part=part,
            node=node,
            gres=gres,
            safe=safe,
            run_root=run_root,
            code=CODE,
            model_id=MODEL_ID,
            model_root=MODEL_ROOT,
            alpha=alpha,
        )
        with open(script, "w") as f:
            f.write(text)
        cmd = [
            "sbatch",
            "--parsable",
            f"--export=ALL,HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')}",
            script,
        ]
        job_id = subprocess.check_output(cmd, text=True).strip()
        rows.append((job_id, alpha, node, gres, run_root))
    print(f"RUN_BASE={BASE}")
    for row in rows:
        print("\t".join(row))


if __name__ == "__main__":
    main()
