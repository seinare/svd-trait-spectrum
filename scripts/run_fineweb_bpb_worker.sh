#!/usr/bin/env bash
set -u

GPU="$1"; shift
RUN_ROOT="$1"; shift
cd /data1/xjh/code/pruning || exit 2

for spec in "$@"; do
  IFS=, read -r MODEL MODEL_PATH ALPHA BATCH <<< "$spec"
  SAFE_ALPHA=${ALPHA/-/m}
  SAFE_ALPHA=${SAFE_ALPHA/./p}
  NAME="${MODEL}_alpha_${SAFE_ALPHA}"
  OUT="$RUN_ROOT/outputs/${NAME}.json"
  LOG="$RUN_ROOT/logs/gpu${GPU}_${NAME}.log"
  DONE="$RUN_ROOT/state/${NAME}.done"
  FAIL="$RUN_ROOT/state/${NAME}.fail"
  if [[ -f "$DONE" || -f "$OUT" ]]; then
    echo "[$(date -Is)] SKIP $NAME" | tee -a "$LOG"
    continue
  fi
  echo "[$(date -Is)] START gpu=$GPU model=$MODEL alpha=$ALPHA batch=$BATCH" | tee -a "$LOG"
  /data1/xjh/code/pruning/.venv/bin/python scripts/eval_fineweb_bpb_matthew.py \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL" \
    --alpha "$ALPHA" \
    --gpu "$GPU" \
    --output "$OUT" \
    --cache_dir /data1/xjh/runs/pruning_codex/fineweb_bpb_cache \
    --batch_size "$BATCH" \
    --seq_len 2048 >> "$LOG" 2>&1
  rc=$?
  if [[ $rc -eq 0 && -f "$OUT" ]]; then
    echo "[$(date -Is)] END $NAME rc=0" | tee -a "$LOG"
    echo "$(date -Is)" > "$DONE"
  else
    echo "[$(date -Is)] FAIL $NAME rc=$rc" | tee -a "$LOG"
    echo "$(date -Is) rc=$rc" > "$FAIL"
  fi
done
