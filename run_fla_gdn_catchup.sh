#!/bin/bash
# Catch up fla-gdn: run stages 1 → 2 → 3 on GPU 0.
# The main orchestrator's fla-gdn subshell died on OOM at stage 1 (no gc at 8K).
# This runner uses gc=True at all stages >=8K, matching the fixed flags.

set -u

DATA="/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile.txt"
OUTDIR="benchmark_results/full_progressive_128k/fla-gdn"
STAGE_MIN=30
GPU=0
BS=7
LR=0.0004794217925780434
BASE_ARGS="--level fla-gdn --dim 1664 --depth 13 --expansion 3 --n_heads 12"

STAGES=(512 8192 32768 131072)

# Start from stage 0 checkpoint
prev_ckpt=$(find "$OUTDIR/stage0_ctx512" -name 'checkpoint_step_*.pt' | sort | tail -1)
[ -z "$prev_ckpt" ] && { echo "no stage 0 checkpoint"; exit 1; }
echo "starting from: $prev_ckpt"

for idx in 1 2 3; do
  ctx=${STAGES[$idx]}
  seed=$((42 + idx))
  stage_dir="$OUTDIR/stage${idx}_ctx${ctx}"
  logfile="$OUTDIR/stage${idx}_ctx${ctx}.log"

  # Skip if already done
  existing=$(find "$stage_dir" -name 'checkpoint_step_*.pt' 2>/dev/null | sort | tail -1)
  if [ -n "$existing" ]; then
    echo "[fla-gdn] stage $idx SKIP (exists: $(basename "$existing"))"
    prev_ckpt="$existing"
    continue
  fi

  # Flags: gc for ctx >= 8K (fla-gdn has no projection chunking)
  extra=""
  [ $ctx -ge 8192 ] && extra="--gradient_checkpointing"

  # Clean any partial stage dir
  rm -rf "$stage_dir"
  mkdir -p "$stage_dir"

  echo "[fla-gdn] stage $idx ctx=$ctx seed=$seed bs=$BS flags=\"$extra\""
  CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --data "$DATA" \
    $BASE_ARGS \
    --lr "$LR" --bf16 \
    --batch_size "$BS" \
    --chunk_size "$ctx" \
    --train_minutes "$STAGE_MIN" \
    --output "$stage_dir" \
    --optimizer schedulefree \
    --seed "$seed" \
    --save_every 999999 --keep_checkpoints 1 --log_every 10 \
    $extra \
    --resume "$prev_ckpt" \
    > "$logfile" 2>&1

  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[fla-gdn] stage $idx FAILED (rc=$rc)"
    tail -5 "$logfile"
    exit 1
  fi

  prev_ckpt=$(find "$stage_dir" -name 'checkpoint_step_*.pt' | sort | tail -1)
  loss=$(grep "FINAL_LOSS_LAST100" "$logfile" | tail -1 | grep -oP '[0-9]+\.[0-9]+')
  echo "[fla-gdn] stage $idx done: loss=${loss:-?} ckpt=$(basename "$prev_ckpt")"
done

echo "[fla-gdn] ALL STAGES COMPLETE"
