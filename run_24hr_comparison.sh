#!/bin/bash
# 24-hour loss-curve comparison at 128K context.
# Resumes from each model's stage 3 checkpoint (final stage of the progressive run)
# and trains for 1440 minutes. All three models start simultaneously on GPUs 0, 1, 2.
#
# Log every 10 steps, save every 500, keep 5 checkpoints for post-hoc analysis.

set -u

DATA="/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile.txt"
PROG_DIR="benchmark_results/full_progressive_128k"
OUTDIR="benchmark_results/24hr_128k"
TRAIN_MIN=1440
SEED=46

mkdir -p "$OUTDIR"

# Model configs — same as progressive, bs=1 at 128K (matches CMA-ES memory-probed actual)
# format: name|gpu|base_args|lr
MODELS=(
  "fla-gdn|0|--level fla-gdn --dim 1664 --depth 13 --expansion 3 --n_heads 12|0.0004794217925780434"
  "e88_n32|1|--level E88 --dim 2432 --depth 10 --n_heads 112 --n_state 32 --expansion 1.0 --use_gate 1 --gate_activation silu|0.0006361350114482852"
  "e1h_n32|2|--level E1H --dim 1792 --depth 13 --n_heads 196 --n_state 32|0.0006030367837449147"
)

# Find each model's stage 3 checkpoint (final progressive stage)
declare -A CKPT
for entry in "${MODELS[@]}"; do
  IFS='|' read -r name _ _ _ <<< "$entry"
  ckpt=$(find "$PROG_DIR/$name/stage3_ctx131072" -name 'checkpoint_step_*.pt' 2>/dev/null | sort | tail -1)
  if [ -z "$ckpt" ]; then
    echo "ERROR: no stage 3 checkpoint for $name — progressive run not complete?"
    exit 1
  fi
  CKPT[$name]="$ckpt"
  echo "$name resume from: $ckpt"
done

# Per-model extra flags at 128K
extra_flags() {
  local name=$1
  if [[ "$name" == e88* ]]; then
    echo "--gradient_checkpointing --projection_chunk_size 512"
  else
    echo "--gradient_checkpointing"
  fi
}

run_model() {
  local name=$1 gpu=$2 args=$3 lr=$4
  local stage_dir="$OUTDIR/$name"
  mkdir -p "$stage_dir"
  local logfile="$OUTDIR/${name}.log"
  local extra
  extra=$(extra_flags "$name")
  local ckpt="${CKPT[$name]}"

  echo "[$name] starting on GPU $gpu (24hr @ 128K, bs=1, seed=$SEED)"
  CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --data "$DATA" \
    $args \
    --lr "$lr" --bf16 \
    --batch_size 1 \
    --chunk_size 131072 \
    --train_minutes "$TRAIN_MIN" \
    --output "$stage_dir" \
    --optimizer schedulefree \
    --seed "$SEED" \
    --save_every 500 --keep_checkpoints 5 --log_every 10 \
    $extra \
    --resume "$ckpt" \
    > "$logfile" 2>&1
  echo "[$name] complete (rc=$?)"
}

echo ""
echo "Starting 24hr comparison at $(date)" | tee "$OUTDIR/run.log"
START=$(date +%s)

PIDS=()
for entry in "${MODELS[@]}"; do
  IFS='|' read -r name gpu args lr <<< "$entry"
  run_model "$name" "$gpu" "$args" "$lr" &
  PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

END=$(date +%s)
echo "Finished 24hr comparison at $(date) (elapsed $((END - START))s)" | tee -a "$OUTDIR/run.log"

# Quick summary of final losses
echo "" | tee -a "$OUTDIR/run.log"
echo "==================== FINAL LOSSES ====================" | tee -a "$OUTDIR/run.log"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r name _ _ _ <<< "$entry"
  loss=$(grep "FINAL_LOSS_LAST100" "$OUTDIR/${name}.log" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+')
  printf "%-10s %s\n" "$name" "${loss:-FAIL}" | tee -a "$OUTDIR/run.log"
done
