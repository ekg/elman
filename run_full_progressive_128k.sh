#!/bin/bash
# Full progressive training: 512 → 8K → 32K → 128K
# 3 models (FLA-GDN, E88 n32, E1H n32) using their best 128K CMA-ES configs.
# Each stage 30 min. Different seed per stage (distinct document streams).
# Checkpoints chain stage-to-stage. Models run in parallel on GPUs 0, 1, 2.

set -u

DATA="/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile.txt"
OUTDIR="benchmark_results/full_progressive_128k"
STAGE_MIN=30

mkdir -p "$OUTDIR"

# Model configs — best 128K CMA-ES winners. target_bs is what's in results.json;
# actual bs used at 128K was 1 (memory probe downgrade). bs_for_stage() matches that.
# format: name|gpu|base_args|lr|target_bs
MODELS=(
  "fla-gdn|0|--level fla-gdn --dim 1664 --depth 13 --expansion 3 --n_heads 12|0.0004794217925780434|7"
  "e88_n32|1|--level E88 --dim 2432 --depth 10 --n_heads 112 --n_state 32 --expansion 1.0 --use_gate 1 --gate_activation silu|0.0006361350114482852|8"
  "e1h_n32|2|--level E1H --dim 1792 --depth 13 --n_heads 196 --n_state 32|0.0006030367837449147|4"
)

STAGES=(512 8192 32768 131072)

# Stage-specific batch size:
# ctx=512/8K: use target bs (fits easily)
# ctx=32K: bs=2 (conservative; target bs OOMs)
# ctx=128K: bs=1 (matches what CMA-ES memory probing actually used)
bs_for_stage() {
  local target=$1 ctx=$2
  case $ctx in
    512|8192) echo "$target" ;;
    32768)    echo 2 ;;
    131072)   echo 1 ;;
  esac
}

# Return extra flags for a given (model, ctx) — matches cmaes_search_v2.get_phase_settings
stage_flags() {
  local model=$1 ctx=$2
  local flags=""
  local base="${model%_*}"  # strip _n32 suffix -> e88 / e1h / fla-gdn
  # fla-gdn doesn't get stripped; handle it
  [[ "$model" == fla-gdn ]] && base="fla-gdn"

  # Using 128K-optimized configs at all stages — enable gc for all ctx >= 8K
  # (the original get_phase_settings assumed stage-optimized configs; ours are bigger)
  if [[ $ctx -ge 8192 ]]; then
    flags="--gradient_checkpointing"
  fi
  if [[ $ctx -ge 32768 && "$base" == "e88" ]]; then
    flags="$flags --projection_chunk_size 512"
  fi
  echo "$flags"
}

run_model() {
  local name=$1 gpu=$2 args=$3 lr=$4 target_bs=$5
  local prev_ckpt=""
  local mdir="$OUTDIR/$name"
  mkdir -p "$mdir"

  for idx in 0 1 2 3; do
    local ctx=${STAGES[$idx]}
    local seed=$((42 + idx))
    local stage_dir="$mdir/stage${idx}_ctx${ctx}"
    local logfile="$mdir/stage${idx}_ctx${ctx}.log"
    local extra
    extra=$(stage_flags "$name" "$ctx")
    local bs
    bs=$(bs_for_stage "$target_bs" "$ctx")
    local resume=""
    [ -n "$prev_ckpt" ] && resume="--resume $prev_ckpt"

    # Skip if this stage already produced a checkpoint (resume after interruption)
    local existing
    existing=$(find "$stage_dir" -name 'checkpoint_step_*.pt' 2>/dev/null | sort | tail -1)
    if [ -n "$existing" ]; then
      echo "[$name] stage $idx ctx=$ctx SKIP (checkpoint exists: $(basename "$existing"))" | tee -a "$mdir/progress.log"
      prev_ckpt="$existing"
      continue
    fi

    # Clear any partial stage dir from a failed prior run
    rm -rf "$stage_dir"
    mkdir -p "$stage_dir"

    echo "[$name] stage $idx ctx=$ctx seed=$seed bs=$bs ${extra:+flags=\"$extra\"}" | tee -a "$mdir/progress.log"

    CUDA_VISIBLE_DEVICES=$gpu python train.py \
      --data "$DATA" \
      $args \
      --lr "$lr" --bf16 \
      --batch_size "$bs" \
      --chunk_size "$ctx" \
      --train_minutes "$STAGE_MIN" \
      --output "$stage_dir" \
      --optimizer schedulefree \
      --seed "$seed" \
      --save_every 999999 \
      --keep_checkpoints 1 \
      --log_every 10 \
      $extra \
      $resume \
      > "$logfile" 2>&1

    local rc=$?
    if [ $rc -ne 0 ]; then
      echo "[$name] stage $idx FAILED (rc=$rc) — aborting" | tee -a "$mdir/progress.log"
      return 1
    fi

    # Pick up the final checkpoint written by train.py at end-of-training
    prev_ckpt=$(find "$stage_dir" -name 'checkpoint_step_*.pt' 2>/dev/null | sort | tail -1)
    local loss
    loss=$(grep "FINAL_LOSS_LAST100" "$logfile" | tail -1 | grep -oP '[0-9]+\.[0-9]+')
    echo "[$name] stage $idx done: loss=${loss:-?} ckpt=$(basename "$prev_ckpt")" | tee -a "$mdir/progress.log"

    if [ -z "$prev_ckpt" ]; then
      echo "[$name] no checkpoint produced at stage $idx — aborting" | tee -a "$mdir/progress.log"
      return 1
    fi
  done

  echo "[$name] ALL STAGES COMPLETE" | tee -a "$mdir/progress.log"
}

echo "Starting $(date)" | tee "$OUTDIR/run.log"

PIDS=()
for entry in "${MODELS[@]}"; do
  IFS='|' read -r name gpu args lr bs <<< "$entry"
  run_model "$name" "$gpu" "$args" "$lr" "$bs" &
  PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

echo "Finished $(date)" | tee -a "$OUTDIR/run.log"

# Final summary
echo "" | tee -a "$OUTDIR/run.log"
echo "==================== SUMMARY ====================" | tee -a "$OUTDIR/run.log"
printf "%-10s | %8s %8s %8s %8s\n" "model" "512" "8K" "32K" "128K" | tee -a "$OUTDIR/run.log"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r name _ _ _ _ <<< "$entry"
  losses=""
  for idx in 0 1 2 3; do
    ctx=${STAGES[$idx]}
    l=$(grep "FINAL_LOSS_LAST100" "$OUTDIR/$name/stage${idx}_ctx${ctx}.log" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+')
    losses="$losses $(printf '%8s' "${l:-FAIL}")"
  done
  printf "%-10s |%s\n" "$name" "$losses" | tee -a "$OUTDIR/run.log"
done
