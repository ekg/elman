#!/bin/bash
# 24-hour tokenized training at 512 context.
# Same 3 models, same recurrent hyperparameters, but GPT-2 BPE instead of bytes.
# Runs on GPUs 3, 4, 5 in parallel (GPUs 0-2 are in use by the byte-level run).

set -u

DATA="/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile.txt"
OUTDIR="benchmark_results/24hr_tokenized_512"
TRAIN_MIN=1440
SEED=42
CTX=512
BS=32                  # 512 ctx leaves loads of memory — bigger batch = better signal
TOKENIZER=gpt2

mkdir -p "$OUTDIR"

# Same recurrent hyperparameters as the byte-level 128K runs.
# Note: vocab_size jumps 256 -> 50257 so total params grow by ~83M each
# (embedding/lm_head is tied, so one matrix of 50257 × dim).
MODELS=(
  "fla-gdn|3|--level fla-gdn --dim 1664 --depth 13 --expansion 3 --n_heads 12|0.0004794217925780434"
  "e88_n32|4|--level E88 --dim 2432 --depth 10 --n_heads 112 --n_state 32 --expansion 1.0 --use_gate 1 --gate_activation silu|0.0006361350114482852"
  "e1h_n32|5|--level E1H --dim 1792 --depth 13 --n_heads 196 --n_state 32|0.0006030367837449147"
)

run_model() {
  local name=$1 gpu=$2 args=$3 lr=$4
  local outdir="$OUTDIR/$name"
  mkdir -p "$outdir"
  local logfile="$OUTDIR/${name}.log"

  echo "[$name] starting on GPU $gpu (24hr @ $CTX tokens, bs=$BS, tok=$TOKENIZER)"
  CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --data "$DATA" \
    --tokenizer "$TOKENIZER" \
    $args \
    --lr "$lr" --bf16 \
    --batch_size "$BS" \
    --chunk_size "$CTX" \
    --train_minutes "$TRAIN_MIN" \
    --output "$outdir" \
    --optimizer schedulefree \
    --seed "$SEED" \
    --save_every 500 --keep_checkpoints 5 --log_every 10 \
    > "$logfile" 2>&1
  echo "[$name] complete (rc=$?)"
}

echo "Starting tokenized 24hr comparison at $(date)" | tee "$OUTDIR/run.log"

PIDS=()
for entry in "${MODELS[@]}"; do
  IFS='|' read -r name gpu args lr <<< "$entry"
  run_model "$name" "$gpu" "$args" "$lr" &
  PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

echo "Finished at $(date)" | tee -a "$OUTDIR/run.log"

echo "" | tee -a "$OUTDIR/run.log"
echo "==================== FINAL LOSSES ====================" | tee -a "$OUTDIR/run.log"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r name _ _ _ <<< "$entry"
  loss=$(grep "FINAL_LOSS_LAST100" "$OUTDIR/${name}.log" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+')
  printf "%-10s %s\n" "$name" "${loss:-FAIL}" | tee -a "$OUTDIR/run.log"
done
