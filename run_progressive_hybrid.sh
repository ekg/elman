#!/bin/bash
# Progressive training: 512 → 32K comparison
# E88-n16 HYBRID (Pararnn Triton) vs E88-n16 CUDA vs FLA-GDN
#
# Phase 1: train 10 min at T=512, save checkpoint
# Phase 2: resume 10 min at T=32K from checkpoint
#
# Runs all 3 models in parallel on 3 GPUs.

set -e

DATA="data/pile.txt"
OUTDIR="/tmp/progressive_hybrid"
SEED=42
P1_MIN=10
P2_MIN=10

mkdir -p "$OUTDIR"

run_progressive() {
    local gpu=$1
    local tag=$2
    local use_hybrid=$3
    local model_args=$4
    local lr_p1=$5
    local lr_p2=$6

    local env_prefix=""
    if [ "$use_hybrid" = "1" ]; then
        env_prefix="ELMAN_PARARNN_HYBRID=1"
    fi

    local p1_out="$OUTDIR/$tag/p1"
    local p2_out="$OUTDIR/$tag/p2"
    local log_p1="$OUTDIR/${tag}_p1.log"
    local log_p2="$OUTDIR/${tag}_p2.log"
    mkdir -p "$p1_out" "$p2_out"

    # Phase 1: T=512, B=16
    echo "[$(date +%H:%M:%S)] $tag GPU=$gpu Phase 1 (T=512, ${P1_MIN}min)"
    env $env_prefix CUDA_VISIBLE_DEVICES=$gpu python -u train.py \
        $model_args \
        --data "$DATA" --batch_size 16 --chunk_size 512 \
        --lr $lr_p1 --seed $SEED --bf16 --train_minutes $P1_MIN \
        --log_every 50 --output "$p1_out" \
        > "$log_p1" 2>&1

    # Find the latest checkpoint
    local ckpt=$(ls -t "$p1_out"/*/checkpoint_step_*.pt 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
        echo "[$(date +%H:%M:%S)] $tag: NO PHASE 1 CHECKPOINT FOUND"
        return
    fi
    echo "[$(date +%H:%M:%S)] $tag Phase 1 done, checkpoint: $(basename $ckpt)"

    # Phase 2: T=32K, B=1, grad_ckpt, resume from Phase 1
    echo "[$(date +%H:%M:%S)] $tag GPU=$gpu Phase 2 (T=32K, ${P2_MIN}min)"
    env $env_prefix CUDA_VISIBLE_DEVICES=$gpu python -u train.py \
        $model_args \
        --data "$DATA" --batch_size 1 --chunk_size 32768 \
        --lr $lr_p2 --seed $SEED --bf16 --train_minutes $P2_MIN \
        --log_every 2 --output "$p2_out" \
        --gradient_checkpointing \
        --resume "$ckpt" \
        > "$log_p2" 2>&1

    echo "[$(date +%H:%M:%S)] $tag Phase 2 done"
}

# E88-n16 best CMA-ES config at 512
E88_ARGS="--level E88 --dim 1536 --depth 25 --n_heads 141 --n_state 16 --use_gate 1 --gate_activation silu --expansion 1.0"
# FLA-GDN best CMA-ES config at 512
FLA_ARGS="--level fla-gdn --dim 1920 --depth 17 --expansion 2 --n_heads 24"

# Run 3 in parallel
run_progressive 0 "e88_n16_hybrid" 1 "$E88_ARGS" 7.9e-4 3e-4 &
PID1=$!
run_progressive 1 "e88_n16_cuda"   0 "$E88_ARGS" 7.9e-4 3e-4 &
PID2=$!
run_progressive 2 "fla_gdn"        0 "$FLA_ARGS" 3e-4   3e-4 &
PID3=$!

wait $PID1 $PID2 $PID3
echo ""
echo "=== Final Results ==="
for tag in e88_n16_hybrid e88_n16_cuda fla_gdn; do
    echo "--- $tag ---"
    echo "  Phase 1 (T=512):"
    grep -E "FINAL_LOSS|Training complete" "$OUTDIR/${tag}_p1.log" 2>/dev/null
    echo "  Phase 2 (T=32K resume):"
    grep -E "FINAL_LOSS|Training complete" "$OUTDIR/${tag}_p2.log" 2>/dev/null
done
