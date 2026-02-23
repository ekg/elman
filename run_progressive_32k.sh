#!/bin/bash
# Progressive Context Scaling: 512 → 32K
# Phase 1: Train best CMA-ES configs at 512 context, save checkpoints
# Phase 2: Resume from checkpoints at 32K with LR sweep
# Phase 3: Train from scratch at 32K as control
#
# 7 models: E88-n16, E88-n32, Mamba2, FLA-GDN, MinLSTM, MinGRU, E1

set -e

DATA="/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile.txt"
OUTDIR="benchmark_results/progressive_32k"
SEED=42
PHASE1_MINUTES=10
PHASE2_MINUTES=10
PHASE3_MINUTES=10

mkdir -p "$OUTDIR"

# ============================================================================
# MODEL CONFIGS (best from CMA-ES at 512 context)
# ============================================================================

declare -A MODEL_NAMES
declare -A MODEL_ARGS
declare -A MODEL_LR

MODEL_NAMES[0]="e88_n16"
MODEL_ARGS[0]="--level E88 --dim 1536 --depth 25 --n_heads 141 --n_state 16 --use_gate 1 --gate_activation silu"
MODEL_LR[0]="0.00079"

MODEL_NAMES[1]="e88_n32"
MODEL_ARGS[1]="--level E88 --dim 1920 --depth 17 --n_heads 83 --n_state 32 --use_gate 1 --gate_activation silu"
MODEL_LR[1]="0.00064"

MODEL_NAMES[2]="mamba2"
MODEL_ARGS[2]="--level mamba2 --dim 1792 --depth 25 --mamba_d_state 96 --mamba_expand 2"
MODEL_LR[2]="0.0003"

MODEL_NAMES[3]="fla_gdn"
MODEL_ARGS[3]="--level fla-gdn --dim 1920 --depth 17 --expansion 2 --n_heads 24"
MODEL_LR[3]="0.0003"

MODEL_NAMES[4]="minlstm"
MODEL_ARGS[4]="--level minlstm --dim 2944 --depth 10 --expansion 1"
MODEL_LR[4]="0.0007752"

MODEL_NAMES[5]="mingru"
MODEL_ARGS[5]="--level mingru --dim 3456 --depth 10 --expansion 1"
MODEL_LR[5]="0.0009875"

MODEL_NAMES[6]="e1"
MODEL_ARGS[6]="--level 1 --dim 2816 --depth 11 --expansion 1"
MODEL_LR[6]="0.0001656"

N_MODELS=7

# ============================================================================
# PHASE 1: Train at 512 context, save checkpoints
# ============================================================================
echo "=================================================================="
echo "PHASE 1: Training at 512 context (${PHASE1_MINUTES} min each)"
echo "=================================================================="

PHASE1_DIR="$OUTDIR/phase1_512"
mkdir -p "$PHASE1_DIR"

PHASE1_PIDS=()
for i in $(seq 0 $((N_MODELS - 1))); do
    name="${MODEL_NAMES[$i]}"
    args="${MODEL_ARGS[$i]}"
    lr="${MODEL_LR[$i]}"
    outdir="$PHASE1_DIR/$name"
    logfile="$PHASE1_DIR/${name}.log"

    echo "  Starting $name on GPU $i ..."

    CUDA_VISIBLE_DEVICES=$i python train.py \
        --data "$DATA" \
        $args \
        --lr "$lr" \
        --bf16 \
        --batch_size 16 \
        --chunk_size 512 \
        --train_minutes "$PHASE1_MINUTES" \
        --output "$outdir" \
        --optimizer schedulefree \
        --seed "$SEED" \
        --save_every 999999 \
        --log_every 50 \
        > "$logfile" 2>&1 &

    PHASE1_PIDS+=($!)
done

echo "  Waiting for Phase 1 (${#PHASE1_PIDS[@]} jobs)..."
PHASE1_FAILED=0
for pid in "${PHASE1_PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "  WARNING: PID $pid failed"
        PHASE1_FAILED=$((PHASE1_FAILED + 1))
    fi
done
echo "  Phase 1 complete. Failures: $PHASE1_FAILED"

# Find checkpoints
echo ""
echo "Phase 1 checkpoints:"
declare -A CHECKPOINTS
for i in $(seq 0 $((N_MODELS - 1))); do
    name="${MODEL_NAMES[$i]}"
    # Find the latest checkpoint
    ckpt=$(find "$PHASE1_DIR/$name" -name "checkpoint_*.pt" 2>/dev/null | sort | tail -1)
    if [ -n "$ckpt" ]; then
        loss=$(basename "$ckpt" | sed 's/.*loss_\([0-9.]*\)\.pt/\1/')
        echo "  $name: $ckpt (loss=$loss)"
        CHECKPOINTS[$i]="$ckpt"
    else
        echo "  $name: NO CHECKPOINT FOUND"
        CHECKPOINTS[$i]=""
    fi
done

# Extract Phase 1 losses
echo ""
echo "Phase 1 final losses (FINAL_LOSS_LAST100):"
for i in $(seq 0 $((N_MODELS - 1))); do
    name="${MODEL_NAMES[$i]}"
    logfile="$PHASE1_DIR/${name}.log"
    loss=$(grep "FINAL_LOSS_LAST100" "$logfile" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+')
    echo "  $name: $loss"
done

# ============================================================================
# PHASE 2: Resume at 32K with LR sweep
# ============================================================================
echo ""
echo "=================================================================="
echo "PHASE 2: Resume at 32K context (${PHASE2_MINUTES} min, LR sweep)"
echo "=================================================================="

PHASE2_DIR="$OUTDIR/phase2_32k_resume"
mkdir -p "$PHASE2_DIR"

# LR values to try (log-spaced around typical good values)
LR_VALUES=("0.0001" "0.0003" "0.0006")

# Run LR sweep: 7 models × 3 LRs = 21 runs, 8 GPUs → 3 rounds
gpu_idx=0
PHASE2_PIDS=()
PHASE2_JOBS=()

for i in $(seq 0 $((N_MODELS - 1))); do
    name="${MODEL_NAMES[$i]}"
    args="${MODEL_ARGS[$i]}"
    ckpt="${CHECKPOINTS[$i]}"

    if [ -z "$ckpt" ]; then
        echo "  Skipping $name (no checkpoint)"
        continue
    fi

    for lr in "${LR_VALUES[@]}"; do
        outdir="$PHASE2_DIR/${name}_lr${lr}"
        logfile="$PHASE2_DIR/${name}_lr${lr}.log"

        # Extra args for long-sequence models
        extra_args=""
        if [[ "$name" == e88_* ]]; then
            extra_args="--gradient_checkpointing --projection_chunk_size 512"
        fi

        echo "  Starting $name (lr=$lr) on GPU $gpu_idx ..."

        CUDA_VISIBLE_DEVICES=$gpu_idx python train.py \
            --data "$DATA" \
            $args \
            --lr "$lr" \
            --bf16 \
            --batch_size 4 \
            --chunk_size 32768 \
            --train_minutes "$PHASE2_MINUTES" \
            --output "$outdir" \
            --optimizer schedulefree \
            --seed "$SEED" \
            --save_every 999999 \
            --keep_checkpoints 0 \
            --log_every 10 \
            --resume "$ckpt" \
            $extra_args \
            > "$logfile" 2>&1 &

        PHASE2_PIDS+=($!)
        PHASE2_JOBS+=("${name}_lr${lr}")
        gpu_idx=$(( (gpu_idx + 1) % 8 ))

        # When all GPUs are busy, wait for current batch
        if [ $gpu_idx -eq 0 ] && [ ${#PHASE2_PIDS[@]} -gt 0 ]; then
            echo "  Waiting for batch of ${#PHASE2_PIDS[@]} jobs..."
            for pid in "${PHASE2_PIDS[@]}"; do
                wait "$pid" 2>/dev/null || true
            done
            PHASE2_PIDS=()
        fi
    done
done

# Wait for remaining jobs
if [ ${#PHASE2_PIDS[@]} -gt 0 ]; then
    echo "  Waiting for final batch of ${#PHASE2_PIDS[@]} jobs..."
    for pid in "${PHASE2_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
fi

echo ""
echo "Phase 2 results (32K resume):"
for i in $(seq 0 $((N_MODELS - 1))); do
    name="${MODEL_NAMES[$i]}"
    echo "  $name:"
    for lr in "${LR_VALUES[@]}"; do
        logfile="$PHASE2_DIR/${name}_lr${lr}.log"
        loss=$(grep "FINAL_LOSS_LAST100" "$logfile" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+')
        if [ -z "$loss" ]; then
            loss="FAILED"
        fi
        echo "    lr=$lr: $loss"
    done
done

# ============================================================================
# PHASE 3: Train from scratch at 32K (control)
# ============================================================================
echo ""
echo "=================================================================="
echo "PHASE 3: Train from scratch at 32K (${PHASE3_MINUTES} min, control)"
echo "=================================================================="

PHASE3_DIR="$OUTDIR/phase3_32k_scratch"
mkdir -p "$PHASE3_DIR"

# Use the best LR from Phase 2 for each model? For simplicity, use middle LR (0.0003)
# and also the model's original CMA-ES LR
PHASE3_PIDS=()
gpu_idx=0

for i in $(seq 0 $((N_MODELS - 1))); do
    name="${MODEL_NAMES[$i]}"
    args="${MODEL_ARGS[$i]}"
    lr="0.0003"  # Fixed LR for fair comparison

    outdir="$PHASE3_DIR/$name"
    logfile="$PHASE3_DIR/${name}.log"

    extra_args=""
    if [[ "$name" == e88_* ]]; then
        extra_args="--gradient_checkpointing --projection_chunk_size 512"
    fi

    echo "  Starting $name (scratch) on GPU $gpu_idx ..."

    CUDA_VISIBLE_DEVICES=$gpu_idx python train.py \
        --data "$DATA" \
        $args \
        --lr "$lr" \
        --bf16 \
        --batch_size 4 \
        --chunk_size 32768 \
        --train_minutes "$PHASE3_MINUTES" \
        --output "$outdir" \
        --optimizer schedulefree \
        --seed "$SEED" \
        --save_every 999999 \
        --keep_checkpoints 0 \
        --log_every 10 \
        $extra_args \
        > "$logfile" 2>&1 &

    PHASE3_PIDS+=($!)
    gpu_idx=$((gpu_idx + 1))
done

echo "  Waiting for Phase 3 (${#PHASE3_PIDS[@]} jobs)..."
for pid in "${PHASE3_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo ""
echo "Phase 3 results (32K from scratch):"
for i in $(seq 0 $((N_MODELS - 1))); do
    name="${MODEL_NAMES[$i]}"
    logfile="$PHASE3_DIR/${name}.log"
    loss=$(grep "FINAL_LOSS_LAST100" "$logfile" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+')
    if [ -z "$loss" ]; then
        loss="FAILED"
    fi
    echo "  $name: $loss"
done

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "=================================================================="
echo "FINAL SUMMARY: Progressive Context Scaling (512 → 32K)"
echo "=================================================================="
echo ""
printf "%-12s | %8s | %8s %8s %8s | %8s\n" "Model" "512" "32K@1e-4" "32K@3e-4" "32K@6e-4" "Scratch"
printf "%-12s-+-%8s-+-%8s-%8s-%8s-+-%8s\n" "------------" "--------" "--------" "--------" "--------" "--------"

for i in $(seq 0 $((N_MODELS - 1))); do
    name="${MODEL_NAMES[$i]}"

    # Phase 1 loss
    p1_loss=$(grep "FINAL_LOSS_LAST100" "$PHASE1_DIR/${name}.log" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+')
    [ -z "$p1_loss" ] && p1_loss="--"

    # Phase 2 losses (3 LRs)
    p2_losses=""
    for lr in "${LR_VALUES[@]}"; do
        l=$(grep "FINAL_LOSS_LAST100" "$PHASE2_DIR/${name}_lr${lr}.log" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+')
        [ -z "$l" ] && l="--"
        p2_losses="$p2_losses $(printf '%8s' "$l")"
    done

    # Phase 3 loss
    p3_loss=$(grep "FINAL_LOSS_LAST100" "$PHASE3_DIR/${name}.log" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+')
    [ -z "$p3_loss" ] && p3_loss="--"

    printf "%-12s | %8s |%s | %8s\n" "$name" "$p1_loss" "$p2_losses" "$p3_loss"
done

echo ""
echo "Done! Results in $OUTDIR"
