#!/bin/bash
# Final fix: bs=1 for OOMing models + Phase 3 scratch runs
set -e

DATA="/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile.txt"
OUTDIR="benchmark_results/progressive_32k"
SEED=42
MINUTES=10

PHASE2_DIR="$OUTDIR/phase2_32k_resume"
PHASE3_DIR="$OUTDIR/phase3_32k_scratch"
mkdir -p "$PHASE3_DIR"

LR_VALUES=("0.0001" "0.0003" "0.0006")

# ============================================================================
# Phase 2: bs=1 for remaining OOM models
# ============================================================================
echo "=================================================================="
echo "PHASE 2: Resume at 32K with batch_size=1"
echo "=================================================================="

declare -A CKPTS
CKPTS[mamba2]="benchmark_results/progressive_32k/phase1_512/mamba2/levelmamba2_100m_20260223_051811/checkpoint_step_001484_loss_1.4403.pt"
CKPTS[fla_gdn]="benchmark_results/progressive_32k/phase1_512/fla_gdn/levelfla-gdn_100m_20260223_051810/checkpoint_step_001698_loss_1.4349.pt"
CKPTS[minlstm]="benchmark_results/progressive_32k/phase1_512/minlstm/levelminlstm_100m_20260223_051811/checkpoint_step_002038_loss_1.7100.pt"
CKPTS[mingru]="benchmark_results/progressive_32k/phase1_512/mingru/levelmingru_100m_20260223_051811/checkpoint_step_002035_loss_1.7160.pt"

declare -A MODEL_ARGS
MODEL_ARGS[mamba2]="--level mamba2 --dim 1792 --depth 25 --mamba_d_state 96 --mamba_expand 2"
MODEL_ARGS[fla_gdn]="--level fla-gdn --dim 1920 --depth 17 --expansion 2 --n_heads 24"
MODEL_ARGS[minlstm]="--level minlstm --dim 2944 --depth 10 --expansion 1"
MODEL_ARGS[mingru]="--level mingru --dim 3456 --depth 10 --expansion 1"

MODELS=("mamba2" "fla_gdn" "minlstm" "mingru")

# Round 1: 4 models × 3 LRs = 12 jobs, fits in 2 rounds of 8
gpu_idx=0
PIDS=()

for model in "${MODELS[@]}"; do
    ckpt="${CKPTS[$model]}"
    args="${MODEL_ARGS[$model]}"

    for lr in "${LR_VALUES[@]}"; do
        outdir="$PHASE2_DIR/${model}_lr${lr}_bs1"
        logfile="$PHASE2_DIR/${model}_lr${lr}_bs1.log"

        echo "  Starting $model (lr=$lr, bs=1) on GPU $gpu_idx ..."

        CUDA_VISIBLE_DEVICES=$gpu_idx python train.py \
            --data "$DATA" \
            $args \
            --lr "$lr" \
            --bf16 \
            --batch_size 1 \
            --chunk_size 32768 \
            --train_minutes "$MINUTES" \
            --output "$outdir" \
            --optimizer schedulefree \
            --seed "$SEED" \
            --save_every 999999 \
            --keep_checkpoints 0 \
            --log_every 10 \
            --resume "$ckpt" \
            > "$logfile" 2>&1 &

        PIDS+=($!)
        gpu_idx=$(( (gpu_idx + 1) % 8 ))

        if [ $gpu_idx -eq 0 ] && [ ${#PIDS[@]} -gt 0 ]; then
            echo "  Waiting for batch of ${#PIDS[@]} jobs..."
            for pid in "${PIDS[@]}"; do
                wait "$pid" 2>/dev/null || true
            done
            PIDS=()
        fi
    done
done

if [ ${#PIDS[@]} -gt 0 ]; then
    echo "  Waiting for final batch of ${#PIDS[@]} jobs..."
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
fi

# ============================================================================
# Phase 3: All 7 models from scratch at 32K
# ============================================================================
echo ""
echo "=================================================================="
echo "PHASE 3: Train from scratch at 32K ($MINUTES min)"
echo "=================================================================="

declare -A ALL_ARGS
declare -A ALL_BS

ALL_ARGS[e88_n16]="--level E88 --dim 1536 --depth 25 --n_heads 141 --n_state 16 --use_gate 1 --gate_activation silu --gradient_checkpointing --projection_chunk_size 512"
ALL_BS[e88_n16]=4

ALL_ARGS[e88_n32]="--level E88 --dim 1920 --depth 17 --n_heads 83 --n_state 32 --use_gate 1 --gate_activation silu --gradient_checkpointing --projection_chunk_size 512"
ALL_BS[e88_n32]=4

ALL_ARGS[mamba2]="--level mamba2 --dim 1792 --depth 25 --mamba_d_state 96 --mamba_expand 2"
ALL_BS[mamba2]=1

ALL_ARGS[fla_gdn]="--level fla-gdn --dim 1920 --depth 17 --expansion 2 --n_heads 24"
ALL_BS[fla_gdn]=1

ALL_ARGS[minlstm]="--level minlstm --dim 2944 --depth 10 --expansion 1"
ALL_BS[minlstm]=1

ALL_ARGS[mingru]="--level mingru --dim 3456 --depth 10 --expansion 1"
ALL_BS[mingru]=1

ALL_ARGS[e1]="--level 1 --dim 2816 --depth 11 --expansion 1"
ALL_BS[e1]=2

ALL_MODELS=("e88_n16" "e88_n32" "mamba2" "fla_gdn" "minlstm" "mingru" "e1")

PIDS=()
gpu_idx=0
for model in "${ALL_MODELS[@]}"; do
    args="${ALL_ARGS[$model]}"
    bs="${ALL_BS[$model]}"
    outdir="$PHASE3_DIR/$model"
    logfile="$PHASE3_DIR/${model}.log"

    echo "  Starting $model (scratch, bs=$bs) on GPU $gpu_idx ..."

    CUDA_VISIBLE_DEVICES=$gpu_idx python train.py \
        --data "$DATA" \
        $args \
        --lr 0.0003 \
        --bf16 \
        --batch_size $bs \
        --chunk_size 32768 \
        --train_minutes "$MINUTES" \
        --output "$outdir" \
        --optimizer schedulefree \
        --seed "$SEED" \
        --save_every 999999 \
        --keep_checkpoints 0 \
        --log_every 10 \
        > "$logfile" 2>&1 &

    PIDS+=($!)
    gpu_idx=$((gpu_idx + 1))
done

echo "  Waiting for Phase 3 (${#PIDS[@]} jobs)..."
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

# ============================================================================
# FINAL COMBINED RESULTS
# ============================================================================
echo ""
echo "=================================================================="
echo "ALL RESULTS"
echo "=================================================================="

echo ""
echo "--- Phase 2 bs=1 (32K resume) ---"
for model in "${MODELS[@]}"; do
    echo "  $model:"
    for lr in "${LR_VALUES[@]}"; do
        logfile="$PHASE2_DIR/${model}_lr${lr}_bs1.log"
        loss=$(grep "FINAL_LOSS_LAST100" "$logfile" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+')
        [ -z "$loss" ] && loss="OOM/FAIL"
        echo "    lr=$lr: $loss"
    done
done

echo ""
echo "--- Phase 3 (32K scratch) ---"
for model in "${ALL_MODELS[@]}"; do
    logfile="$PHASE3_DIR/${model}.log"
    loss=$(grep "FINAL_LOSS_LAST100" "$logfile" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+')
    [ -z "$loss" ] && loss="OOM/FAIL"
    echo "  $model: $loss"
done
