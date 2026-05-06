#!/bin/bash
# Stage 0: train CMA-ES 1.27B winners at ctx=512 for 60min, 3 seeds each.
# 5 models × 3 seeds = 15 jobs. Schedule on 7 GPUs (3 batches).

set -e
OUTDIR=/tmp/cmaes_winners_stage0
mkdir -p "$OUTDIR"
PILE=/home/erikg/elman/data/pile.txt

COMMON="--bf16 --tokenizer p50k_base --train_minutes 60 --chunk_size 512 --data $PILE"

# CMA-ES 1.27B winners (from /tmp/cmaes_1B/*.log, top by 5-min AvgLoss)
E88_ARGS="--level E88 --dim 1408 --depth 14 --n_heads 386 --n_state 32 --use_gate 1 --gate_activation silu --use_triton 1 --batch_size 17 --lr 1.054e-3"
E94_ARGS="--level E94 --dim 3328 --depth 28 --n_heads 30 --n_state 16 --use_gate 1 --gate_activation silu --use_permutation 1 --batch_size 7 --lr 2.445e-3"
FLA_ARGS="--level fla-gdn --dim 2688 --depth 21 --expansion 2 --n_heads 44 --batch_size 4 --lr 2.871e-3"
MAM_ARGS="--level mamba2 --dim 2048 --depth 32 --expansion 3 --mamba_d_state 160 --batch_size 3 --lr 3.502e-4"
TX_ARGS="--level llama --dim 2560 --depth 19 --expansion 3 --n_heads 11 --batch_size 2 --lr 5.263e-4"

GPUS=(0 1 2 3 4 5 7)

launch() {
    local gpu=$1; local name=$2; local seed=$3; shift 3
    local margs="$@"
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 -u /home/erikg/elman/train.py \
        $COMMON --seed $seed \
        --output "$OUTDIR/${name}_seed${seed}_ckpt" \
        $margs \
        > "$OUTDIR/${name}_seed${seed}.log" 2>&1 &
    echo "GPU $gpu: $name seed=$seed pid=$!"
}

# Build list of (name, args) pairs
JOBS=(
  "e88|$E88_ARGS"
  "e94|$E94_ARGS"
  "fla-gdn|$FLA_ARGS"
  "mamba2|$MAM_ARGS"
  "transformer|$TX_ARGS"
)
SEEDS=(42 123 456)

# 15 jobs, 7 GPUs → 3 batches
i=0
pids=()
for job in "${JOBS[@]}"; do
    name=$(echo "$job" | cut -d'|' -f1)
    margs=$(echo "$job" | cut -d'|' -f2-)
    for seed in "${SEEDS[@]}"; do
        gpu=${GPUS[$((i % ${#GPUS[@]}))]}
        # If we've launched 7 already, wait for one to finish
        if [ $i -ge ${#GPUS[@]} ]; then
            prior_idx=$((i - ${#GPUS[@]}))
            wait ${pids[$prior_idx]} 2>/dev/null || true
        fi
        launch $gpu $name $seed $margs
        pids+=($!)
        i=$((i + 1))
    done
done

echo "All 15 stage-0 jobs scheduled. Wait for them..."
for pid in "${pids[@]}"; do wait $pid 2>/dev/null || true; done
echo "=== Stage 0 complete ==="
