#!/bin/bash
# Long-run: drive 4 models at ctx=32K for 48h with hourly checkpoints.
# Resumes from /tmp/ctxscale_v2/stage3_ctx32k/ checkpoints (already trained through 4 progressive stages).
# Mamba2 skipped: OOMs at ctx=32K with CMA-ES winner config (depth=32 expand=3 d_state=160).

set -e
OUTDIR=${OUTDIR:-/tmp/long_train_ctx32k_triton_e88}
mkdir -p "$OUTDIR"
PILE=/home/erikg/elman/data/pile.txt
PRIOR=${PRIOR:-/tmp/ctxscale_v3_triton_e88/stage3_ctx32k}

# Common args: 48h, hourly checkpoints, keep last 72 hours of ckpts
COMMON="--bf16 --tokenizer p50k_base --train_minutes 2880 --chunk_size 32768 --batch_size 1 \
        --keep_checkpoints 72 --save_every 3000 --gradient_checkpointing --loss_chunk_size 4096 \
        --data $PILE"

# Model-specific args
E88_ARGS="--level E88 --dim 1408 --depth 14 --n_heads 386 --n_state 32 --use_gate 1 --gate_activation silu --use_triton 1 --lr 1.054e-3 --projection_chunk_size 4096"
E94_ARGS="--level E94 --dim 3328 --depth 28 --n_heads 30 --n_state 16 --use_gate 1 --gate_activation silu --use_permutation 1 --lr 2.445e-3"
FLA_ARGS="--level fla-gdn --dim 2688 --depth 21 --expansion 2 --n_heads 44 --lr 2.871e-3"
TX_ARGS="--level llama --dim 2560 --depth 19 --expansion 3 --n_heads 11 --lr 5.263e-4"

launch() {
    local gpu=$1; local name=$2; local extra="$3"; shift 3
    local margs="$@"
    local prior_ckpt=$(ls "$PRIOR/${name}_seed42_ckpt"/level*/latest.pt 2>/dev/null | head -1)
    if [ -z "$prior_ckpt" ]; then
        echo "MISSING resume ckpt for $name seed=42"
        return
    fi
    $extra CUDA_VISIBLE_DEVICES=$gpu nohup python3 -u /home/erikg/elman/train.py \
        $COMMON --resume "$prior_ckpt" \
        --output "$OUTDIR/${name}_seed42_ckpt" --seed 5042 \
        $margs > "$OUTDIR/${name}_seed42.log" 2>&1 &
    echo "GPU $gpu: $name pid=$! resume=$prior_ckpt"
}

# Use 4 GPUs (skip 6 user-occupied, 5/7 spare)
launch 0 e88         ""                                              $E88_ARGS
launch 1 fla-gdn     ""                                              $FLA_ARGS
launch 2 transformer ""                                              $TX_ARGS
launch 3 e94         "env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"   $E94_ARGS

echo
echo "All 4 long-run jobs launched. Targets: 48h ctx=32K, hourly ckpts."
echo "Monitor: tail -f $OUTDIR/*.log"
