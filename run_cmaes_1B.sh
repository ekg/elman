#!/bin/bash
# Parallel CMA-ES at 1.27B target params, ctx=512, 5min/eval.
# 5 models, 1 GPU each, pop=4 internal. Spare GPUs 5,7 for fast lane.

set -e

OUTDIR=/tmp/cmaes_1B
mkdir -p "$OUTDIR"
PILE=/home/erikg/elman/data/pile.txt

COMMON="--params 1270M --train_minutes 5 --popsize 4 \
        --chunk_size 512 --tokenizer p50k_base \
        --data $PILE --phase both \
        --lhs_samples 16 --min_generations 8"

launch() {
    local gpu=$1; local model=$2; shift 2
    local extra="$@"
    nohup python3 -u /home/erikg/elman/cmaes_search_v2.py \
        --model $model --gpus $gpu \
        --output $OUTDIR/$model \
        $COMMON $extra \
        > $OUTDIR/${model}.log 2>&1 &
    echo "GPU $gpu: $model -> $OUTDIR/${model}.log (pid $!)"
}

launch 0 e88        --fixed_n_state 32 --use_triton_e88
launch 1 fla-gdn
launch 2 mamba2
launch 3 transformer
launch 4 e94        --fixed_n_state 16

echo
echo "All 5 CMA-ES searches launched. Spare GPUs 5,7 available."
echo "Monitor: tail -f $OUTDIR/*.log"
