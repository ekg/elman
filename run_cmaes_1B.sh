#!/bin/bash
# Parallel CMA-ES at 1.27B target params, ctx=512, 5min/eval.
# Default: all 5 models, 1 GPU each, pop=4 internal. Use MODELS=e88 for
# the Triton-specific E88 rerun without rerunning unchanged baselines.

set -e

OUTDIR=${OUTDIR:-/tmp/cmaes_1B_triton_e88}
MODELS=${MODELS:-"e88 fla-gdn mamba2 transformer e94"}
mkdir -p "$OUTDIR"
PILE=/home/erikg/elman/data/pile.txt

COMMON="--params 1270M --train_minutes 5 --popsize 4 \
        --chunk_size 512 --tokenizer p50k_base \
        --data $PILE --phase both \
        --lhs_samples 16 --min_generations 8"

launch() {
    local gpu=$1; local model=$2; shift 2
    local extra="$@"
    setsid nohup python3 -u /home/erikg/elman/cmaes_search_v2.py \
        --model $model --gpus $gpu \
        --output $OUTDIR/$model \
        $COMMON $extra \
        > $OUTDIR/${model}.log 2>&1 &
    echo "GPU $gpu: $model -> $OUTDIR/${model}.log (pid $!)"
}

launched=0
for model in $MODELS; do
    case "$model" in
        e88)
            launch 0 e88 --fixed_n_state 32 --use_triton_e88
            ;;
        fla-gdn)
            launch 1 fla-gdn
            ;;
        mamba2)
            launch 2 mamba2
            ;;
        transformer)
            launch 3 transformer
            ;;
        e94)
            launch 4 e94 --fixed_n_state 16
            ;;
        *)
            echo "Unknown model '$model'. Valid: e88 fla-gdn mamba2 transformer e94" >&2
            exit 1
            ;;
    esac
    launched=$((launched + 1))
done

echo
echo "$launched CMA-ES search(es) launched. MODELS=$MODELS"
echo "Monitor: tail -f $OUTDIR/*.log"
