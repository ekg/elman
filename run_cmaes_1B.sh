#!/bin/bash
# Parallel CMA-ES at 1.27B target params, ctx=512, 5min/eval.
# Default: all 5 models, 1 GPU each. Use MODELS=e88 for the Triton-specific
# E88 rerun; that mode defaults to GPUs 0-6 and popsize=7.

set -e

OUTDIR=${OUTDIR:-/tmp/cmaes_1B_triton_e88}
MODELS=${MODELS:-"e88 fla-gdn mamba2 transformer e94"}
if [ "$MODELS" = "e88" ]; then
    E88_GPUS=${E88_GPUS:-"0,1,2,3,4,5,6"}
    POPSIZE=${POPSIZE:-7}
elif [ "$MODELS" = "m2rnn" ] || [ "$MODELS" = "m2rnn-paper" ]; then
    M2RNN_GPUS=${M2RNN_GPUS:-"0,4,5,6"}
    POPSIZE=${POPSIZE:-4}
else
    E88_GPUS=${E88_GPUS:-"0"}
    M2RNN_GPUS=${M2RNN_GPUS:-"5"}
    POPSIZE=${POPSIZE:-4}
fi
mkdir -p "$OUTDIR"
PILE=/home/erikg/elman/data/pile.txt

if [[ " $MODELS " == *" m2rnn "* || " $MODELS " == *" m2rnn-paper "* ]]; then
    XMA_PATH=${XMA_PATH:-/tmp/m2rnn_xma}
    if [ ! -d "$XMA_PATH/xma" ]; then
        echo "M2RNN requires XMA. Set XMA_PATH to accelerated-model-architectures checkout." >&2
        exit 1
    fi
    export XMA_PATH
    export PYTHONPATH="$XMA_PATH:${PYTHONPATH:-}"
fi

COMMON="--params 1270M --train_minutes 5 --popsize $POPSIZE \
        --chunk_size 512 --tokenizer p50k_base \
        --data $PILE --phase both \
        --lhs_samples 16 --min_generations 8"

launch() {
    local gpus=$1; local model=$2; shift 2
    local extra="$@"
    setsid nohup python3 -u /home/erikg/elman/cmaes_search_v2.py \
        --model $model --gpus $gpus \
        --output $OUTDIR/$model \
        $COMMON $extra \
        > $OUTDIR/${model}.log 2>&1 &
    echo "GPUs $gpus: $model -> $OUTDIR/${model}.log (pid $!)"
}

launched=0
for model in $MODELS; do
    case "$model" in
        e88)
            launch "$E88_GPUS" e88 --fixed_n_state 32 --use_triton_e88
            ;;
        fla-gdn)
            launch 1 fla-gdn
            ;;
        mamba2)
            launch 2 mamba2
            ;;
        m2rnn)
            launch "$M2RNN_GPUS" m2rnn --fixed_n_state 16
            ;;
        m2rnn-paper)
            launch "$M2RNN_GPUS" m2rnn-paper --fixed_n_state 16
            ;;
        transformer)
            launch 3 transformer
            ;;
        e94)
            launch 4 e94 --fixed_n_state 16
            ;;
        *)
            echo "Unknown model '$model'. Valid: e88 fla-gdn mamba2 m2rnn m2rnn-paper transformer e94" >&2
            exit 1
            ;;
    esac
    launched=$((launched + 1))
done

echo
echo "$launched CMA-ES search(es) launched. MODELS=$MODELS POPSIZE=$POPSIZE"
echo "Monitor: tail -f $OUTDIR/*.log"
