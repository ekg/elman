#!/usr/bin/env bash
set -euo pipefail

# Short production-scale probes for the M2RNN paper-shaped head geometry.
#
# The paper-shaped 1.27B config uses H=759 value/forget/gate/weight heads.
# q/k head counts must divide 759 because M2RNN repeats smaller head groups
# up to the maximum head count.
#
# Usage:
#   GPUS="0 6" STEPS=1500 LR=1e-4 ./run_m2rnn_qk_ablation.sh
#
# Optional:
#   NORMALIZE_QK=1 ./run_m2rnn_qk_ablation.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${DATA:-$ROOT/data/pile.txt}"
XMA_PATH="${XMA_PATH:-/tmp/m2rnn_xma}"
OUT_BASE="${OUT_BASE:-/tmp/m2rnn_qk_ablation/ctx2k_$(date -u +%Y%m%d_%H%M%S)}"
GPUS=(${GPUS:-0 1 2 3 4 5 6})
STEPS="${STEPS:-1500}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-2048}"
NORMALIZE_QK="${NORMALIZE_QK:-0}"
USE_RESIDUAL="${USE_RESIDUAL:-1}"
FREEZE_STATE_WEIGHT="${FREEZE_STATE_WEIGHT:-0}"
LAUNCH_DELAY="${LAUNCH_DELAY:-0}"

# Higher divisors like 253/759 are useful stress tests, but at fixed dim they
# add hundreds of millions to billions of q/k projection parameters. Keep the
# default sweep in the regime that is still close enough to the paper-shaped
# 1.27B run to diagnose conditioning.
QK_HEADS=(${QK_HEADS:-1 3 11 23 69})

mkdir -p "$OUT_BASE"
echo "output: $OUT_BASE"
echo "gpus: ${GPUS[*]}"
echo "qk_heads: ${QK_HEADS[*]}"
echo "steps=$STEPS lr=$LR batch_size=$BATCH_SIZE grad_accum=$GRAD_ACCUM normalize_qk=$NORMALIZE_QK use_residual=$USE_RESIDUAL freeze_state_weight=$FREEZE_STATE_WEIGHT launch_delay=$LAUNCH_DELAY"

running=0
gpu_i=0
pids=()

launch_one() {
  local qkh="$1"
  local gpu="${GPUS[$gpu_i]}"
  gpu_i=$(( (gpu_i + 1) % ${#GPUS[@]} ))

  local tag="qk${qkh}_norm${NORMALIZE_QK}_lr${LR}"
  local log="$OUT_BASE/${tag}.log"
  local out="$OUT_BASE/${tag}_ckpt"

  echo "[launch] gpu=$gpu qk_heads=$qkh log=$log"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" XMA_PATH="$XMA_PATH" PYTHONPATH="$XMA_PATH${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -u "$ROOT/train.py" \
    --bf16 \
    --tokenizer p50k_base \
    --params 1270M \
    --steps "$STEPS" \
    --chunk_size "$CHUNK_SIZE" \
    --optimizer schedulefree \
    --save_every 100000 \
    --keep_checkpoints 1 \
    --log_every 50 \
    --data "$DATA" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --output "$out" \
    --level m2rnn \
    --dim 3072 \
    --depth 10 \
    --n_heads 759 \
    --n_state 16 \
    --expansion 1.0 \
    --use_gate 1 \
    --use_conv 1 \
    --d_conv 4 \
    --m2rnn_paper_shape \
    --m2rnn_output_norm 1 \
    --m2rnn_state_grad_clip 1.0 \
    --m2rnn_q_heads "$qkh" \
    --m2rnn_k_heads "$qkh" \
    --m2rnn_normalize_qk "$NORMALIZE_QK" \
    --m2rnn_use_residual "$USE_RESIDUAL" \
    --m2rnn_freeze_state_weight "$FREEZE_STATE_WEIGHT" \
    --lr "$LR" \
    > "$log" 2>&1 &

  pids+=("$!")
  running=$((running + 1))
}

for qkh in "${QK_HEADS[@]}"; do
  launch_one "$qkh"
  if (( LAUNCH_DELAY > 0 )); then
    sleep "$LAUNCH_DELAY"
  fi
  if (( running >= ${#GPUS[@]} )); then
    wait -n
    running=$((running - 1))
  fi
done

wait

echo "=== final lines ==="
for log in "$OUT_BASE"/*.log; do
  echo "== $log =="
  grep -E "step|FINAL|Traceback|CUDA|OOM|Error" "$log" | tail -20 || true
done
