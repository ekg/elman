#!/bin/bash
# E5 Hidden Dimension Benchmark - Parallel on multiple GPUs
# Tests E5 with different hidden state dimensions

set -e

OUTPUT_BASE="output/e5_dim_benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_BASE"

DATA="data/pile.txt"
STEPS=1000
BATCH=32
CHUNK=512

echo "=============================================="
echo "E5 Hidden Dimension Benchmark"
echo "=============================================="
echo "Output: $OUTPUT_BASE"
echo "Steps: $STEPS, Batch: $BATCH, Chunk: $CHUNK"
echo ""

# Define experiments: GPU CONFIG_ARGS OUTPUT_NAME
# E5 configs: different hidden dims with adjusted rank/depth
# params_per_layer = dim * (6 * rank + 1) + 2 * dim (layernorm)
# For ~50M params with vocab=256
declare -a EXPERIMENTS=(
    # E5 with different hidden dimensions
    "0|--level 5 --dim 512 --rank 64 --depth 252|e5_d512_r64"   # Baseline
    "1|--level 5 --dim 768 --rank 48 --depth 168|e5_d768_r48"   # 1.5x hidden
    "2|--level 5 --dim 1024 --rank 32 --depth 144|e5_d1024_r32" # 2x hidden
    "3|--level 5 --dim 1024 --rank 48 --depth 112|e5_d1024_r48" # 2x hidden, more rank
    "4|--level 5 --dim 1536 --rank 24 --depth 82|e5_d1536_r24"  # 3x hidden

    # Baselines for comparison
    "5|--level 1|e1_baseline"                                    # E1: 21 layers
    "6|--level mamba2|mamba2"                                    # Mamba2: 18 layers
    "7|--level 6 --rank 64|e6_baseline"                          # E6: 755 layers
)

# Launch all experiments in parallel
PIDS=()
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r GPU ARGS NAME <<< "$exp"

    LOG="$OUTPUT_BASE/${NAME}.log"

    echo "Launching $NAME on GPU $GPU..."

    # Need to handle dim/rank/depth args for E5
    if [[ "$ARGS" == *"--dim"* ]]; then
        # E5 with explicit dim/rank/depth - use direct Python invocation
        CUDA_VISIBLE_DEVICES=$GPU python -c "
import sys
sys.path.insert(0, '.')
import torch
import time
import math
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM
from elman.data import DocumentStreamDataset

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Parse args from: $ARGS
args = '$ARGS'.split()
dim = int(args[args.index('--dim') + 1])
rank = int(args[args.index('--rank') + 1])
depth = int(args[args.index('--depth') + 1])
level = int(args[args.index('--level') + 1])

print(f'Creating E{level} model: dim={dim}, rank={rank}, depth={depth}')

model = LadderLM(
    vocab_size=256,
    dim=dim,
    depth=depth,
    level=level,
    rank=rank,
).cuda().bfloat16()

num_params = model.get_num_params()
print(f'Parameters: {num_params:,}')

dataset = DocumentStreamDataset('$DATA', chunk_size=$CHUNK+1, seed=42)

optimizer = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train()
optimizer.train()

start = time.time()
tokens = 0

for step in range(1, $STEPS+1):
    batch = torch.stack([dataset[0][0] for _ in range($BATCH)]).cuda()
    optimizer.zero_grad()
    loss = model(batch, return_loss=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    tokens += $BATCH * $CHUNK

    if step % 100 == 0 or step == 1:
        elapsed = time.time() - start
        ppl = math.exp(min(loss.item(), 20))
        print(f'Step {step:4d} | Loss {loss.item():.4f} | PPL {ppl:.1f} | {tokens/elapsed:.0f} tok/s')

elapsed = time.time() - start
print(f'Final: Loss {loss.item():.4f} | Tok/s {tokens/elapsed:.0f}')
" > "$LOG" 2>&1 &
    else
        # Use train_ladder.py for baselines
        CUDA_VISIBLE_DEVICES=$GPU python train_ladder.py \
            $ARGS \
            --params 50m \
            --data "$DATA" \
            --tokenizer byte \
            --batch_size $BATCH \
            --chunk_size $CHUNK \
            --max_steps $STEPS \
            --output "$OUTPUT_BASE/$NAME" \
            --log_interval 100 \
            --no-ddp \
            > "$LOG" 2>&1 &
    fi

    PIDS+=($!)
    sleep 0.5
done

echo ""
echo "All ${#EXPERIMENTS[@]} experiments launched on 8 GPUs!"
echo "Waiting for completion..."
echo ""

# Wait and collect results
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    IFS='|' read -r GPU ARGS NAME <<< "${EXPERIMENTS[$i]}"

    if wait $PID; then
        echo "✓ $NAME completed"
    else
        echo "✗ $NAME FAILED"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=============================================="
echo "Results Summary"
echo "=============================================="

# Parse results from logs
printf "%-18s %10s %8s %10s %8s\n" "Model" "Params" "Loss" "Tok/s" "Depth"
printf "%-18s %10s %8s %10s %8s\n" "-----" "------" "----" "-----" "-----"

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r GPU ARGS NAME <<< "$exp"
    LOG="$OUTPUT_BASE/${NAME}.log"

    if [ -f "$LOG" ]; then
        # Extract final metrics
        LOSS=$(grep -oP 'Loss [0-9.]+' "$LOG" | tail -1 | awk '{print $2}' || echo "N/A")
        TOKS=$(grep -oP '[0-9]+ tok/s' "$LOG" | tail -1 | awk '{print $1}' || echo "N/A")
        PARAMS=$(grep -oP 'Parameters: [0-9,]+' "$LOG" | head -1 | sed 's/Parameters: //' || echo "-")
        DEPTH=$(grep -oP 'depth=[0-9]+' "$LOG" | head -1 | sed 's/depth=//' || echo "-")

        printf "%-18s %10s %8s %10s %8s\n" "$NAME" "$PARAMS" "$LOSS" "$TOKS" "$DEPTH"
    else
        printf "%-18s %10s %8s %10s %8s\n" "$NAME" "ERROR" "-" "-" "-"
    fi
done

echo ""
echo "Logs saved to: $OUTPUT_BASE/"
echo "Failed: $FAILED/${#EXPERIMENTS[@]}"
