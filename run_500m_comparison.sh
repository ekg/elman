#!/bin/bash
# E88 500M 30-minute comparison study

OUTDIR="benchmark_results/e88_500m_30min/20260123_123714"
DATA="data/pile.txt"
MINUTES=30

echo "=== E88 500M 30-minute Comparison Study ==="
echo "Output: $OUTDIR"
echo ""

# Wait for any running training to complete
while pgrep -f "train.py.*train_minutes" > /dev/null; do
    echo "Waiting for current training to complete..."
    sleep 60
done

# 1. E88 Winner: d20_h44_n48 (already started, wait for completion)
echo "[1/5] E88 Winner (d20_h44_n48) - already running"
while pgrep -f "e88_winner" > /dev/null; do
    sleep 30
done
echo "  Done."

# 2. E88 #2: d24_h32_n48 (low ratio)
echo "[2/5] E88 #2 (d24_h32_n48, ratio=0.46)"
python train.py \
    --data "$DATA" \
    --level E88 \
    --dim 3328 --depth 24 --n_heads 32 --n_state 48 \
    --expansion 1.0 --use_gate 0 --bf16 \
    --batch_size 16 --chunk_size 512 \
    --lr 3e-4 --optimizer schedulefree \
    --train_minutes $MINUTES \
    --output "$OUTDIR/e88_second" \
    > "$OUTDIR/e88_second.log" 2>&1
echo "  Done."

# 3. E88 #3: d20_h52_n40
echo "[3/5] E88 #3 (d20_h52_n40)"
python train.py \
    --data "$DATA" \
    --level E88 \
    --dim 2944 --depth 20 --n_heads 52 --n_state 40 \
    --expansion 1.0 --use_gate 0 --bf16 \
    --batch_size 16 --chunk_size 512 \
    --lr 3e-4 --optimizer schedulefree \
    --train_minutes $MINUTES \
    --output "$OUTDIR/e88_third" \
    > "$OUTDIR/e88_third.log" 2>&1
echo "  Done."

# 4. FLA-GDN 500M
echo "[4/5] FLA-GDN 500M (d=1664, depth=24)"
python train.py \
    --data "$DATA" \
    --level fla-gdn \
    --dim 1664 --depth 24 \
    --expansion 2.0 --bf16 \
    --batch_size 16 --chunk_size 512 \
    --lr 3e-4 --optimizer schedulefree \
    --train_minutes $MINUTES \
    --output "$OUTDIR/fla_gdn" \
    > "$OUTDIR/fla_gdn.log" 2>&1
echo "  Done."

# 5. Mamba2 500M
echo "[5/5] Mamba2 500M (d=1792, depth=24)"
python train.py \
    --data "$DATA" \
    --level mamba2 \
    --dim 1792 --depth 24 \
    --bf16 \
    --batch_size 16 --chunk_size 512 \
    --lr 3e-4 --optimizer schedulefree \
    --train_minutes $MINUTES \
    --output "$OUTDIR/mamba2" \
    > "$OUTDIR/mamba2.log" 2>&1
echo "  Done."

echo ""
echo "=== All experiments complete ==="
echo "Results in: $OUTDIR"
