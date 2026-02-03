#!/bin/bash
# E74 Ablation Benchmark - Simplified version

cd /home/erikg/elman
OUTPUT_DIR="benchmark_results/e74_100m"
mkdir -p $OUTPUT_DIR

TRAIN_MINUTES=10
SEED=42

run_config() {
    local id=$1
    local gpu=$2
    local dim=$3
    local state=$4
    local proj=$5
    local nonlin=$6
    local gate=$7
    local update=$8
    local rank=$9

    echo "$(date '+%H:%M:%S') Config $id on GPU $gpu: $state/$proj/$nonlin/$update"

    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 python train_e74_ablation.py \
        --data data/pile.txt --dim $dim --depth 20 --n_state 96 \
        --state_type $state --proj_type $proj --nonlin_type $nonlin \
        --gate_type $gate --update_type $update --rank $rank --block_size 8 \
        --batch_size 32 --chunk_size 512 --steps 999999 --train_minutes $TRAIN_MINUTES \
        --lr 3e-4 --log_every 10 --output $OUTPUT_DIR/config$(printf "%02d" $id) \
        --bf16 --expansion 2.0 --seed $SEED \
        > $OUTPUT_DIR/config$(printf "%02d" $id).log 2>&1 &
}

echo "E74 Ablation Benchmark - 24 configs, 3 rounds of 8"
echo "Started: $(date)"
echo ""

# ROUND 1 (configs 1-8)
echo "=== ROUND 1 ==="
run_config 1 0 1408 full full tanh output delta 8
run_config 2 1 1408 full no_z tanh output delta 8
run_config 3 2 1408 full tied_kq tanh output delta 8
run_config 4 3 1536 full tied_kvq tanh output delta 8
run_config 5 4 1408 diagonal full tanh output delta 8
run_config 6 5 1408 diagonal no_z tanh output delta 8
run_config 7 6 1408 diagonal tied_kq tanh output delta 8
run_config 8 7 1536 diagonal tied_kvq tanh output delta 8
echo "Waiting for Round 1..."
wait
echo "Round 1 complete: $(date)"

# ROUND 2 (configs 9-16)
echo ""
echo "=== ROUND 2 ==="
run_config 9 0 1408 lowrank no_z tanh output delta 4
run_config 10 1 1408 lowrank no_z tanh output delta 8
run_config 11 2 1408 blockdiag no_z tanh output delta 8
run_config 12 3 1408 diagonal tied_kq linear output delta 8
run_config 13 4 1408 diagonal tied_kq rmsnorm output delta 8
run_config 14 5 1408 full no_z linear output delta 8
run_config 15 6 1408 diagonal no_z tanh retain delta 8
run_config 16 7 1408 diagonal no_z tanh state delta 8
echo "Waiting for Round 2..."
wait
echo "Round 2 complete: $(date)"

# ROUND 3 (configs 17-24)
echo ""
echo "=== ROUND 3 ==="
run_config 17 0 1408 diagonal tied_kq linear retain delta 8
run_config 18 1 1408 lowrank tied_kq tanh output delta 4
run_config 19 2 1408 lowrank tied_kq linear output delta 8
run_config 20 3 1536 diagonal tied_kvq linear output delta 8
run_config 21 4 1536 full tied_kvq tanh output simple 8
run_config 22 5 1536 diagonal tied_kvq tanh output simple 8
run_config 23 6 1536 full tied_kvq linear output simple 8
run_config 24 7 1536 diagonal tied_kvq linear output simple 8
echo "Waiting for Round 3..."
wait
echo "Round 3 complete: $(date)"

echo ""
echo "=== ALL COMPLETE ==="
echo "$(date)"
echo ""
echo "Results:"
for i in $(seq -w 1 24); do
    log="$OUTPUT_DIR/config${i}.log"
    if [ -f "$log" ]; then
        last=$(grep "^step" "$log" | tail -1)
        echo "Config $i: $last"
    fi
done
