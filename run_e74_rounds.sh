#!/bin/bash
# E74 Ablation Benchmark - All 24 configs in 3 rounds of 8

cd /home/erikg/elman
OUTPUT_DIR="benchmark_results/e74_100m"
mkdir -p $OUTPUT_DIR

EXPANSION=2.0
N_STATE=96
DEPTH=20
TRAIN_MINUTES=10
SEED=42

# Config dimensions (pre-computed)
declare -A DIMS
DIMS[1]=1408; DIMS[2]=1408; DIMS[3]=1408; DIMS[4]=1536
DIMS[5]=1408; DIMS[6]=1408; DIMS[7]=1408; DIMS[8]=1536
DIMS[9]=1408; DIMS[10]=1408; DIMS[11]=1408; DIMS[12]=1408
DIMS[13]=1408; DIMS[14]=1408; DIMS[15]=1408; DIMS[16]=1408
DIMS[17]=1408; DIMS[18]=1408; DIMS[19]=1408; DIMS[20]=1536
DIMS[21]=1536; DIMS[22]=1536; DIMS[23]=1536; DIMS[24]=1536

# Config details: state_type|proj_type|nonlin|gate|update
declare -A CONFIGS
CONFIGS[1]="full|full|tanh|output|delta"
CONFIGS[2]="full|no_z|tanh|output|delta"
CONFIGS[3]="full|tied_kq|tanh|output|delta"
CONFIGS[4]="full|tied_kvq|tanh|output|delta"
CONFIGS[5]="diagonal|full|tanh|output|delta"
CONFIGS[6]="diagonal|no_z|tanh|output|delta"
CONFIGS[7]="diagonal|tied_kq|tanh|output|delta"
CONFIGS[8]="diagonal|tied_kvq|tanh|output|delta"
CONFIGS[9]="lowrank|no_z|tanh|output|delta"
CONFIGS[10]="lowrank|no_z|tanh|output|delta"
CONFIGS[11]="blockdiag|no_z|tanh|output|delta"
CONFIGS[12]="diagonal|tied_kq|linear|output|delta"
CONFIGS[13]="diagonal|tied_kq|rmsnorm|output|delta"
CONFIGS[14]="full|no_z|linear|output|delta"
CONFIGS[15]="diagonal|no_z|tanh|retain|delta"
CONFIGS[16]="diagonal|no_z|tanh|state|delta"
CONFIGS[17]="diagonal|tied_kq|linear|retain|delta"
CONFIGS[18]="lowrank|tied_kq|tanh|output|delta"
CONFIGS[19]="lowrank|tied_kq|linear|output|delta"
CONFIGS[20]="diagonal|tied_kvq|linear|output|delta"
CONFIGS[21]="full|tied_kvq|tanh|output|simple"
CONFIGS[22]="diagonal|tied_kvq|tanh|output|simple"
CONFIGS[23]="full|tied_kvq|linear|output|simple"
CONFIGS[24]="diagonal|tied_kvq|linear|output|simple"

# Ranks for lowrank configs
declare -A RANKS
RANKS[9]=4; RANKS[10]=8; RANKS[18]=4; RANKS[19]=8

launch_config() {
    local config_id=$1
    local gpu=$2

    IFS='|' read -r state_type proj_type nonlin gate update <<< "${CONFIGS[$config_id]}"
    local dim=${DIMS[$config_id]}
    local rank=${RANKS[$config_id]:-8}

    echo "$(date '+%H:%M:%S') Launching config $config_id on GPU $gpu: $state_type/$proj_type/$nonlin"

    CUDA_VISIBLE_DEVICES=$gpu python train_e74_ablation.py \
        --data data/pile.txt \
        --dim $dim \
        --depth $DEPTH \
        --n_state $N_STATE \
        --state_type $state_type \
        --proj_type $proj_type \
        --nonlin_type $nonlin \
        --gate_type $gate \
        --update_type $update \
        --rank $rank \
        --block_size 8 \
        --batch_size 32 \
        --chunk_size 512 \
        --steps 999999 \
        --train_minutes $TRAIN_MINUTES \
        --lr 3e-4 \
        --log_every 10 \
        --output $OUTPUT_DIR/config$(printf "%02d" $config_id) \
        --bf16 \
        --expansion $EXPANSION \
        --seed $SEED \
        2>&1 | tee $OUTPUT_DIR/config$(printf "%02d" $config_id).log &
}

run_round() {
    local round=$1
    local start_config=$2
    local end_config=$3

    echo ""
    echo "=========================================="
    echo "ROUND $round: Configs $start_config-$end_config"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    local gpu=0
    for config_id in $(seq $start_config $end_config); do
        launch_config $config_id $gpu
        gpu=$((gpu + 1))
    done

    echo "Waiting for Round $round to complete (~10 minutes)..."
    wait
    echo "Round $round complete at $(date '+%H:%M:%S')"
}

# Run all 3 rounds
echo "E74 Ablation Benchmark - 24 configs, 3 rounds"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"

run_round 1 1 8
run_round 2 9 16
run_round 3 17 24

echo ""
echo "=========================================="
echo "ALL ROUNDS COMPLETE"
echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Extract results
echo ""
echo "RESULTS SUMMARY:"
echo "----------------"
for i in $(seq 1 24); do
    log="$OUTPUT_DIR/config$(printf "%02d" $i).log"
    if [ -f "$log" ]; then
        last_step=$(grep "^step" "$log" | tail -1)
        if [ -n "$last_step" ]; then
            echo "Config $i: $last_step"
        else
            echo "Config $i: (no training output)"
        fi
    else
        echo "Config $i: (no log file)"
    fi
done
