#!/bin/bash
# Context-scaling sweep on stage-0 CMA-ES-winner checkpoints.
# 5 models × 3 seeds × 4 stages = 60 jobs across ctx=2K → 8K → 32K → 128K.
# Resume from /tmp/cmaes_winners_stage0/<model>_seed<s>_ckpt for stage 1.
# Each subsequent stage resumes from prior stage output.

set -e
OUTDIR=/tmp/ctxscale_v2
mkdir -p "$OUTDIR"
PILE=/home/erikg/elman/data/pile.txt
PRIOR=/tmp/cmaes_winners_stage0

# Model architectures (must match stage-0). Append --batch_size dynamically per stage.
E88_ARGS="--level E88 --dim 1408 --depth 14 --n_heads 386 --n_state 32 --use_gate 1 --gate_activation silu --use_triton 1 --lr 1.054e-3"
E94_ARGS="--level E94 --dim 3328 --depth 28 --n_heads 30 --n_state 16 --use_gate 1 --gate_activation silu --use_permutation 1 --lr 2.445e-3"
FLA_ARGS="--level fla-gdn --dim 2688 --depth 21 --expansion 2 --n_heads 44 --lr 2.871e-3"
MAM_ARGS="--level mamba2 --dim 2048 --depth 32 --expansion 3 --mamba_d_state 160 --lr 3.502e-4"
TX_ARGS="--level llama --dim 2560 --depth 19 --expansion 3 --n_heads 11 --lr 5.263e-4"

GPUS=(0 1 2 3 4 5 7)

# Per-(model, stage) bs heuristics (safe values; will probe later if needed).
# stage1 ctx=2K: bs reduced from stage 0 bs by ~4x where possible
# stage2 ctx=8K: bs=1 for memory safety
# stage3 ctx=32K: bs=1
# stage4 ctx=128K: bs=1 (transformer may OOM, accepted)

stage_run() {
    local stage=$1; local ctx=$2; local prior_dir=$3; local seed_off=$4
    # Use grad_checkpointing for ALL models at long ctx (≥8K) for symmetric memory budget.
    # Transformer needs it for attention mem; other models pay same throughput cost for fairness.
    local grad_ckpt_flag=""
    if [ "$ctx" -ge 8192 ]; then grad_ckpt_flag="--gradient_checkpointing"; fi
    local -A model_bs=(
        ["e88|2048"]=4    ["e88|8192"]=1    ["e88|32768"]=1   ["e88|65536"]=1
        ["e94|2048"]=2    ["e94|8192"]=1    ["e94|32768"]=1   ["e94|65536"]=1
        ["fla-gdn|2048"]=2 ["fla-gdn|8192"]=1 ["fla-gdn|32768"]=1 ["fla-gdn|65536"]=1
        ["mamba2|2048"]=2 ["mamba2|8192"]=1 ["mamba2|32768"]=1
        ["transformer|2048"]=1 ["transformer|8192"]=1 ["transformer|32768"]=1
    )

    echo "=== Stage $stage: ctx=$ctx seed_off=$seed_off (prior=$prior_dir) ==="
    mkdir -p "$OUTDIR/$stage"
    local jobs=()
    for entry in "e88|$E88_ARGS" "e94|$E94_ARGS" "fla-gdn|$FLA_ARGS" "mamba2|$MAM_ARGS" "transformer|$TX_ARGS"; do
        local mname=$(echo "$entry" | cut -d'|' -f1)
        local margs=$(echo "$entry" | cut -d'|' -f2-)
        local bs=${model_bs["$mname|$ctx"]:-}
        # Skip models that don't fit at this ctx (no bs entry)
        if [ -z "$bs" ]; then
            echo "  SKIP $mname at ctx=$ctx (memory-infeasible)"
            continue
        fi
        for seed in 42 123 456; do
            local resume_ckpt=$(ls -t "$prior_dir/${mname}_seed${seed}"*/level*/latest.pt 2>/dev/null | head -1)
            if [ -z "$resume_ckpt" ]; then
                echo "  MISSING ckpt for $mname seed=$seed; SKIP"
                continue
            fi
            jobs+=("${mname}|$seed|$bs|$margs|$resume_ckpt")
        done
    done
    echo "  ${#jobs[@]} jobs queued"

    local pids=()
    local i=0
    for job in "${jobs[@]}"; do
        local mname=$(echo "$job" | cut -d'|' -f1)
        local seed=$(echo "$job" | cut -d'|' -f2)
        local bs=$(echo "$job" | cut -d'|' -f3)
        local margs=$(echo "$job" | cut -d'|' -f4)
        local jckpt=$(echo "$job" | cut -d'|' -f5)
        local gpu=${GPUS[$((i % ${#GPUS[@]}))]}
        local logf="$OUTDIR/$stage/${mname}_seed${seed}.log"
        local outd="$OUTDIR/$stage/${mname}_seed${seed}_ckpt"

        # E88 only: per-chunk projection recomputation (saves ~5GB/layer at long ctx, throughput-positive vs grad_ckpt)
        local proj_flag=""
        if [ "$mname" = "e88" ] && [ "$ctx" -ge 8192 ]; then
            proj_flag="--projection_chunk_size 4096"
        fi
        # Loss chunking when ctx is long enough that lm_head*V*T tensor matters
        local loss_chunk_flag=""
        if [ "$ctx" -ge 8192 ]; then
            loss_chunk_flag="--loss_chunk_size 4096"
        fi
        # E94 at ctx>=64K needs expandable_segments to fit
        local alloc_env=""
        if [ "$mname" = "e94" ] && [ "$ctx" -ge 65536 ]; then
            alloc_env="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            loss_chunk_flag="--loss_chunk_size 2048"
        fi

        if [ $i -ge ${#GPUS[@]} ]; then
            local prior_idx=$((i - ${#GPUS[@]}))
            wait ${pids[$prior_idx]} 2>/dev/null || true
        fi

        env $alloc_env CUDA_VISIBLE_DEVICES=$gpu nohup python3 -u /home/erikg/elman/train.py \
            --bf16 --tokenizer p50k_base --train_minutes 60 \
            --chunk_size $ctx --batch_size $bs \
            --keep_checkpoints 1 --save_every 5000 \
            $grad_ckpt_flag $proj_flag $loss_chunk_flag \
            --data "$PILE" \
            --resume "$jckpt" \
            --output "$outd" \
            $margs --seed $((seed + seed_off)) \
            > "$logf" 2>&1 &
        pids+=($!)
        echo "  GPU $gpu: $mname seed=$seed bs=$bs flags=[$grad_ckpt_flag $proj_flag] pid=${pids[-1]}"
        i=$((i + 1))
    done

    for pid in "${pids[@]}"; do
        wait $pid 2>/dev/null || true
    done
    echo "=== Stage $stage complete ==="
}

# Skip already-completed stages (idempotent). Expected per stage:
#   stage1/2/3 = 15 jobs (5 models × 3 seeds), stage4 = 9 (3 models × 3 seeds)
stage_done() {
    local stage=$1
    local expected=${2:-15}
    [ -d "$OUTDIR/$stage" ] && \
        [ "$(grep -l 'FINAL_LOSS_LAST100' $OUTDIR/$stage/*.log 2>/dev/null | wc -l)" -ge $expected ]
}

if stage_done stage1_ctx2k 15; then echo "Skip stage1 (done)"; else stage_run stage1_ctx2k 2048 "$PRIOR" 1000; fi
if stage_done stage2_ctx8k 15; then echo "Skip stage2 (done)"; else stage_run stage2_ctx8k 8192 "$OUTDIR/stage1_ctx2k" 2000; fi
# Stage 3: mamba2 OOMs at ctx=32K with 1.27B (skipped), so expect 12 (e88×3 + e94×3 + fla-gdn×3 + transformer×3)
if stage_done stage3_ctx32k 12; then echo "Skip stage3 (done)"; else stage_run stage3_ctx32k 32768 "$OUTDIR/stage2_ctx8k" 3000; fi
if stage_done stage4_ctx64k 9; then echo "Skip stage4 (done)"; else stage_run stage4_ctx64k 65536 "$OUTDIR/stage3_ctx32k" 4000; fi
echo "Stage 4 attempts ctx=64K. Mamba2 + Transformer skipped (memory). E88 + FLA-GDN + E94 only."

echo "=== ALL CONTEXT SCALING STAGES COMPLETE ==="
