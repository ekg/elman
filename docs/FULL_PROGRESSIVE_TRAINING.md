# Full Progressive Context Training: 512 → 8K → 32K → 128K

## Motivation

The current multi-context CMA-ES campaign uses a two-stage progressive approach:
1. Train at 512 context for 10 min (warm start)
2. Train at target context (8K/32K/128K) for 20 min

This means a 32K search jumps directly from 512 → 32K, a 64x increase in context length. At 8K, E88 beat Mamba2 (1.1206 vs 1.1385). But at 32K, Mamba2 pulled ahead (1.2674 LHS vs E88's 1.3495 final). The jump from 512 to 32K may be too large for sequential RNNs — they need more gradient steps to adapt their hidden state dynamics to longer dependencies.

**Hypothesis**: Incremental context scaling (512 → 8K → 32K → 128K) will disproportionately benefit sequential nonlinear RNNs (E88, E1H) compared to parallel-scan models (Mamba2, FLA-GDN), because:

1. Sequential RNNs get fewer gradient updates per minute at long context (no parallel scan)
2. Nonlinear state dynamics need gradual adaptation — abrupt context jumps waste the warm start
3. Parallel models can immediately leverage long context via their scan, so they don't need the intermediate steps as much

## Proposed Protocol

### Phase Structure

```
Stage 1: 512 context,  10 min  → checkpoint
Stage 2: 8K context,   10 min  → checkpoint (resume from stage 1)
Stage 3: 32K context,  10 min  → checkpoint (resume from stage 2)
Stage 4: 128K context, 10 min  → final loss  (resume from stage 3)
```

Total: 40 min per eval (vs current 30 min for two-stage).

### Comparison Conditions

For each model, run three conditions:
- **Full progressive**: 512 → 8K → 32K → 128K (10 min each)
- **Two-stage**: 512 → 128K (10 min + 30 min, matched total 40 min)
- **Direct**: 128K from scratch (40 min, control)

### CMA-ES Integration

Two options:

**Option A**: Search at each stage independently (current approach but chained)
- Run CMA-ES at 512, take best config
- Use that config, run CMA-ES at 8K (varying only LR + batch_size?)
- Chain forward to 32K, 128K
- Pro: finds optimal config per stage
- Con: expensive, config changes between stages

**Option B**: Search at final target, but with full progressive warm-up
- CMA-ES optimizes for 128K loss
- Each eval does the full 512→8K→32K→128K pipeline internally
- Pro: optimizes for actual target
- Con: 40 min per eval instead of 30 min

Option B is cleaner — it finds configs that work best for the full progressive pipeline.

## Evidence from Current Campaign

### Gap Analysis (E88 n32 vs FLA-GDN, relative)

| Context | Gap |
|---------|-----|
| 512     | 12.0% |
| 8K      | 9.0% |
| 32K     | 9.0% |

The gap narrowed from 512 to 8K but stalled from 8K to 32K. Full progressive training might allow the gap to continue narrowing at 32K and 128K.

### E88 vs Mamba2 Reversal

| Context | E88 n16 | Mamba2 | Winner |
|---------|---------|--------|--------|
| 8K      | 1.1206  | 1.1385 | E88    |
| 32K     | 1.3570  | ~1.26* | Mamba2 |

*Mamba2 32K still in progress

E88 beat Mamba2 at 8K but falls behind at 32K. The 512→32K jump may be the cause — Mamba2's parallel scan handles the transition smoothly while E88's sequential processing struggles to adapt from 512-length representations to 32K in one shot.

### Throughput Penalty

At 32K, each training step processes 32K tokens sequentially for E88 vs parallel scan for Mamba2. Fewer steps = fewer gradient updates = slower adaptation. Intermediate checkpoints at 8K give E88 more gradient steps during the transition.

## Implementation

Modify `cmaes_search_v2.py` `run_training_progressive()` to support multi-stage:

```python
PROGRESSIVE_STAGES = [512, 8192, 32768, 131072]

def run_training_full_progressive(params, model_type, total_minutes, output_dir, ...):
    """Multi-stage progressive training: 512 → 8K → 32K → 128K"""
    minutes_per_stage = total_minutes / len(PROGRESSIVE_STAGES)

    ckpt = None
    for stage_ctx in PROGRESSIVE_STAGES:
        ckpt = train_stage(params, stage_ctx, minutes_per_stage, resume_from=ckpt)

    return parse_loss(ckpt)
```

## Timeline

Run after current 28-search multi-context campaign completes (~mid April 2026). Use the same 7 models × CMA-ES methodology but with the full progressive pipeline.
