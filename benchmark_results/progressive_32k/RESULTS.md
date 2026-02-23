# Progressive Context Scaling: 512 → 32K

## Experiment Design

**Hypothesis**: Architecture search at short context (512) + progressive training to long context (32K) is more efficient than searching directly at 32K.

**Protocol**:
- Phase 1: Train best CMA-ES configs at 512 context for 10 min → save checkpoints
- Phase 2: Resume from checkpoints at 32K with LR sweep (1e-4, 3e-4, 6e-4) for 10 min
- Phase 3 (control): Train from scratch at 32K for 10 min

**Setup**: ~480M params, seed=42, bf16, schedulefree optimizer, commapile data

## Results

| Model | Type | 512 Loss | 32K Resume | Best LR | 32K Scratch | Resume vs Scratch |
|-------|------|----------|------------|---------|-------------|-------------------|
| **FLA-GDN** | Assoc scan | 1.435 | **1.179** | 3e-4 | 1.748 | -0.569 |
| **E1** | Sequential RNN | 1.633 | **1.205** | 6e-4 | 3.892 | -2.687 |
| **Mamba2** | SSM par scan | 1.440 | **1.217** | 6e-4 | 1.839 | -0.622 |
| **E88 n16** | Sequential RNN | 1.470 | **1.230** | 3e-4 | 2.846 | -1.616 |
| **E88 n32** | Sequential RNN | 1.473 | **1.373** | 3e-4 | 2.884 | -1.511 |
| MinGRU | Par scan | 1.716 | 1.728 | 3e-4 | 2.825 | -1.097 |
| MinLSTM | Par scan | 1.710 | 1.737 | 1e-4 | 2.702 | -0.965 |

### Batch sizes at 32K (48GB GPU)
| Model | bs (resume) | Peak mem |
|-------|-------------|----------|
| E88 n16 | 4 | ~34 GB (grad_ckpt + proj_chunk) |
| E88 n32 | 4 | ~40 GB (grad_ckpt + proj_chunk) |
| E1 | 2 | ~47 GB |
| Mamba2 | 1 | ~32 GB |
| FLA-GDN | 1 | ~42 GB |
| MinLSTM | 1 | ~47 GB |
| MinGRU | 1 | ~47 GB |

## Key Findings

### 1. Progressive training is massively better than training from scratch at 32K
Every model benefits hugely from the 512→32K progressive approach. The gap ranges from 0.5 nats (FLA-GDN) to 2.7 nats (E1). At 10 min training time, models simply don't get enough steps at 32K to learn from scratch.

### 2. FLA-GDN leads at 32K (1.179)
Consistent with its dominance at shorter contexts. The associative scan / chunked attention approach handles long context efficiently.

### 3. E1 is the surprise performer (1.205)
The simple gated Elman RNN nearly matches Mamba2 (1.217) at 32K, despite being the worst-performing model at 512 among the top group. E1 benefits the most from progressive training — its 512 representations transfer exceptionally well to long context.

### 4. E88 n16 > E88 n32 at 32K
The many-small-heads strategy (141 heads × 16×16) transfers better to long context than fewer-larger-heads (83 heads × 32×32). n16 reaches 1.230 vs n32's 1.373.

### 5. MinGRU and MinLSTM can't use long context
Their 32K resume losses (1.728, 1.737) are actually WORSE than their 512 losses (1.716, 1.710). These parallel scan architectures fail to exploit the additional context. This is a significant negative finding.

### 6. Sequential models need progressive training most
E88/E1 trained from scratch at 32K are terrible (2.8-3.9 loss) but excellent with resume (1.2-1.4). Their sequential nature means very few gradient updates at 32K, making warm starts critical.

## Model Configurations (CMA-ES optimized at 512)

```
E88 n16: --dim 1536 --depth 25 --n_heads 141 --n_state 16 --use_gate 1 --gate_activation silu --lr 7.9e-4
E88 n32: --dim 1920 --depth 17 --n_heads 83 --n_state 32 --use_gate 1 --gate_activation silu --lr 6.4e-4
Mamba2:  --dim 1792 --depth 25 --mamba_d_state 96 --mamba_expand 2 --lr 3e-4
FLA-GDN: --dim 1920 --depth 17 --expansion 2 --n_heads 24 --lr 3e-4
MinLSTM: --dim 2944 --depth 10 --expansion 1 --lr 7.8e-4
MinGRU:  --dim 3456 --depth 10 --expansion 1 --lr 9.9e-4
E1:      --dim 2816 --depth 11 --expansion 1 --lr 1.7e-4
```

## Reproduction

```bash
# Full pipeline
bash run_progressive_32k.sh

# Fix for OOMing models (bs=2 and bs=1)
bash run_progressive_32k_bs1.sh
```

## Date: February 23, 2026
