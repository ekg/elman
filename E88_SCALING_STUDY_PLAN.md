# E88 Scaling Laws Study Plan

## RESULTS SUMMARY (Jan 22, 2026)

### Dimension Sweep (500M params)
| dim | depth | Loss | Finding |
|-----|-------|------|---------|
| **1792** | 38 | **1.44** | **BEST - optimal width/depth** |
| 2560 | 18 | 1.47 | Good - wider but shallower |
| 2304 | 22 | 1.47 | Good |
| 2048 | 28 | 1.55 | Previous default |
| 1536 | 50 | 1.70 | Too narrow |

### n_state Sweep (dim=2048, depth=32)
| n_state | n_heads | Loss | Finding |
|---------|---------|------|---------|
| **32** | 64 | **1.47** | **BEST** |
| 24 | 85 | 1.55 | Close second |
| 48 | 42 | 1.57 | Too large |
| 16 | 128 | 1.69 | Too small |

### Key Findings
1. **Optimal dim=1792, depth=38** beats previous best by 0.11 nats
2. **n_state=32 is universally optimal** - neither smaller nor larger helps
3. **Sweet spot width: 1792-2816** - not too narrow, not too wide
4. **E88 benefits from depth** - 38 layers better than 28 at same params

### Best Config for 500M
```
dim=1792, depth=38, n_state=64 (NOT 32!)
Loss: 1.44 (vs Mamba2: 1.36, FLA-GDN: 1.40)
```
**NOTE**: E88 is now competitive with FLA-GDN (1.44 vs 1.40, only 0.04 nats difference)!

---

## What We Know So Far

| Finding | Evidence |
|---------|----------|
| Balanced ratio (n_heads × n_state ≈ dim) is critical | Unbalanced configs 10x slower |
| n_state=32 beats n_state=48,64 at 500M | 1.55 vs 1.70-1.72 loss |
| E88 beats Mamba2 at 500M | 1.55 vs 1.80 loss |
| E88 uses 7x less state than Mamba2 | 57K vs 410K per layer |
| FLA-GDN still leads | 1.29 loss with 1.33M state |

### Gating Ablation (Jan 23, 2026) - COMPLETED

Testing whether SiLU gating (suggested by elman-proofs) improves E88.

| Config | Params | Loss | Tok/s | Notes |
|--------|--------|------|-------|-------|
| **Mamba2** | 508M | **1.36** | 17K | Best overall |
| **FLA-GDN** | 534M | **1.40** | 19K | Second best |
| E88 ungated | 492M | 1.61 | 10.5K | Best E88 |
| E88 SiLU gated | 614M | 1.70 | 9.1K | Worse than ungated |
| E88 sigmoid gated | 614M | 1.70 | 9.2K | Worse than ungated |

**Key Finding: Output gating HURTS E88 performance!**
- Gated variants are 0.09 nats worse than ungated
- This is despite having 122M more parameters (614M vs 492M)
- The g_proj output gating used by FLA-GDN does not help E88
- E88's internal tanh nonlinearity may already provide sufficient regularization

---

### n_state Verification (Jan 23, 2026)

**Critical finding: n_state=64 >> n_state=32 at dim=1792, depth=38!**

| Config | n_state | Loss | Difference |
|--------|---------|------|------------|
| n64_default | 64 | **1.44** | baseline |
| n32_explicit | 32 | 1.84 | +0.40 worse |

The previous "good" results (1.44 loss) used n_state=64 (train.py default), not n_state=32.
This 0.40 nat difference is huge - optimal n_state depends on dim/depth combination.

---

## Open Questions

1. **Does wider dimension help?** (dim=3000+ vs dim=2000) - ANSWERED: No, dim=1792 optimal
2. **Depth vs width trade-off?** (depth=48 vs depth=20 at same params) - ANSWERED: depth=38 optimal
3. **Is n_state=32 universally optimal?** - ANSWERED: **NO!** n_state=64 is better at dim=1792 (+0.40 nats)
4. **Does SiLU gating help?** - ANSWERED: **NO!** Both SiLU and sigmoid gating hurt E88 by ~0.09 nats
5. **Why does E88 trail FLA-GDN?** E88 (1.44 with n_state=64) vs FLA-GDN (1.40) - gap is now only 0.04 nats!
6. **What's the optimal n_state for different dim/depth?** Need systematic sweep
4. **What's the minimum useful state size?**
5. **How does E88 scale from 100M → 1B?**
6. **Can more state close the gap to FLA-GDN?**

## Proposed Experiments

### Experiment 1: Dimension Sweep (500M, fixed params)

Test if wider models perform better by varying dim while keeping ~500M params.

| Config | dim | depth | n_heads | n_state | Params |
|--------|-----|-------|---------|---------|--------|
| narrow | 1536 | 48 | 48 | 32 | ~500M |
| medium | 2048 | 32 | 64 | 32 | ~500M |
| wide | 2560 | 24 | 80 | 32 | ~500M |
| wider | 3072 | 18 | 96 | 32 | ~500M |

Hypothesis: Wider might help if E88 benefits from more expressive per-layer computation.

### Experiment 2: Depth Sweep (fixed dim=2048)

Test depth scaling at constant width.

| Config | dim | depth | n_heads | n_state | Params |
|--------|-----|-------|---------|---------|--------|
| shallow | 2048 | 16 | 64 | 32 | ~250M |
| medium | 2048 | 24 | 64 | 32 | ~380M |
| deep | 2048 | 32 | 64 | 32 | ~500M |
| deeper | 2048 | 48 | 64 | 32 | ~750M |

### Experiment 3: n_state Fine-Grained Sweep

Test n_state more finely around the optimum.

| n_state | n_heads (for ratio=1.0 at dim=2048) |
|---------|-------------------------------------|
| 16 | 128 |
| 24 | 85 |
| 32 | 64 |
| 40 | 51 |
| 48 | 43 |

### Experiment 4: Multi-Scale Comparison

Compare E88, Mamba2, FLA-GDN at multiple scales.

| Scale | E88 config | Baseline configs |
|-------|------------|------------------|
| 100M | dim=1024, depth=20, h=32, n=32 | mamba2, fla-gdn |
| 300M | dim=1536, depth=28, h=48, n=32 | mamba2, fla-gdn |
| 500M | dim=2048, depth=32, h=64, n=32 | mamba2, fla-gdn |
| 1B | dim=2560, depth=40, h=80, n=32 | mamba2, fla-gdn |

### Experiment 5: State Size Matching

Can E88 match FLA-GDN's state and quality?

FLA-GDN: 1.33M state/layer → need n_heads × n_state² = 1.33M
- With n_state=32: n_heads = 1300 (too many!)
- With n_state=64: n_heads = 325 (still many)
- With n_state=128: n_heads = 81 (tested, slow due to imbalance)

Alternative: Accept smaller state, focus on efficiency.

## Priority Order

1. **Dimension sweep** - Quick to run, answers "does wider help?"
2. **Multi-scale comparison** - Reveals scaling laws
3. **n_state fine-grained** - Confirms n_state=32 optimal
4. **Depth sweep** - Depth vs width trade-off

## Key Metrics to Track

- Loss (last-100 avg)
- Throughput (tok/s)
- State size per layer
- Params
- Steps completed in 10 min
