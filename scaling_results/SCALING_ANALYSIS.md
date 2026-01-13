# Scaling Study: E1 vs E33 vs E42 vs Mamba2

## Summary

**Critical Finding**: Elman advantage diminishes with scale and reverses at 1B parameters.

| Scale | Best Model | Loss | Delta vs Mamba2 |
|-------|------------|------|-----------------|
| 50M   | E42        | 1.33 | -0.47 (47% better) |
| 100M  | E1         | 1.42 | -0.32 (23% better) |
| 250M  | E42        | 1.41 | -0.22 (16% better) |
| 500M  | E1         | 1.52 | -0.16 (11% better) |
| 1B    | **Mamba2** | 1.73 | +0.03 (Mamba wins!) |

## Complete Results (Last-100 Avg Loss)

| Scale | E1    | E33   | E42   | Mamba2 |
|-------|-------|-------|-------|--------|
| 50M   | 1.44  | 1.44  | **1.33** | 1.80 |
| 100M  | **1.42** | 1.50 | 1.44  | 1.74 |
| 250M  | 1.58  | 1.55  | **1.41** | 1.64 |
| 500M  | **1.52** | 1.58 | 1.66  | 1.68 |
| 1B    | 1.76  | 1.80  | 1.81  | **1.73** |

## Throughput (tok/s)

| Scale | E1      | E33     | E42    | Mamba2 |
|-------|---------|---------|--------|--------|
| 50M   | 231K    | **234K** | 191K   | 106K   |
| 100M  | 119K    | **132K** | 93K    | 58K    |
| 250M  | 52K     | **58K**  | 43K    | 36K    |
| 500M  | **30K** | 28K     | 21K    | 16K    |
| 1B    | 14K     | **15K** | 9K     | 11K    |

## Key Insights

### 1. Elman Advantage Diminishes with Scale
- At 50M: Elman is 0.47 nats better (47% lower loss)
- At 100M: Elman is 0.32 nats better (23% lower)
- At 250M: Elman is 0.22 nats better (16% lower)
- At 500M: Elman is 0.16 nats better (11% lower)
- At 1B: **Mamba2 wins by 0.03 nats**

### 2. E42 (Linear Tied) Wins at Smaller Scales
- Best at 50M (1.33), 250M (1.41)
- Degrades significantly at 500M+ (1.66, 1.81)
- Tied weight sharing may limit capacity at scale

### 3. E1 (Mamba-Gated) Scales Better than E33/E42
- Best at 100M, 500M
- Most consistent across scales
- Closest to Mamba2 at 1B (1.76 vs 1.73)

### 4. Throughput Advantage Narrows
- At 50M: E33 is 2.2x faster than Mamba2
- At 1B: E33 is only 1.3x faster than Mamba2

### 5. E33 Has Consistent Throughput Advantage
- Highest throughput at all scales except 500M
- Self-gating is more efficient than Mamba-gating

## Hypotheses for Scale Degradation

1. **Spectral Radius**: Larger hidden states may destabilize nonlinear recurrence
2. **Gradient Flow**: Nonlinear recurrence compounds gradient issues at depth
3. **Linear Recurrence Advantage**: Mamba2's linear recurrence enables parallel scan optimization

## Conclusion

For production deployment:
- **< 500M params**: Use Elman (E42 for quality, E33 for throughput)
- **>= 1B params**: Consider Mamba2 (better loss, competitive throughput)

The nonlinear recurrence advantage disappears at scale. Linear state-space models like Mamba2 scale more gracefully.

---
Generated: 2026-01-13T06:21
Training: 60 minutes per configuration
Metric: Last-100 step average loss
