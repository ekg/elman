# E74 Ablation Study Plan

Systematic ablation of E73 matrix state RNN to find optimal architecture without redundancy.

---

## 1. Ablation Dimensions

### A. Projection Ablations (4 variants)
| ID | Config | Projections | Params (n=64, d=1024) |
|----|--------|-------------|----------------------|
| P0 | baseline | k, v, q, z separate | 4×n×d = 262K |
| P1 | no_z | k, v, q separate | 3×n×d = 196K |
| P2 | tied_kq | k=q, v separate | 2×n×d = 131K |
| P3 | tied_kvq | k=v=q | 1×n×d = 65K |

### B. State Structure Ablations (5 variants)
| ID | Structure | Shape | Memory per batch |
|----|-----------|-------|------------------|
| S0 | full_matrix | [B, n, n] | n² = 4096 |
| S1 | diagonal | [B, n] | n = 64 |
| S2 | lowrank_4 | [B, n, 4] + [B, 4, n] | 2×n×r = 512 |
| S3 | lowrank_8 | [B, n, 8] + [B, 8, n] | 2×n×r = 1024 |
| S4 | block_diag | [B, n/8, 8, 8] | (n/b)×b² = 512 |

### C. Nonlinearity Ablations (4 variants)
| ID | After Update | Stability Mechanism |
|----|--------------|---------------------|
| N0 | tanh(S) | tanh bounds S |
| N1 | none (linear) | spectral_norm on W |
| N2 | rmsnorm(S) | RMS normalization |
| N3 | S / (||S||_F + ε) | Frobenius normalization |

### D. Gate Ablations (3 variants)
| ID | Gating | Description |
|----|--------|-------------|
| G0 | output_only | out = x * silu(x) |
| G1 | retain_gate | α*S_prev + (1-α)*delta |
| G2 | state_gate | delta * sigmoid(f(S)) |

### E. Update Rule Ablations (2 variants)
| ID | Update Rule | Formula |
|----|-------------|---------|
| U0 | delta | S = f(S + outer(v - S@k, k)) - erase before write |
| U1 | simple | S = f(α*S + outer(v, k)) - just decay + write |

The **delta rule** does retrieval-based erasure before writing (like content-addressable memory).
The **simple rule** just decays old state and writes new input (like standard RNN/GRU).

---

## 2. Ablation Priority

### Phase 1: State Structure (Highest Impact)
S0→S1 (diagonal) gives **64x memory reduction** for state.
This is the biggest win if expressivity holds.

Test: S0 vs S1 vs S2 vs S3 vs S4 with P0 (baseline projections)

### Phase 2: Projections (Medium Impact)
Once best state structure found, ablate projections.
P0→P3 gives **4x parameter reduction** in cell.

Test: P0 vs P1 vs P2 vs P3 with best state structure

### Phase 3: Nonlinearity (Gradient Impact)
Linear (N1) may improve gradient flow like E42.

Test: N0 vs N1 vs N2 vs N3 with best structure + projections

### Phase 4: Gates (Fine-tuning)
Add E68-style state gating if helpful.

Test: G0 vs G1 vs G2 with best config

---

## 3. Memory Analysis

### Current E73 Checkpointed (S0, P0)
```
Per sequence (T=512, B=32, n=64, K=32):
  S_checkpoints: [T/K+1, B, n, n] = 17 × 32 × 64 × 64 × 2B = 4.5 MB
  k_norm_cache:  [T, B, n] = 512 × 32 × 64 × 2B = 2.1 MB
  v_cache:       [T, B, n] = 2.1 MB
  q_cache:       [T, B, n] = 2.1 MB
  z_cache:       [T, B, n] = 2.1 MB
  Sq_cache:      [T, B, n] = 2.1 MB
  Total: ~15 MB per sequence
```

### Diagonal State (S1, P1)
```
Per sequence (T=512, B=32, n=64, K=32):
  S_checkpoints: [T/K+1, B, n] = 17 × 32 × 64 × 2B = 70 KB (64x smaller!)
  k_norm_cache:  [T, B, n] = 2.1 MB
  v_cache:       [T, B, n] = 2.1 MB
  q_cache:       [T, B, n] = 2.1 MB (or tie with k)
  Sq_cache:      [T, B, n] = 2.1 MB (becomes S*q element-wise)
  Total: ~8.4 MB per sequence (without z)
```

### Diagonal + Tied k=q (S1, P2)
```
  S_checkpoints: 70 KB
  w_cache:       [T, B, n] = 2.1 MB (k=q)
  v_cache:       [T, B, n] = 2.1 MB
  Sw_cache:      [T, B, n] = 2.1 MB
  Total: ~6.4 MB per sequence (57% reduction)
```

---

## 4. Diagonal State Formulation

### Full Matrix Delta Rule
```
S[i,j] = tanh(S[i,j] + (v[i] - retrieved[i]) * k_norm[j])
retrieved[i] = Σ_j S[i,j] * k_norm[j]
out[i] = Σ_j S[i,j] * q[j]
```

### Diagonal Delta Rule
```
S[i] = tanh(S[i] + (v[i] - S[i] * k_norm[i]) * k_norm[i])
     = tanh(S[i] * (1 - k_norm[i]²) + v[i] * k_norm[i])
out[i] = S[i] * q[i]
```

Key insight: Diagonal is actually an EMA variant!
- If k_norm[i]² = 1-α, then S[i] = tanh(α*S[i] + (1-α)*v[i])
- But k_norm is input-dependent → **input-dependent decay**

This is similar to Mamba's selective SSM but simpler.

### Low-Rank Delta Rule (rank r)
```
S = U @ V^T  where U∈[B,n,r], V∈[B,n,r]

# Update: factor the outer product
delta = v - (U @ V^T) @ k_norm
      = v - U @ (V^T @ k_norm)  # Only r×n matmul

# Option A: Update U only
U = U + outer(delta, some_projection)

# Option B: Alternate U/V updates
# Even timesteps: U = U + delta @ V^T @ k
# Odd timesteps:  V = V + ...
```

---

## 5. CUDA Kernel Strategy

### Option A: Separate Kernels per Variant
- e74_diagonal_gpu.cu.cc
- e74_lowrank_gpu.cu.cc
- e74_blockdiag_gpu.cu.cc

**Pros**: Clean, optimized per structure
**Cons**: Code duplication, maintenance burden

### Option B: Generic Checkpointed Kernel
Template the state structure, keep checkpoint logic shared.

```cpp
template<typename StateType>
class E74CheckpointedForward {
    void Run(..., StateType* S, ...);
};

// Specializations
template<> void E74CheckpointedForward<DiagonalState>::Run(...);
template<> void E74CheckpointedForward<LowRankState>::Run(...);
```

**Pros**: Shared checkpoint logic, less duplication
**Cons**: Template complexity, harder to optimize

### Option C: Recommended Hybrid
1. **Copy e73_checkpointed_gpu.cu.cc → e74_checkpointed_gpu.cu.cc**
2. Add `state_type` parameter (0=full, 1=diagonal, 2=lowrank)
3. Branch only in state update/retrieval kernels
4. Keep checkpoint save/load generic

```cpp
// In forward loop:
if (state_type == DIAGONAL) {
    E74DiagonalDeltaKernel<<<...>>>(S_diag, v, k_norm);
} else if (state_type == FULL) {
    E74FullDeltaKernel<<<...>>>(S_full, v, k_norm);
}

// Checkpoint save is just memcpy - state_size varies
cudaMemcpyAsync(S_checkpoint, S, state_size, ...);
```

---

## 6. Implementation Order

### Step 1: Python Ablation Framework
Create `e74_ablations.py` with all variants as PyTorch modules.
Run quick perplexity comparisons to identify promising configs.

### Step 2: Top Candidates → CUDA
Pick 2-3 best configs from Python experiments.
Implement optimized CUDA kernels for those.

### Step 3: Full Benchmark
Run full training comparisons at 50M-500M scale.
Measure: perplexity, throughput, memory, gradient norms.

---

## 7. Ablation Matrix

Full combinatorial is 4×5×4×3 = 240 variants (too many).

**Recommended subset (20 experiments):**

| # | State | Proj | Nonlin | Gate | Description |
|---|-------|------|--------|------|-------------|
| 1 | S0 | P0 | N0 | G0 | E73 baseline |
| 2 | S0 | P1 | N0 | G0 | Remove z |
| 3 | S0 | P2 | N0 | G0 | Tie k=q |
| 4 | S0 | P3 | N0 | G0 | Tie k=v=q |
| 5 | S1 | P0 | N0 | G0 | Diagonal baseline |
| 6 | S1 | P1 | N0 | G0 | Diagonal, no z |
| 7 | S1 | P2 | N0 | G0 | Diagonal, tied k=q |
| 8 | S1 | P3 | N0 | G0 | Diagonal, k=v=q |
| 9 | S2 | P1 | N0 | G0 | Lowrank-4, no z |
| 10 | S3 | P1 | N0 | G0 | Lowrank-8, no z |
| 11 | S4 | P1 | N0 | G0 | Block-diag, no z |
| 12 | S1 | P2 | N1 | G0 | Diag, linear (E42-style) |
| 13 | S1 | P2 | N2 | G0 | Diag, rmsnorm |
| 14 | S0 | P1 | N1 | G0 | Full, linear |
| 15 | S1 | P1 | N0 | G1 | Diag, retain gate |
| 16 | S1 | P1 | N0 | G2 | Diag, state gate |
| 17 | S1 | P2 | N1 | G1 | Diag, linear, retain (best combo?) |
| 18 | S2 | P2 | N0 | G0 | Lowrank-4, tied k=q |
| 19 | S3 | P2 | N1 | G0 | Lowrank-8, linear |
| 20 | S1 | P3 | N1 | G0 | Minimal: diag, tied, linear |

---

## 8. Success Metrics

For each ablation, measure:

1. **Perplexity** (primary): 10-min training runs
2. **Parameters**: Total count
3. **Memory**: Peak GPU memory during training
4. **Throughput**: Tokens/second
5. **Gradient health**: Mean/max gradient norms over training

**Decision criteria**:
- Perplexity within 5% of baseline → prefer simpler
- Memory reduction > 2x → strong consideration
- Throughput improvement > 20% → strong consideration

---

## 9. Expected Outcomes

Based on E42/E68 success patterns:

**High confidence wins:**
- P1 (remove z) ≈ P0 perplexity (z adds complexity, not expressivity)
- S1 (diagonal) competitive for many tasks (like E42's vector state)

**Medium confidence:**
- P2/P3 (tied projections) may hurt complex reasoning tasks
- N1 (linear) needs careful spectral control but improves gradients
- S2/S3 (low-rank) good tradeoff between diagonal and full

**Worth testing:**
- G1/G2 (gates) may help long-range tasks
- S4 (block-diag) may capture local structure efficiently

---

## 10. Next Steps

1. **Implement Python ablation framework** (this session)
2. **Run Phase 1 experiments** (state structure)
3. **Analyze results, pick winners**
4. **Write CUDA design doc for winners**
5. **Implement CUDA kernels**
6. **Full-scale benchmarks**

Ready to implement?
