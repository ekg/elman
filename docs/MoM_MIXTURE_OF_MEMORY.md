# Mixture of Memory (MoM) for E88

**Date**: 2026-01-27
**Status**: Proposal for implementation

## Summary

Mixture of Memory (MoM) applies the Mixture of Experts (MoE) principle to E88's memory heads. Instead of updating all 104 heads every timestep, we route each token to a subset of heads (e.g., top-32). This allows scaling to 300+ heads with the same compute budget.

## Motivation

**Updated E88 optimal config (CMA-ES Jan 27, 2026):**
- n_heads=98, n_state=32, depth=14, dim=2176 → loss 1.39
- Key finding: **shallow+wide beats deep+narrow** (d=14 > d=32)
- Gap to Mamba2/FLA-GDN: ~0.12 nats (1.39 vs 1.27)

The bottleneck is sequential recurrence - we can't parallelize across time. But we CAN reduce work per timestep by only updating relevant heads.

## Core Idea

```
Standard E88:  All H heads updated per token  → O(H) compute
MoM E88:       Top-k heads updated per token  → O(k) compute, k << H
```

With k=32 and H=312 (3x current), we get:
- 3x more memory capacity (312 heads vs 104)
- Same compute per step (32 updates vs 104 → actually cheaper)
- Each head specializes in certain token patterns

## Algorithm

### Forward Pass

```python
def mom_e88_forward(x, S, router, k=32):
    """
    Args:
        x: [B, T, dim] input
        S: [B, H, n_state, n_state] all head states
        router: nn.Linear(dim, H) routing network
        k: number of heads to activate per token

    Returns:
        output: [B, T, dim]
        S: updated states (only top-k modified per token)
    """
    B, T, dim = x.shape
    H = S.shape[1]

    outputs = []
    for t in range(T):
        x_t = x[:, t]  # [B, dim]

        # 1. ROUTE: Select top-k heads for this token
        router_logits = router(x_t)  # [B, H]
        router_weights, top_k_indices = topk(softmax(router_logits), k)  # [B, k]

        # 2. PROJECT: Compute k, v, q only for selected heads
        # This is the key efficiency gain - skip projection for inactive heads
        k_selected = project_k(x_t, top_k_indices)  # [B, k, n_state]
        v_selected = project_v(x_t, top_k_indices)  # [B, k, n_state]
        q_selected = project_q(x_t, top_k_indices)  # [B, k, n_state]
        decay_selected = compute_decay(x_t, top_k_indices)  # [B, k]

        # 3. UPDATE: Only update selected heads' states
        for b in range(B):
            for i, h in enumerate(top_k_indices[b]):
                k_h = k_selected[b, i]
                v_h = v_selected[b, i]
                decay_h = decay_selected[b, i]

                # Standard E88 update
                retrieved = S[b, h] @ k_h
                delta = v_h - retrieved
                S[b, h] = tanh(decay_h * S[b, h] + outer(delta, k_h))

        # 4. RETRIEVE: Query selected heads and combine
        head_outputs = []
        for b in range(B):
            for i, h in enumerate(top_k_indices[b]):
                q_h = q_selected[b, i]
                Sq = S[b, h] @ q_h
                head_outputs.append(router_weights[b, i] * Sq)

        # Weighted combination of head outputs
        out_t = sum(head_outputs)  # [B, n_state] or project to dim
        outputs.append(out_t)

    return stack(outputs, dim=1), S
```

### Router Design

The router determines which heads activate for each token. Options:

**Option A: Simple Linear Router (recommended to start)**
```python
self.router = nn.Linear(dim, H, bias=False)

def route(x):
    logits = self.router(x)  # [B, H]
    weights = softmax(logits / temperature)
    return topk(weights, k)
```

**Option B: Hash-based Router (faster, no learning)**
```python
def route(x):
    # Project to low-dim, hash to bucket
    h = hash(x @ W_hash) % H
    # Select k consecutive heads starting at h
    return [(h + i) % H for i in range(k)]
```

**Option C: Content-based Router (most expressive)**
```python
# Each head has a "key" that tokens match against
self.head_keys = nn.Parameter(torch.randn(H, key_dim))

def route(x):
    query = self.route_proj(x)  # [B, key_dim]
    scores = query @ self.head_keys.T  # [B, H]
    return topk(softmax(scores), k)
```

### Load Balancing

MoE models need load balancing to prevent collapse (all tokens routing to same heads). Apply auxiliary loss:

```python
def load_balance_loss(router_logits, top_k_indices):
    """Encourage uniform head usage across batch."""
    H = router_logits.shape[-1]

    # Count how often each head is selected
    head_counts = bincount(top_k_indices.flatten(), minlength=H)

    # Target: uniform distribution
    target = top_k_indices.numel() / H

    # L2 penalty for deviation from uniform
    return ((head_counts - target) ** 2).mean()
```

Add to training loss: `loss = ce_loss + 0.01 * load_balance_loss`

## CUDA Implementation Strategy

### Kernel Structure

The CUDA kernel needs modification to handle variable head indices per batch element.

**Current E88 kernel:**
```cuda
// Each block handles one (batch, head) pair
// Block (b, h) always processes head h
__global__ void E88Forward(int B, int H, ...) {
    int b = blockIdx.x / H;
    int h = blockIdx.x % H;
    // ... process head h for batch b
}
```

**MoM kernel:**
```cuda
// Each block handles one (batch, slot) pair
// Slot i maps to different heads for different batch elements
__global__ void MoME88Forward(
    int B, int k,
    const int* head_indices,  // [B, k] which head for each slot
    const float* router_weights,  // [B, k] weights for combining
    ...
) {
    int b = blockIdx.x / k;
    int slot = blockIdx.x % k;
    int h = head_indices[b * k + slot];  // Look up actual head index
    float weight = router_weights[b * k + slot];

    // ... process head h for batch b, scale output by weight
}
```

### Memory Access Pattern

**Challenge**: Non-contiguous head access. If batch 0 uses heads [3, 17, 42, ...] and batch 1 uses heads [5, 12, 88, ...], memory access is scattered.

**Solution**: Pre-gather states into contiguous buffer:
```cuda
// Before kernel: gather selected states
for (int b = 0; b < B; b++) {
    for (int i = 0; i < k; i++) {
        int h = head_indices[b * k + i];
        // Copy S[b, h] to S_gathered[b, i]
        memcpy(&S_gathered[b * k + i], &S[b * H + h], state_size);
    }
}

// Run kernel on contiguous S_gathered
MoME88Forward<<<B * k, threads>>>(..., S_gathered, ...);

// After kernel: scatter back to original positions
for (int b = 0; b < B; b++) {
    for (int i = 0; i < k; i++) {
        int h = head_indices[b * k + i];
        memcpy(&S[b * H + h], &S_gathered[b * k + i], state_size);
    }
}
```

This adds O(B * k * state_size) memcpy overhead but ensures coalesced access in the main kernel.

### Backward Pass

Gradients flow only through selected heads:
- `d_router`: gradient through routing decision
- `d_S[selected]`: gradient to selected head states
- `d_S[unselected]`: zero (these heads weren't used)

For router gradient, use straight-through estimator or Gumbel-softmax for differentiable top-k.

## Implementation Plan

### Phase 1: Python Prototype (verify algorithm)
1. Add `MoME88FLAHybrid` class in `elman/models/mom_e88.py`
2. Implement routing + selective update in pure PyTorch
3. Test on small scale (dim=256, H=32, k=8)
4. Verify loss improves with more heads at fixed k

### Phase 2: CUDA Forward Kernel
1. Copy `e88_fla_hybrid_gpu.cu.cc` to `mom_e88_gpu.cu.cc`
2. Add head_indices parameter
3. Implement gather/scatter for state access
4. Benchmark: should be ~3x faster than full E88 with 3x heads

### Phase 3: CUDA Backward Kernel
1. Implement gradient checkpointing for selected heads only
2. Handle router gradient (may need separate kernel)
3. Verify gradients match PyTorch reference

### Phase 4: Scaling Experiments
1. Test at 480M scale: H=312, k=32 (same compute as H=104)
2. Compare loss vs standard E88
3. Analyze head specialization (which tokens route where?)

## Expected Outcomes

**If MoM works well:**
- 3x more memory capacity with same compute
- Potential loss improvement of 0.05-0.10 nats
- Head specialization: some heads for syntax, others for semantics, etc.

**Potential issues:**
- Router collapse (all tokens to same heads) → need load balancing
- Training instability from discrete routing → use Gumbel-softmax
- Overhead from gather/scatter → may limit speedup

## Comparison to MoE

| Aspect | MoE (Mixture of Experts) | MoM (Mixture of Memory) |
|--------|--------------------------|-------------------------|
| Routes to | FFN experts | Memory heads |
| State | Stateless | Stateful (persistent memory) |
| Sparsity benefit | Compute (skip FFN) | Compute + Memory updates |
| Load balancing | Critical | Critical |
| Typical k | 1-2 experts | 16-64 heads |

## References

- Mixture of Experts: Shazeer et al., "Outrageously Large Neural Networks" (2017)
- Switch Transformer: Fedus et al. (2021) - simplified MoE routing
- GLaM: Du et al. (2022) - MoE at scale
- Current E88: `elman/models/e88_fla_hybrid.py`, `elman/cuda/lib/e88_fla_hybrid_gpu.cu.cc`

## Files to Create/Modify

```
elman/models/mom_e88.py              # New: MoM E88 Python implementation
elman/cuda/lib/mom_e88_gpu.cu.cc     # New: MoM E88 CUDA kernel
elman/cuda/lib/hasty/elman_ladder.h  # Add: MoM function declarations
elman/cuda/pytorch/elman_ladder.cc   # Add: MoM Python bindings
elman/models/ladder_lm.py            # Add: MoM E88 to level registry
```

## Quick Start Commands

```bash
# After implementing mom_e88.py:

# Test Python prototype (small scale)
python -c "
from elman.models.mom_e88 import MoME88
model = MoME88(dim=256, n_heads=64, n_state=16, top_k=16)
x = torch.randn(2, 32, 256)
out, _ = model(x)
print(f'Output shape: {out.shape}')
"

# Benchmark vs standard E88 (after CUDA kernel)
python train.py --level MoME88 --dim 896 --depth 32 --n_heads 312 --n_state 32 --top_k 32 \
  --data data/pile.txt --batch_size 16 --bf16 --train_minutes 10
```
