"""
Level 8: Low-Rank Elman with Big Hidden State

Like Mamba2's insight: use small recurrent state but big projections.
W matrices are decomposed as U @ V (low-rank) to allow much larger hidden states
without quadratic parameter explosion.

Key idea:
    - Standard Elman: W_x is (d_inner x d_inner) = O(d²) params
    - Low-rank: W_x = U @ V where U is (d_inner x rank), V is (rank x d_inner)
    - This gives O(2 * d_inner * rank) params
    - Can have d_inner = 4x bigger with same param budget!

Architecture:
    v = r_h * h_prev + (W_x_u @ W_x_v) @ x + b
    delta = sigmoid((W_delta_u @ W_delta_v) @ x + r_delta * h_prev + b_delta)
    h_new = (1 - delta) * h_prev + delta * tanh(v)
    output = h_new * silu(h_new + x_proj + b_gate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LEVEL_8_AVAILABLE = True


class LowRankLinear(nn.Module):
    """Low-rank linear layer: y = (U @ V) @ x = U @ (V @ x)"""

    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.U = nn.Parameter(torch.empty(out_dim, rank))
        self.V = nn.Parameter(torch.empty(rank, in_dim))
        nn.init.xavier_uniform_(self.U, gain=0.5)
        nn.init.xavier_uniform_(self.V, gain=0.5)

    def forward(self, x):
        # More efficient: (V @ x) first since rank << dim
        return F.linear(F.linear(x, self.V), self.U)


class LowRankElmanCell(nn.Module):
    """
    Low-Rank Elman cell - Level 8.

    Uses low-rank decomposition of W matrices to allow much larger hidden states.

    Args:
        dim: Hidden dimension (can be very large!)
        rank: Rank for W decomposition (small, e.g., 64-256)
        delta_init: Initial bias for delta gate
        r_h_init: Initial value for r_h (decay factor)
    """

    def __init__(self, dim, rank=64, delta_init=-2.0, r_h_init=0.9, **kwargs):
        super().__init__()
        self.dim = dim
        self.rank = rank

        # Diagonal r_h (decay factor) - like Mamba2's diagonal A
        self.r_h = nn.Parameter(torch.full((dim,), float(r_h_init)))

        # Diagonal r_delta
        self.r_delta = nn.Parameter(torch.full((dim,), 0.1))

        # Low-rank W_x: instead of (dim, dim), use (dim, rank) @ (rank, dim)
        self.W_x = LowRankLinear(dim, dim, rank)

        # Low-rank W_delta
        self.W_delta = LowRankLinear(dim, dim, rank)

        # Biases
        self.b = nn.Parameter(torch.zeros(dim))
        self.b_delta = nn.Parameter(torch.full((dim,), delta_init))

        # h+x selective gate bias
        self.b_gate = nn.Parameter(torch.zeros(dim))

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state, or None

        Returns:
            output: [T, B, dim] selective output
            h: [T+1, B, dim] hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.dim, device=x.device, dtype=x.dtype)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]

            # v = r_h * h_prev + W_x @ x + b (diagonal r_h, low-rank W_x)
            v = self.r_h * h_prev + self.W_x(x_t) + self.b
            candidate = torch.tanh(v)

            # Delta gate: sigmoid(W_delta @ x + r_delta * h_prev + b_delta)
            delta_raw = self.W_delta(x_t) + self.r_delta * h_prev + self.b_delta
            delta = torch.sigmoid(delta_raw)

            # Gated update
            h_new = (1 - delta) * h_prev + delta * candidate

            # h+x selective output: output = h * silu(h + x + b_gate)
            gate = F.silu(h_new + x_t + self.b_gate)
            output = h_new * gate

            h_list.append(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)

        return output, h


class LowRankElman(nn.Module):
    """
    Low-Rank Elman layer for use in LadderLM.

    Level 8: Uses low-rank W decomposition for very large hidden states.

    The key insight from Mamba2: you can have a big projection space
    with a small recurrent "bottleneck". Here we use rank << d_inner
    to achieve the same effect.

    Args:
        dim: Model dimension
        expansion: Hidden state expansion factor (can be large, e.g., 4-8x!)
        rank: Rank for W matrix decomposition
        delta_init: Initial bias for delta gate
        dropout: Dropout rate
    """

    def __init__(self, dim, expansion=4.0, rank=128, delta_init=-2.0,
                 dropout=0.0, **kwargs):
        # Level 8 ignores the default expansion from configs - we always use 4x
        # because low-rank W matrices allow much larger hidden states
        expansion = max(expansion, 4.0)  # Force at least 4x expansion
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.rank = rank

        # Input projection (still full-rank for expressivity)
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Low-rank Elman cell with big hidden state
        self.cell = LowRankElmanCell(
            self.d_inner, rank=rank, delta_init=delta_init
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, h0=None):
        """
        Args:
            x: [B, T, dim] input sequence
            h0: [B, d_inner] initial hidden state or None

        Returns:
            output: [B, T, dim] output for residual connection
            h_final: [B, d_inner] final hidden state for TBPTT
        """
        B, T, D = x.shape

        # Input projection
        x_proj = self.in_proj(x)  # [B, T, d_inner]

        # Transpose to [T, B, d_inner] for cell
        x_proj = x_proj.transpose(0, 1).contiguous()

        # Run cell
        output, h = self.cell(x_proj, h0)

        # Transpose back
        output = output.transpose(0, 1).contiguous()

        # Output projection
        output = self.out_proj(output)
        output = self.dropout(output)

        # Final hidden state for TBPTT
        h_final = h[-1]

        return output, h_final


__all__ = [
    'LowRankElmanCell',
    'LowRankElman',
    'LowRankLinear',
    'LEVEL_8_AVAILABLE',
]


if __name__ == "__main__":
    print("Testing Level 8 (Low-Rank Elman)...")

    dim = 768

    # Compare param counts at different expansions and ranks
    print(f"\nParam counts for dim={dim}:")
    print("-" * 60)

    for expansion in [1.0, 2.0, 4.0, 8.0]:
        for rank in [32, 64, 128]:
            model = LowRankElman(dim, expansion=expansion, rank=rank)
            params = sum(p.numel() for p in model.parameters())
            d_inner = int(dim * expansion)
            print(f"expansion={expansion:.1f} (d_inner={d_inner}), rank={rank}: {params:,} params")

    print()

    # Forward pass test
    model = LowRankElman(dim, expansion=4.0, rank=64)
    x = torch.randn(2, 16, dim)
    out, h = model(x)
    print(f"Forward pass: input {x.shape} -> output {out.shape}, hidden {h.shape}")
    print("OK!")
