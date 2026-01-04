"""
Level 3: Diagonal Selective - Diagonal r_h (like Mamba2's diagonal A)

Same recurrence as Level 2, but W_h becomes diagonal r_h:
    delta = sigmoid(W_delta @ x_t + b_delta)
    h_t = (1 - delta) * h_{t-1} + delta * tanh(W_x @ x_t + r_h * h_{t-1} + b)

The hidden-to-hidden transition is now element-wise (diagonal matrix).
This matches Mamba2's approach with diagonal A matrix.

With learned gate projection:
    output_t = h_t * silu(W_gate @ x_t + b_gate)

Key insight: Diagonal r_h prevents eigenvalue blowup at depth.
"""

import torch
import os
REQUIRE_CUDA = os.environ.get('ELMAN_REQUIRE_CUDA', '0') == '1'
import torch.nn as nn
import torch.nn.functional as F

# Try to import Haste CUDA kernel
try:
    import hasty_pytorch_lib
    HASTE_AVAILABLE = hasattr(hasty_pytorch_lib, 'diagonal_selective_forward')
except ImportError:
    HASTE_AVAILABLE = False

LEVEL_3_AVAILABLE = True  # PyTorch fallback always available


class DiagonalSelectiveFunction(torch.autograd.Function):
    """Autograd function for Diagonal Selective Elman with learned gate projection."""

    @staticmethod
    def forward(ctx, training, x, h0, W_x, r_h, W_delta, W_gate, b, b_delta, b_gate):
        h, output, v, delta_cache, gate_cache = hasty_pytorch_lib.diagonal_selective_forward(
            training,
            x.contiguous(),
            h0.contiguous(),
            W_x.contiguous(),
            r_h.contiguous(),
            W_delta.contiguous(),
            W_gate.contiguous(),
            b.contiguous(),
            b_delta.contiguous(),
            b_gate.contiguous()
        )
        if training:
            ctx.save_for_backward(x, W_x, r_h, W_delta, W_gate, h, v, delta_cache, gate_cache)
        return output, h

    @staticmethod
    def backward(ctx, d_output, dh_unused):
        x, W_x, r_h, W_delta, W_gate, h, v, delta_cache, gate_cache = ctx.saved_tensors
        dx, dW_x, dr_h, dW_delta, dW_gate, db, db_delta, db_gate = hasty_pytorch_lib.diagonal_selective_backward(
            W_x, r_h, W_delta, W_gate, x, h, v, delta_cache, gate_cache,
            d_output.contiguous()
        )
        return None, dx, None, dW_x, dr_h, dW_delta, dW_gate, db, db_delta, db_gate


class DiagonalSelectiveCell(nn.Module):
    """
    Diagonal Selective cell - Level 3 of ablation ladder.

    Recurrence (diagonal W_h):
        delta = sigmoid(W_delta @ x_t + b_delta)
        h_t = (1 - delta) * h_{t-1} + delta * tanh(W_x @ x_t + r_h * h_{t-1} + b)

    Learned gate projection output:
        output_t = h_t * silu(W_gate @ x_t + b_gate)

    Args:
        dim: Hidden dimension
        delta_init: Initial bias for delta gate
    """

    def __init__(self, dim, delta_init=-2.0):
        super().__init__()
        self.dim = dim

        # Candidate computation weights
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        # DIAGONAL: r_h is a vector, not a matrix
        self.r_h = nn.Parameter(torch.zeros(dim))  # Initialize to 0 for stability
        self.b = nn.Parameter(torch.zeros(dim))

        # Delta (gate) computation
        self.W_delta = nn.Parameter(torch.empty(dim, dim))
        self.b_delta = nn.Parameter(torch.full((dim,), delta_init))

        # Learned gate projection: output = h * silu(W_gate @ x + b_gate)
        self.W_gate = nn.Parameter(torch.empty(dim, dim))
        self.b_gate = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_gate)
        # r_h already initialized to 0 (stable default)
        nn.init.xavier_uniform_(self.W_delta, gain=0.1)

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] selective output with learned gate projection
            h: [T+1, B, dim] all hidden states including h0
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.dim, device=x.device, dtype=x.dtype)

        # Use Haste kernel if available
        if HASTE_AVAILABLE and x.is_cuda:
            return DiagonalSelectiveFunction.apply(
                self.training, x, h0,
                self.W_x, self.r_h, self.W_delta, self.W_gate,
                self.b, self.b_delta, self.b_gate
            )

        # PyTorch fallback
        return self._forward_pytorch(x, h0)

    def _forward_pytorch(self, x, h0):
        """Pure PyTorch implementation with learned gate projection."""
        T, B, D = x.shape
        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]

            # Delta gate
            delta_raw = x_t @ self.W_delta.T + self.b_delta
            delta = torch.sigmoid(delta_raw)

            # Candidate with DIAGONAL r_h (element-wise multiply)
            candidate_raw = x_t @ self.W_x.T + self.r_h * h_prev + self.b
            candidate = torch.tanh(candidate_raw)

            # State update
            h_new = (1 - delta) * h_prev + delta * candidate
            h_list.append(h_new)

            # Learned gate projection: output = h * silu(W_gate @ x + b_gate)
            gate_proj = x_t @ self.W_gate.T + self.b_gate
            gate = F.silu(gate_proj)
            output = h_new * gate
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class DiagonalSelective(nn.Module):
    """
    Diagonal Selective layer - Level 3 with projections.

    Same as Level 2 but with diagonal r_h instead of full W_h.
    This is a key architectural choice matching Mamba2's diagonal A.

    Uses learned gate projection: output = h * silu(W_gate @ x + b_gate)
    """

    def __init__(self, dim, expansion=1.0, delta_init=-2.0, dropout=0.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Diagonal Selective cell
        self.cell = DiagonalSelectiveCell(
            self.d_inner,
            delta_init=delta_init
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, h0=None, **kwargs):
        """
        Args:
            x: [B, T, dim] input sequence
            h0: [B, d_inner] initial hidden state

        Returns:
            output: [B, T, dim] output sequence
            h_final: [B, d_inner] final hidden state
        """
        B, T, D = x.shape

        # Project input
        x_proj = self.in_proj(x)  # [B, T, d_inner]

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        # Run cell - returns (output, h_all) with h+x gating
        cell_out, h_all = self.cell(x_rnn, h0)
        h_final = h_all[-1]

        # Transpose back: [B, T, d_inner]
        cell_out = cell_out.permute(1, 0, 2).contiguous()

        # Apply dropout and project
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, LEVEL=3_DIAGONAL'


if __name__ == "__main__":
    print("Testing DiagonalSelective (Level 3)...")
    print("=" * 60)
    print(f"Haste CUDA kernel available: {HASTE_AVAILABLE}")

    # Test layer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DiagonalSelective(dim=512, expansion=2.0).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Verify r_h is diagonal (vector, not matrix) and W_gate is matrix
    print(f"r_h shape: {model.cell.r_h.shape} (should be [{model.d_inner}])")
    print(f"W_gate shape: {model.cell.W_gate.shape} (should be [{model.d_inner}, {model.d_inner}])")
    print(f"b_gate shape: {model.cell.b_gate.shape} (should be [{model.d_inner}])")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print("Level 3 (Diagonal Selective) test passed!")
