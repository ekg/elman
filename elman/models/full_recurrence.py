"""
Level 4: Full Recurrence - Full R_h matrix (instead of diagonal)

Same as Level 3 (Diagonal Selective), but with FULL R_h matrix:
    delta = sigmoid(W_delta @ x_t + b_delta)
    h_t = (1 - delta) * h_{t-1} + delta * tanh(W_x @ x_t + R_h @ h_{t-1} + b)

The hidden-to-hidden transition uses a full dense matrix.
This increases expressivity vs diagonal but potentially less stable.

Output selectivity unchanged:
    compete = softmax(h_t.view(groups), dim=-1)
    output = compete * silu(W_out @ h_t)

Key question: Does full R_h improve over diagonal r_h at the cost of stability?
"""

import torch
import os
REQUIRE_CUDA = os.environ.get('ELMAN_REQUIRE_CUDA', '0') == '1'
import torch.nn as nn
import torch.nn.functional as F

# Try to import Haste CUDA kernel
try:
    import hasty_pytorch_lib
    HASTE_AVAILABLE = hasattr(hasty_pytorch_lib, 'full_recurrence_forward')
except ImportError:
    HASTE_AVAILABLE = False

LEVEL_4_AVAILABLE = True  # PyTorch fallback always available


class FullRecurrenceFunction(torch.autograd.Function):
    """Autograd function for Full Recurrence Elman (Haste kernel)."""

    @staticmethod
    def forward(ctx, training, x, h0, W_x, R_h, W_delta, W_out, b, b_delta, n_groups):
        h, output, v, delta_cache, compete_cache = hasty_pytorch_lib.full_recurrence_forward(
            training,
            x.contiguous(),
            h0.contiguous(),
            W_x.contiguous(),
            R_h.contiguous(),
            W_delta.contiguous(),
            W_out.contiguous(),
            b.contiguous(),
            b_delta.contiguous(),
            n_groups
        )
        if training:
            ctx.save_for_backward(x, W_x, R_h, W_delta, W_out, h, v, delta_cache, compete_cache)
            ctx.n_groups = n_groups
        return h, output

    @staticmethod
    def backward(ctx, dh_out, d_output):
        x, W_x, R_h, W_delta, W_out, h, v, delta_cache, compete_cache = ctx.saved_tensors
        dx, dW_x, dR_h, dW_delta, dW_out, db, db_delta = hasty_pytorch_lib.full_recurrence_backward(
            W_x, R_h, W_delta, W_out, x, h, v, delta_cache, compete_cache,
            d_output.contiguous(), ctx.n_groups
        )
        return None, dx, None, dW_x, dR_h, dW_delta, dW_out, db, db_delta, None


class FullRecurrenceCell(nn.Module):
    """
    Full Recurrence cell - Level 4 of ablation ladder.

    Recurrence (full R_h matrix):
        delta = sigmoid(W_delta @ x_t + b_delta)
        h_t = (1 - delta) * h_{t-1} + delta * tanh(W_x @ x_t + R_h @ h_{t-1} + b)

    Output selectivity:
        compete = softmax(h_t.view(n_groups, group_size), dim=-1)
        output = compete * silu(W_out @ h_t)

    Args:
        dim: Hidden dimension
        n_groups: Number of groups for compete softmax
        delta_init: Initial bias for delta gate
    """

    def __init__(self, dim, n_groups=32, delta_init=-2.0):
        super().__init__()
        self.dim = dim
        self.n_groups = n_groups
        self.group_size = dim // n_groups

        assert dim % n_groups == 0, f"dim ({dim}) must be divisible by n_groups ({n_groups})"

        # Candidate computation weights
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        # FULL R_h matrix (unlike diagonal in Level 3)
        self.R_h = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        # Delta (gate) computation
        self.W_delta = nn.Parameter(torch.empty(dim, dim))
        self.b_delta = nn.Parameter(torch.full((dim,), delta_init))

        # Output projection
        self.W_out = nn.Parameter(torch.empty(dim, dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x)
        # Initialize R_h with small values for stability
        nn.init.xavier_uniform_(self.R_h, gain=0.1)
        nn.init.xavier_uniform_(self.W_delta, gain=0.1)
        nn.init.xavier_uniform_(self.W_out)

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state

        Returns:
            h: [T+1, B, dim] all hidden states including h0
            output: [T, B, dim] selective outputs
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.dim, device=x.device, dtype=x.dtype)

        # Use Haste kernel if available
        if HASTE_AVAILABLE and x.is_cuda:
            return FullRecurrenceFunction.apply(
                self.training, x, h0,
                self.W_x, self.R_h, self.W_delta, self.W_out,
                self.b, self.b_delta, self.n_groups
            )

        # PyTorch fallback
        return self._forward_pytorch(x, h0)

    def _forward_pytorch(self, x, h0):
        """Pure PyTorch implementation."""
        T, B, D = x.shape
        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]

            # Delta gate
            delta_raw = x_t @ self.W_delta.T + self.b_delta
            delta = torch.sigmoid(delta_raw)

            # Candidate with FULL R_h matrix
            candidate_raw = x_t @ self.W_x.T + h_prev @ self.R_h.T + self.b
            candidate = torch.tanh(candidate_raw)

            # State update
            h_new = (1 - delta) * h_prev + delta * candidate
            h_list.append(h_new)

            # Output selectivity: compete x silu
            h_grouped = h_new.view(B, self.n_groups, self.group_size)
            compete = F.softmax(h_grouped, dim=-1)
            compete = compete.view(B, D)

            out_proj = h_new @ self.W_out.T
            output = compete * F.silu(out_proj)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return h, output


class FullRecurrence(nn.Module):
    """
    Full Recurrence layer - Level 4 with projections.

    Same as Level 3 but with full R_h matrix instead of diagonal r_h.
    This increases expressivity but may be less stable at depth.
    """

    def __init__(self, dim, expansion=1.0, n_groups=32, delta_init=-2.0, dropout=0.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_groups = n_groups

        # Adjust n_groups if needed
        while self.d_inner % self.n_groups != 0:
            self.n_groups -= 1

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Full Recurrence cell
        self.cell = FullRecurrenceCell(
            self.d_inner,
            n_groups=self.n_groups,
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

        # Run cell
        h_all, selective_out = self.cell(x_rnn, h0)
        h_final = h_all[-1]

        # Transpose back: [B, T, d_inner]
        selective_out = selective_out.permute(1, 0, 2).contiguous()

        # Apply dropout and project
        selective_out = self.dropout(selective_out)
        output = self.out_proj(selective_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, n_groups={self.n_groups}, LEVEL=4_FULL_RECURRENCE'


if __name__ == "__main__":
    print("Testing FullRecurrence (Level 4)...")
    print("=" * 60)
    print(f"Haste CUDA kernel available: {HASTE_AVAILABLE}")

    # Test layer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FullRecurrence(dim=512, expansion=2.0, n_groups=32).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Verify R_h is full matrix (not diagonal)
    print(f"R_h shape: {model.cell.R_h.shape} (should be [{model.d_inner}, {model.d_inner}])")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print("Level 4 (Full Recurrence) test passed!")
