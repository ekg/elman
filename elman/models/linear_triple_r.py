"""
Level 5: Linear Triple R - Three R matrices for maximum expressivity

Uses three full recurrence matrices:
    v = R_x @ x_t + R_h @ h_{t-1} + b
    delta = sigmoid(W_delta @ x_t + R_delta @ h_{t-1} + b_delta)
    h_t = (1 - delta) * h_{t-1} + delta * tanh(v)

R_x: Input transformation (replaces W_x)
R_h: Hidden recurrence
R_delta: Hidden-to-gate modulation

Output selectivity unchanged:
    compete = softmax(h_t.view(groups), dim=-1)
    output = compete * silu(W_out @ h_t)

This is the most expressive linear-space architecture.
Key question: Does triple R improve over single R_h?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Haste CUDA kernel
try:
    import hasty_pytorch_lib
    HASTE_AVAILABLE = hasattr(hasty_pytorch_lib, 'linear_triple_r_forward')
except ImportError:
    HASTE_AVAILABLE = False

LEVEL_5_AVAILABLE = True  # PyTorch fallback always available


class LinearTripleRFunction(torch.autograd.Function):
    """Autograd function for Linear Triple R Elman (Haste kernel)."""

    @staticmethod
    def forward(ctx, training, x, h0, R_h, R_x, R_delta, W_delta, W_out, b, b_delta, n_groups):
        h, output, v, delta_cache, compete_cache = hasty_pytorch_lib.linear_triple_r_forward(
            training,
            x.contiguous(),
            h0.contiguous(),
            R_h.contiguous(),
            R_x.contiguous(),
            R_delta.contiguous(),
            W_delta.contiguous(),
            W_out.contiguous(),
            b.contiguous(),
            b_delta.contiguous(),
            n_groups
        )
        if training:
            ctx.save_for_backward(x, R_h, R_x, R_delta, W_delta, W_out, h, v, delta_cache, compete_cache)
            ctx.n_groups = n_groups
        return h, output

    @staticmethod
    def backward(ctx, dh_out, d_output):
        x, R_h, R_x, R_delta, W_delta, W_out, h, v, delta_cache, compete_cache = ctx.saved_tensors
        dx, dR_h, dR_x, dR_delta, dW_delta, dW_out, db, db_delta = hasty_pytorch_lib.linear_triple_r_backward(
            R_h, R_x, R_delta, W_delta, W_out, x, h, v, delta_cache, compete_cache,
            d_output.contiguous(), ctx.n_groups
        )
        return None, dx, None, dR_h, dR_x, dR_delta, dW_delta, dW_out, db, db_delta, None


class LinearTripleRCell(nn.Module):
    """
    Linear Triple R cell - Level 5 of ablation ladder.

    Recurrence (three R matrices):
        v = R_x @ x_t + R_h @ h_{t-1} + b
        delta = sigmoid(W_delta @ x_t + R_delta @ h_{t-1} + b_delta)
        h_t = (1 - delta) * h_{t-1} + delta * tanh(v)

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

        # Triple R matrices
        self.R_h = nn.Parameter(torch.empty(dim, dim))  # Hidden recurrence
        self.R_x = nn.Parameter(torch.empty(dim, dim))  # Input transformation
        self.R_delta = nn.Parameter(torch.empty(dim, dim))  # Hidden-to-delta
        self.b = nn.Parameter(torch.zeros(dim))

        # Delta (gate) computation
        self.W_delta = nn.Parameter(torch.empty(dim, dim))
        self.b_delta = nn.Parameter(torch.full((dim,), delta_init))

        # Output projection
        self.W_out = nn.Parameter(torch.empty(dim, dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.R_x)
        nn.init.xavier_uniform_(self.R_h, gain=0.1)  # Small for stability
        nn.init.xavier_uniform_(self.R_delta, gain=0.1)
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
            return LinearTripleRFunction.apply(
                self.training, x, h0,
                self.R_h, self.R_x, self.R_delta, self.W_delta, self.W_out,
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

            # Candidate with triple R
            v = x_t @ self.R_x.T + h_prev @ self.R_h.T + self.b
            candidate = torch.tanh(v)

            # Delta gate with R_delta modulation
            delta_raw = x_t @ self.W_delta.T + h_prev @ self.R_delta.T + self.b_delta
            delta = torch.sigmoid(delta_raw)

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


class LinearTripleR(nn.Module):
    """
    Linear Triple R layer - Level 5 with projections.

    Uses three full R matrices for maximum expressivity in linear space.
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

        # Linear Triple R cell
        self.cell = LinearTripleRCell(
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
        return f'dim={self.dim}, d_inner={self.d_inner}, n_groups={self.n_groups}, LEVEL=5_TRIPLE_R'


if __name__ == "__main__":
    print("Testing LinearTripleR (Level 5)...")
    print("=" * 60)
    print(f"Haste CUDA kernel available: {HASTE_AVAILABLE}")

    # Test layer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LinearTripleR(dim=512, expansion=2.0, n_groups=32).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Verify all R matrices exist
    print(f"R_h shape: {model.cell.R_h.shape}")
    print(f"R_x shape: {model.cell.R_x.shape}")
    print(f"R_delta shape: {model.cell.R_delta.shape}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print("Level 5 (Linear Triple R) test passed!")
