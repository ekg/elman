"""
Level 0: Pure Elman - Basic tanh recurrence with NO output gating

h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
output_t = h_t  (direct output, no gating)

This is the true simplest recurrent cell - just tanh activation.
No gating, no output selectivity, no log-space.

Uses haste_pytorch_lib (6-arg version without b_gate).
"""

import torch
import torch.nn as nn

# Try to import Haste CUDA kernel (the 6-arg pure version)
import os
REQUIRE_CUDA = os.environ.get('ELMAN_REQUIRE_CUDA', '0') == '1'

try:
    import haste_pytorch_lib
    HASTE_AVAILABLE = hasattr(haste_pytorch_lib, 'stock_elman_forward')
except ImportError as e:
    if REQUIRE_CUDA:
        raise ImportError(f"CUDA kernels required but not available: {e}")
    HASTE_AVAILABLE = False

PURE_ELMAN_AVAILABLE = True


class PureElmanFunction(torch.autograd.Function):
    """Autograd function for Pure Elman (no output gating)."""

    @staticmethod
    def forward(ctx, training, x, h0, W_x, W_h, b):
        # haste_pytorch_lib.stock_elman_forward returns [h, v]
        h, v = haste_pytorch_lib.stock_elman_forward(
            training,
            x.contiguous(),
            h0.contiguous(),
            W_x.contiguous(),
            W_h.contiguous(),
            b.contiguous()
        )
        # Output is just h[1:] (hidden states, no gating)
        output = h[1:]
        if training:
            ctx.save_for_backward(x, W_x, W_h, h, v)
        return output, h

    @staticmethod
    def backward(ctx, d_output, dh_unused):
        x, W_x, W_h, h, v = ctx.saved_tensors
        dx, dW_x, dW_h, db = haste_pytorch_lib.stock_elman_backward(
            W_x, W_h, x, h, v, d_output.contiguous()
        )
        return None, dx, None, dW_x, dW_h, db


class PureElmanCell(nn.Module):
    """
    Pure Elman cell - true Level 0 with NO output gating.

    h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
    output_t = h_t

    Args:
        dim: Hidden dimension
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Input projection
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        # Recurrence matrix
        self.W_h = nn.Parameter(torch.empty(dim, dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_h, gain=0.5)

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state (optional)

        Returns:
            output: [T, B, dim] = hidden states (no gating)
            h: [T+1, B, dim] all hidden states including h0
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.dim, device=x.device, dtype=x.dtype)

        # Use CUDA kernel
        if HASTE_AVAILABLE and x.is_cuda:
            return PureElmanFunction.apply(
                self.training, x, h0,
                self.W_x, self.W_h, self.b
            )

        if REQUIRE_CUDA:
            raise RuntimeError("CUDA kernels required (ELMAN_REQUIRE_CUDA=1) but not available")

        # PyTorch fallback (shouldn't be used per user request)
        raise RuntimeError("Pure Elman requires CUDA kernel - no PyTorch fallback")


class PureElman(nn.Module):
    """
    Pure Elman layer - true Level 0 with NO output gating.

    Wraps PureElmanCell with input/output projections for use in LM.
    Output is just the hidden state - no selectivity.
    """

    def __init__(self, dim, expansion=1.0, dropout=0.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Pure Elman cell (no gating)
        self.cell = PureElmanCell(self.d_inner)

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

        # Run cell - returns (output, h_all) - output = h directly
        cell_out, h_all = self.cell(x_rnn, h0)  # [T, B, d_inner], [T+1, B, d_inner]
        h_final = h_all[-1]  # [B, d_inner]

        # Transpose back: [B, T, d_inner]
        cell_out = cell_out.permute(1, 0, 2).contiguous()

        # Apply dropout and project
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, LEVEL=0_PURE'


__all__ = ['PureElman', 'PureElmanCell', 'PURE_ELMAN_AVAILABLE']


if __name__ == "__main__":
    print("Testing PureElman (True Level 0 - NO output gating)...")
    print("=" * 60)

    # Test layer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PureElman(dim=512, expansion=2.0).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print("Pure Elman (no output gating) test passed!")
