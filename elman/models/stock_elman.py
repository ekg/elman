"""
Level 0: Stock Elman - Basic tanh recurrence

h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)

This is the simplest recurrent cell - just tanh activation on linear
combination of input and previous hidden state. No gating, no output
selectivity, no log-space.

Serves as baseline for ablation ladder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Haste CUDA kernel
import os
REQUIRE_CUDA = os.environ.get('ELMAN_REQUIRE_CUDA', '0') == '1'

try:
    import hasty_pytorch_lib
    HASTE_AVAILABLE = hasattr(hasty_pytorch_lib, 'stock_elman_forward')
except ImportError as e:
    if REQUIRE_CUDA:
        raise ImportError(f"CUDA kernels required but not available: {e}")
    HASTE_AVAILABLE = False

LEVEL_0_AVAILABLE = True  # PyTorch fallback always available


class StockElmanFunction(torch.autograd.Function):
    """Autograd function for Stock Elman (Haste kernel)."""

    @staticmethod
    def forward(ctx, training, x, h0, W_x, W_h, b):
        h, v = hasty_pytorch_lib.stock_elman_forward(
            training,
            x.contiguous(),
            h0.contiguous(),
            W_x.contiguous(),
            W_h.contiguous(),
            b.contiguous()
        )
        if training:
            ctx.save_for_backward(x, W_x, W_h, h, v)
        return h

    @staticmethod
    def backward(ctx, dh_out):
        x, W_x, W_h, h, v = ctx.saved_tensors
        dh = dh_out[1:].contiguous()
        dx, dW_x, dW_h, db = hasty_pytorch_lib.stock_elman_backward(
            W_x, W_h, x, h, v, dh
        )
        return None, dx, None, dW_x, dW_h, db


class StockElmanCell(nn.Module):
    """
    Stock Elman cell - Level 0 of ablation ladder.

    h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)

    Args:
        dim: Hidden dimension
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Full weight matrices
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        # Xavier init for stability
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_h)

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state (optional)

        Returns:
            h: [T+1, B, dim] all hidden states including h0
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.dim, device=x.device, dtype=x.dtype)

        # Use Haste kernel if available
        if HASTE_AVAILABLE and x.is_cuda:
            return StockElmanFunction.apply(
                self.training, x, h0,
                self.W_x, self.W_h, self.b
            )

        # PyTorch fallback
        if REQUIRE_CUDA:
            raise RuntimeError("CUDA kernels required (ELMAN_REQUIRE_CUDA=1) but not available")
        return self._forward_pytorch(x, h0)

    def _forward_pytorch(self, x, h0):
        """Pure PyTorch implementation."""
        T, B, D = x.shape
        h_list = [h0]

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]

            # h_t = tanh(W_x @ x + W_h @ h + b)
            raw = x_t @ self.W_x.T + h_prev @ self.W_h.T + self.b
            h_new = torch.tanh(raw)
            h_list.append(h_new)

        return torch.stack(h_list, dim=0)


class StockElman(nn.Module):
    """
    Stock Elman layer - Level 0 with projections.

    Wraps StockElmanCell with input/output projections for use in LM.
    NO gating, NO output selectivity - pure baseline.
    """

    def __init__(self, dim, expansion=1.0, dropout=0.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Elman cell
        self.cell = StockElmanCell(self.d_inner)

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
        h_all = self.cell(x_rnn, h0)  # [T+1, B, d_inner]
        h_out = h_all[1:]  # [T, B, d_inner]
        h_final = h_all[-1]  # [B, d_inner]

        # Transpose back: [B, T, d_inner]
        h_out = h_out.permute(1, 0, 2).contiguous()

        # Apply dropout and project
        h_out = self.dropout(h_out)
        output = self.out_proj(h_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, LEVEL=0_STOCK'


if __name__ == "__main__":
    print("Testing StockElman (Level 0)...")
    print("=" * 60)
    print(f"Haste CUDA kernel available: {HASTE_AVAILABLE}")

    # Test layer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = StockElman(dim=512, expansion=2.0).to(device).bfloat16()
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
    print("Level 0 (Stock Elman) test passed!")
