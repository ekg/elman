"""
Auto Elman - Autonomous hidden state with input-only gating

Architecture:
    h_t = tanh(W_h @ h_{t-1} + b_h)           -- hidden evolves autonomously (NO input)
    output_t = h_t * silu(W_gate @ x_t + b_gate)  -- input only selects output

Key insight: Tests whether input needs to affect state directly,
or if it's sufficient for input to just SELECT which hidden dims to expose.

The hidden state is a auto dynamical system - it evolves independently of input.
Input only affects the output through the gating mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Haste CUDA kernel
import os
REQUIRE_CUDA = os.environ.get('ELMAN_REQUIRE_CUDA', '0') == '1'

try:
    import hasty_pytorch_lib
    HASTE_AVAILABLE = hasattr(hasty_pytorch_lib, 'auto_elman_forward')
except ImportError as e:
    if REQUIRE_CUDA:
        raise ImportError(f"CUDA kernels required but not available: {e}")
    HASTE_AVAILABLE = False

AUTO_ELMAN_AVAILABLE = True


class AutoElmanFunction(torch.autograd.Function):
    """Autograd function for Auto Elman (autonomous hidden, input-only gating)."""

    @staticmethod
    def forward(ctx, training, x, h0, W_h, W_gate, b_h, b_gate):
        h, output, v, gate_cache = hasty_pytorch_lib.auto_elman_forward(
            training,
            x.contiguous(),
            h0.contiguous(),
            W_h.contiguous(),
            W_gate.contiguous(),
            b_h.contiguous(),
            b_gate.contiguous()
        )
        if training:
            ctx.save_for_backward(x, W_h, W_gate, h, v, gate_cache)
        return output, h

    @staticmethod
    def backward(ctx, d_output, dh_unused):
        x, W_h, W_gate, h, v, gate_cache = ctx.saved_tensors
        dx, dW_h, dW_gate, db_h, db_gate = hasty_pytorch_lib.auto_elman_backward(
            W_h, W_gate, x, h, v, gate_cache, d_output.contiguous()
        )
        return None, dx, None, dW_h, dW_gate, db_h, db_gate


class AutoElmanCell(nn.Module):
    """
    Auto Elman cell - Autonomous hidden state with input-only gating.

    Hidden update (autonomous - NO input):
        h_t = tanh(W_h @ h_{t-1} + b_h)

    Output (input-only gating):
        output_t = h_t * silu(W_gate @ x_t + b_gate)

    Args:
        dim: Hidden dimension
        w_h_init_gain: Initial gain for W_h (spectral radius control)
    """

    def __init__(self, dim, w_h_init_gain=0.9):
        super().__init__()
        self.dim = dim

        # Hidden-to-hidden (autonomous dynamics)
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        # Initialize b_h non-zero so autonomous dynamics can bootstrap from h0=0
        self.b_h = nn.Parameter(torch.empty(dim))

        # Gate projection (input → gate)
        self.W_gate = nn.Parameter(torch.empty(dim, dim))
        self.b_gate = nn.Parameter(torch.zeros(dim))

        self.w_h_init_gain = w_h_init_gain
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_h, gain=self.w_h_init_gain)
        nn.init.xavier_uniform_(self.W_gate)
        # Initialize b_h to small random values so h can bootstrap from h0=0
        nn.init.uniform_(self.b_h, -0.1, 0.1)

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state (optional)

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, dim] all hidden states including h0
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.dim, device=x.device, dtype=x.dtype)

        # Use CUDA kernel
        if HASTE_AVAILABLE and x.is_cuda:
            return AutoElmanFunction.apply(
                self.training, x, h0,
                self.W_h, self.W_gate, self.b_h, self.b_gate
            )

        if REQUIRE_CUDA:
            raise RuntimeError("CUDA kernels required (ELMAN_REQUIRE_CUDA=1) but not available")

        # PyTorch fallback
        return self._forward_pytorch(x, h0)

    def _forward_pytorch(self, x, h0):
        """Auto PyTorch implementation."""
        T, B, D = x.shape
        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]

            # Autonomous hidden update: h_t = tanh(W_h @ h + b_h)
            # NO input in hidden state!
            raw = h_prev @ self.W_h.T + self.b_h
            h_new = torch.tanh(raw)
            h_list.append(h_new)

            # Input-only gating: output = h * silu(W_gate @ x + b_gate)
            gate_proj = x_t @ self.W_gate.T + self.b_gate
            gate = F.silu(gate_proj)
            output = h_new * gate
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class AutoElman(nn.Module):
    """
    Auto Elman layer - Autonomous hidden with input-only gating.

    Wraps AutoElmanCell with input/output projections for use in LM.
    The hidden state evolves independently - input only affects output selection.
    """

    def __init__(self, dim, expansion=1.0, dropout=0.0,
                 w_h_init_gain=0.9, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Auto Elman cell (autonomous hidden, input-only gating)
        self.cell = AutoElmanCell(self.d_inner, w_h_init_gain=w_h_init_gain)

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
        cell_out, h_all = self.cell(x_rnn, h0)
        h_final = h_all[-1]

        # Transpose back: [B, T, d_inner]
        cell_out = cell_out.permute(1, 0, 2).contiguous()

        # Apply dropout and project
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, LEVEL=AUTO_AUTONOMOUS'


__all__ = ['AutoElman', 'AutoElmanCell', 'AUTO_ELMAN_AVAILABLE']


if __name__ == "__main__":
    print("Testing AutoElman (Autonomous Hidden, Input-Only Gating)...")
    print("=" * 60)
    print(f"Haste CUDA kernel available: {HASTE_AVAILABLE}")

    # Test layer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoElman(dim=512, expansion=2.0).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Verify architecture
    print(f"\nArchitecture:")
    print(f"  W_h shape: {model.cell.W_h.shape} (hidden-to-hidden)")
    print(f"  W_gate shape: {model.cell.W_gate.shape} (input→gate)")
    print(f"  Note: NO W_x - input doesn't affect hidden state!")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params:,}")
    print("Auto Elman test passed!")
