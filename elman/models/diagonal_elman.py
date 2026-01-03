"""
Diagonal X-Gated Elman - tanh recurrence with diagonal decay + x-only selective output

h_t = tanh(W_x @ x_t + α ⊙ h_{t-1} + b)  -- KEEP tanh, diagonal α instead of dense W_h
output_t = h_t * silu(x_t + b_gate)  -- x-only selective gating

Key insight:
- Dense W_h is 30% of layer params, but diagonal α is 1000x smaller
- KEEP tanh for expressivity! (removing it hurts loss significantly)
- Faster than dense W_h (no W_h @ h GEMM in recurrence)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
import os
REQUIRE_CUDA = os.environ.get('ELMAN_REQUIRE_CUDA', '0') == '1'

try:
    import hasty_pytorch_lib
    HASTE_AVAILABLE = hasattr(hasty_pytorch_lib, 'diagonal_elman_forward')
except ImportError as e:
    if REQUIRE_CUDA:
        raise ImportError(f"CUDA kernels required but not available: {e}")
    HASTE_AVAILABLE = False

DIAGONAL_ELMAN_AVAILABLE = True  # PyTorch fallback always available


class DiagonalElmanFunction(torch.autograd.Function):
    """Autograd function for Diagonal Elman with CUDA kernel."""

    @staticmethod
    def forward(ctx, training, x, h0, W_x, alpha, b, b_gate):
        h, output, v_cache, gate_cache = hasty_pytorch_lib.diagonal_elman_forward(
            training,
            x.contiguous(),
            h0.contiguous(),
            W_x.contiguous(),
            alpha.contiguous(),
            b.contiguous(),
            b_gate.contiguous()
        )
        if training:
            ctx.save_for_backward(x, W_x, alpha, b_gate, h, v_cache, gate_cache)
        return output, h

    @staticmethod
    def backward(ctx, d_output, dh_unused):
        x, W_x, alpha, b_gate, h, v_cache, gate_cache = ctx.saved_tensors
        dx, dW_x, dalpha, db, d_b_gate = hasty_pytorch_lib.diagonal_elman_backward(
            W_x, alpha, b_gate, x, h, v_cache, gate_cache, d_output.contiguous()
        )
        return None, dx, None, dW_x, dalpha, db, d_b_gate


class DiagonalElmanCell(nn.Module):
    """
    Diagonal Elman cell with tanh + x-only gating.

    h_t = tanh(W_x @ x_t + α ⊙ h_{t-1} + b)  -- KEEP tanh!
    output_t = h_t * silu(x_t + b_gate)

    Args:
        dim: Hidden dimension
        alpha_init: Initial value for diagonal decay (default -2.0, sigmoid → ~0.12)
    """

    def __init__(self, dim, alpha_init=-2.0):
        super().__init__()
        self.dim = dim

        # Input projection (only dense matrix in recurrence!)
        self.W_x = nn.Parameter(torch.empty(dim, dim))

        # Diagonal decay - learnable per-dimension
        # Using log-sigmoid parameterization: α = sigmoid(alpha_raw)
        # This ensures α ∈ (0, 1) for stability
        self.alpha_raw = nn.Parameter(torch.full((dim,), alpha_init))

        # Recurrence bias
        self.b = nn.Parameter(torch.zeros(dim))

        # x-only selective gate bias
        self.b_gate = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x)

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state (optional)

        Returns:
            output: [T, B, dim] x-only selective output
            h: [T+1, B, dim] all hidden states including h0
        """
        T, B, D = x.shape
        device, dtype = x.device, x.dtype

        if h0 is None:
            h0 = torch.zeros(B, self.dim, device=device, dtype=dtype)

        # Use CUDA kernel if available
        if HASTE_AVAILABLE and x.is_cuda:
            return DiagonalElmanFunction.apply(
                self.training, x, h0,
                self.W_x, self.alpha_raw, self.b, self.b_gate
            )

        # PyTorch fallback (for validation/debugging only)
        if REQUIRE_CUDA:
            raise RuntimeError("CUDA kernels required (ELMAN_REQUIRE_CUDA=1) but not available")
        return self._forward_pytorch(x, h0)

    def _forward_pytorch(self, x, h0):
        """Pure PyTorch implementation for validation."""
        T, B, D = x.shape

        # Get diagonal decay (stable in (0, 1))
        alpha = torch.sigmoid(self.alpha_raw)  # [dim]

        # Pre-compute x projections for all timesteps
        x_proj = x @ self.W_x.T  # [T, B, dim]

        # Sequential recurrence
        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            # tanh recurrence: h = tanh(W_x @ x + α ⊙ h_prev + b)
            v = x_proj[t] + alpha * h_prev + self.b
            h_new = torch.tanh(v)
            h_list.append(h_new)

            # X-only selective output: output = h * silu(x + b_gate)
            gate = F.silu(x[t] + self.b_gate)
            output = h_new * gate
            output_list.append(output)

        h = torch.stack(h_list, dim=0)  # [T+1, B, dim]
        output = torch.stack(output_list, dim=0)  # [T, B, dim]

        return output, h


class DiagonalElman(nn.Module):
    """
    Diagonal Elman layer with tanh + x-only gating.

    h = tanh(W_x @ x + α ⊙ h_prev + b)  -- tanh kept, diagonal α
    output = h * silu(x + b_gate)       -- x-only gating

    Much more parameter-efficient than dense W_h, keeps tanh for expressivity.
    """

    def __init__(self, dim, expansion=1.0, dropout=0.0,
                 alpha_init=-2.0, use_conv=False, d_conv=4, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.use_conv = use_conv

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Optional conv1d for local context
        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner,
                bias=True,
            )
        else:
            self.conv1d = None

        # Diagonal Elman cell
        self.cell = DiagonalElmanCell(self.d_inner, alpha_init=alpha_init)

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

        # Optional conv1d for local context
        if self.use_conv and self.conv1d is not None:
            x_conv = x_proj.transpose(1, 2)  # [B, d_inner, T]
            x_conv = self.conv1d(x_conv)
            x_conv = x_conv[:, :, :T]  # Causal
            x_conv = x_conv.transpose(1, 2)
            x_proj = F.silu(x_conv)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        # Run cell
        cell_out, h_all = self.cell(x_rnn, h0)  # [T, B, d_inner], [T+1, B, d_inner]
        h_final = h_all[-1]  # [B, d_inner]

        # Transpose back: [B, T, d_inner]
        cell_out = cell_out.permute(1, 0, 2).contiguous()

        # Apply dropout and project
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, DIAGONAL_LINEAR, CUDA={HASTE_AVAILABLE}'


__all__ = ['DiagonalElman', 'DiagonalElmanCell', 'DIAGONAL_ELMAN_AVAILABLE']


if __name__ == "__main__":
    print("Testing DiagonalElman...")
    print("=" * 60)
    print(f"CUDA kernel available: {HASTE_AVAILABLE}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DiagonalElman(dim=512, expansion=2.0).to(device).bfloat16()
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

    print("\nDiagonalElman test passed!")
