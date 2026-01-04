"""
SSM Elman - Mamba2-style State Space Model with Elman mixing

Cell Architecture:
    B_t = W_B @ x_t                      -- input projection
    C_t = W_C @ x_t                      -- output projection
    dt_t = softplus(W_dt @ x_t + b_dt)   -- input-dependent timestep
    A = sigmoid(a_log)                   -- learned diagonal decay (0-1)
    h_t = A * h_{t-1} + W_h @ h_{t-1} + dt_t * B_t  -- SSM + Elman mixing
    y_t = C_t * h_t                      -- selective output

Layer Architecture (SSMElman wrapper):
    - in_proj: project input to d_inner
    - SSMElmanCell: run the SSM
    - out_proj: project back to dim
    - output_gate: output * silu(W_gate @ x + b_gate)

Key features:
    - Diagonal A (stable decay)
    - W_h @ h (cross-dimension mixing like Elman)
    - Input-dependent B, C, dt (selectivity like Mamba2)
    - Learned output gate (like other Elman levels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Haste CUDA kernel
import os
REQUIRE_CUDA = os.environ.get('ELMAN_REQUIRE_CUDA', '0') == '1'

try:
    import hasty_pytorch_lib
    HASTE_AVAILABLE = hasattr(hasty_pytorch_lib, 'ssm_elman_forward')
except ImportError as e:
    if REQUIRE_CUDA:
        raise ImportError(f"CUDA kernels required but not available: {e}")
    HASTE_AVAILABLE = False

SSM_ELMAN_AVAILABLE = True


class SSMElmanFunction(torch.autograd.Function):
    """Autograd function for SSM Elman."""

    @staticmethod
    def forward(ctx, training, x, h0, W_B, W_C, W_h, W_dt, a_log, b_dt):
        h, y, B_proj_cache, C_proj_cache, dt_cache = hasty_pytorch_lib.ssm_elman_forward(
            training,
            x.contiguous(),
            h0.contiguous(),
            W_B.contiguous(),
            W_C.contiguous(),
            W_h.contiguous(),
            W_dt.contiguous(),
            a_log.contiguous(),
            b_dt.contiguous()
        )
        if training:
            ctx.save_for_backward(W_B, W_C, W_h, W_dt, a_log, x, h, B_proj_cache, C_proj_cache, dt_cache)
        return y, h

    @staticmethod
    def backward(ctx, dy, dh_unused):
        W_B, W_C, W_h, W_dt, a_log, x, h, B_proj_cache, C_proj_cache, dt_cache = ctx.saved_tensors
        dx, dW_B, dW_C, dW_h, dW_dt, da_log, db_dt = hasty_pytorch_lib.ssm_elman_backward(
            W_B, W_C, W_h, W_dt, a_log, x, h, B_proj_cache, C_proj_cache, dt_cache, dy.contiguous()
        )
        return None, dx, None, dW_B, dW_C, dW_h, dW_dt, da_log, db_dt


class SSMElmanCell(nn.Module):
    """
    SSM Elman cell - SSM with cross-dimension mixing.

    Update equations:
        B = W_B @ x                      (input projection)
        C = W_C @ x                      (output projection)
        dt = softplus(W_dt @ x + b_dt)   (input-dependent timestep)
        A = sigmoid(a_log)               (diagonal decay)
        h_t = A * h_{t-1} + W_h @ h_{t-1} + dt * B  (SSM + Elman mixing)
        y_t = C * h_t                    (selective output)

    Args:
        dim: Hidden dimension
        dt_init: Initial value for dt bias (controls initial decay)
    """

    def __init__(self, dim, dt_init=1.0):
        super().__init__()
        self.dim = dim

        # Input-dependent projections
        self.W_B = nn.Parameter(torch.empty(dim, dim))
        self.W_C = nn.Parameter(torch.empty(dim, dim))
        self.W_dt = nn.Parameter(torch.empty(dim, dim))

        # Hidden-to-hidden mixing (cross-dimension, like Elman)
        self.W_h = nn.Parameter(torch.empty(dim, dim))

        # Diagonal decay (stored as log before sigmoid)
        # Initialize to give A ≈ 0.9 (decay of 0.1 per step)
        self.a_log = nn.Parameter(torch.zeros(dim))

        # dt bias (controls initial timestep scale)
        self.b_dt = nn.Parameter(torch.full((dim,), dt_init))

        self.dt_init = dt_init
        self._init_weights()

    def _init_weights(self):
        # Initialize projections
        nn.init.xavier_uniform_(self.W_B)
        nn.init.xavier_uniform_(self.W_C)
        nn.init.xavier_uniform_(self.W_dt, gain=0.1)  # Small to start with moderate dt
        nn.init.xavier_uniform_(self.W_h, gain=0.05)  # Small for stability

        # a_log initialized to 0 gives A = sigmoid(0) = 0.5
        # Start with lower A to leave room for W_h contribution
        nn.init.constant_(self.a_log, 0.0)  # A = 0.5

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state (optional)

        Returns:
            y: [T, B, dim] output
            h: [T+1, B, dim] all hidden states including h0
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.dim, device=x.device, dtype=x.dtype)

        # Use CUDA kernel
        if HASTE_AVAILABLE and x.is_cuda:
            return SSMElmanFunction.apply(
                self.training, x, h0,
                self.W_B, self.W_C, self.W_h, self.W_dt, self.a_log, self.b_dt
            )

        if REQUIRE_CUDA:
            raise RuntimeError("CUDA kernels required (ELMAN_REQUIRE_CUDA=1) but not available")

        # PyTorch fallback
        return self._forward_pytorch(x, h0)

    def _forward_pytorch(self, x, h0):
        """Pure PyTorch implementation."""
        T, B, D = x.shape
        h_list = [h0]
        y_list = []

        # Pre-compute A = sigmoid(a_log)
        A = torch.sigmoid(self.a_log)  # [dim]

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]

            # Input-dependent projections
            B_t = x_t @ self.W_B.T           # [B, dim]
            C_t = x_t @ self.W_C.T           # [B, dim]
            dt_raw = x_t @ self.W_dt.T + self.b_dt  # [B, dim]
            dt_t = F.softplus(dt_raw)        # [B, dim]

            # Hidden mixing
            H_t = h_prev @ self.W_h.T        # [B, dim]

            # SSM + Elman: h_t = A * h_prev + W_h @ h_prev + dt * B
            h_new = A * h_prev + H_t + dt_t * B_t
            h_list.append(h_new)

            # Selective output: y = C * h
            y_t = C_t * h_new
            y_list.append(y_t)

        h = torch.stack(h_list, dim=0)
        y = torch.stack(y_list, dim=0)
        return y, h


class SSMElman(nn.Module):
    """
    SSM Elman layer - Mamba2-style diagonal SSM with learned output gate.

    Wraps SSMElmanCell with input/output projections for use in LM.
    Final output is gated by silu(W_gate @ x + b_gate).
    """

    def __init__(self, dim, expansion=1.0, dropout=0.0, dt_init=1.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # SSM cell
        self.cell = SSMElmanCell(self.d_inner, dt_init=dt_init)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # Learned output gate: silu(W_gate @ x + b_gate)
        self.W_gate = nn.Parameter(torch.empty(dim, dim))
        self.b_gate = nn.Parameter(torch.zeros(dim))

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.W_gate)

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

        # Apply learned output gate: output * silu(W_gate @ x + b_gate)
        gate = F.silu(x @ self.W_gate.T + self.b_gate)  # [B, T, dim]
        output = output * gate

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, LEVEL=SSM'


__all__ = ['SSMElman', 'SSMElmanCell', 'SSM_ELMAN_AVAILABLE']


if __name__ == "__main__":
    print("Testing SSM Elman (Mamba2-style diagonal SSM)...")
    print("=" * 60)
    print(f"Haste CUDA kernel available: {HASTE_AVAILABLE}")

    # Test layer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SSMElman(dim=512, expansion=2.0).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Verify architecture
    print(f"\nCell Architecture:")
    print(f"  W_B shape: {model.cell.W_B.shape} (input -> B)")
    print(f"  W_C shape: {model.cell.W_C.shape} (input -> C)")
    print(f"  W_h shape: {model.cell.W_h.shape} (hidden -> hidden mixing)")
    print(f"  W_dt shape: {model.cell.W_dt.shape} (input -> dt)")
    print(f"  a_log shape: {model.cell.a_log.shape} (diagonal decay)")
    print(f"  A (decay) mean: {torch.sigmoid(model.cell.a_log).mean():.4f}")
    print(f"\nLayer Gate:")
    print(f"  W_gate shape: {model.W_gate.shape} (output gate)")
    print(f"  b_gate shape: {model.b_gate.shape} (output gate bias)")

    print("\nGradients (cell):")
    for name, p in model.cell.named_parameters():
        if p.grad is not None:
            print(f"  {name}: norm={p.grad.float().norm():.4f}")
    print("\nGradients (layer gate):")
    if model.W_gate.grad is not None:
        print(f"  W_gate: norm={model.W_gate.grad.float().norm():.4f}")
    if model.b_gate.grad is not None:
        print(f"  b_gate: norm={model.b_gate.grad.float().norm():.4f}")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params:,}")
    print("SSM Elman test passed!")
