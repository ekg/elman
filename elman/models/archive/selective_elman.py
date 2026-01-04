"""
Level 9: Selective Elman with Input-Dependent B, C, dt (like Mamba2)

Adds Mamba2-style input-dependent selectivity to stock Elman:
- B: input-dependent write gating (what to write to hidden state)
- C: input-dependent read gating (what to read from hidden state)
- dt: input-dependent decay/update rate (how much to update hidden state)

Architecture:
    # Input-dependent projections
    B = sigmoid(W_B @ x + b_B)           # write gate [0,1]
    C = sigmoid(W_C @ x + b_C)           # read gate [0,1]
    dt = sigmoid(W_dt @ x + b_dt)        # update rate [0,1]

    # Recurrence with B modulating write
    candidate = tanh(B * (W_x @ x) + W_h @ h_prev + b)
    h_new = (1 - dt) * h_prev + dt * candidate

    # Output with C modulating read
    output = C * h_new * silu(h_new + x + b_gate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Haste CUDA kernel
import os
REQUIRE_CUDA = os.environ.get('ELMAN_REQUIRE_CUDA', '0') == '1'

try:
    import haste_pytorch_lib
    HASTE_AVAILABLE = hasattr(haste_pytorch_lib, 'input_selective_elman_forward')
except ImportError as e:
    if REQUIRE_CUDA:
        raise ImportError(f"CUDA kernels required but not available: {e}")
    HASTE_AVAILABLE = False

SELECTIVE_ELMAN_AVAILABLE = True
LEVEL_2_AVAILABLE = True  # For backwards compat


class InputSelectiveElmanFunction(torch.autograd.Function):
    """Autograd function for Input-Selective Elman with B, C, dt gates."""

    @staticmethod
    def forward(ctx, training, x, h0, W_x, W_h, W_B, W_C, W_dt, b, b_B, b_C, b_dt, b_gate):
        h, output, v_cache, B_cache, C_cache, dt_cache = haste_pytorch_lib.input_selective_elman_forward(
            training,
            x.contiguous(),
            h0.contiguous(),
            W_x.contiguous(),
            W_h.contiguous(),
            W_B.contiguous(),
            W_C.contiguous(),
            W_dt.contiguous(),
            b.contiguous(),
            b_B.contiguous(),
            b_C.contiguous(),
            b_dt.contiguous(),
            b_gate.contiguous()
        )
        if training:
            ctx.save_for_backward(x, W_x, W_h, W_B, W_C, W_dt, b_gate,
                                  h, v_cache, B_cache, C_cache, dt_cache)
        return output, h

    @staticmethod
    def backward(ctx, d_output, dh_unused):
        x, W_x, W_h, W_B, W_C, W_dt, b_gate, h, v_cache, B_cache, C_cache, dt_cache = ctx.saved_tensors
        dx, dW_x, dW_h, dW_B, dW_C, dW_dt, db, db_B, db_C, db_dt, db_gate = haste_pytorch_lib.input_selective_elman_backward(
            W_x, W_h, W_B, W_C, W_dt, b_gate, x, h,
            v_cache, B_cache, C_cache, dt_cache, d_output.contiguous()
        )
        return None, dx, None, dW_x, dW_h, dW_B, dW_C, dW_dt, db, db_B, db_C, db_dt, db_gate


class SelectiveElmanCell(nn.Module):
    """
    Selective Elman cell with input-dependent B, C, dt.

    Like Mamba2's input-dependent selectivity but for Elman RNN.
    """

    def __init__(self, dim, dt_init=-2.0):
        super().__init__()
        self.dim = dim

        # Standard Elman weights
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        # Input-dependent B (write gate)
        self.W_B = nn.Parameter(torch.empty(dim, dim))
        self.b_B = nn.Parameter(torch.zeros(dim))

        # Input-dependent C (read gate)
        self.W_C = nn.Parameter(torch.empty(dim, dim))
        self.b_C = nn.Parameter(torch.zeros(dim))

        # Input-dependent dt (update rate)
        self.W_dt = nn.Parameter(torch.empty(dim, dim))
        self.b_dt = nn.Parameter(torch.full((dim,), dt_init))

        # Output gate bias (for h+x gating)
        self.b_gate = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        # Main weights
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_h, gain=0.5)

        # Selectivity weights - small init
        nn.init.xavier_uniform_(self.W_B, gain=0.1)
        nn.init.xavier_uniform_(self.W_C, gain=0.1)
        nn.init.xavier_uniform_(self.W_dt, gain=0.1)

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] selective output
            h: [T+1, B, dim] hidden states
        """
        T, batch, D = x.shape

        if h0 is None:
            h0 = torch.zeros(batch, self.dim, device=x.device, dtype=x.dtype)

        # Use Haste kernel if available
        if HASTE_AVAILABLE and x.is_cuda:
            return InputSelectiveElmanFunction.apply(
                self.training, x, h0,
                self.W_x, self.W_h, self.W_B, self.W_C, self.W_dt,
                self.b, self.b_B, self.b_C, self.b_dt, self.b_gate
            )

        # PyTorch fallback
        if REQUIRE_CUDA:
            raise RuntimeError("CUDA kernels required (ELMAN_REQUIRE_CUDA=1) but not available")
        return self._forward_pytorch(x, h0)

    def _forward_pytorch(self, x, h0):
        """PyTorch fallback implementation."""
        T, batch, D = x.shape

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]

            # Input-dependent gates (like Mamba2's B, C, dt)
            B = torch.sigmoid(x_t @ self.W_B.T + self.b_B)  # write gate
            C = torch.sigmoid(x_t @ self.W_C.T + self.b_C)  # read gate
            dt = torch.sigmoid(x_t @ self.W_dt.T + self.b_dt)  # update rate

            # Candidate with B-modulated write
            raw = B * (x_t @ self.W_x.T) + h_prev @ self.W_h.T + self.b
            candidate = torch.tanh(raw)

            # Gated update with input-dependent dt
            h_new = (1 - dt) * h_prev + dt * candidate

            # Output with C-modulated read and h+x gating
            gate = F.silu(h_new + x_t + self.b_gate)
            output = C * h_new * gate

            h_list.append(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class SelectiveElman(nn.Module):
    """
    Selective Elman layer with input-dependent B, C, dt.

    Level 9 with Mamba2-style selectivity.
    """

    def __init__(self, dim, expansion=1.0, dt_init=-2.0, dropout=0.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Selective Elman cell
        self.cell = SelectiveElmanCell(self.d_inner, dt_init=dt_init)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, h0=None, **kwargs):
        """
        Args:
            x: [B, T, dim] input
            h0: [B, d_inner] initial hidden state

        Returns:
            output: [B, T, dim]
            h_final: [B, d_inner]
        """
        B, T, D = x.shape

        # Project
        x_proj = self.in_proj(x)

        # Transpose for cell
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        # Run cell
        cell_out, h_all = self.cell(x_rnn, h0)
        h_final = h_all[-1]

        # Transpose back
        cell_out = cell_out.permute(1, 0, 2).contiguous()

        # Project and dropout
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final


__all__ = ['SelectiveElman', 'SelectiveElmanCell', 'SELECTIVE_ELMAN_AVAILABLE', 'LEVEL_2_AVAILABLE']


if __name__ == "__main__":
    print("Testing Selective Elman (Level 9 with B, C, dt)...")
    print(f"CUDA kernel available: {HASTE_AVAILABLE}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dim = 512
    model = SelectiveElman(dim, expansion=1.5).to(device).bfloat16()
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    x = torch.randn(2, 32, dim, device=device, dtype=torch.bfloat16)
    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}, Hidden: {h.shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Speed comparison
    if device == 'cuda':
        import time
        # Warmup
        for _ in range(3):
            out, h = model(x)
        torch.cuda.synchronize()

        # Time
        start = time.time()
        for _ in range(10):
            out, h = model(x)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 10

        tokens = 2 * 32
        print(f"Time: {elapsed*1000:.2f}ms, {tokens/elapsed:.0f} tok/s")
