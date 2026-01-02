"""
Level 1: Gated Elman - Input-dependent delta gate

delta = sigmoid(W_delta @ x_t + b_delta)
h_t = (1 - delta) * h_{t-1} + delta * tanh(W_x @ x_t + W_h @ h_{t-1} + b)

This adds a discretization gate to control memory/update tradeoff per position.
The gate decides how much to keep from previous state vs new candidate.

Note: This is NOT a GRU (which has 3 gates). It's a simple discretization.

Key question: Does input-dependent gating help compared to stock Elman?
"""

import torch
import os
REQUIRE_CUDA = os.environ.get('ELMAN_REQUIRE_CUDA', '0') == '1'
import torch.nn as nn
import torch.nn.functional as F

# Try to import Haste CUDA kernel
try:
    import hasty_pytorch_lib
    HASTE_AVAILABLE = hasattr(hasty_pytorch_lib, 'gated_elman_forward')
except ImportError:
    HASTE_AVAILABLE = False

LEVEL_1_AVAILABLE = True  # PyTorch fallback always available


class GatedElmanFunction(torch.autograd.Function):
    """Autograd function for Gated Elman with h+x selective gating."""

    @staticmethod
    def forward(ctx, training, x, h0, W_x, W_h, W_delta, b, b_delta, b_gate):
        h, output, v, delta_cache, gate_cache = hasty_pytorch_lib.gated_elman_forward(
            training,
            x.contiguous(),
            h0.contiguous(),
            W_x.contiguous(),
            W_h.contiguous(),
            W_delta.contiguous(),
            b.contiguous(),
            b_delta.contiguous(),
            b_gate.contiguous()
        )
        if training:
            ctx.save_for_backward(x, W_x, W_h, W_delta, b_gate, h, v, delta_cache, gate_cache)
        return output, h

    @staticmethod
    def backward(ctx, d_output, dh_unused):
        x, W_x, W_h, W_delta, b_gate, h, v, delta_cache, gate_cache = ctx.saved_tensors
        dx, dW_x, dW_h, dW_delta, db, db_delta, d_b_gate = hasty_pytorch_lib.gated_elman_backward(
            W_x, W_h, W_delta, b_gate, x, h, v, delta_cache, gate_cache, d_output.contiguous()
        )
        return None, dx, None, dW_x, dW_h, dW_delta, db, db_delta, d_b_gate


class GatedElmanCell(nn.Module):
    """
    Gated Elman cell - Level 1 with h+x selective gating.

    delta = sigmoid(W_delta @ x_t + b_delta)
    h_t = (1 - delta) * h_{t-1} + delta * tanh(W_x @ x_t + W_h @ h_{t-1} + b)
    output_t = h_t * silu(h_t + x_t + b_gate)  -- h+x selective gating

    Args:
        dim: Hidden dimension
        delta_init: Initial bias for delta gate (negative = keep more state)
        w_h_mode: Constraint mode for W_h matrix
        w_h_init_gain: Initial gain for W_h initialization
    """

    def __init__(self, dim, delta_init=-2.0, w_h_mode='spectral_norm', w_h_init_gain=1.0):
        super().__init__()
        self.dim = dim
        self.w_h_mode = w_h_mode
        self.w_h_init_gain = w_h_init_gain

        # Candidate computation weights
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        # h+x selective gate bias
        self.b_gate = nn.Parameter(torch.zeros(dim))

        # Recurrence matrix with optional constraints
        if w_h_mode == 'scaled_orthogonal':
            self.register_buffer('W_h_base', torch.empty(dim, dim))
            self.w_h_log_scale = nn.Parameter(torch.tensor(-0.01))
        else:
            self.W_h = nn.Parameter(torch.empty(dim, dim))

        # Delta (gate) computation
        self.W_delta = nn.Parameter(torch.empty(dim, dim))
        self.b_delta = nn.Parameter(torch.full((dim,), delta_init))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x)
        if self.w_h_mode == 'scaled_orthogonal':
            nn.init.orthogonal_(self.W_h_base)
        else:
            nn.init.xavier_uniform_(self.W_h, gain=self.w_h_init_gain)
        nn.init.xavier_uniform_(self.W_delta, gain=0.1)

    def get_W_h(self):
        """Get the effective W_h matrix with constraints applied."""
        if self.w_h_mode == 'scaled_orthogonal':
            scale = torch.sigmoid(self.w_h_log_scale)
            return scale * self.W_h_base
        elif self.w_h_mode == 'spectral_norm':
            target_radius = 0.99
            u = getattr(self, '_spectral_u', None)
            if u is None or u.shape[0] != self.dim:
                u = torch.randn(self.dim, device=self.W_h.device, dtype=self.W_h.dtype)
                u = u / u.norm()
            with torch.no_grad():
                for _ in range(3):
                    v = self.W_h.T @ u
                    v = v / (v.norm() + 1e-8)
                    u = self.W_h @ v
                    u = u / (u.norm() + 1e-8)
                self._spectral_u = u
            sigma = (u @ self.W_h @ v).abs()
            return self.W_h * (target_radius / (sigma + 1e-8))
        else:
            return self.W_h

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] h+x selective output
            h: [T+1, B, dim] all hidden states including h0
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.dim, device=x.device, dtype=x.dtype)

        # Get constrained W_h
        W_h = self.get_W_h()

        # Use Haste kernel if available
        if HASTE_AVAILABLE and x.is_cuda:
            return GatedElmanFunction.apply(
                self.training, x, h0,
                self.W_x, W_h, self.W_delta,
                self.b, self.b_delta, self.b_gate
            )

        # PyTorch fallback
        return self._forward_pytorch(x, h0, W_h)

    def _forward_pytorch(self, x, h0, W_h):
        """Pure PyTorch implementation with h+x selective gating."""
        T, B, D = x.shape
        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]

            # Delta gate: sigmoid(W_delta @ x + b_delta)
            delta_raw = x_t @ self.W_delta.T + self.b_delta
            delta = torch.sigmoid(delta_raw)

            # Candidate: tanh(W_x @ x + W_h @ h + b)
            candidate_raw = x_t @ self.W_x.T + h_prev @ W_h.T + self.b
            candidate = torch.tanh(candidate_raw)

            # State update: interpolation between h_prev and candidate
            h_new = (1 - delta) * h_prev + delta * candidate
            h_list.append(h_new)

            # h+x selective output: output = h * silu(h + x + b_gate)
            gate = F.silu(h_new + x_t + self.b_gate)
            output = h_new * gate
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class GatedElman(nn.Module):
    """
    Gated Elman layer - Level 1 with projections and h+x selective gating.

    Adds input-dependent gating to stock Elman.
    Now includes h+x selective output: output = h * silu(h + x + b_gate)
    """

    def __init__(self, dim, expansion=1.0, delta_init=-2.0, dropout=0.0,
                 r_h_mode='spectral_norm', r_h_init_gain=1.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.r_h_mode = r_h_mode

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Gated Elman cell
        self.cell = GatedElmanCell(self.d_inner, delta_init=delta_init,
                                   w_h_mode=r_h_mode, w_h_init_gain=r_h_init_gain)

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
        cell_out, h_all = self.cell(x_rnn, h0)  # [T, B, d_inner], [T+1, B, d_inner]
        h_final = h_all[-1]  # [B, d_inner]

        # Transpose back: [B, T, d_inner]
        cell_out = cell_out.permute(1, 0, 2).contiguous()

        # Apply dropout and project
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, LEVEL=1_GATED'


if __name__ == "__main__":
    print("Testing GatedElman (Level 1)...")
    print("=" * 60)
    print(f"Haste CUDA kernel available: {HASTE_AVAILABLE}")

    # Test layer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GatedElman(dim=512, expansion=2.0).to(device).bfloat16()
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
    print("Level 1 (Gated Elman) test passed!")
