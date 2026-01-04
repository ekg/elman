"""
Level log_3: Log-Storage Diagonal Elman

Log-space state storage with diagonal recurrence. Uses tanh activation
(not polynomial) to match the CUDA kernel interface.

Architecture:
    # Diagonal recurrence in log space
    candidate = tanh(W_x @ x + r_h * h_prev + b)
    delta = sigmoid(W_delta @ x + b_delta)
    h_new = (1-delta) * h_prev + delta * candidate

    # h+x selective output (like Mamba2)
    output = h * silu(h + x + b_gate)

Key features:
- Log-space storage for numerical stability
- Diagonal r_h for efficient recurrence
- h+x selective output: input-dependent gating like Mamba2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import log-space utilities
from .logspace_polynomial import (
    LOG_ZERO, LOG_EPS, to_log_space, from_log_space,
    signed_log_add
)

# Try to import Haste CUDA kernel
try:
    import hasty_pytorch_lib
    HASTE_LOG_STORAGE_DIAG_AVAILABLE = hasattr(hasty_pytorch_lib, 'log_storage_diagonal_forward')
except ImportError:
    hasty_pytorch_lib = None
    HASTE_LOG_STORAGE_DIAG_AVAILABLE = False

LOG_STORAGE_DIAGONAL_AVAILABLE = True  # PyTorch fallback always available


class LogStorageDiagonalFunction(torch.autograd.Function):
    """Autograd function for Log-Storage Diagonal Elman with h+x gating (Haste kernel)."""

    @staticmethod
    def forward(ctx, training, x, log_h0, sign_h0, W_x, r_h,
                W_delta, b, b_delta, b_gate):
        results = hasty_pytorch_lib.log_storage_diagonal_forward(
            training,
            x.contiguous(),
            log_h0.contiguous(),
            sign_h0.contiguous(),
            W_x.contiguous(),
            r_h.contiguous(),
            W_delta.contiguous(),
            b.contiguous(),
            b_delta.contiguous(),
            b_gate.contiguous()
        )
        log_h, sign_h, output, v_cache, delta_cache, gate_cache, \
            weight1_cache, log_term1_cache, log_term2_cache = results

        if training:
            ctx.save_for_backward(
                x, W_x, r_h, W_delta, b_gate,
                log_h, sign_h, v_cache, delta_cache, gate_cache,
                weight1_cache, log_term1_cache, log_term2_cache
            )

        # Return h_linear as None since it's not computed in CUDA kernel
        return output, log_h, sign_h, None

    @staticmethod
    def backward(ctx, d_output, d_log_h, d_sign_h, d_h_linear):
        (x, W_x, r_h, W_delta, b_gate,
         log_h, sign_h, v_cache, delta_cache, gate_cache,
         weight1_cache, log_term1_cache, log_term2_cache) = ctx.saved_tensors

        dx, dW_x, d_r_h, dW_delta, db, db_delta, db_gate = \
            hasty_pytorch_lib.log_storage_diagonal_backward(
                W_x, r_h, W_delta, b_gate,
                x, log_h, sign_h, v_cache, delta_cache, gate_cache,
                weight1_cache, log_term1_cache, log_term2_cache,
                d_output.contiguous()
            )

        return (None, dx, None, None, dW_x, d_r_h,
                dW_delta, db, db_delta, db_gate)


class LogStorageDiagonalCell(nn.Module):
    """
    Log-Storage Diagonal Elman cell - Level log_3.

    Log-space state storage with diagonal recurrence and tanh activation.
    h+x selective output: output = h * silu(h + x + b_gate)

    Args:
        dim: Hidden dimension
        delta_init: Initial bias for delta gate
    """

    def __init__(self, dim, delta_init=-2.0, **kwargs):
        super().__init__()
        self.dim = dim

        # Input projection
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        # Diagonal recurrence (linear space)
        self.r_h = nn.Parameter(torch.full((dim,), 0.9))

        # Delta gate
        self.W_delta = nn.Parameter(torch.empty(dim, dim))
        self.b_delta = nn.Parameter(torch.full((dim,), delta_init))

        # h+x selective gate bias
        self.b_gate = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_delta, gain=0.1)

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: tuple of (log_h0, sign_h0) each [B, dim], or None

        Returns:
            output: [T, B, dim] h+x selective output
            log_h: [T+1, B, dim] log magnitudes
            sign_h: [T+1, B, dim] signs
            h_linear: [T, B, dim] linear hidden states (None for CUDA)
        """
        T, B, D = x.shape

        if h0 is None:
            log_h0 = torch.full((B, self.dim), LOG_ZERO, device=x.device, dtype=x.dtype)
            sign_h0 = torch.ones(B, self.dim, device=x.device, dtype=x.dtype)
        else:
            log_h0, sign_h0 = h0

        # Use CUDA kernel when available
        if HASTE_LOG_STORAGE_DIAG_AVAILABLE and x.is_cuda:
            return LogStorageDiagonalFunction.apply(
                self.training, x, log_h0, sign_h0,
                self.W_x, self.r_h, self.W_delta,
                self.b, self.b_delta, self.b_gate
            )
        else:
            return self._forward_pytorch(x, log_h0, sign_h0)

    def _forward_pytorch(self, x, log_h0, sign_h0):
        """Pure PyTorch implementation with log-space storage and h+x gating."""
        T, B, D = x.shape

        log_h_list = [log_h0]
        sign_h_list = [sign_h0]
        output_list = []
        h_linear_list = []

        for t in range(T):
            log_h_prev = log_h_list[-1]
            sign_h_prev = sign_h_list[-1]
            x_t = x[t]

            # Convert h_prev from log to linear
            h_prev_linear = from_log_space(log_h_prev, sign_h_prev)

            # Candidate: tanh(W_x @ x + r_h * h_prev + b)
            v = x_t @ self.W_x.T + self.r_h * h_prev_linear + self.b
            candidate = torch.tanh(v)

            # Delta gate
            delta_raw = x_t @ self.W_delta.T + self.b_delta
            delta = torch.sigmoid(delta_raw)

            # Gated update in linear space
            h_new = (1 - delta) * h_prev_linear + delta * candidate

            # Convert back to log space
            log_h_new, sign_h_new = to_log_space(h_new)

            # h+x selective output: output = h * silu(h + x + b_gate)
            gate = F.silu(h_new + x_t + self.b_gate)
            output = h_new * gate

            log_h_list.append(log_h_new)
            sign_h_list.append(sign_h_new)
            output_list.append(output)
            h_linear_list.append(h_new)

        log_h = torch.stack(log_h_list, dim=0)
        sign_h = torch.stack(sign_h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        h_linear = torch.stack(h_linear_list, dim=0)

        return output, log_h, sign_h, h_linear


class LogStorageDiagonal(nn.Module):
    """
    Log-Storage Diagonal Elman layer for use in LadderLM.

    Log-space state storage with diagonal recurrence.
    h+x selective output: output = h * silu(h + x + b_gate)

    Args:
        dim: Model dimension
        expansion: Hidden state expansion factor
        delta_init: Initial bias for delta gate
        dropout: Dropout rate
        **kwargs: Ignored (for API compatibility)
    """

    def __init__(self, dim, expansion=1.0, delta_init=-2.0,
                 dropout=0.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Log-storage diagonal cell
        self.cell = LogStorageDiagonalCell(
            self.d_inner, delta_init=delta_init
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, h0=None):
        """
        Args:
            x: [B, T, dim] input sequence
            h0: tuple of (log_h0, sign_h0) or None

        Returns:
            output: [B, T, dim] output for residual connection
            h_final: tuple of (log_h_final, sign_h_final) for TBPTT
        """
        B, T, D = x.shape

        # Input projection
        x_proj = self.in_proj(x)

        # Transpose to [T, B, d_inner] for cell
        x_proj = x_proj.transpose(0, 1).contiguous()

        # Run cell
        output, log_h, sign_h, h_linear = self.cell(x_proj, h0)

        # Transpose back to [B, T, d_inner]
        output = output.transpose(0, 1).contiguous()

        # Output projection
        output = self.out_proj(output)
        output = self.dropout(output)

        # Final hidden state for TBPTT
        log_h_final = log_h[-1]
        sign_h_final = sign_h[-1]
        h_final = (log_h_final, sign_h_final)

        return output, h_final


__all__ = [
    'LogStorageDiagonalCell',
    'LogStorageDiagonal',
    'LOG_STORAGE_DIAGONAL_AVAILABLE',
    'HASTE_LOG_STORAGE_DIAG_AVAILABLE',
]


if __name__ == "__main__":
    print("Testing LogStorageDiagonal (log_3)...")
    print("=" * 60)
    print(f"Haste CUDA kernel available: {HASTE_LOG_STORAGE_DIAG_AVAILABLE}")

    # Test layer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LogStorageDiagonal(dim=512, expansion=2.0).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print(f"Hidden state: log_h={h[0].shape}, sign_h={h[1].shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Verify h+x gating components
    print(f"b_gate shape: {model.cell.b_gate.shape} (should be [{model.d_inner}])")
    print(f"r_h shape: {model.cell.r_h.shape}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print("Log-Storage Diagonal (log_3) test passed!")
