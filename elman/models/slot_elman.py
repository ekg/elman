"""
E2: Slot-Based Elman - Mamba2-style multi-slot memory

Architecture insight from Mamba2:
- Mamba2 uses h ∈ R^(d × d_state) with d_state=64
- Each slot has diagonal decay (no O(d²) matmul)
- This gives 64x more memory capacity while keeping O(d) compute per slot

E2 Design:
    h ∈ R^(B, d, n_slots)           # n_slots independent memory vectors
    a ∈ R^(d, n_slots)              # Diagonal decay per slot (0-1)
    B ∈ R^(d, n_slots)              # Input-to-slot projection (per-element)

    h_t[:,i] = a[:,i] * h_{t-1}[:,i] + B[:,i] * x_t  # Diagonal recurrence
    output = h_t.sum(dim=-1) * silu(z)              # Combine slots + gate

Key differences from e1:
- e1: h ∈ R^d with full W_h matmul (O(d²) per step)
- e2: h ∈ R^(d, n_slots) with diagonal decay (O(d * n_slots) per step)
- e2 has n_slots × more memory with similar compute (when n_slots << d)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    SLOT_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'slot_elman_forward')
except ImportError:
    SLOT_CUDA_AVAILABLE = False


class SlotElmanFunction(torch.autograd.Function):
    """CUDA-accelerated slot elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, z, h0, decay, B, C):
        h, output = hasty_pytorch_lib.slot_elman_forward(training, x, z, h0, decay, B, C)
        ctx.save_for_backward(x, z, h, decay, B, C)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        x, z, h, decay, B, C = ctx.saved_tensors
        dx, dz, d_decay, dB, dC = hasty_pytorch_lib.slot_elman_backward(
            x, z, h, decay, B, C, d_output.contiguous()
        )
        return None, dx, dz, None, d_decay, dB, dC


class SlotElmanCell(nn.Module):
    """
    E2 Elman cell with slot-based memory.

    Each slot has:
    - Diagonal decay a[:,i] ∈ (0, 1) learned per dimension
    - Input contribution B[:,i] * x_t

    h_t[:,i] = a[:,i] * h_{t-1}[:,i] + B[:,i] * silu(x_t)
    output = sum(h_t, dim=-1) * silu(z_t)
    """

    def __init__(self, dim, n_slots=64, init_decay=0.9):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots

        # Diagonal decay per slot - learned, constrained to (0, 1) via sigmoid
        # Initialize to achieve init_decay after sigmoid
        init_logit = torch.logit(torch.tensor(init_decay))
        self.decay_logits = nn.Parameter(torch.full((dim, n_slots), init_logit.item()))

        # Input-to-slot projection (element-wise, not matmul)
        # Each slot gets a different weighted view of the input
        self.B = nn.Parameter(torch.empty(dim, n_slots))

        # Slot combination weights (optional, can just sum)
        self.C = nn.Parameter(torch.ones(n_slots) / n_slots)

        self._init_weights()

    def _init_weights(self):
        # Initialize B with small values - each slot gets different input mixing
        nn.init.normal_(self.B, mean=0.0, std=0.02)

    def forward(self, x, z, h0=None):
        """
        Args:
            x: [T, B, dim] input for RNN (pre-activated)
            z: [T, B, dim] input for gating
            h0: [B, dim, n_slots] initial hidden state

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, dim, n_slots] all hidden states
        """
        T, B_size, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B_size, D, self.n_slots, device=x.device, dtype=x.dtype)

        # Get decay in (0, 1) range
        decay = torch.sigmoid(self.decay_logits)  # [dim, n_slots]

        # Use CUDA kernel if available
        if SLOT_CUDA_AVAILABLE and x.is_cuda:
            h, output = SlotElmanFunction.apply(
                self.training, x.contiguous(), z.contiguous(),
                h0.contiguous(), decay.contiguous(),
                self.B.contiguous(), self.C.contiguous()
            )
            return output, h

        # PyTorch fallback
        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]  # [B, dim, n_slots]
            x_t = x[t]  # [B, dim]
            z_t = z[t]  # [B, dim]

            # Diagonal recurrence per slot:
            # h_new[:,i] = decay[:,i] * h_prev[:,i] + B[:,i] * x_t
            # Expand x_t to [B, dim, 1] for broadcasting
            x_expanded = x_t.unsqueeze(-1)  # [B, dim, 1]

            # h_new = decay * h_prev + B * x_t
            h_new = decay * h_prev + self.B * x_expanded  # [B, dim, n_slots]
            h_list.append(h_new)

            # Combine slots with learned weights
            # h_combined = sum_i(C[i] * h_new[:,:,i])
            h_combined = (h_new * self.C).sum(dim=-1)  # [B, dim]

            # Mamba2-style gating
            output = h_combined * F.silu(z_t)  # [B, dim]
            output_list.append(output)

        h = torch.stack(h_list, dim=0)  # [T+1, B, dim, n_slots]
        output = torch.stack(output_list, dim=0)  # [T, B, dim]
        return output, h


class SlotElman(nn.Module):
    """
    E2: Slot-Based Elman with Mamba2-style multi-slot memory.

    Architecture:
        x, z = split(in_proj(x))    # Split into RNN input and gate
        x = conv1d(x) if use_conv   # Optional local context
        x = silu(x)                 # Pre-activation
        h = slot_cell(x, z)         # Multi-slot RNN with gated output
        output = out_proj(h)        # Project back to dim
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        n_slots=64,
        init_decay=0.9,
        dropout=0.0,
        use_conv=False,
        d_conv=4,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_slots = n_slots
        self.use_conv = use_conv

        # Mamba2-style: project to 2*d_inner, then split
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

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

        # Slot-based Elman cell
        self.cell = SlotElmanCell(
            self.d_inner,
            n_slots=n_slots,
            init_decay=init_decay
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, h0=None, **kwargs):
        """
        Args:
            x: [B, T, dim] input sequence
            h0: [B, d_inner, n_slots] initial hidden state

        Returns:
            output: [B, T, dim] output sequence
            h_final: [B, d_inner, n_slots] final hidden state
        """
        B, T, D = x.shape

        # Mamba2-style: project and split
        xz = self.in_proj(x)  # [B, T, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # Each [B, T, d_inner]

        # Optional conv1d
        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)  # [B, d_inner, T]
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)  # [B, T, d_inner]

        # Pre-activation
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()
        z_rnn = z.permute(1, 0, 2).contiguous()

        # Run slot-based cell
        cell_out, h_all = self.cell(x_rnn, z_rnn, h0)
        h_final = h_all[-1]  # [B, d_inner, n_slots]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, n_slots={self.n_slots}, use_conv={self.use_conv}, LEVEL=2_SLOT'


if __name__ == "__main__":
    print("Testing SlotElman (E2)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test basic
    model = SlotElman(dim=512, expansion=2.0, n_slots=64).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print(f"Testing forward with n_slots=64...")
    out, h = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Hidden: {h.shape} (includes slots dimension)")

    print("\nTesting backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Compare parameter counts
    print("\n" + "=" * 60)
    print("Parameter comparison:")

    from mamba_gated_elman import MambaGatedElman
    e1 = MambaGatedElman(dim=512, expansion=2.0).to(device)
    e2 = SlotElman(dim=512, expansion=2.0, n_slots=64).to(device)

    e1_params = sum(p.numel() for p in e1.parameters())
    e2_params = sum(p.numel() for p in e2.parameters())

    print(f"E1 (Mamba-Gated): {e1_params:,} params")
    print(f"E2 (Slot, 64 slots): {e2_params:,} params")
    print(f"E2 memory slots: 64x more memory capacity")

    # Memory per step comparison
    d_inner = 1024  # 512 * 2
    print(f"\nMemory per step:")
    print(f"  E1: {d_inner} floats")
    print(f"  E2: {d_inner * 64} floats (64 slots)")

    print("\nE2 (Slot Elman) test passed!")
