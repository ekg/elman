"""
E26: Parallel Dual-Memory Elman

Key insight: Separate "what" (content via GEMMs) from "where" (routing via attention).

Architecture:
  PARALLEL PHASE (batched cuBLAS):
    x_proj[0:T] = x[0:T] @ W_x.T   # One big GEMM for all timesteps

  SEQUENTIAL PHASE (cheap routing):
    for t in range(T):
      read = softmax(h_work @ tape.T) @ tape    # O(N×D) dots
      h_work = tanh(x_proj[t] + W_h @ h_work + read + b)
      tape = sparse_write(tape, h_work)          # O(N×D) dots

The W_h @ h_work is still sequential but unavoidable for RNN semantics.
Attention is cheap: O(N×D) vs O(D²) for GEMM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def e26_forward_step_python(
    x_proj_t: torch.Tensor,     # [B, D] - pre-projected input
    h_tape: torch.Tensor,       # [B, N, D] - tape memory
    h_work: torch.Tensor,       # [B, D] - working memory
    W_h: torch.Tensor,          # [D, D] - recurrence weight
    b_h: torch.Tensor,          # [D] - bias
    W_write: torch.Tensor,      # [D, D] - write projection
    scale: float                # 1/sqrt(D) for attention
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single E26 timestep.

    Returns: (h_work_new, h_tape_new, read_attn, write_attn)
    """
    B, N, D = h_tape.shape

    # Read attention: h_work @ tape.T -> [B, N]
    read_scores = torch.einsum('bd,bnd->bn', h_work.float(), h_tape.float()) * scale
    read_attn = F.softmax(read_scores, dim=-1).to(h_work.dtype)

    # Read value: weighted sum over tape
    read_val = torch.einsum('bn,bnd->bd', read_attn.float(), h_tape.float()).to(h_work.dtype)

    # Update h_work: tanh(x_proj + W_h @ h_work + read + b)
    Rh = h_work @ W_h.T  # [B, D] - the sequential GEMM
    h_work_new = torch.tanh(x_proj_t + Rh + read_val + b_h)

    # Write attention
    write_val = h_work_new @ W_write.T  # [B, D]
    write_scores = torch.einsum('bd,bnd->bn', write_val.float(), h_tape.float()) * scale
    write_attn = F.softmax(write_scores, dim=-1).to(h_work.dtype)

    # Update tape: h_tape = h_tape * (1 - w) + write_val * w
    h_tape_new = (h_tape * (1 - write_attn.unsqueeze(-1)) +
                  write_val.unsqueeze(1) * write_attn.unsqueeze(-1))

    return h_work_new, h_tape_new, read_attn, write_attn


def e26_forward_python(
    x: torch.Tensor,            # [B, T, D] - input sequence
    h_tape_init: torch.Tensor,  # [B, N, D] - initial tape
    h_work_init: torch.Tensor,  # [B, D] - initial working memory
    W_h: torch.Tensor,          # [D, D]
    W_x: torch.Tensor,          # [D, D]
    b_h: torch.Tensor,          # [D]
    W_write: torch.Tensor       # [D, D]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    E26 forward pass.

    Returns: (h_work_all, h_tape_final, read_attn_all, write_attn_all)
    """
    B, T, D = x.shape
    N = h_tape_init.size(1)
    scale = 1.0 / (D ** 0.5)

    # PARALLEL PHASE: Pre-compute x_proj for ALL timesteps
    x_proj = x @ W_x.T  # [B, T, D] - one big GEMM

    # SEQUENTIAL PHASE: Routing
    h_tape = h_tape_init.clone()
    h_work = h_work_init.clone()

    h_work_list = []
    read_attn_list = []
    write_attn_list = []

    for t in range(T):
        h_work, h_tape, read_attn, write_attn = e26_forward_step_python(
            x_proj[:, t, :], h_tape, h_work, W_h, b_h, W_write, scale
        )
        h_work_list.append(h_work)
        read_attn_list.append(read_attn)
        write_attn_list.append(write_attn)

    h_work_all = torch.stack(h_work_list, dim=1)      # [B, T, D]
    read_attn_all = torch.stack(read_attn_list, dim=1)  # [B, T, N]
    write_attn_all = torch.stack(write_attn_list, dim=1)  # [B, T, N]

    return h_work_all, h_tape, read_attn_all, write_attn_all


class E26DualMemoryElmanCell(nn.Module):
    """Single E26 cell for stacking."""

    def __init__(self, dim: int, n_slots: int = 8):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.scale = 1.0 / (dim ** 0.5)

        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.b_h = nn.Parameter(torch.zeros(dim))
        self.W_write = nn.Parameter(torch.empty(dim, dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.W_h)
        self.W_h.data.mul_(0.9)
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_write)

    def forward(
        self,
        x: torch.Tensor,
        h_tape: Optional[torch.Tensor] = None,
        h_work: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        if h_tape is None:
            h_tape = torch.zeros(B, self.n_slots, D, device=device, dtype=dtype)
        if h_work is None:
            h_work = torch.zeros(B, D, device=device, dtype=dtype)

        h_work_all, h_tape_final, _, _ = e26_forward_python(
            x, h_tape, h_work, self.W_h, self.W_x, self.b_h, self.W_write
        )

        return h_work_all, h_tape_final, h_work_all[:, -1, :]


class E26DualMemoryElman(nn.Module):
    """Full E26 model with embedding and output projection."""

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 512,
        n_slots: int = 8,
        depth: int = 1
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            E26DualMemoryElmanCell(dim, n_slots) for _ in range(depth)
        ])
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)
        self.out_proj.weight = self.embed.weight  # Weight tying

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            h, _, _ = layer(h)
        return self.out_proj(h)


if __name__ == '__main__':
    # Quick test
    B, T, D, N = 2, 8, 256, 8
    device = 'cuda'
    dtype = torch.bfloat16

    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1
    h_tape = torch.zeros(B, N, D, device=device, dtype=dtype)
    h_work = torch.zeros(B, D, device=device, dtype=dtype)

    W_h = torch.randn(D, D, device=device, dtype=dtype) * 0.01
    W_x = torch.randn(D, D, device=device, dtype=dtype) * 0.01
    b_h = torch.zeros(D, device=device, dtype=dtype)
    W_write = torch.randn(D, D, device=device, dtype=dtype) * 0.01

    # Make W_h well-conditioned
    with torch.no_grad():
        W_h_f32 = torch.empty_like(W_h, dtype=torch.float32)
        nn.init.orthogonal_(W_h_f32)
        W_h.copy_(W_h_f32.to(dtype) * 0.9)

    h_work_all, h_tape_final, read_attn, write_attn = e26_forward_python(
        x, h_tape, h_work, W_h, W_x, b_h, W_write
    )

    print(f"E26 Python forward test:")
    print(f"  h_work_all shape: {h_work_all.shape}")
    print(f"  h_tape_final shape: {h_tape_final.shape}")
    print(f"  read_attn shape: {read_attn.shape}")
    print(f"  write_attn shape: {write_attn.shape}")
    print(f"  h_work_all stats: min={h_work_all.min():.4f}, max={h_work_all.max():.4f}")
    print(f"  PASS" if not torch.isnan(h_work_all).any() else "  FAIL: NaN")
