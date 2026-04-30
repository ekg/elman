"""
E94: Symmetric matrix-state RNN — heads as M-chunks, time/depth recurrence.

Key idea (clean version):

  State per (B, T) position:  S ∈ ℝ^(N × M)         where N = head_dim = 16
                              M = H · head_dim       (M chunked into H 16-wide heads)

  Each "head" h is a 16×16 SLICE of S:
      S_h = S[:, h·16 : (h+1)·16] ∈ ℝ^(16, 16)

  Per-(layer, head) structured matrices (16×16 each):
      W_h_time[l, h]  ∈ ℝ^(16, 16)  — time recurrence within layer l, head h
      W_h_layer[l, h] ∈ ℝ^(16, 16)  — depth recurrence (state from layer l → l+1)

Forward (per layer l, applied to all heads in parallel via reshape):

  Time recurrence within layer l (sequential over t):
      For t = 0..T-1:
          # 16×16 chunk-wise mixing of previous time step
          wh_t  = W_h_time[l, h] · S_h^{l, t-1}             (per head)

          # Input enters: at l=0 from token embedding (delta-rule write),
          # at l>0 from previous layer's state at this t (W_h_layer mix)
          if l == 0:
              write = outer(k[t], v[t] - retrieved)         (per head — delta rule)
          else:
              write = W_h_layer[l-1, h] · S_h^{l-1, t}      (per head)

          S^{l, t} = tanh( wh_t + write )

Final readout: tanh-merge across heads, then project to vocab.

NO out_proj that collapses N·M to dim. NO residual stream of dim. State stays in
state-space throughout the model.

Parameter budget (each layer):
  W_h_time  : H · 16 · 16 = 256 · H  per layer
  W_h_layer : H · 16 · 16 = 256 · H  per layer (zero on last layer)
  + token embedding (vocab → small initial input) once
  + final readout (state → vocab) once
"""
import math
import os, sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton fast path
try:
    _PARARNN_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'experiments', 'pararnn_kernel', 'tree_scan'
    )
    if _PARARNN_PATH not in sys.path:
        sys.path.insert(0, _PARARNN_PATH)
    from e94_autograd import E94TimeFunction, E94TimeWriteFunction
    E94_TRITON_AVAILABLE = True
except Exception as e:
    E94TimeFunction = None
    E94TimeWriteFunction = None
    E94_TRITON_AVAILABLE = False


class E94Model(nn.Module):
    """E94 full model — heads as M-chunks, dual recurrence."""

    HEAD_DIM = 16

    def __init__(
        self,
        vocab_size: int = 256,
        n_heads: int = 32,                # H — number of head-chunks (M = H · 16)
        depth: int = 6,
        head_dim: int = 16,               # N (= head_dim by design symmetry)
        dropout: float = 0.0,
        share_layer_weights: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.H = n_heads
        self.N = head_dim                  # rows of state = head_dim
        self.head_dim = head_dim           # cols of head slice
        self.M = n_heads * head_dim
        self.L = depth
        self.share_layer_weights = share_layer_weights

        H, N, hd, M, L = n_heads, head_dim, head_dim, self.M, depth

        # Token embedding for layer 0 input via delta-rule:
        #   k: H · N (per-head row selector, normalized)
        #   v: H · hd (per-head column write)
        self.embed_k = nn.Embedding(vocab_size, H * N)
        self.embed_v = nn.Embedding(vocab_size, H * hd)

        # Per-(layer, head) time recurrence matrices [N, N] per head
        if share_layer_weights:
            self.W_h_time = nn.Parameter(self._init_eye(H, N))
            # Cross-head layer transition matrix [H, H] (single, shared across all transitions)
            self.W_h_layer = nn.Parameter(self._init_eye_HH(H))
        else:
            self.W_h_time = nn.Parameter(
                torch.stack([self._init_eye(H, N) for _ in range(L)], dim=0)
            )  # [L, H, N, N]
            # Cross-head matrix per transition: [L-1, H, H]
            self.W_h_layer = nn.Parameter(
                torch.stack([self._init_eye_HH(H) for _ in range(L - 1)], dim=0)
            ) if L > 1 else None

        # Final readout: state → vocab. Uses tanh-merge across heads then linear.
        # Merged shape: [N, hd] (averaged H heads). Flatten and project.
        self.head = nn.Linear(N * hd, vocab_size, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.normal_(self.embed_k.weight, std=0.02)
        nn.init.normal_(self.embed_v.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02 / math.sqrt(N * hd))

    @staticmethod
    def _init_eye(H, N):
        eye = torch.eye(N).unsqueeze(0).expand(H, -1, -1).contiguous()
        return eye + 0.01 * torch.randn(H, N, N)

    @staticmethod
    def _init_eye_HH(H):
        # Cross-head [H, H] init near identity (so layer transition is near-passthrough)
        return torch.eye(H) + 0.01 * torch.randn(H, H)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_num_params(self):
        # train.py uses this name
        return self.num_params()

    def forward(self, tokens: torch.Tensor, return_loss: bool = False):
        """tokens: [B, T] long tensor."""
        B, T = tokens.shape
        H, N, hd, M, L = self.H, self.N, self.head_dim, self.M, self.L

        # Layer 0 input embeddings (k for row selection, v for column write)
        k_raw = self.embed_k(tokens).view(B, T, H, N)         # [B, T, H, N]
        v_emb = self.embed_v(tokens).view(B, T, H, hd)        # [B, T, H, hd]
        k = F.normalize(k_raw, dim=-1)                         # delta-rule stability

        # Process layer 0 first (input via embeddings), then subsequent layers.
        # state_l: [B, T, H, N, hd] — current layer's full trajectory
        state_l = None

        # Initial S0 for time recurrence (zeros at every layer entry)
        S0_zeros = torch.zeros(B, H, N, hd, device=tokens.device, dtype=torch.float32)

        for l in range(L):
            if self.share_layer_weights:
                W_t = self.W_h_time   # [H, N, N]
                W_d = self.W_h_layer if (l > 0 and self.W_h_layer is not None) else None
            else:
                W_t = self.W_h_time[l]                                      # [H, N, N]
                W_d = self.W_h_layer[l - 1] if (l > 0 and self.W_h_layer is not None) else None

            use_triton = E94_TRITON_AVAILABLE and tokens.is_cuda

            if l == 0:
                # Layer 0: delta-rule input via embedding
                if use_triton:
                    state_l = E94TimeFunction.apply(
                        S0_zeros, W_t.contiguous(),
                        k.contiguous(), v_emb.contiguous(),
                    )
                else:
                    # Python fallback
                    new_state = torch.zeros(B, T, H, N, hd, device=tokens.device, dtype=torch.float32)
                    s_prev = S0_zeros
                    for t in range(T):
                        wh_t = torch.einsum('hnp,bhpc->bhnc', W_t, s_prev)
                        retrieved = torch.einsum('bhnc,bhn->bhc', s_prev, k[:, t])
                        delta = v_emb[:, t] - retrieved
                        write = torch.einsum('bhn,bhc->bhnc', k[:, t], delta)
                        s_new = torch.tanh(wh_t + write)
                        new_state[:, t] = s_new
                        s_prev = s_new
                    state_l = new_state
            else:
                # Layer l>0: cross-head mix of prev-layer state
                # W_d shape [H, H] mixes ACROSS heads at fixed (n, hd) positions.
                # writes[b, t, h_out, n, c] = sum_{h_in} W_d[h_out, h_in] * state_l[b, t, h_in, n, c]
                writes = torch.einsum('oi,btinc->btonc', W_d, state_l)  # [B, T, H, N, hd]
                if use_triton:
                    state_l = E94TimeWriteFunction.apply(
                        S0_zeros, W_t.contiguous(),
                        writes.contiguous(),
                    )
                else:
                    new_state = torch.zeros(B, T, H, N, hd, device=tokens.device, dtype=torch.float32)
                    s_prev = S0_zeros
                    for t in range(T):
                        wh_t = torch.einsum('hnp,bhpc->bhnc', W_t, s_prev)
                        s_new = torch.tanh(wh_t + writes[:, t])
                        new_state[:, t] = s_new
                        s_prev = s_new
                    state_l = new_state

        state_l = self.dropout(state_l)

        # Readout: normalized tanh-merge across H heads, then linear → vocab
        # merged: [B, T, N, hd]
        merged = torch.tanh(state_l.mean(dim=2))
        logits = self.head(merged.reshape(B, T, N * hd))                     # [B, T, vocab]

        if return_loss:
            shift_logits = logits[:, :-1].contiguous()
            shift_targets = tokens[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, self.vocab_size),
                shift_targets.reshape(-1),
            )
            return loss
        return logits


def count_params(vocab_size=256, H=32, head_dim=16, L=6):
    """Rough breakdown of E94 parameters."""
    N = head_dim
    embed = vocab_size * H * N + vocab_size * H * head_dim
    w_h_time = L * H * N * N
    w_h_layer = (L - 1) * H * H   # cross-head [H, H] per transition
    head = N * head_dim * vocab_size
    total = embed + w_h_time + w_h_layer + head
    print(f"E94 params (vocab={vocab_size}, H={H}, head_dim={head_dim}, L={L}):")
    print(f"  embed (k+v):  {embed:>12,}")
    print(f"  W_h_time:     {w_h_time:>12,}")
    print(f"  W_h_layer:    {w_h_layer:>12,}")
    print(f"  head:         {head:>12,}")
    print(f"  TOTAL:        {total:>12,}  (~{total/1e6:.1f}M)")
    return total


if __name__ == '__main__':
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=== E94 Smoke Test ===\n")
    count_params(vocab_size=256, H=32, head_dim=16, L=6)
    print()

    model = E94Model(vocab_size=256, n_heads=32, head_dim=16, depth=6).to(device)
    print(f"Actual params: {model.num_params():,}\n")

    B, T = 2, 32
    tokens = torch.randint(0, 256, (B, T), device=device)

    logits = model(tokens)
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (B, T, 256)

    loss = model(tokens, return_loss=True)
    print(f"Loss (random init): {loss.item():.4f}")
    loss.backward()
    print(f"W_h_time.grad: {model.W_h_time.grad.norm().item():.4f}")
    print(f"W_h_layer.grad: {model.W_h_layer.grad.norm().item():.4f}")
    print(f"embed_k.grad: {model.embed_k.weight.grad.norm().item():.4f}")
    print(f"embed_v.grad: {model.embed_v.weight.grad.norm().item():.4f}")
    print(f"head.grad: {model.head.weight.grad.norm().item():.4f}")
    print("PASS" if all(p.grad is not None and p.grad.norm().item() > 0 for p in model.parameters()) else "FAIL")
