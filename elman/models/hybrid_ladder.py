"""HybridLadderLM — LadderLM with per-layer architecture.

Accepts a `layer_pattern` (list of level names) instead of a single `level`.
Each entry is one layer's architecture. The pattern repeats to fill `depth`.

Examples:
  ['E88', 'fla-gdn']                  → alternating E88/FLA every layer
  ['E88', 'E88', 'fla-gdn', 'fla-gdn'] → 2 E88 then 2 FLA, repeat
  ['E88']                             → all E88 (equivalent to LadderLM(level='E88'))

Each layer can have different shape kwargs by passing `layer_kwargs`
(list-of-dicts, one per pattern entry).

Same residual + RMSNorm wrapping as LadderLM.
"""
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn

from .ladder_lm import RMSNorm, get_ladder_level


class HybridLadderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 512,
        depth: int = 12,
        layer_pattern: Optional[List[str]] = None,
        layer_kwargs: Optional[List[Dict[str, Any]]] = None,
        # Defaults shared across all layers (overridden by per-layer kwargs)
        n_state: int = 16,
        n_heads: int = 4,
        expansion: float = 1.0,
        rank: Optional[int] = None,
        use_gate: bool = True,
        gate_activation: str = 'silu',
        dropout: float = 0.0,
        **extra_kwargs,
    ):
        super().__init__()
        if layer_pattern is None:
            layer_pattern = ['E88']
        if layer_kwargs is None:
            layer_kwargs = [{}] * len(layer_pattern)
        assert len(layer_pattern) == len(layer_kwargs), \
            f"layer_pattern ({len(layer_pattern)}) and layer_kwargs ({len(layer_kwargs)}) must match"

        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.layer_pattern = layer_pattern

        self.embed = nn.Embedding(vocab_size, dim)

        self.layer_norms = nn.ModuleList([RMSNorm(dim) for _ in range(depth)])

        layers = []
        actual_pattern = []
        for i in range(depth):
            level = layer_pattern[i % len(layer_pattern)]
            kw = layer_kwargs[i % len(layer_kwargs)]
            actual_pattern.append(level)

            LayerClass = get_ladder_level(level)
            base_kwargs = {
                'dim': dim,
                'n_state': n_state,
                'n_heads': n_heads,
                'expansion': expansion,
                'use_gate': use_gate,
                'gate_activation': gate_activation,
                'dropout': dropout,
            }
            if rank is not None:
                base_kwargs['rank'] = rank
            base_kwargs.update(kw)

            try:
                layer = LayerClass(**base_kwargs)
            except TypeError as e:
                # Some layer classes don't accept all of these kwargs.
                # Try with a smaller set.
                minimal = {'dim': dim}
                for k in ('n_state', 'n_heads', 'expansion', 'rank', 'use_gate',
                          'gate_activation', 'dropout'):
                    if k in base_kwargs:
                        try:
                            test = LayerClass(**minimal, **{k: base_kwargs[k]})
                            minimal[k] = base_kwargs[k]
                        except TypeError:
                            pass
                layer = LayerClass(**minimal, **kw)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.actual_pattern = actual_pattern

        self.out_norm = RMSNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)
        # Tie output to embedding (saves params)
        self.out_proj.weight = self.embed.weight

    def forward(self, x: torch.Tensor, return_loss: bool = False, targets: Optional[torch.Tensor] = None):
        h = self.embed(x)  # [B, T, dim]
        for ln, layer in zip(self.layer_norms, self.layers):
            normed = ln(h)
            out = layer(normed)
            if isinstance(out, tuple):
                out = out[0]
            h = h + out  # residual
        h = self.out_norm(h)
        logits = self.out_proj(h)
        if return_loss:
            tgt = targets if targets is not None else x
            import torch.nn.functional as F
            return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                    tgt[:, 1:].reshape(-1))
        return logits


# ============================================================================
# Self-test
# ============================================================================
if __name__ == '__main__':
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Hybrid: alternating E88 and FLA-GDN
    model = HybridLadderLM(
        vocab_size=4, dim=128, depth=4,
        layer_pattern=['E88', 'fla-gdn'],
        n_state=16, n_heads=4,
    ).to(device)
    print(f"layer pattern: {model.actual_pattern}")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randint(0, 4, (2, 32), device=device)
    logits = model(x)
    print(f"logits: {tuple(logits.shape)}")

    # Backward
    loss = model(x, return_loss=True)
    loss.backward()
    print(f"loss: {loss.item():.4f} — backward OK")
