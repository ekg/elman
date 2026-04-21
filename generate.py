#!/usr/bin/env python3
"""
Autoregressive text generation from trained Elman model checkpoints.

Usage:
    python generate.py --checkpoint output/levelE88_100m_.../checkpoint_step_001000_loss_1.3060.pt
    python generate.py --checkpoint output/levelE88_100m_.../checkpoint_step_001000_loss_1.3060.pt --prompt "The meaning of life"
    python generate.py --checkpoint output/levelE88_100m_.../checkpoint_step_001000_loss_1.3060.pt --temperature 0.5 --max_tokens 1000 --top_k 20
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add elman package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add CUDA extension directory for hasty_pytorch_lib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elman', 'cuda'))

import torch
import torch.nn.functional as F


def parse_level(level_str):
    """Parse level string to int or keep as string for log-space levels."""
    if level_str.startswith('log_'):
        return level_str
    try:
        return int(level_str)
    except ValueError:
        return level_str


def load_model(checkpoint_path):
    """Load model from checkpoint, auto-discovering args.json in the parent directory."""
    checkpoint_path = Path(checkpoint_path).resolve()

    # Handle 'latest.pt' symlink
    if checkpoint_path.is_symlink():
        checkpoint_path = checkpoint_path.parent / os.readlink(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Find args.json in checkpoint's parent directory
    checkpoint_dir = checkpoint_path.parent
    args_path = checkpoint_dir / 'args.json'
    if not args_path.exists():
        raise FileNotFoundError(
            f"args.json not found in {checkpoint_dir}. "
            f"Expected it alongside the checkpoint file."
        )

    with open(args_path) as f:
        args = json.load(f)

    level_str = args['level']
    level = parse_level(level_str)

    # Determine model type and construct
    if level_str.lower() == 'mamba2':
        print(f"WARNING: Mamba2 models use a different LM wrapper (Mamba2LM).")
        print(f"         Generation may not work correctly with return_prev_hiddens.")
        print(f"         Attempting to load anyway...")
        from elman.models.mamba2_baseline import Mamba2LM
        model = Mamba2LM(
            vocab_size=256,
            dim=args['dim'],
            depth=args['depth'],
            d_state=args.get('mamba_d_state', 64),
            expand=args.get('mamba_expand', 2),
        )
        model_type = 'mamba2'

    elif level_str.lower() == 'e88_fused':
        from elman.models.e88_fused import E88FusedLM
        model = E88FusedLM(
            vocab_size=256,
            dim=args['dim'],
            depth=args['depth'],
            n_heads=args.get('n_heads'),
            n_state=args.get('n_state', 64),
            expansion=args.get('expansion', 1.0),
            use_gate=bool(args.get('use_gate', 1)),
            checkpoint_interval=args.get('checkpoint_interval', 16),
        )
        model_type = 'e88_fused'

    elif level_str.lower() in ('mingru', 'minlstm', 'cudagru', 'cudalstm'):
        print(f"WARNING: {level_str} models use specialized LM wrappers.")
        print(f"         Generation with return_prev_hiddens is not supported.")
        sys.exit(1)

    else:
        # Standard LadderLM models (E0, E1, E88, E42, fla-gdn, llama, etc.)
        from elman.models import LadderLM
        model = LadderLM(
            vocab_size=256,
            dim=args['dim'],
            depth=args['depth'],
            level=level,
            expansion=args.get('expansion', 1.0),
            n_state=args.get('n_state', 64),
            n_heads=args.get('n_heads', None),
            use_gate=bool(args.get('use_gate', 1)),
            gate_activation=args.get('gate_activation', 'sigmoid'),
            linear_state=bool(args.get('linear_state', 0)),
            use_write_gate=bool(args.get('use_write_gate', 0)),
            checkpoint_interval=args.get('checkpoint_interval', 16),
            gradient_checkpointing=False,
            projection_chunk_size=0,
        )
        model_type = 'ladder'

    # Load checkpoint weights
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])

    # Schedule-free quirk: in this version, the x-mode (eval) extrapolated weights
    # produce catastrophic loss at inference (~20 nats), while the y-mode weights
    # the optimizer uses during training match the reported training loss (~0.8).
    # train.py saves x-mode (via optimizer.eval() before save), so we must swap
    # back to y-mode via optimizer.train() to get usable weights.
    if args.get('optimizer', 'adamw') == 'schedulefree' and 'optimizer_state_dict' in ckpt:
        try:
            import schedulefree
            optimizer = schedulefree.AdamWScheduleFree(
                model.parameters(),
                lr=args.get('lr', 3e-4),
                weight_decay=args.get('weight_decay', 0.01),
            )
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            optimizer.train()  # swap params from loaded x-mode back to y-mode (the good weights)
            print("  Schedule-free: swapped params to y-mode (training weights) for inference")
        except Exception as e:
            print(f"  WARNING: failed to apply schedule-free swap: {e}")

    step = ckpt.get('step', '?')
    loss = ckpt.get('loss', '?')
    print(f"  Step: {step}, Loss: {loss}")
    print(f"  Model: {level_str}, dim={args['dim']}, depth={args['depth']}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    return model, model_type, args


def sample_top_k(logits, temperature, top_k):
    """Sample from logits with temperature and top-k filtering."""
    if temperature <= 0:
        # Greedy
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_k > 0 and top_k < logits.size(-1):
        # Zero out everything outside top-k
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_value = values[..., -1:]
        logits = torch.where(logits < min_value, torch.full_like(logits, float('-inf')), logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(probs.shape[:-1])


@torch.no_grad()
def generate_ladder(model, prompt_tokens, max_tokens, temperature, top_k, device):
    """Stateful autoregressive generation: O(T) per step amortized.

    Feeds the whole prompt in one forward (chunked/parallel kernels), then
    feeds single tokens with carried per-layer state (fast recurrent path).
    FLA-GDN uses its Cache; E88/E1H use their hidden state tensors.
    """
    model.eval()
    generated = list(prompt_tokens)
    hiddens = None

    # Ingest the prompt in one forward (uses chunked kernels for efficiency)
    if len(prompt_tokens) > 0:
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        logits, (hiddens, _) = model(
            prompt_tensor, return_loss=False,
            return_prev_hiddens=True, prev_hiddens=None,
        )
        next_logits = logits[0, -1, :]
        next_token = sample_top_k(next_logits.unsqueeze(0), temperature, top_k).item()
        generated.append(next_token)
        sys.stdout.write(bytes([next_token]).decode('utf-8', errors='replace'))
        sys.stdout.flush()

    # Single-token steps with carried state
    remaining = max_tokens - (1 if len(prompt_tokens) > 0 else 0)
    for _ in range(remaining):
        token_tensor = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
        logits, (hiddens, _) = model(
            token_tensor, return_loss=False,
            return_prev_hiddens=True, prev_hiddens=hiddens,
        )
        next_logits = logits[0, -1, :]
        next_token = sample_top_k(next_logits.unsqueeze(0), temperature, top_k).item()
        generated.append(next_token)
        sys.stdout.write(bytes([next_token]).decode('utf-8', errors='replace'))
        sys.stdout.flush()

    return generated


@torch.no_grad()
def generate_e88_fused(model, prompt_tokens, max_tokens, temperature, top_k, device):
    """Generate tokens using E88FusedLM (no return_prev_hiddens, uses layer-level hidden)."""
    model.eval()

    generated = list(prompt_tokens)

    # E88FusedLM does not support return_prev_hiddens at the LM level,
    # so we manually manage hidden states per layer.
    layer_hiddens = [None] * len(model.layers)

    def forward_one_token(token_id, layer_hiddens):
        """Forward a single token through E88FusedLM, managing hidden states manually."""
        token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)
        h = model.embed(token_tensor)  # [1, 1, dim]

        new_hiddens = []
        for idx, (norm, layer) in enumerate(zip(model.norms, model.layers)):
            residual = h
            h_normed = norm(h)
            out, h_final = layer(h_normed, hidden=layer_hiddens[idx])
            h = residual + out
            new_hiddens.append(h_final)

        h = model.final_norm(h)
        if model.tie_embeddings:
            logits = F.linear(h, model.embed.weight)
        else:
            logits = model.head(h)

        return logits, new_hiddens

    # Process prompt
    if len(prompt_tokens) > 0:
        # Feed prompt tokens one at a time to accumulate hidden state
        for tok in prompt_tokens[:-1]:
            _, layer_hiddens = forward_one_token(tok, layer_hiddens)

        # Last prompt token: get logits for next prediction
        logits, layer_hiddens = forward_one_token(prompt_tokens[-1], layer_hiddens)
        next_logits = logits[0, -1, :]
        next_token = sample_top_k(next_logits.unsqueeze(0), temperature, top_k).item()
        generated.append(next_token)
        sys.stdout.write(bytes([next_token]).decode('utf-8', errors='replace'))
        sys.stdout.flush()
        tokens_to_generate = max_tokens - 1
    else:
        # No prompt: start with a zero token or newline
        tokens_to_generate = max_tokens

    # Generate remaining tokens
    for i in range(tokens_to_generate):
        logits, layer_hiddens = forward_one_token(generated[-1], layer_hiddens)
        next_logits = logits[0, -1, :]
        next_token = sample_top_k(next_logits.unsqueeze(0), temperature, top_k).item()
        generated.append(next_token)
        sys.stdout.write(bytes([next_token]).decode('utf-8', errors='replace'))
        sys.stdout.flush()

    return generated


def main():
    parser = argparse.ArgumentParser(description='Generate text from trained Elman model checkpoints')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint .pt file')
    parser.add_argument('--prompt', type=str, default='',
                        help='Text prompt to condition generation on')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (0 = greedy)')
    parser.add_argument('--max_tokens', type=int, default=500,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--top_k', type=int, default=40,
                        help='Top-k sampling (0 = no filtering)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Device: {device}")

    # Load model
    model, model_type, model_args = load_model(args.checkpoint)
    model = model.to(device)
    if device.type == 'cuda':
        model = model.bfloat16()

    # Encode prompt as bytes (byte-level, vocab_size=256)
    prompt_tokens = list(args.prompt.encode('utf-8')) if args.prompt else []

    print(f"\nGeneration config: temperature={args.temperature}, top_k={args.top_k}, max_tokens={args.max_tokens}")
    if args.prompt:
        print(f"Prompt ({len(prompt_tokens)} bytes): {repr(args.prompt)}")
    print(f"\n--- Generated text ---")

    # Print prompt first
    if args.prompt:
        sys.stdout.write(args.prompt)
        sys.stdout.flush()

    # Generate
    if model_type == 'e88_fused':
        generated = generate_e88_fused(model, prompt_tokens, args.max_tokens, args.temperature, args.top_k, device)
    else:
        # LadderLM and mamba2 both support return_prev_hiddens-style interface
        generated = generate_ladder(model, prompt_tokens, args.max_tokens, args.temperature, args.top_k, device)

    print(f"\n--- End ({len(generated)} bytes total) ---")


if __name__ == '__main__':
    main()
