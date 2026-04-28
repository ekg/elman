"""Generic trainer: one (model, task) pair, fixed steps, report accuracy.

Uses LadderLM from elman.models for the backbone (handles E88/FLA-GDN/etc.
uniformly with proper prenorm + residual).

Output written to {output_dir}/{label}.json with task accuracy curve.

Usage:
    python train_task.py --task parity --model E88 --dim 256 --depth 4 \\
        --steps 5000 --seq_len 256 --batch_size 64 --label parity_e88
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'elman', 'cuda'))

import torch
import torch.nn.functional as F

from experiments.expressivity_tasks.tasks import ALL_TASKS


def build_model(level, dim, depth, vocab_size, **kwargs):
    """Build a small LadderLM model for the given level."""
    from elman.models import LadderLM
    # Common kwargs
    common = dict(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
    )

    if level == 'E88':
        common.update(
            n_heads=kwargs.get('n_heads', 8),
            n_state=kwargs.get('n_state', 16),
            expansion=1.0,
            use_gate=1,
            gate_activation='silu',
        )
    elif level == 'E91':
        common.update(
            n_heads=kwargs.get('n_heads', 8),
            n_state=kwargs.get('n_state', 16),
            rank=kwargs.get('rank', None),  # None defaults to n_state in E91MatMat
            use_gate=True,
            gate_activation='silu',
        )
    elif level == 'E92':
        common.update(
            n_heads=kwargs.get('n_heads', 8),
            n_state=kwargs.get('n_state', 16),
        )
    elif level == 'fla-gdn':
        common.update(
            expansion=kwargs.get('expansion', 2),
            n_heads=kwargs.get('n_heads', 4),
        )
    elif level == 'mamba2':
        common.update(
            mamba_d_state=kwargs.get('mamba_d_state', 64),
            mamba_expand=kwargs.get('mamba_expand', 2),
        )
    elif level == 'llama':
        common.update(
            n_heads=kwargs.get('n_heads', 4),
            expansion=kwargs.get('expansion', 4),
        )

    return LadderLM(level=level, **common)


def evaluate_accuracy(model, task, B, T, n_batches, rng, device):
    """Compute task accuracy over n_batches batches."""
    model.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            inputs_np, targets_np, mask_np = task.generate_batch(B, T, rng)
            x = torch.from_numpy(inputs_np).to(device)
            y = torch.from_numpy(targets_np).to(device)
            m = torch.from_numpy(mask_np).to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)  # [B, T, V]
            preds = logits.argmax(dim=-1)
            correct += ((preds == y) & m).sum().item()
            total += m.sum().item()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                y.view(-1),
                reduction='none',
            ).view_as(m)
            loss = (loss * m).sum().item() / max(m.sum().item(), 1)
            losses.append(loss)
    return correct / max(total, 1), np.mean(losses)


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Build task
    task_kwargs = {}
    if args.task == 'modular_counter':
        task_kwargs['K'] = args.K
    elif args.task == 'dyck':
        task_kwargs['max_depth'] = args.K
    elif args.task == 'fsm_tracking':
        task_kwargs['n_states'] = args.K
    elif args.task == 'selective_copy':
        task_kwargs['n_to_copy'] = args.K
    elif args.task == 'assoc_recall':
        task_kwargs['n_pairs'] = args.K

    task = ALL_TASKS[args.task](**task_kwargs)
    print(f"Task: {task.name}, vocab_size={task.vocab_size}", flush=True)

    # Build model
    extra_model_kwargs = {}
    if args.n_heads is not None: extra_model_kwargs['n_heads'] = args.n_heads
    if args.n_state is not None: extra_model_kwargs['n_state'] = args.n_state
    if args.expansion is not None: extra_model_kwargs['expansion'] = args.expansion
    if args.rank is not None: extra_model_kwargs['rank'] = args.rank
    model = build_model(args.model, args.dim, args.depth, task.vocab_size, **extra_model_kwargs)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, dim={args.dim}, depth={args.depth}, params={n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    log = {'task': task.name, 'model': args.model, 'dim': args.dim, 'depth': args.depth,
           'seq_len': args.seq_len, 'batch_size': args.batch_size, 'lr': args.lr,
           'seed': args.seed, 'params': n_params,
           'random_baseline_acc': task.random_baseline_acc(),
           'steps': []}

    t0 = time.time()
    eval_interval = max(50, args.steps // 20)

    model.train()
    for step in range(args.steps):
        inputs_np, targets_np, mask_np = task.generate_batch(args.batch_size, args.seq_len, rng)
        x = torch.from_numpy(inputs_np).to(device)
        y = torch.from_numpy(targets_np).to(device)
        m = torch.from_numpy(mask_np).to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
        loss_per = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            y.view(-1),
            reduction='none',
        ).view_as(m)
        loss = (loss_per * m).sum() / m.sum().clamp_min(1)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == args.steps - 1:
            acc, eval_loss = evaluate_accuracy(model, task, args.batch_size, args.seq_len,
                                                4, rng, device)
            elapsed = time.time() - t0
            print(f"  step {step:>5d}  train_loss={loss.item():.4f}  eval_acc={acc:.4f}  eval_loss={eval_loss:.4f}  ({elapsed:.0f}s)", flush=True)
            log['steps'].append({'step': step, 'train_loss': float(loss.item()),
                                  'eval_acc': float(acc), 'eval_loss': float(eval_loss),
                                  'elapsed_s': float(elapsed)})
            model.train()

    # Final eval (more batches for stable estimate)
    acc, eval_loss = evaluate_accuracy(model, task, args.batch_size, args.seq_len,
                                        16, rng, device)
    log['final_acc'] = float(acc)
    log['final_loss'] = float(eval_loss)
    log['elapsed_total_s'] = float(time.time() - t0)
    print(f"\nFINAL: acc={acc:.4f}  loss={eval_loss:.4f}  baseline={task.random_baseline_acc():.4f}", flush=True)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'{args.label}.json')
    with open(out_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"Saved to {out_path}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=list(ALL_TASKS.keys()))
    ap.add_argument('--model', required=True, help='Level for LadderLM (E88, fla-gdn, mamba2, llama)')
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--n_heads', type=int, default=None)
    ap.add_argument('--n_state', type=int, default=None)
    ap.add_argument('--expansion', type=float, default=None)
    ap.add_argument('--rank', type=int, default=None, help='Rank for E91 matrix-matrix update')
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--seq_len', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--K', type=int, default=5, help='Task-specific K (mod base, max depth, n_states, n_to_copy, n_pairs)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--label', required=True)
    ap.add_argument('--output_dir', default='experiments/expressivity_tasks/results')
    args = ap.parse_args()
    train(args)
