"""Train a HybridLadderLM (per-layer architecture) on a task. Mirrors
train_task.py but uses HybridLadderLM directly so we can pass layer_pattern.

Usage:
    python train_hybrid.py --task parity --layer_pattern E88 fla-gdn \\
        --dim 128 --depth 4 --steps 500 --label hybrid_parity
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from elman.models.hybrid_ladder import HybridLadderLM
from experiments.expressivity_tasks.tasks import ALL_TASKS


def evaluate(model, task, B, T, n_batches, rng, device):
    model.eval()
    correct = total = 0
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            inp, tgt, mask = task.generate_batch(B, T, rng)
            x = torch.from_numpy(inp).to(device)
            y = torch.from_numpy(tgt).to(device)
            m = torch.from_numpy(mask).to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
            preds = logits.argmax(dim=-1)
            correct += ((preds == y) & m).sum().item()
            total += m.sum().item()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                    y.view(-1), reduction='none').view_as(m)
            losses.append((loss * m).sum().item() / max(m.sum().item(), 1))
    return correct / max(total, 1), float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=list(ALL_TASKS.keys()))
    ap.add_argument('--layer_pattern', nargs='+', required=True,
                    help='List of layer levels, e.g. E88 fla-gdn')
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--n_heads', type=int, default=4)
    ap.add_argument('--n_state', type=int, default=16)
    ap.add_argument('--rank', type=int, default=None)
    ap.add_argument('--expansion', type=float, default=1.0)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--seq_len', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'schedulefree'])
    ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--label', required=True)
    ap.add_argument('--output_dir', default='experiments/expressivity_tasks/results')
    ap.add_argument('--eval_lengths', type=int, nargs='+', default=None,
                    help='If set, after training, eval at each of these T values '
                         '(Délétang length-extrapolation protocol). Records per-T '
                         "accuracy under log['length_extrap'].")
    ap.add_argument('--eval_lengths_n_batches', type=int, default=8,
                    help='Number of eval batches per length in --eval_lengths.')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Build task
    task_kwargs = {}
    if args.task == 'modular_counter':       task_kwargs['K'] = args.K
    elif args.task == 'dyck':                 task_kwargs['max_depth'] = args.K
    elif args.task == 'fsm_tracking':         task_kwargs['n_states'] = args.K
    elif args.task == 'selective_copy':       task_kwargs['n_to_copy'] = args.K
    elif args.task == 'assoc_recall':         task_kwargs['n_pairs'] = args.K
    task = ALL_TASKS[args.task](**task_kwargs)
    print(f"Task: {task.name}, vocab_size={task.vocab_size}", flush=True)

    # Build hybrid model
    model = HybridLadderLM(
        vocab_size=task.vocab_size,
        dim=args.dim, depth=args.depth,
        layer_pattern=args.layer_pattern,
        n_state=args.n_state, n_heads=args.n_heads,
        expansion=args.expansion,
        rank=args.rank,
    ).to(device)
    print(f"Pattern: {model.actual_pattern}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}", flush=True)

    if args.optimizer == 'schedulefree':
        import schedulefree
        optimizer = schedulefree.AdamWScheduleFree(
            model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
        print(f"Using schedule-free AdamW (lr={args.lr})", flush=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        print(f"Using vanilla AdamW (lr={args.lr})", flush=True)

    log = {'task': task.name, 'pattern': model.actual_pattern, 'dim': args.dim, 'depth': args.depth,
           'seq_len': args.seq_len, 'batch_size': args.batch_size, 'lr': args.lr,
           'seed': args.seed, 'params': n_params,
           'random_baseline_acc': task.random_baseline_acc(),
           'steps': []}

    t0 = time.time()
    eval_interval = max(50, args.steps // 20)
    model.train()
    if hasattr(optimizer, 'train'): optimizer.train()
    for step in range(args.steps):
        inp, tgt, mask = task.generate_batch(args.batch_size, args.seq_len, rng)
        x = torch.from_numpy(inp).to(device)
        y = torch.from_numpy(tgt).to(device)
        m = torch.from_numpy(mask).to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
        loss_per = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                    y.view(-1), reduction='none').view_as(m)
        loss = (loss_per * m).sum() / m.sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == args.steps - 1:
            if hasattr(optimizer, 'eval'): optimizer.eval()
            acc, eval_loss = evaluate(model, task, args.batch_size, args.seq_len, 4, rng, device)
            if hasattr(optimizer, 'train'): optimizer.train()
            elapsed = time.time() - t0
            print(f"  step {step:>5d}  train_loss={loss.item():.4f}  eval_acc={acc:.4f}  eval_loss={eval_loss:.4f}  ({elapsed:.0f}s)", flush=True)
            log['steps'].append({'step': step, 'train_loss': float(loss.item()),
                                  'eval_acc': float(acc), 'eval_loss': float(eval_loss),
                                  'elapsed_s': float(elapsed)})
            model.train()

    if hasattr(optimizer, 'eval'): optimizer.eval()
    acc, eval_loss = evaluate(model, task, args.batch_size, args.seq_len, 16, rng, device)
    log['final_acc'] = float(acc); log['final_loss'] = float(eval_loss)
    log['elapsed_total_s'] = float(time.time() - t0)
    print(f"\nFINAL: acc={acc:.4f}  loss={eval_loss:.4f}  baseline={task.random_baseline_acc():.4f}", flush=True)

    # Length-extrapolation eval (Délétang protocol): test at lengths the
    # model never trained on. A model that learned the algorithm
    # extrapolates; a model that memorized the training-length
    # distribution does not.
    if args.eval_lengths is not None:
        log['length_extrap'] = {}
        # Use a smaller per-batch B at very long T to avoid OOM.
        for T_eval in args.eval_lengths:
            B_eval = args.batch_size
            # Cap memory: scale batch down for very long sequences.
            if T_eval > 4 * args.seq_len:
                B_eval = max(2, args.batch_size // (T_eval // (4 * args.seq_len)))
            try:
                acc_T, loss_T = evaluate(
                    model, task, B_eval, T_eval,
                    args.eval_lengths_n_batches, rng, device,
                )
                print(f"  length_extrap T={T_eval:>5d} (B={B_eval}): "
                      f"acc={acc_T:.4f}  loss={loss_T:.4f}", flush=True)
                log['length_extrap'][str(T_eval)] = {
                    'acc': float(acc_T),
                    'loss': float(loss_T),
                    'B_eval': int(B_eval),
                }
            except Exception as e:
                print(f"  length_extrap T={T_eval}: ERROR {type(e).__name__}: {e}",
                      flush=True)
                log['length_extrap'][str(T_eval)] = {'error': str(e)}

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'{args.label}.json')
    json.dump(log, open(out_path, 'w'), indent=2)
    print(f"Saved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
