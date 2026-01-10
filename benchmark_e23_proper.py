#!/usr/bin/env python3
"""
Proper E23 vs E1 vs Mamba2 benchmark with optimal configs.
Uses batch_size=256, dim=1280, depth=6 for E1, matching the standard setup.
"""
import sys; sys.path.insert(0, '/home/erikg/elman')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM

# Import E23 model
sys.path.insert(0, 'elman/cuda')
import hasty_pytorch_lib as hasty

torch.manual_seed(42); np.random.seed(42)
batch_size, time_limit = 64, 600  # 64 to fit E23's larger memory footprint
seq_len = 512

# Data setup
with open('/home/erikg/elman/data/pile.txt', 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

def get_batch(buf, mm, data_len, batch_size, seq_len):
    pos = np.random.randint(0, data_len - seq_len - 1, size=batch_size)
    for j, p in enumerate(pos):
        buf[j] = np.frombuffer(mm[p:p+seq_len+1], dtype=np.uint8)
    return torch.from_numpy(buf.astype(np.int64)).cuda()

def train_model(name, model, batch_size, time_limit):
    """Train a model for time_limit seconds."""
    model = model.cuda().bfloat16()
    print(f'{name}: params={model.get_num_params():,}', flush=True)

    opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
    model.train(); opt.train()
    buf = np.zeros((batch_size, seq_len+1), dtype=np.uint8)
    losses = []; start = time.time(); step = 0

    while time.time() - start < time_limit:
        step += 1
        batch = get_batch(buf, mm, data_len, batch_size, seq_len)
        opt.zero_grad()
        loss = model(batch, return_loss=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 50 == 0:
            elapsed = time.time() - start
            tokens = step * batch_size * seq_len
            print(f'[{name}] Step {step} | {elapsed:.0f}s | Loss {loss.item():.4f} | Avg100 {np.mean(losses[-100:]):.4f} | {int(tokens/elapsed)} tok/s', flush=True)

    elapsed = time.time() - start
    tokens = step * batch_size * seq_len
    avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
    tps = int(tokens/elapsed)
    print(f'FINAL {name}: steps={step}, tokens={tokens/1e6:.1f}M, loss={avg_loss:.4f}, tok/s={tps/1000:.1f}K', flush=True)

    del model, opt
    torch.cuda.empty_cache()
    return {'name': name, 'loss': avg_loss, 'tps': tps, 'steps': step}


if __name__ == '__main__':
    import argparse
    from elman.models import create_ladder_model
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['e1', 'e23', 'mamba2'])
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--time', type=int, default=600)
    args = parser.parse_args()

    batch_size = args.batch
    time_limit = args.time

    if args.model == 'e1':
        # E1 d1280×6: ~50M params, optimal config
        model = LadderLM(vocab_size=256, dim=1280, depth=6, level=1)
        train_model('E1_d1280x6', model, batch_size, time_limit)

    elif args.model == 'e23':
        # E23: Use create_ladder_model to get correct params
        model = create_ladder_model(target_params='50m', level=23, vocab_size=256)
        train_model('E23_50M', model, batch_size, time_limit)

    elif args.model == 'mamba2':
        from elman.models.mamba2_baseline import create_mamba2_model
        model = create_mamba2_model(target_params='50m', vocab_size=256)
        train_model('Mamba2_50M', model, batch_size, time_limit)

    mm.close()
