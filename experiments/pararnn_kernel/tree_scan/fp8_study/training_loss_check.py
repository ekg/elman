"""Step 4b: brief training-loss impact check.

Run ~50 training steps of a small E88 model with bf16 hybrid vs fp8
hybrid.  Compare loss trajectories.

Uses real pile.txt data (no mock data).
"""
import os, sys, time
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mmap
import torch

# We install the hybrid; toggling FP8 via env.
# Two subprocess runs so FP8_STORAGE is set at import time for each.

def run_training(use_fp8, n_steps=60, seed=42):
    import os as _os
    _os.environ['CUDA_VISIBLE_DEVICES'] = _os.environ.get('CUDA_VISIBLE_DEVICES', '2')
    if use_fp8:
        _os.environ['ELMAN_PARARNN_FP8'] = '1'
    else:
        _os.environ.pop('ELMAN_PARARNN_FP8', None)

    # Install the hybrid
    from install_hybrid import install
    install()

    from elman.models import LadderLM

    torch.manual_seed(seed)
    # Small E88 model
    dim = 512
    depth = 4
    n_heads = 16  # dim/n_heads = 32, n_state=16 -> balanced
    n_state = 16
    model = LadderLM(
        vocab_size=256, dim=dim, depth=depth, level='E88',
        n_heads=n_heads, n_state=n_state, use_gate=1,
        gate_activation='silu',
    ).to('cuda').to(torch.bfloat16)
    # Force training mode
    model.train()

    # Data loading (same seed so both runs see same batches)
    with open('/home/erikg/elman/data/pile.txt', 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    seq_len = 512
    batch_size = 8
    rng = np.random.RandomState(seed)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    t0 = time.time()
    for step in range(n_steps):
        pos = rng.randint(0, len(mm) - seq_len - 1, size=batch_size)
        buf = np.empty((batch_size, seq_len + 1), dtype=np.uint8)
        for i, p in enumerate(pos):
            buf[i] = np.frombuffer(mm[p:p + seq_len + 1], dtype=np.uint8)
        x = torch.from_numpy(buf).long().to('cuda')
        loss = model(x, return_loss=True)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return losses, elapsed


if __name__ == '__main__':
    # Invoked as subprocess; pick bf16 or fp8 via env var.
    mode = os.environ.get('MODE', 'bf16')
    use_fp8 = (mode == 'fp8')
    losses, elapsed = run_training(use_fp8=use_fp8, n_steps=int(os.environ.get('N_STEPS', '60')))
    # Write to a file tagged by mode
    out = f'/tmp/fp8_train_loss_{mode}.txt'
    with open(out, 'w') as f:
        for l in losses:
            f.write(f'{l}\n')
    print(f'MODE={mode} final_loss={losses[-1]:.4f} avg_last_20={sum(losses[-20:])/20:.4f} elapsed={elapsed:.1f}s')
