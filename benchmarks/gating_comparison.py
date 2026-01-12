#!/usr/bin/env python
"""Compare gating mechanisms: silu vs gelu vs mish vs squared_relu"""
import sys
import os
import time
import torch
import torch.nn.functional as F

sys.stdout = sys.stderr
sys.path.insert(0, "/home/erikg/elman")

import warnings
warnings.filterwarnings("ignore")

from elman.models.ladder_lm import LadderLM
from elman.data.dataset import DocumentStreamDataset
from schedulefree import AdamWScheduleFree

GATE_TYPE = os.environ.get("GATE_TYPE", "silu")
NAME = os.environ.get("NAME", GATE_TYPE.upper())
MAX_STEPS = int(os.environ.get("MAX_STEPS", 500))

DIM, DEPTH, BATCH, SEQ = 1280, 6, 32, 512
SEED = 42

device = torch.device("cuda:0")
dtype = torch.bfloat16

torch.manual_seed(SEED)
dataset = DocumentStreamDataset("data/pile.txt", SEQ, seed=SEED)

# Create E1 model
model = LadderLM(vocab_size=256, dim=DIM, depth=DEPTH, level=1, mamba2_init=True)

# Patch the gating function based on GATE_TYPE
def mish(x):
    return x * torch.tanh(F.softplus(x))

def squared_relu(x):
    return F.relu(x) ** 2

gate_fns = {
    'silu': F.silu,
    'gelu': F.gelu,
    'mish': mish,
    'squared_relu': squared_relu,
}

gate_fn = gate_fns[GATE_TYPE]

# Monkey-patch the cell's forward to use different gating
for layer in model.layers:
    cell = layer.cell  # MambaGatedElman has .cell directly

    def make_patched_forward(gfn):
        def patched_forward(self, x, z, h0=None):
            T, B, D = x.shape
            if h0 is None:
                h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

            h_list = [h0]
            output_list = []

            for t in range(T):
                h_prev = h_list[-1]
                x_t = x[t]
                z_t = z[t]

                raw = x_t @ self.W_x.T + h_prev @ self.W_h.T + self.b
                h_new = torch.tanh(raw)
                h_list.append(h_new)

                # Use patched gate function instead of silu
                gate = gfn(z_t)
                output = h_new * gate
                output_list.append(output)

            h = torch.stack(h_list, dim=0)
            output = torch.stack(output_list, dim=0)
            return output, h

        import types
        return types.MethodType(patched_forward, cell)

    cell.forward = make_patched_forward(gate_fn)

model = model.to(device=device, dtype=dtype)
num_params = sum(p.numel() for p in model.parameters())

print(f"{NAME}: dim={DIM}, depth={DEPTH}, params={num_params/1e6:.1f}M, gate={GATE_TYPE}", flush=True)

optimizer = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train()
optimizer.train()

start = time.time()
tokens_seen = 0
step = 0
losses = []
checkpoints = {}

while step < MAX_STEPS:
    step += 1

    batch_chunks = [dataset[0][0] for _ in range(BATCH)]
    batch = torch.stack(batch_chunks).to(device)

    optimizer.zero_grad()
    loss = model(batch, return_loss=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    tokens_seen += BATCH * SEQ
    losses.append(loss.item())

    if step % 50 == 0 or step == 1:
        avg = sum(losses[-50:]) / min(50, len(losses))
        print(f"  Step {step}: loss={avg:.4f}", flush=True)
        if step in [50, 100, 200, 300, 400, 500]:
            checkpoints[step] = avg

elapsed = time.time() - start
final_loss = sum(losses[-100:]) / min(100, len(losses))
throughput = tokens_seen / elapsed

print(f"RESULT|{NAME}|{num_params/1e6:.1f}|{final_loss:.4f}|{throughput/1000:.1f}|{step}|{checkpoints}", flush=True)
