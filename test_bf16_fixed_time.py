"""Fixed-time comparison: which approach gets better loss in 60 seconds?"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import mmap
import numpy as np
sys.path.insert(0, '/home/erikg/elman')

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

class SimpleGatedElman(nn.Module):
    def __init__(self, vocab_size, dim, depth, use_f32_internal=True):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_f32_internal = use_f32_internal
        
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                'W_x': nn.Linear(dim, dim, bias=False),
                'W_h': nn.Linear(dim, dim, bias=False),
                'W_z': nn.Linear(dim, dim * 2, bias=False),
                'ln': nn.LayerNorm(dim),
            }))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        
    def forward(self, x, return_loss=False):
        if return_loss:
            inp, target = x[:, :-1], x[:, 1:]
        else:
            inp = x
            
        B, T = inp.shape
        x = self.embed(inp)
        
        for layer in self.layers:
            residual = x
            x = layer['ln'](x)
            Wx = layer['W_x'](x)
            z = layer['W_z'](x)
            
            h = torch.zeros(B, self.dim, device=x.device, dtype=x.dtype)
            outputs = []
            
            for t in range(T):
                if self.use_f32_internal:
                    pre = Wx[:, t].float() + layer['W_h'](h).float()
                    h_new = torch.tanh(pre).to(x.dtype)
                else:
                    pre = Wx[:, t] + layer['W_h'](h)
                    h_new = torch.tanh(pre)
                
                z_t = z[:, t]
                z_h, z_x = z_t.chunk(2, dim=-1)
                if self.use_f32_internal:
                    gate_h = torch.sigmoid(z_h.float()).to(x.dtype)
                    gate_x = torch.sigmoid(z_x.float()).to(x.dtype)
                else:
                    gate_h = torch.sigmoid(z_h)
                    gate_x = torch.sigmoid(z_x)
                    
                out = gate_h * h_new + gate_x * Wx[:, t]
                outputs.append(out)
                h = h_new
                
            x = residual + torch.stack(outputs, dim=1)
        
        x = self.norm(x)
        logits = self.head(x)
        
        if return_loss:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.reshape(-1))
            return loss
        return logits

data_path = '/home/erikg/elman/data/pile.txt'
with open(data_path, 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

def get_batch(batch_size, seq_len):
    positions = np.random.randint(0, data_len - seq_len - 1, size=batch_size)
    buf = np.zeros((batch_size, seq_len + 1), dtype=np.uint8)
    for i, pos in enumerate(positions):
        buf[i] = np.frombuffer(mm[pos:pos + seq_len + 1], dtype=np.uint8)
    return torch.from_numpy(buf.astype(np.int64)).cuda()

dim, depth = 256, 4
batch_size, seq_len = 32, 256
time_limit = 60  # seconds

device = 'cuda'
dtype = torch.bfloat16

print(f"Fixed-time comparison: {time_limit} seconds each")
print(f"Config: dim={dim}, depth={depth}, batch={batch_size}, seq={seq_len}\n")

results = {}
for use_f32 in [True, False]:
    mode = "f32-internal" if use_f32 else "pure-bf16"
    print(f"{'='*60}")
    print(f"Mode: {mode}")
    
    torch.manual_seed(42)
    model = SimpleGatedElman(256, dim, depth, use_f32_internal=use_f32).to(device).to(dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    losses = []
    step = 0
    torch.cuda.synchronize()
    start = time.time()
    
    while time.time() - start < time_limit:
        x = get_batch(batch_size, seq_len)
        loss = model(x, return_loss=True)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        step += 1
        
        if step % 50 == 0:
            elapsed = time.time() - start
            avg = sum(losses[-50:]) / 50
            print(f"  Step {step} ({elapsed:.1f}s): loss={avg:.4f}")
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    final_loss = sum(losses[-50:]) / 50
    results[mode] = {
        'steps': step,
        'loss': final_loss,
        'tok_per_s': (step * batch_size * seq_len) / elapsed
    }
    print(f"\nTotal steps: {step}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Throughput: {results[mode]['tok_per_s']/1000:.1f}K tok/s")
    print()

print("="*60)
print("SUMMARY")
print("="*60)
for mode, r in results.items():
    print(f"{mode:15s}: {r['steps']} steps, loss={r['loss']:.4f}, {r['tok_per_s']/1000:.1f}K tok/s")

f32_loss = results['f32-internal']['loss']
bf16_loss = results['pure-bf16']['loss']
print(f"\nIn {time_limit}s, pure-bf16 achieved {100*(f32_loss-bf16_loss)/f32_loss:+.1f}% loss difference")
print(f"Steps ratio: {results['pure-bf16']['steps']/results['f32-internal']['steps']:.2f}x")

mm.close()
