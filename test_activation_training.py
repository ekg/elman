"""Fixed-time training comparison of activation functions."""

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

# Activation functions (defined to work with autograd)
class PadeTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x_clamped = x.clamp(-4, 4)
        x2 = x_clamped * x_clamped
        y = x_clamped * (27.0 + x2) / (27.0 + 9.0 * x2)
        ctx.save_for_backward(x_clamped, y)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        # d/dx[x(27+x²)/(27+9x²)] = (27+x²)/(27+9x²) + x*d/dx[(27+x²)/(27+9x²)]
        # Simplified: d/dx = (729 - 27x² - 9x⁴) / (27+9x²)²
        # Or use: d(tanh)/dx ≈ 1 - y² (approximation)
        dy = 1.0 - y * y
        return grad_output * dy

def pade_tanh(x):
    return PadeTanh.apply(x)

class Softsign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        denom = 1.0 + x.abs()
        y = x / denom
        ctx.save_for_backward(denom)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        denom, = ctx.saved_tensors
        # d/dx[x/(1+|x|)] = 1/(1+|x|)²
        return grad_output / (denom * denom)

def softsign(x):
    return Softsign.apply(x)

class GatedElman(nn.Module):
    def __init__(self, vocab_size, dim, depth, activation='tanh'):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.activation = activation
        
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
        
        if activation == 'tanh':
            self.act_fn = torch.tanh
        elif activation == 'pade':
            self.act_fn = pade_tanh
        elif activation == 'softsign':
            self.act_fn = softsign
        
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
                pre = Wx[:, t] + layer['W_h'](h)
                h = self.act_fn(pre)
                
                z_t = z[:, t]
                z_h, z_x = z_t.chunk(2, dim=-1)
                gate_h = torch.sigmoid(z_h)
                gate_x = torch.sigmoid(z_x)
                    
                out = gate_h * h + gate_x * Wx[:, t]
                outputs.append(out)
                
            x = residual + torch.stack(outputs, dim=1)
        
        x = self.norm(x)
        logits = self.head(x)
        
        if return_loss:
            loss = F.cross_entropy(logits.view(-1, self.embed.num_embeddings), target.reshape(-1))
            return loss
        return logits

# Data
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

# Config
dim, depth = 256, 4
batch_size, seq_len = 32, 256
time_limit = 90  # seconds each

device = 'cuda'
dtype = torch.bfloat16

print("=" * 70)
print(f"FIXED-TIME ACTIVATION COMPARISON ({time_limit}s each)")
print(f"Config: dim={dim}, depth={depth}, batch={batch_size}, seq={seq_len}")
print("=" * 70)

activations = ['tanh', 'pade', 'softsign']
results = {}

for act in activations:
    print(f"\n{'='*60}")
    print(f"Activation: {act}")
    
    torch.manual_seed(42)
    model = GatedElman(256, dim, depth, activation=act).to(device).to(dtype)
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
    tok_per_s = (step * batch_size * seq_len) / elapsed
    
    results[act] = {
        'steps': step,
        'loss': final_loss,
        'tok_per_s': tok_per_s
    }
    
    print(f"\nTotal: {step} steps, loss={final_loss:.4f}, {tok_per_s/1000:.1f}K tok/s")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n{'Activation':<15} {'Steps':>8} {'Loss':>10} {'Tok/s':>12} {'vs tanh':>12}")
print("-" * 60)

tanh_loss = results['tanh']['loss']
tanh_tps = results['tanh']['tok_per_s']

for act, r in results.items():
    loss_diff = (r['loss'] - tanh_loss) / tanh_loss * 100
    tps_diff = (r['tok_per_s'] - tanh_tps) / tanh_tps * 100
    print(f"{act:<15} {r['steps']:>8} {r['loss']:>10.4f} {r['tok_per_s']/1000:>10.1f}K {loss_diff:>+10.1f}%")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

# Determine best
best_act = min(results.items(), key=lambda x: x[1]['loss'])
fastest_act = max(results.items(), key=lambda x: x[1]['tok_per_s'])

print(f"""
Best loss:      {best_act[0]} ({best_act[1]['loss']:.4f})
Fastest:        {fastest_act[0]} ({fastest_act[1]['tok_per_s']/1000:.1f}K tok/s)

If {best_act[0]} == {fastest_act[0]}: Clear winner!
Otherwise: Trade-off between speed and quality.
""")

mm.close()
