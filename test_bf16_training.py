"""Compare training with f32-internal vs pure-bf16 element-wise ops."""

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

# Simple gated Elman for testing (pure PyTorch, we control precision)
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
        x = self.embed(inp)  # [B, T, dim]
        
        for layer in self.layers:
            residual = x
            x = layer['ln'](x)
            
            # Pre-compute projections
            Wx = layer['W_x'](x)  # [B, T, dim]
            z = layer['W_z'](x)   # [B, T, 2*dim]
            
            # Sequential RNN
            h = torch.zeros(B, self.dim, device=x.device, dtype=x.dtype)
            outputs = []
            
            for t in range(T):
                if self.use_f32_internal:
                    # Current approach: f32 internal
                    pre = Wx[:, t].float() + layer['W_h'](h).float()
                    h_new = torch.tanh(pre).to(x.dtype)
                else:
                    # Pure bf16
                    pre = Wx[:, t] + layer['W_h'](h)
                    h_new = torch.tanh(pre)
                
                # Gating
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

# Data loading
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

# Training params
dim, depth = 256, 4  # Small model for quick test
batch_size, seq_len = 32, 256
n_steps = 200

device = 'cuda'
dtype = torch.bfloat16

print(f"Testing dim={dim}, depth={depth}, batch={batch_size}, seq={seq_len}")
print(f"Training for {n_steps} steps each\n")

for use_f32 in [True, False]:
    mode = "f32-internal" if use_f32 else "pure-bf16"
    print(f"{'='*60}")
    print(f"Mode: {mode}")
    print(f"{'='*60}")
    
    torch.manual_seed(42)
    model = SimpleGatedElman(256, dim, depth, use_f32_internal=use_f32).to(device).to(dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    losses = []
    torch.cuda.synchronize()
    start = time.time()
    
    for step in range(n_steps):
        x = get_batch(batch_size, seq_len)
        loss = model(x, return_loss=True)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        
        if (step + 1) % 50 == 0:
            avg = sum(losses[-50:]) / 50
            print(f"  Step {step+1}: loss={avg:.4f}")
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    final_loss = sum(losses[-50:]) / 50
    print(f"\nFinal loss (last 50): {final_loss:.4f}")
    print(f"Time: {elapsed:.1f}s ({n_steps/elapsed:.1f} steps/s)")
    print()

mm.close()
