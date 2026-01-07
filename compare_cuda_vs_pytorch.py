"""Compare CUDA kernel vs PyTorch fallback for E1."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.insert(0, '/home/erikg/elman')

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

# Pure PyTorch E1 implementation (no CUDA kernels)
class PureE1Cell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_h)
        
    def forward(self, x, z, h0):
        # x: [T, B, D], z: [T, B, D], h0: [B, D]
        T, B, D = x.shape
        h = h0
        h_all = []
        outputs = []
        
        # Pre-compute W_x @ x for all timesteps
        Wx = F.linear(x, self.W_x)  # [T, B, D]
        
        for t in range(T):
            # h = tanh(Wx[t] + W_h @ h + b)
            pre = Wx[t] + F.linear(h, self.W_h) + self.b
            h = torch.tanh(pre)
            
            # output = h * silu(z[t])
            out = h * F.silu(z[t])
            
            h_all.append(h)
            outputs.append(out)
        
        return torch.stack(h_all), torch.stack(outputs)

# CUDA E1 cell
from elman.models.mamba_gated_elman import MambaGatedElmanCell, MAMBA_CUDA_AVAILABLE

print("=" * 60)
print("CUDA vs PyTorch Comparison for E1")
print("=" * 60)
print(f"CUDA kernels available: {MAMBA_CUDA_AVAILABLE}")

dim = 1024
batch_size = 32
seq_len = 512
n_iters = 100

device = 'cuda'
dtype = torch.bfloat16

# Test data
x = torch.randn(seq_len, batch_size, dim, device=device, dtype=dtype)
z = torch.randn(seq_len, batch_size, dim, device=device, dtype=dtype)
h0 = torch.zeros(batch_size, dim, device=device, dtype=dtype)

print(f"\nConfig: dim={dim}, batch={batch_size}, seq={seq_len}")
print(f"Testing {n_iters} forward+backward passes each\n")

# Test Pure PyTorch
print("Testing Pure PyTorch E1...")
torch.manual_seed(42)
pure_cell = PureE1Cell(dim).to(device).to(dtype)

# Warmup
for _ in range(5):
    h_all, outputs = pure_cell(x, z, h0)
    loss = outputs.sum()
    loss.backward()
torch.cuda.synchronize()

# Benchmark
torch.cuda.synchronize()
start = time.time()
for _ in range(n_iters):
    h_all, outputs = pure_cell(x, z, h0)
    loss = outputs.sum()
    loss.backward()
torch.cuda.synchronize()
pytorch_time = time.time() - start
pytorch_tps = (n_iters * batch_size * seq_len) / pytorch_time

print(f"  Time: {pytorch_time:.2f}s")
print(f"  Throughput: {pytorch_tps/1000:.1f}K tok/s")

# Test CUDA kernel
if MAMBA_CUDA_AVAILABLE:
    print("\nTesting CUDA E1...")
    torch.manual_seed(42)
    cuda_cell = MambaGatedElmanCell(dim).to(device).to(dtype)
    
    # Warmup
    for _ in range(5):
        h_all, outputs = cuda_cell(x, z, h0)
        loss = outputs.sum()
        loss.backward()
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        h_all, outputs = cuda_cell(x, z, h0)
        loss = outputs.sum()
        loss.backward()
    torch.cuda.synchronize()
    cuda_time = time.time() - start
    cuda_tps = (n_iters * batch_size * seq_len) / cuda_time
    
    print(f"  Time: {cuda_time:.2f}s")
    print(f"  Throughput: {cuda_tps/1000:.1f}K tok/s")
    
    speedup = cuda_tps / pytorch_tps
    print(f"\n{'='*60}")
    print(f"CUDA speedup over PyTorch: {speedup:.2f}x")
    print(f"{'='*60}")
else:
    print("CUDA kernels not available for comparison")
