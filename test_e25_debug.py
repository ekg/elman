"""Debug script for E25 CUDA kernel vs Python reference."""

import torch
import math

torch.manual_seed(42)

# Test parameters
B = 2  # batch size
T = 4  # sequence length
D = 256  # dimension
N = 8  # slots

device = 'cuda'
dtype = torch.bfloat16

print("="*60)
print("E25 CUDA vs Python Debug")
print("="*60)

# Check if CUDA kernel is available
print("\n1. Checking CUDA kernel availability...")
try:
    import hasty_pytorch_lib
    has_e25_forward = hasattr(hasty_pytorch_lib, 'e25_entmax_forward')
    has_e25_backward = hasattr(hasty_pytorch_lib, 'e25_entmax_backward')
    print(f"   hasty_pytorch_lib loaded: True")
    print(f"   e25_entmax_forward: {has_e25_forward}")
    print(f"   e25_entmax_backward: {has_e25_backward}")
except ImportError as e:
    print(f"   hasty_pytorch_lib import failed: {e}")
    has_e25_forward = False

# Import the Python reference implementation
from elman.models.e25_entmax import (
    e25_forward_step_python,
    e25_sequence_python,
    entmax_1_5,
    E25DualMemoryElmanCell,
    E25DualMemoryElmanFunction
)

print("\n2. Testing 1.5-entmax Python implementation...")
z = torch.randn(B, N, device=device, dtype=torch.float32)
p = entmax_1_5(z, dim=-1)
print(f"   Input scores (sample): {z[0, :4].tolist()}")
print(f"   Output probs (sample): {p[0, :4].tolist()}")
print(f"   Sum to 1: {p.sum(dim=-1).tolist()}")
print(f"   Sparsity: {(p == 0).float().mean().item():.1%}")

print("\n3. Testing forward pass (Python vs CUDA)...")

# Create test inputs
x_seq = torch.randn(B, T, D, device=device, dtype=dtype)
h_tape_init = torch.zeros(B, N, D, device=device, dtype=dtype)
h_work_init = torch.zeros(B, D, device=device, dtype=dtype)
W_h = torch.randn(D, D, device=device, dtype=dtype) * 0.1
W_x = torch.randn(D, D, device=device, dtype=dtype) * 0.1
b_h = torch.zeros(D, device=device, dtype=dtype)
W_write = torch.randn(D, D, device=device, dtype=dtype) * 0.1

scale = 1.0 / math.sqrt(D)

# Run Python reference
h_work_py, h_tape_final_py, _, read_attn_py, write_attn_py = e25_sequence_python(
    x_seq.float(),
    h_tape_init.float(),
    h_work_init.float(),
    W_h.float(), W_x.float(), b_h.float(), W_write.float()
)
h_work_py = h_work_py.to(dtype)
h_tape_final_py = h_tape_final_py.to(dtype)

print(f"   Python h_work_all shape: {h_work_py.shape}")
print(f"   Python h_work_all sample: {h_work_py[0, 0, :4].float().tolist()}")

# Run CUDA kernel if available
if has_e25_forward:
    print("\n   Running CUDA kernel...")
    # Pre-compute x_proj for CUDA path
    x_proj = (x_seq @ W_x.T).contiguous()

    h_work_cuda, h_tape_final_cuda, h_tape_all_cuda, read_attn_cuda, write_attn_cuda = \
        hasty_pytorch_lib.e25_entmax_forward(
            True,  # training
            x_proj,  # Note: CUDA expects x_proj, not x_seq
            h_tape_init.contiguous(),
            h_work_init.contiguous(),
            W_h.contiguous(),
            W_x.contiguous(),  # Not used in CUDA since we pass x_proj
            b_h.contiguous(),
            W_write.contiguous()
        )

    print(f"   CUDA h_work_all shape: {h_work_cuda.shape}")
    print(f"   CUDA h_work_all sample: {h_work_cuda[0, 0, :4].float().tolist()}")

    # Compare
    diff = (h_work_py.float() - h_work_cuda.float()).abs()
    print(f"\n   Max diff (h_work): {diff.max().item():.6f}")
    print(f"   Mean diff (h_work): {diff.mean().item():.6f}")

    # Check attention weights
    read_attn_py_stacked = torch.stack([r.to(dtype) for r in read_attn_py], dim=1)  # [B, T, N]
    print(f"\n   Python read_attn sample (t=0): {read_attn_py_stacked[0, 0, :4].float().tolist()}")
    print(f"   CUDA read_attn sample (t=0): {read_attn_cuda[0, 0, :4].float().tolist()}")

    attn_diff = (read_attn_py_stacked.float() - read_attn_cuda.float()).abs()
    print(f"   Max diff (read_attn): {attn_diff.max().item():.6f}")

    write_attn_py_stacked = torch.stack([w.to(dtype) for w in write_attn_py], dim=1)
    print(f"\n   Python write_attn sample (t=0): {write_attn_py_stacked[0, 0, :4].float().tolist()}")
    print(f"   CUDA write_attn sample (t=0): {write_attn_cuda[0, 0, :4].float().tolist()}")

    write_attn_diff = (write_attn_py_stacked.float() - write_attn_cuda.float()).abs()
    print(f"   Max diff (write_attn): {write_attn_diff.max().item():.6f}")

else:
    print("   CUDA kernel not available, skipping CUDA test")

print("\n4. Testing E25DualMemoryElmanCell (use_cuda=False)...")
cell = E25DualMemoryElmanCell(dim=D, n_slots=N).to(device).to(dtype)
x_seq_test = torch.randn(B, T, D, device=device, dtype=dtype)

# Python path
h_work_out_py, h_tape_final_py, h_work_final_py = cell(x_seq_test, use_cuda=False)
print(f"   Python output shape: {h_work_out_py.shape}")
print(f"   Python output sample: {h_work_out_py[0, 0, :4].float().tolist()}")

print("\n5. Testing E25DualMemoryElmanCell (use_cuda=True)...")
# CUDA path
h_work_out_cuda, h_tape_final_cuda, h_work_final_cuda = cell(x_seq_test, use_cuda=True)
print(f"   CUDA output shape: {h_work_out_cuda.shape}")
print(f"   CUDA output sample: {h_work_out_cuda[0, 0, :4].float().tolist()}")

# Compare cell outputs
cell_diff = (h_work_out_py.float() - h_work_out_cuda.float()).abs()
print(f"\n   Max diff: {cell_diff.max().item():.6f}")
print(f"   Mean diff: {cell_diff.mean().item():.6f}")

# Check if there's a significant discrepancy
if cell_diff.max().item() > 0.1:
    print("\n   WARNING: Large difference detected between Python and CUDA!")
    print("   This could be the source of the loss explosion.")
else:
    print("\n   Forward pass looks OK!")

print("\n6. Testing backward pass...")
cell.zero_grad()
x_seq_grad = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)

# Python backward
h_work_out_py2, _, _ = cell(x_seq_grad, use_cuda=False)
loss_py = h_work_out_py2.sum()
loss_py.backward()
grad_py = x_seq_grad.grad.clone()
print(f"   Python grad norm: {grad_py.norm().item():.6f}")

cell.zero_grad()
x_seq_grad2 = x_seq_grad.detach().clone().requires_grad_(True)

# CUDA backward
h_work_out_cuda2, _, _ = cell(x_seq_grad2, use_cuda=True)
loss_cuda = h_work_out_cuda2.sum()
loss_cuda.backward()
grad_cuda = x_seq_grad2.grad.clone()
print(f"   CUDA grad norm: {grad_cuda.norm().item():.6f}")

grad_diff = (grad_py.float() - grad_cuda.float()).abs()
print(f"\n   Max grad diff: {grad_diff.max().item():.6f}")
print(f"   Mean grad diff: {grad_diff.mean().item():.6f}")

if grad_diff.max().item() > 0.1:
    print("\n   WARNING: Large gradient difference detected!")
else:
    print("\n   Backward pass looks OK!")

print("\n7. Full model test...")
from elman.models.ladder_lm import create_ladder_model

model_py = create_ladder_model('50m', vocab_size=256, level=25).cuda().bfloat16()
model_cuda = create_ladder_model('50m', vocab_size=256, level=25).cuda().bfloat16()

# Copy weights
model_cuda.load_state_dict(model_py.state_dict())

x = torch.randint(0, 256, (4, 64), device='cuda')

# Force Python path
for layer in model_py.layers:
    if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'cell'):
        layer.mixer.cell.use_cuda = False

loss_py = model_py(x, return_loss=True)
print(f"   Python model loss: {loss_py.item():.4f}")

loss_cuda = model_cuda(x, return_loss=True)
print(f"   CUDA model loss: {loss_cuda.item():.4f}")

print("\n" + "="*60)
print("Debug complete!")
print("="*60)
