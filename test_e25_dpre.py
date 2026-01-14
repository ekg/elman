"""Compare d_pre_act between Python and CUDA."""

import torch
import math

torch.manual_seed(42)

B = 1
T = 1
D = 256
N = 8

device = 'cuda'
dtype = torch.bfloat16

from elman.models.e25_entmax import E25DualMemoryElmanCell

cell = E25DualMemoryElmanCell(dim=D, n_slots=N).to(device).to(dtype)
x = torch.randn(B, T, D, device=device, dtype=dtype)

# Run Python forward
h_tape_init = torch.zeros(B, N, D, device=device, dtype=dtype)
h_work_init = torch.zeros(B, D, device=device, dtype=dtype)

from elman.models.e25_entmax import e25_forward_step_python
scale = 1.0 / math.sqrt(D)

h_work_py, h_tape_py, read_attn_py, write_attn_py, _ = e25_forward_step_python(
    x[:, 0].float(),
    h_tape_init.float(),
    h_work_init.float(),
    cell.W_h.float(), cell.W_x.float(), cell.b_h.float(), cell.W_write.float(),
    scale
)

print("Forward h_work (Python):", h_work_py[0, :4].tolist())

# Compute Python d_pre_act
d_h_work = torch.ones_like(h_work_py)  # Upstream gradient is all ones
d_pre_act_py = d_h_work * (1 - h_work_py ** 2)
print("Python d_pre_act norm:", d_pre_act_py.norm().item())
print("Python d_pre_act sample:", d_pre_act_py[0, :4].tolist())

# Compute Python dW_x
dW_x_py = d_pre_act_py.T @ x[:, 0].float()
print("Python dW_x norm:", dW_x_py.norm().item())
print("Python dW_x sample:", dW_x_py[0, :4].tolist())

# Now run CUDA forward
print("\n--- CUDA ---")
out_cuda, h_tape_cuda, h_work_cuda = cell(x, use_cuda=True)
print("Forward h_work (CUDA):", out_cuda[0, 0, :4].float().tolist())

# Run CUDA backward
out_cuda_grad = out_cuda.clone().requires_grad_(True)
loss = out_cuda.sum()
loss.backward()

print("CUDA dW_x norm:", cell.W_x.grad.float().norm().item())
print("CUDA dW_x sample:", cell.W_x.grad[0, :4].float().tolist())

# Compare
print("\n--- Comparison ---")
print("Forward diff:", (h_work_py.to(dtype).float() - out_cuda.float()).abs().max().item())
print("dW_x diff:", (dW_x_py.to(dtype).float() - cell.W_x.grad.float()).abs().max().item())
