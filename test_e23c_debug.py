"""
Debug E23c: Step-by-step comparison of Python vs CUDA
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['LD_LIBRARY_PATH'] = f"/home/erikg/.local/lib/python3.12/site-packages/torch/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import torch
import torch.nn.functional as F
import hasty_pytorch_lib
from elman.models.dual_memory_elman_chunked import DualMemoryElmanChunkedCell


def debug_first_timestep():
    """Debug just the first timestep h_work computation."""
    B, D, N, K = 2, 8, 4, 4
    device = 'cuda'
    dtype = torch.bfloat16

    print(f"Testing first timestep: B={B}, D={D}")
    print("="*60)

    # Create cell
    cell = DualMemoryElmanChunkedCell(dim=D, n_slots=N, chunk_size=K).to(device).to(dtype)

    # Simple input
    torch.manual_seed(0)
    x = torch.randn(B, 1, D, device=device, dtype=dtype) * 0.1
    h_tape = torch.zeros(B, N, D, device=device, dtype=dtype)
    h_work = torch.zeros(B, D, device=device, dtype=dtype)

    # Python forward
    with torch.no_grad():
        py_out, py_tape, py_h = cell(x, h_tape.clone(), h_work.clone())

    print(f"Python output[0,0,:4]: {py_out[0,0,:4].tolist()}")

    # CUDA forward - check weight handling
    # nn.Linear stores weight as [out, in] and computes x @ W^T
    # For CUDA, we need to pass weights correctly

    # Check what the Python cell computes for h_work
    # h_work = tanh(W_h @ h_work + W_x @ x + b)
    # With h_work=0: h_work_new = tanh(W_x @ x + b)
    with torch.no_grad():
        Wx_x = F.linear(x[:, 0], cell.W_x.weight)  # [B, D]
        h_work_py = torch.tanh(Wx_x + cell.b_h)
        print(f"Python h_work[0,:4]: {h_work_py[0,:4].tolist()}")

    # Now test CUDA
    # The binding pre-computes x_proj = x @ W_x^T using a GEMM
    # Let's check if the weight transpose is correct

    # In binding: passes W_x directly to cublasGemmEx with CUBLAS_OP_T
    # This computes: W_x @ x^T (treating row-major as col-major)
    # Let me trace through manually

    # PyTorch nn.Linear: output = input @ weight.T
    # So W_x(x) = x @ W_x.weight^T

    # For CUDA binding: we call cublasGemmEx(OP_T, OP_N, ...)
    # Passing W_x.weight as A and x as B
    # In cuBLAS col-major: A_cublas = W_x.weight row-major interpreted = W_x.weight^T col-major
    # With OP_T: op(A) = (W_x.weight^T)^T = W_x.weight
    # B_cublas = x row-major = x^T col-major
    # With OP_N: op(B) = x^T
    # Result: W_x.weight @ x^T col-major = (x @ W_x.weight^T)^T col-major = x @ W_x.weight^T row-major
    # But nn.Linear computes x @ W_x.weight^T !
    # So this should be correct...

    # BUT the validation script transposes the weight before passing:
    # W_x = py_cell.W_x.weight.T.contiguous()
    # So CUDA gets W_x.weight.T, and computes x @ (W_x.weight.T)^T = x @ W_x.weight
    # While Python computes x @ W_x.weight^T
    # This is wrong!

    print("\nChecking weight transpose issue:")
    print(f"W_x.weight shape: {cell.W_x.weight.shape}")

    # Don't transpose - just pass weight directly
    W_x_direct = cell.W_x.weight.contiguous()
    W_h_direct = cell.W_h.weight.contiguous()
    W_write_direct = cell.W_write.weight.contiguous()

    with torch.no_grad():
        cuda_results = hasty_pytorch_lib.e23c_chunked_forward(
            False, x, h_tape.clone(), h_work.clone(),
            W_h_direct, W_x_direct, cell.b_h, W_write_direct, K
        )

    cuda_out = cuda_results[0]
    cuda_h_work = cuda_results[2]

    print(f"CUDA output[0,0,:4] (direct): {cuda_out[0,0,:4].tolist()}")
    print(f"CUDA h_work[0,0,:4] (direct): {cuda_h_work[0,0,:4].tolist()}")

    diff = torch.abs(py_out - cuda_out)
    max_idx = diff.argmax()
    b_idx = max_idx // (py_out.shape[1] * py_out.shape[2])
    t_idx = (max_idx // py_out.shape[2]) % py_out.shape[1]
    d_idx = max_idx % py_out.shape[2]
    print(f"Output diff (direct): {diff.max().item():.6f} at [{b_idx}, {t_idx}, {d_idx}]")
    print(f"  Python: {py_out[b_idx, t_idx, d_idx].item():.6f}")
    print(f"  CUDA:   {cuda_out[b_idx, t_idx, d_idx].item():.6f}")

    # Also try with transpose (what the original validation did)
    W_x_T = cell.W_x.weight.T.contiguous()
    W_h_T = cell.W_h.weight.T.contiguous()
    W_write_T = cell.W_write.weight.T.contiguous()

    with torch.no_grad():
        cuda_results_T = hasty_pytorch_lib.e23c_chunked_forward(
            False, x, h_tape.clone(), h_work.clone(),
            W_h_T, W_x_T, cell.b_h, W_write_T, K
        )

    cuda_out_T = cuda_results_T[0]
    cuda_h_work_T = cuda_results_T[2]

    print(f"\nCUDA output[0,0,:4] (transposed): {cuda_out_T[0,0,:4].tolist()}")
    print(f"CUDA h_work[0,:4] (transposed): {cuda_h_work_T[0,0,:4].tolist()}")
    print(f"Output diff (transposed): {torch.abs(py_out - cuda_out_T).max().item():.6f}")


def debug_attention():
    """Debug the attention computation."""
    B, D, N, K = 2, 8, 4, 4
    T = 4
    device = 'cuda'
    dtype = torch.bfloat16

    print(f"\n\nTesting attention: B={B}, T={T}, D={D}, N={N}, K={K}")
    print("="*60)

    cell = DualMemoryElmanChunkedCell(dim=D, n_slots=N, chunk_size=K).to(device).to(dtype)

    torch.manual_seed(1)
    x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1
    # Non-zero tape to test attention
    h_tape = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
    h_work = torch.zeros(B, D, device=device, dtype=dtype)

    # Python forward
    with torch.no_grad():
        py_out, py_tape, py_h = cell(x, h_tape.clone(), h_work.clone())

    print(f"Python output[0,0,:4]: {py_out[0,0,:4].tolist()}")
    print(f"Python tape[0,0,:4]: {py_tape[0,0,:4].tolist()}")

    # CUDA forward (with direct weights)
    W_x_direct = cell.W_x.weight.contiguous()
    W_h_direct = cell.W_h.weight.contiguous()
    W_write_direct = cell.W_write.weight.contiguous()

    with torch.no_grad():
        cuda_results = hasty_pytorch_lib.e23c_chunked_forward(
            False, x, h_tape.clone(), h_work.clone(),
            W_h_direct, W_x_direct, cell.b_h, W_write_direct, K
        )

    cuda_out = cuda_results[0]
    cuda_tape = cuda_results[1]

    print(f"\nCUDA output[0,0,:4] (direct): {cuda_out[0,0,:4].tolist()}")
    print(f"CUDA tape[0,0,:4] (direct): {cuda_tape[0,0,:4].tolist()}")
    print(f"Output diff: {torch.abs(py_out - cuda_out).max().item():.6f}")
    print(f"Tape diff: {torch.abs(py_tape - cuda_tape).max().item():.6f}")


if __name__ == '__main__':
    debug_first_timestep()
    debug_attention()
