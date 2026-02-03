#!/usr/bin/env python3
"""
Test with Python matching CUDA's precision pattern EXACTLY:
- State stays in fp32 throughout loop (CUDA uses shared memory)
- Only convert to bf16 at checkpoint boundaries and final output
"""

import torch
import torch.nn.functional as F
import hasty_pytorch_lib as elman_ladder_cuda

CHECKPOINT_INTERVAL = 16

def test_exact_match():
    torch.manual_seed(42)
    device = torch.device('cuda')
    dtype = torch.bfloat16

    T, B, dim, n_state = 32, 2, 64, 32  # More timesteps to test checkpointing

    W = torch.randn(n_state, dim, device=device, dtype=dtype)
    x = torch.randn(T, B, dim, device=device, dtype=dtype)
    S0 = torch.zeros(B, n_state, n_state, device=device, dtype=dtype)

    # ============ CUDA VERSION ============
    print("=" * 60)
    print("CUDA FORWARD")
    print("=" * 60)

    S_cuda = S0.clone()
    empty = torch.empty(0, device=device, dtype=dtype)

    results = elman_ladder_cuda.e74_full_matrix_forward(
        True, x, S_cuda, 0, True, W, empty, empty, empty)

    S_final_cuda = results[0]
    output_cuda = results[1]
    k_cache = results[2]

    # ============ PYTHON VERSION matching CUDA precision exactly ============
    print("\n" + "=" * 60)
    print("PYTHON FORWARD (matching CUDA precision)")
    print("=" * 60)

    # Projection (matches cuBLAS)
    x_flat = x.reshape(T * B, dim)
    kvq_py = (x_flat @ W.T)  # bf16
    kvq_py = kvq_py.reshape(T, B, n_state)

    # Initial state in fp32 (CUDA loads bf16 S0 into fp32 shared memory)
    S_py = S0.float()  # fp32 throughout loop
    outputs_py = []

    for t in range(T):
        # Load k from bf16 to fp32 (like CUDA does)
        k_bf16 = kvq_py[t]  # bf16
        v_bf16 = kvq_py[t]  # bf16
        q_bf16 = kvq_py[t]  # bf16

        # Convert to fp32 (CUDA: k_shared[tid] = __bfloat162float(k_all[...]))
        k_fp32 = k_bf16.float()
        v_fp32 = v_bf16.float()
        q_fp32 = q_bf16.float()

        # Copy v and q before k normalization (CUDA does this assignment)
        # In CUDA for tied_kvq: v_shared[tid] = k_shared[tid] BEFORE normalization
        # So v and q have the ORIGINAL (unnormalized) values

        # Normalize k in fp32
        k_norm_sq = (k_fp32 * k_fp32).sum(dim=-1, keepdim=True)
        k_norm_val = torch.sqrt(k_norm_sq) + 1e-6
        k_norm = k_fp32 / k_norm_val  # fp32

        # retrieved = S @ k_norm (fp32)
        retrieved = torch.einsum('bij,bj->bi', S_py, k_norm)

        # delta = v - retrieved (v is original k, not normalized)
        delta = v_fp32 - retrieved

        # State update in fp32: S = tanh(S + outer(delta, k_norm))
        for i in range(n_state):
            for j in range(n_state):
                update = S_py[:, i, j] + delta[:, i] * k_norm[:, j]
                S_py[:, i, j] = torch.tanh(update)

        # Output: Sq = S @ q, then Sq * Sq * sigmoid(Sq)
        Sq = torch.einsum('bij,bj->bi', S_py, q_fp32)
        sig = 1.0 / (1.0 + torch.exp(-Sq))
        out_fp32 = Sq * Sq * sig
        out_bf16 = out_fp32.to(dtype)
        outputs_py.append(out_bf16)

    output_py = torch.stack(outputs_py, dim=0)

    # Convert final state to bf16 (CUDA: S[...] = __float2bfloat16(S_shared[i]))
    S_py = S_py.to(dtype)

    # ============ COMPARE ============
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    S_diff = (S_py.float() - S_final_cuda.float()).abs()
    out_diff = (output_py.float() - output_cuda.float()).abs()

    print(f"Final state diff: max={S_diff.max():.8f}, mean={S_diff.mean():.8f}")
    print(f"Output diff: max={out_diff.max():.8f}, mean={out_diff.mean():.8f}")

    # Print some sample values
    print(f"\nPython S[0,0,:5]: {S_py[0,0,:5].float()}")
    print(f"CUDA   S[0,0,:5]: {S_final_cuda[0,0,:5].float()}")

    print(f"\nPython output[-1,0,:5]: {output_py[-1,0,:5].float()}")
    print(f"CUDA   output[-1,0,:5]: {output_cuda[-1,0,:5].float()}")

    if S_diff.max() < 1e-6 and out_diff.max() < 1e-6:
        print("\n*** EXACT MATCH! ***")
    elif S_diff.max() < 0.001 and out_diff.max() < 0.1:
        print("\n*** VERY CLOSE (within bf16 rounding) ***")
    else:
        # Find where max state diff is
        max_s_idx = S_diff.argmax()
        b, i, j = max_s_idx // (n_state * n_state), (max_s_idx % (n_state * n_state)) // n_state, max_s_idx % n_state
        print(f"\nMax state diff at [{b},{i},{j}]:")
        print(f"  Python: {S_py[b,i,j].float():.8f}")
        print(f"  CUDA:   {S_final_cuda[b,i,j].float():.8f}")

        # Find where max output diff is
        max_o_idx = out_diff.argmax()
        t = max_o_idx // (B * n_state)
        b = (max_o_idx % (B * n_state)) // n_state
        n = max_o_idx % n_state
        print(f"\nMax output diff at [{t},{b},{n}]:")
        print(f"  Python: {output_py[t,b,n].float():.8f}")
        print(f"  CUDA:   {output_cuda[t,b,n].float():.8f}")

if __name__ == '__main__':
    test_exact_match()
