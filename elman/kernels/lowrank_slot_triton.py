"""
Triton/PyTorch optimized E3 (LowRank Slot Elman).

Key insight: torch.einsum with batched operations is faster than
sequential per-slot CUDA kernels.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor


def lowrank_slot_forward_einsum(
    x: Tensor,      # [T, B, D]
    z: Tensor,      # [T, B, D]
    h0: Tensor,     # [n_slots, B, D]
    W_x: Tensor,    # [D, D]
    U: Tensor,      # [n_slots, D, rank]
    V: Tensor,      # [n_slots, rank, D]
    b: Tensor,      # [D]
    C: Tensor,      # [n_slots]
    training: bool = True,
):
    """
    Optimized forward using batched matrix ops via bmm.

    Uses torch.bmm for batched GEMMs across slots.
    """
    T, B, D = x.shape
    n_slots = U.shape[0]
    rank = U.shape[2]

    # Pre-compute W_x @ x for all timesteps [T, B, D]
    Wx_all = torch.matmul(x, W_x.T)

    # Allocate outputs
    h_all = torch.zeros(T + 1, n_slots, B, D, device=x.device, dtype=x.dtype)
    h_all[0] = h0
    output = torch.zeros(T, B, D, device=x.device, dtype=x.dtype)
    Vh_cache = torch.zeros(T, n_slots, B, rank, device=x.device, dtype=x.dtype) if training else None

    for t in range(T):
        h_prev = h_all[t]  # [n_slots, B, D]
        Wx_t = Wx_all[t]   # [B, D]
        z_t = z[t]         # [B, D]

        # Batched V @ h_prev: [n_slots, B, rank]
        # V: [n_slots, rank, D], h_prev: [n_slots, B, D]
        # Vh[s] = h_prev[s] @ V[s].T -> [B, D] @ [D, rank] = [B, rank]
        Vh = torch.bmm(h_prev, V.transpose(1, 2))  # [n_slots, B, rank]

        if training:
            Vh_cache[t] = Vh

        # Batched U @ Vh: [n_slots, B, D]
        # U: [n_slots, D, rank], Vh: [n_slots, B, rank]
        # Uh[s] = Vh[s] @ U[s].T -> [B, rank] @ [rank, D] = [B, D]
        Uh = torch.bmm(Vh, U.transpose(1, 2))  # [n_slots, B, D]

        # h_new = tanh(Wx + Uh + b) for each slot
        pre_act = Wx_t.unsqueeze(0) + Uh + b.view(1, 1, -1)
        h_new = torch.tanh(pre_act)  # [n_slots, B, D]

        h_all[t + 1] = h_new

        # output = sum_s(C[s] * h_new[s]) * silu(z)
        weighted_h = (C.view(-1, 1, 1) * h_new).sum(dim=0)  # [B, D]
        silu_z = z_t * torch.sigmoid(z_t)
        output[t] = weighted_h * silu_z

    return h_all, output, Vh_cache


def lowrank_slot_backward_einsum(
    d_output: Tensor,   # [T, B, D]
    h_all: Tensor,      # [T+1, n_slots, B, D]
    Vh_cache: Tensor,   # [T, n_slots, B, rank]
    z: Tensor,          # [T, B, D]
    W_x: Tensor,        # [D, D]
    U: Tensor,          # [n_slots, D, rank]
    V: Tensor,          # [n_slots, rank, D]
    C: Tensor,          # [n_slots]
):
    """
    Backward pass using einsum for batched operations.
    """
    T, B, D = d_output.shape
    n_slots = U.shape[0]
    rank = U.shape[2]

    # Allocate gradient tensors
    dx = torch.zeros(T, B, D, device=d_output.device, dtype=d_output.dtype)
    dz = torch.zeros_like(d_output)
    dW_x = torch.zeros_like(W_x)
    dU = torch.zeros_like(U)
    dV = torch.zeros_like(V)
    db = torch.zeros(D, device=d_output.device, dtype=torch.float32)
    dC = torch.zeros_like(C)

    # For BPTT
    dh_next = torch.zeros(n_slots, B, D, device=d_output.device, dtype=d_output.dtype)

    # Store dv for each timestep for batched gradient computation
    dv_all = torch.zeros(T, n_slots, B, D, device=d_output.device, dtype=d_output.dtype)

    for t in range(T - 1, -1, -1):
        d_out_t = d_output[t]  # [B, D]
        z_t = z[t]             # [B, D]
        h_t = h_all[t + 1]     # [n_slots, B, D]
        h_prev = h_all[t]      # [n_slots, B, D]
        Vh_t = Vh_cache[t]     # [n_slots, B, rank]

        # Backward through output gate: out = weighted_h * silu(z)
        sigmoid_z = torch.sigmoid(z_t)
        silu_z = z_t * sigmoid_z
        dsilu = sigmoid_z * (1 + z_t * (1 - sigmoid_z))

        weighted_h = torch.einsum('s,sbd->bd', C, h_t)
        dweighted_h = d_out_t * silu_z
        dz[t] = d_out_t * weighted_h * dsilu

        # Backward through weighted sum: weighted_h = sum_s(C[s] * h[s])
        # dC[s] += sum_{b,d} dweighted_h[b,d] * h[s,b,d]
        dC += torch.einsum('bd,sbd->s', dweighted_h, h_t)
        dh = C.view(-1, 1, 1) * dweighted_h.unsqueeze(0)  # [n_slots, B, D]

        # Add recurrent gradient
        dh = dh + dh_next

        # Backward through tanh: h = tanh(v), dv = dh * (1 - h^2)
        dv = dh * (1 - h_t * h_t)
        dv_all[t] = dv

        # Accumulate db
        db += dv.sum(dim=(0, 1)).float()

        # Backward through Uh = U @ Vh
        # dU[s,d,r] += sum_b dv[s,b,d] * Vh[s,b,r]
        dU += torch.einsum('sbd,sbr->sdr', dv, Vh_t)

        # dVh = U.T @ dv: [n_slots, B, rank]
        dVh = torch.einsum('sdr,sbd->sbr', U, dv)

        # Backward through Vh = V @ h_prev
        # dV[s,r,d] += sum_b dVh[s,b,r] * h_prev[s,b,d]
        dV += torch.einsum('sbr,sbd->srd', dVh, h_prev)

        # dh_prev for recurrence
        dh_next = torch.einsum('srd,sbr->sbd', V, dVh)

    # Batched dx = dv_all @ W_x
    dv_flat = dv_all.permute(0, 2, 3, 1).reshape(T * B, D, n_slots)  # This is wrong shape
    # Actually: dx[t,b] = sum_s dv[t,s,b] @ W_x
    dx = torch.einsum('tsbd,dD->tbD', dv_all, W_x)

    # Batched dW_x = x.T @ dv
    # dW_x[d,D] = sum_{t,s,b} x[t,b,D] * dv[t,s,b,d]
    # This needs the input x - we don't have it here
    # For now, return None for dW_x

    return dx, dz, None, dU, dV, db.to(d_output.dtype), dC


# Autograd function
class LowRankSlotEinsumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, training, x, z, h0, W_x, U, V, b, C):
        h_all, output, Vh_cache = lowrank_slot_forward_einsum(
            x, z, h0, W_x, U, V, b, C, training=training
        )
        if training:
            ctx.save_for_backward(x, z, h_all, Vh_cache, W_x, U, V, b, C)
        return h_all, output, Vh_cache

    @staticmethod
    def backward(ctx, dh_all, d_output, dVh_cache):
        x, z, h_all, Vh_cache, W_x, U, V, b, C = ctx.saved_tensors

        T, B, D = x.shape
        n_slots = U.shape[0]
        rank = U.shape[2]

        # Allocate gradients
        dx = torch.zeros_like(x)
        dz = torch.zeros_like(z)
        dW_x = torch.zeros_like(W_x)
        dU = torch.zeros_like(U)
        dV = torch.zeros_like(V)
        db = torch.zeros(D, device=x.device, dtype=torch.float32)
        dC = torch.zeros_like(C)

        dh_next = torch.zeros(n_slots, B, D, device=x.device, dtype=x.dtype)

        # Pre-transpose for bmm
        U_t = U.transpose(1, 2)  # [n_slots, rank, D]
        V_t = V.transpose(1, 2)  # [n_slots, D, rank]

        for t in range(T - 1, -1, -1):
            d_out_t = d_output[t]
            z_t = z[t]
            h_t = h_all[t + 1]
            h_prev = h_all[t]
            Vh_t = Vh_cache[t]
            x_t = x[t]

            # Backward through output gate
            sigmoid_z = torch.sigmoid(z_t)
            silu_z = z_t * sigmoid_z
            dsilu = sigmoid_z * (1 + z_t * (1 - sigmoid_z))

            # Use bmm for weighted sum: C[s] * h_t[s] summed
            weighted_h = (C.view(-1, 1, 1) * h_t).sum(dim=0)  # [B, D]
            dweighted_h = d_out_t * silu_z
            dz[t] = d_out_t * weighted_h * dsilu

            # dC: sum over B,D of dweighted_h * h_t
            dC += (dweighted_h.unsqueeze(0) * h_t).sum(dim=(1, 2))
            dh = C.view(-1, 1, 1) * dweighted_h.unsqueeze(0) + dh_next

            # Backward through tanh
            dv = dh * (1 - h_t * h_t)

            # Accumulate db
            db += dv.sum(dim=(0, 1)).float()

            # dx: sum_s dv[s] @ W_x
            dv_sum = dv.sum(dim=0)  # [B, D]
            dx[t] = torch.matmul(dv_sum, W_x)

            # dW_x: outer product
            dW_x += torch.matmul(dv_sum.T, x_t)

            # dU: dv @ Vh.T using bmm
            # dv: [n_slots, B, D], Vh_t: [n_slots, B, rank]
            # dU[s] = dv[s].T @ Vh_t[s] -> [D, B] @ [B, rank] = [D, rank]
            dU += torch.bmm(dv.transpose(1, 2), Vh_t)  # [n_slots, D, rank]

            # dVh: U.T @ dv using bmm
            # U_t: [n_slots, rank, D], dv: [n_slots, B, D]
            # dVh[s] = (U_t[s] @ dv[s].T).T = dv[s] @ U_t[s].T
            dVh = torch.bmm(dv, U_t.transpose(1, 2))  # [n_slots, B, rank]

            # dV: dVh.T @ h_prev using bmm
            # dVh: [n_slots, B, rank], h_prev: [n_slots, B, D]
            # dV[s] = dVh[s].T @ h_prev[s] -> [rank, B] @ [B, D] = [rank, D]
            dV += torch.bmm(dVh.transpose(1, 2), h_prev)  # [n_slots, rank, D]

            # dh_prev: V @ dVh using bmm
            # V: [n_slots, rank, D], dVh: [n_slots, B, rank]
            # dh_prev[s] = (V[s].T @ dVh[s].T).T = dVh[s] @ V[s]
            dh_next = torch.bmm(dVh, V)  # [n_slots, B, D]

        return None, dx, dz, None, dW_x, dU, dV, db.to(x.dtype), dC


# Test
if __name__ == "__main__":
    import time

    device = 'cuda'
    T, B, D = 512, 32, 768
    n_slots, rank = 8, 256

    # Create inputs
    x = torch.randn(T, B, D, device=device, dtype=torch.bfloat16)
    z = torch.randn(T, B, D, device=device, dtype=torch.bfloat16)
    h0 = torch.zeros(n_slots, B, D, device=device, dtype=torch.bfloat16)
    W_x = torch.randn(D, D, device=device, dtype=torch.bfloat16)
    U = torch.randn(n_slots, D, rank, device=device, dtype=torch.bfloat16)
    V = torch.randn(n_slots, rank, D, device=device, dtype=torch.bfloat16)
    b = torch.zeros(D, device=device, dtype=torch.bfloat16)
    C = torch.ones(n_slots, device=device, dtype=torch.bfloat16) / n_slots

    print(f"Testing LowRank Slot Elman Kernels")
    print(f"T={T}, B={B}, D={D}, n_slots={n_slots}, rank={rank}")
    print("=" * 60)

    # Test einsum version
    print("\n1. Einsum version (forward only):")
    for _ in range(3):
        h, out, _ = lowrank_slot_forward_einsum(x, z, h0, W_x, U, V, b, C, training=False)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        h, out, _ = lowrank_slot_forward_einsum(x, z, h0, W_x, U, V, b, C, training=False)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"   Forward: {elapsed/10*1000:.1f}ms, per step: {elapsed/10/T*1000:.3f}ms")

    # Compare with CUDA kernel
    print("\n2. CUDA kernel (forward only):")
    import hasty_pytorch_lib
    for _ in range(3):
        h, out, _ = hasty_pytorch_lib.lowrank_slot_elman_forward(True, x, z, h0, W_x, U, V, b, C)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        h, out, _ = hasty_pytorch_lib.lowrank_slot_elman_forward(True, x, z, h0, W_x, U, V, b, C)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"   Forward: {elapsed/10*1000:.1f}ms, per step: {elapsed/10/T*1000:.3f}ms")

    # Test with backward
    print("\n3. Einsum version (forward + backward):")
    x.requires_grad_(True)
    U_grad = U.clone().requires_grad_(True)
    V_grad = V.clone().requires_grad_(True)

    for _ in range(3):
        h, out, Vh = LowRankSlotEinsumFunction.apply(True, x, z, h0, W_x, U_grad, V_grad, b, C)
        out.sum().backward()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(5):
        h, out, Vh = LowRankSlotEinsumFunction.apply(True, x, z, h0, W_x, U_grad, V_grad, b, C)
        out.sum().backward()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"   Fwd+Bwd: {elapsed/5*1000:.1f}ms, per step: {elapsed/5/T*1000:.3f}ms")
    print(f"   Tokens/s: {T*B*5/elapsed/1000:.1f}k")
