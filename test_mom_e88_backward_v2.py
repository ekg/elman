#!/usr/bin/env python
"""
Test MoM E88 backward pass by comparing CUDA backward with PyTorch autograd.
"""

import torch
import sys
sys.path.insert(0, '/home/erikg/elman')

import hasty_pytorch_lib

device = 'cuda'
dtype = torch.bfloat16


class MoME88PythonRef(torch.autograd.Function):
    """Python reference implementation with autograd support."""

    @staticmethod
    def forward(ctx, k, v, q, decay, head_indices, router_weights, S0, H, K):
        """Forward pass - pure Python, float32 for precision."""
        B, T = k.shape[:2]
        head_v_dim = v.shape[-1]
        n_state = k.shape[-1]

        # Work in float32 for accuracy
        k_f = k.float()
        v_f = v.float()
        q_f = q.float()
        decay_f = decay.float()
        router_weights_f = router_weights.float()
        S0_f = S0.float()

        outputs = []
        S_slots = S0_f.clone()
        all_S_prev = []  # Save states for backward

        for t in range(T):
            all_S_prev.append(S_slots.clone())
            output_t = torch.zeros(B, head_v_dim, device=device, dtype=torch.float32)
            S_next = S_slots.clone()

            for b in range(B):
                for slot in range(K):
                    h = head_indices[b, t, slot].item()
                    w = router_weights_f[b, t, slot]

                    k_h = k_f[b, t, h]
                    v_h = v_f[b, t, h]
                    q_h = q_f[b, t, h]
                    decay_h = decay_f[b, t, h]

                    S_slot = S_slots[b, slot]
                    retrieved = S_slot.T @ k_h
                    delta = v_h - retrieved
                    outer = torch.outer(k_h, delta)
                    new_state = torch.tanh(decay_h * S_slot + outer)
                    S_next[b, slot] = new_state

                    Sq = new_state.T @ q_h
                    output_t[b] += w * Sq

            S_slots = S_next
            outputs.append(output_t)

        output = torch.stack(outputs, dim=1)

        # Save for backward
        ctx.save_for_backward(k_f, v_f, q_f, decay_f, head_indices, router_weights_f)
        ctx.all_S_prev = all_S_prev
        ctx.H = H
        ctx.K = K
        ctx.output = output

        return output.to(dtype), S_slots.to(dtype)

    @staticmethod
    def backward(ctx, d_output, d_S_final):
        """Backward pass - pure Python."""
        k_f, v_f, q_f, decay_f, head_indices, router_weights_f = ctx.saved_tensors
        all_S_prev = ctx.all_S_prev
        H, K = ctx.H, ctx.K

        B, T = k_f.shape[:2]
        n_state = k_f.shape[-1]
        head_v_dim = v_f.shape[-1]

        d_output = d_output.float()

        d_k = torch.zeros_like(k_f)
        d_v = torch.zeros_like(v_f)
        d_q = torch.zeros_like(q_f)
        d_decay = torch.zeros_like(decay_f)
        d_router_weights = torch.zeros_like(router_weights_f)

        # Initialize dS to zero
        dS_slots = torch.zeros(B, K, n_state, head_v_dim, device=device, dtype=torch.float32)

        for t in range(T - 1, -1, -1):
            S_prev = all_S_prev[t]

            for b in range(B):
                for slot in range(K):
                    h = head_indices[b, t, slot].item()
                    w = router_weights_f[b, t, slot]

                    k_h = k_f[b, t, h]
                    v_h = v_f[b, t, h]
                    q_h = q_f[b, t, h]
                    decay_h = decay_f[b, t, h]

                    S_slot_prev = S_prev[b, slot]

                    # Recompute forward quantities
                    retrieved = S_slot_prev.T @ k_h
                    delta = v_h - retrieved
                    outer = torch.outer(k_h, delta)
                    pre_tanh = decay_h * S_slot_prev + outer
                    S_new = torch.tanh(pre_tanh)
                    dtanh = 1 - S_new ** 2

                    Sq = S_new.T @ q_h

                    # d_router_weight
                    d_router_weights[b, t, slot] += (Sq * d_output[b, t]).sum()

                    # d_Sq = w * d_output
                    d_Sq = w * d_output[b, t]

                    # d_q: Sq[j] = sum_i S_new[i,j] * q[i]
                    # d_q[i] = sum_j S_new[i,j] * d_Sq[j]
                    d_q[b, t, h] += S_new @ d_Sq

                    # dS_new from output: dS_new[i,j] += q[i] * d_Sq[j]
                    dS_new = dS_slots[b, slot] + torch.outer(q_h, d_Sq)

                    # Backward through tanh
                    d_pre = dS_new * dtanh

                    # d_decay = sum(d_pre * S_prev)
                    d_decay[b, t, h] += (d_pre * S_slot_prev).sum()

                    # d_delta[j] = sum_i d_pre[i,j] * k[i]
                    d_delta = d_pre.T @ k_h

                    # d_k from outer: d_k[i] = sum_j d_pre[i,j] * delta[j]
                    d_k_from_outer = d_pre @ delta

                    # d_v = d_delta
                    d_v[b, t, h] += d_delta

                    # d_retrieved = -d_delta
                    d_retrieved = -d_delta

                    # d_k from retrieved: retrieved[j] = sum_i S_prev[i,j] * k[i]
                    # d_k[i] += sum_j S_prev[i,j] * d_retrieved[j]
                    d_k_from_retrieved = S_slot_prev @ d_retrieved

                    d_k[b, t, h] += d_k_from_outer + d_k_from_retrieved

                    # dS_prev = d_pre * decay + outer(k, -d_delta)
                    dS_slots[b, slot] = d_pre * decay_h + torch.outer(k_h, -d_delta)

        return d_k.to(dtype), d_v.to(dtype), d_q.to(dtype), d_decay.to(dtype), \
               None, d_router_weights.to(dtype), None, None, None


def test_backward():
    """Compare CUDA backward with Python reference backward."""
    print("Testing MoM E88 backward: CUDA vs Python Reference")
    print("=" * 60)

    B, T, H, K = 1, 2, 4, 2
    n_state = 16
    head_v_dim = 16

    torch.manual_seed(42)

    k = torch.randn(B, T, H, n_state, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, head_v_dim, device=device, dtype=dtype, requires_grad=True)
    q = torch.randn(B, T, H, n_state, device=device, dtype=dtype, requires_grad=True)
    decay = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype)).requires_grad_(True)
    router_weights = torch.softmax(torch.randn(B, T, K, device=device, dtype=dtype), dim=-1).requires_grad_(True)

    # Normalize k and q
    k.data = k.data / (k.data.norm(dim=-1, keepdim=True) + 1e-6)
    q.data = q.data / (q.data.norm(dim=-1, keepdim=True) + 1e-6)

    head_indices = torch.tensor([[[0, 1], [2, 3]]], device=device, dtype=torch.int32)
    S0 = torch.zeros(B, K, n_state, head_v_dim, device=device, dtype=dtype)

    # Random gradient
    d_output = torch.randn(B, T, head_v_dim, device=device, dtype=dtype)

    # === Python reference backward ===
    output_py, S_py = MoME88PythonRef.apply(k, v, q, decay, head_indices, router_weights, S0, H, K)
    loss_py = (output_py.float() * d_output.float()).sum()
    loss_py.backward()

    d_k_py = k.grad.clone()
    d_v_py = v.grad.clone()
    d_q_py = q.grad.clone()
    d_decay_py = decay.grad.clone()
    d_router_py = router_weights.grad.clone()

    # Reset grads
    k.grad = None
    v.grad = None
    q.grad = None
    decay.grad = None
    router_weights.grad = None

    # === CUDA backward ===
    k_t = k.detach().transpose(0, 1).contiguous()
    v_t = v.detach().transpose(0, 1).contiguous()
    q_t = q.detach().transpose(0, 1).contiguous()
    decay_t = decay.detach().transpose(0, 1).contiguous()
    head_indices_t = head_indices.transpose(0, 1).contiguous()
    router_weights_t = router_weights.detach().transpose(0, 1).contiguous()

    output_t, S_cuda, S_cache = hasty_pytorch_lib.mom_e88_forward(
        True, k_t, v_t, q_t, decay_t,
        head_indices_t, router_weights_t,
        S0, H, K
    )

    d_output_t = d_output.transpose(0, 1).contiguous()

    d_k_cuda, d_v_cuda, d_q_cuda, d_decay_cuda, d_router_cuda = hasty_pytorch_lib.mom_e88_backward(
        k_t, v_t, q_t, decay_t,
        head_indices_t, router_weights_t,
        S_cache, d_output_t, H, K
    )

    # Transpose back to [B, T, ...]
    d_k_cuda = d_k_cuda.transpose(0, 1)
    d_v_cuda = d_v_cuda.transpose(0, 1)
    d_q_cuda = d_q_cuda.transpose(0, 1)
    d_decay_cuda = d_decay_cuda.transpose(0, 1)
    d_router_cuda = d_router_cuda.transpose(0, 1)

    # Compare gradients
    def compare(name, cuda_grad, py_grad):
        diff = (cuda_grad.float() - py_grad.float()).abs()
        rel_err = diff / (py_grad.float().abs() + 1e-8)
        print(f"{name}:")
        print(f"  Max abs diff: {diff.max().item():.6f}")
        print(f"  Max rel err:  {rel_err.max().item():.4f}")
        print(f"  Mean diff:    {diff.mean().item():.6f}")
        print(f"  CUDA range:   [{cuda_grad.min().item():.4f}, {cuda_grad.max().item():.4f}]")
        print(f"  Python range: [{py_grad.min().item():.4f}, {py_grad.max().item():.4f}]")
        return diff.max().item()

    print("\n=== Gradient Comparison ===")
    diff_k = compare("d_k", d_k_cuda, d_k_py)
    print()
    diff_v = compare("d_v", d_v_cuda, d_v_py)
    print()
    diff_q = compare("d_q", d_q_cuda, d_q_py)
    print()
    diff_decay = compare("d_decay", d_decay_cuda, d_decay_py)
    print()
    diff_router = compare("d_router", d_router_cuda, d_router_py)

    print("\n" + "=" * 60)
    threshold = 0.1
    max_diff = max(diff_k, diff_v, diff_q, diff_decay, diff_router)
    if max_diff < threshold:
        print(f"PASS: All gradient differences < {threshold}")
        return True
    else:
        print(f"FAIL: Max gradient difference = {max_diff:.4f} (threshold: {threshold})")
        return False


if __name__ == '__main__':
    success = test_backward()
    sys.exit(0 if success else 1)
