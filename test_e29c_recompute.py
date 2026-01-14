"""
Test E29c read_val recomputation to debug CUDA backward.
"""
import torch
import torch.nn.functional as F


def test_read_val_recompute():
    """Compare Python vs CUDA read_val recomputation."""
    B, T, N, D = 4, 8, 8, 64
    device = 'cuda'
    dtype = torch.bfloat16

    torch.manual_seed(42)

    # Create test data matching forward output shapes
    # h_tape_all: [B, T+1, N, D] - tape states at each timestep
    h_tape_all = torch.randn(B, T+1, N, D, device=device, dtype=dtype) * 0.1
    # read_attn_all: [B, T, N] - attention weights
    read_attn_all = F.softmax(torch.randn(B, T, N, device=device), dim=-1).to(dtype)

    # Pick a timestep to test
    t = 3

    # Python computation of read_val
    h_tape_t = h_tape_all[:, t]  # [B, N, D]
    read_attn = read_attn_all[:, t]  # [B, N]
    read_val_python = torch.einsum('bn,bnd->bd', read_attn.float(), h_tape_t.float()).to(dtype)

    print(f"Testing read_val recomputation at timestep {t}")
    print(f"  h_tape_t shape: {h_tape_t.shape}, range: [{h_tape_t.min():.4f}, {h_tape_t.max():.4f}]")
    print(f"  read_attn shape: {read_attn.shape}, range: [{read_attn.min():.4f}, {read_attn.max():.4f}]")
    print(f"  read_val_python shape: {read_val_python.shape}, range: [{read_val_python.min():.4f}, {read_val_python.max():.4f}]")

    # Now test what the CUDA kernel would see
    # The backward transposes h_tape_all to [T+1, B, N, D]
    h_tape_all_t = h_tape_all.permute(1, 0, 2, 3).contiguous()  # [T+1, B, N, D]
    read_attn_all_t = read_attn_all.permute(1, 0, 2).contiguous()  # [T, B, N]

    print(f"\nTransposed shapes:")
    print(f"  h_tape_all_t: {h_tape_all_t.shape}")
    print(f"  read_attn_all_t: {read_attn_all_t.shape}")

    # The CUDA kernel accesses:
    # h_tape_t = h_tape_all + t * batch_size * n_slots * dim
    # For [T+1, B, N, D] contiguous layout, this gives element [t, 0, 0, 0] through [t, B-1, N-1, D-1]
    # Which is h_tape_all_t[t] with shape [B, N, D]
    h_tape_cuda = h_tape_all_t[t]  # [B, N, D]

    # read_attn_t = read_attn_all + t * batch_size * n_slots
    # For [T, B, N] contiguous layout, this gives element [t, 0, 0] through [t, B-1, N-1]
    # Which is read_attn_all_t[t] with shape [B, N]
    read_attn_cuda = read_attn_all_t[t]  # [B, N]

    # Recompute read_val the way CUDA kernel does
    # for d in range(D):
    #     for n in range(N):
    #         val += attn[b * N + n] * tape[b * N * D + n * D + d]

    # For layout [B, N, D], index b * N * D + n * D + d = (b * N + n) * D + d
    # This is equivalent to tape[b, n, d] for [B, N, D] layout
    read_val_cuda = torch.einsum('bn,bnd->bd', read_attn_cuda.float(), h_tape_cuda.float()).to(dtype)

    print(f"\nCUDA-style computation:")
    print(f"  h_tape_cuda shape: {h_tape_cuda.shape}, matches: {torch.allclose(h_tape_cuda, h_tape_t)}")
    print(f"  read_attn_cuda shape: {read_attn_cuda.shape}, matches: {torch.allclose(read_attn_cuda, read_attn)}")
    print(f"  read_val_cuda shape: {read_val_cuda.shape}, range: [{read_val_cuda.min():.4f}, {read_val_cuda.max():.4f}]")

    # Compare
    diff = (read_val_python.float() - read_val_cuda.float()).abs().max()
    print(f"\n  Max difference: {diff:.6f}")
    print(f"  PASS: {diff < 1e-3}")

    # Now test actual CUDA kernel if available
    try:
        import hasty_pytorch_lib
        if hasattr(hasty_pytorch_lib, 'e29c_diagonal_backward'):
            print("\n\nTesting actual CUDA backward...")
            test_cuda_backward(B, T, N, D)
    except ImportError:
        print("\n(CUDA kernel not available for direct testing)")


def test_cuda_backward(B, T, N, D):
    """Test CUDA backward directly (bypassing Python fallback)."""
    device = 'cuda'
    dtype = torch.bfloat16

    torch.manual_seed(42)

    import hasty_pytorch_lib
    from elman.models.e29c_diagonal import e29c_forward_python, e29c_backward_python

    # Create test weights
    W_h = torch.randn(D, D, device=device, dtype=dtype) * 0.1
    W_xz = torch.randn(2*D, D, device=device, dtype=dtype) * 0.1
    b_h = torch.zeros(D, device=device, dtype=dtype)
    W_write = torch.randn(D, D, device=device, dtype=dtype) * 0.1
    g_z = torch.ones(D, device=device, dtype=dtype)
    g_r = torch.ones(D, device=device, dtype=dtype)
    g_h = torch.ones(D, device=device, dtype=dtype)
    b_gate = torch.zeros(D, device=device, dtype=dtype)

    # Initial states
    h_tape_init = torch.zeros(B, N, D, device=device, dtype=dtype)
    h_work_init = torch.zeros(B, D, device=device, dtype=dtype)

    # Input
    x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1

    # Run forward (use CUDA to get proper h_tape_all layout)
    output_all, h_work_all, h_tape_final, h_tape_all, read_attn_all, write_attn_all = \
        hasty_pytorch_lib.e29c_diagonal_forward(
            True, x, h_tape_init, h_work_init, W_h, W_xz, b_h, W_write, g_z, g_r, g_h, b_gate)

    print(f"  Forward output: {output_all.shape}, range: [{output_all.min():.4f}, {output_all.max():.4f}]")
    print(f"  h_tape_all: {h_tape_all.shape}")
    print(f"  read_attn_all: {read_attn_all.shape}")

    # Debug: check h_tape_all values at each timestep
    print("\n  h_tape_all values at each timestep:")
    for t in range(min(4, T+1)):
        ht = h_tape_all[:, t]  # [B, N, D]
        print(f"    t={t}: mean={ht.abs().mean():.6f}, max={ht.abs().max():.6f}")

    print("\n  read_attn_all values at each timestep:")
    for t in range(min(4, T)):
        ra = read_attn_all[:, t]  # [B, N]
        print(f"    t={t}: mean={ra.abs().mean():.6f}, max={ra.abs().max():.6f}, sum={ra.sum(dim=-1).mean():.4f}")

    # Check what CUDA backward will actually see after permutation
    print("\n  After permutation (what CUDA backward sees):")
    h_tape_all_t = h_tape_all.permute(1, 0, 2, 3).contiguous()  # [T+1, B, N, D]
    read_attn_all_t = read_attn_all.permute(1, 0, 2).contiguous()  # [T, B, N]
    print(f"    h_tape_all_t shape: {h_tape_all_t.shape}")
    print(f"    read_attn_all_t shape: {read_attn_all_t.shape}")

    for t in range(min(4, T+1)):
        if t < T+1:
            ht = h_tape_all_t[t]  # [B, N, D]
            print(f"    h_tape_all_t[{t}]: mean={ht.abs().mean():.6f}, max={ht.abs().max():.6f}")
        if t < T:
            ra = read_attn_all_t[t]  # [B, N]
            # Compute expected read_val
            expected_rv = torch.einsum('bn,bnd->bd', ra.float(), h_tape_all_t[t].float())
            print(f"    expected read_val[{t}]: mean={expected_rv.abs().mean():.6f}, max={expected_rv.abs().max():.6f}")

    # d_output = all ones
    d_output_all = torch.ones_like(output_all)
    d_h_tape_final = torch.zeros(B, N, D, device=device, dtype=dtype)

    # Call CUDA backward directly
    print("\n  Calling CUDA backward directly...")
    torch.cuda.synchronize()  # Ensure forward is complete

    result = hasty_pytorch_lib.e29c_diagonal_backward(
        x.contiguous(),
        h_work_all.contiguous(),
        h_work_init.contiguous(),
        h_tape_all.contiguous(),
        read_attn_all.contiguous(),
        write_attn_all.contiguous(),
        W_h.contiguous(),
        W_xz.contiguous(),
        W_write.contiguous(),
        g_z.contiguous(),
        g_r.contiguous(),
        g_h.contiguous(),
        b_gate.contiguous(),
        d_output_all.contiguous(),
        d_h_tape_final.contiguous())

    dx_cuda, dW_h_cuda, dW_xz_cuda, db_h_cuda, dW_write_cuda, dg_z_cuda, dg_r_cuda, dg_h_cuda, db_gate_cuda = result

    torch.cuda.synchronize()  # Ensure backward is complete
    print(f"\n  CUDA backward results (synchronized):")
    print(f"    dg_z: mean={dg_z_cuda.abs().mean():.6f}, max={dg_z_cuda.abs().max():.6f}")
    print(f"    dg_r: mean={dg_r_cuda.abs().mean():.6f}, max={dg_r_cuda.abs().max():.6f}")
    print(f"    dg_h: mean={dg_h_cuda.abs().mean():.6f}, max={dg_h_cuda.abs().max():.6f}")
    print(f"    db_gate: mean={db_gate_cuda.abs().mean():.6f}, max={db_gate_cuda.abs().max():.6f}")
    print(f"    db_h: mean={db_h_cuda.abs().mean():.6f}, max={db_h_cuda.abs().max():.6f}")

    # Compare with Python backward
    scale = 1.0 / (D ** 0.5)
    dx_py, dW_h_py, dW_xz_py, db_h_py, dW_write_py, dg_z_py, dg_r_py, dg_h_py, db_gate_py = \
        e29c_backward_python(x, h_work_all, h_tape_all, read_attn_all, write_attn_all,
                             W_h, W_xz, W_write, g_z, g_r, g_h, b_gate, h_work_init,
                             d_output_all, d_h_tape_final, scale)

    print(f"\n  Python backward results:")
    print(f"    dg_z: mean={dg_z_py.abs().mean():.6f}, max={dg_z_py.abs().max():.6f}")
    print(f"    dg_r: mean={dg_r_py.abs().mean():.6f}, max={dg_r_py.abs().max():.6f}")
    print(f"    dg_h: mean={dg_h_py.abs().mean():.6f}, max={dg_h_py.abs().max():.6f}")
    print(f"    db_gate: mean={db_gate_py.abs().mean():.6f}, max={db_gate_py.abs().max():.6f}")
    print(f"    db_h: mean={db_h_py.abs().mean():.6f}, max={db_h_py.abs().max():.6f}")

    # Check if CUDA dg_r is zero (the bug)
    if dg_r_cuda.abs().mean() < 1e-8:
        print("\n  BUG CONFIRMED: CUDA dg_r is zero!")
    else:
        diff = (dg_r_cuda.float() - dg_r_py.float()).abs().max()
        print(f"\n  dg_r diff between CUDA and Python: {diff:.6f}")


if __name__ == '__main__':
    test_read_val_recompute()
