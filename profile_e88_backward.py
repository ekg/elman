"""
Profile E88 backward kernel to identify bottlenecks.
"""

import torch
import time

def profile_e88_backward():
    print("E88 Backward Kernel Profiling")
    print("=" * 60)

    try:
        import hasty_pytorch_lib
        from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction
    except ImportError as e:
        print(f"Import error: {e}")
        return

    torch.manual_seed(42)

    # Configuration matching 100M model
    T, B, H = 512, 32, 16
    n_state, head_v_dim = 32, 64
    checkpoint_interval = 32

    print(f"Config: T={T}, B={B}, H={H}, n_state={n_state}, head_v_dim={head_v_dim}")
    print(f"Checkpoint interval: {checkpoint_interval}")
    print(f"Num segments: {(T + checkpoint_interval - 1) // checkpoint_interval}")

    # Generate test data
    k = torch.randn(T, B, H, n_state, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(T, B, H, head_v_dim, device='cuda', dtype=torch.bfloat16)
    q = torch.randn(T, B, H, n_state, device='cuda', dtype=torch.bfloat16)
    decay = torch.sigmoid(torch.randn(T, B, H, device='cuda', dtype=torch.bfloat16))
    S0 = torch.zeros(B, H, n_state, head_v_dim, device='cuda', dtype=torch.bfloat16)

    # L2 normalize
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)

    k.requires_grad_(True)
    v.requires_grad_(True)
    q.requires_grad_(True)
    decay.requires_grad_(True)

    # Forward pass
    print("\n--- Forward Pass ---")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        results = hasty_pytorch_lib.e88_fla_hybrid_forward(
            True, k, v, q, decay, S0, H
        )
    torch.cuda.synchronize()
    fwd_time = (time.perf_counter() - start) * 1000 / 10
    print(f"Forward time: {fwd_time:.2f} ms")

    output = results[1]
    S_cache = results[2]
    d_output = torch.randn_like(output)

    # Parse S_cache
    num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
    checkpoints_size = num_checkpoints * B * H * n_state * head_v_dim
    S_checkpoints = S_cache[:checkpoints_size].view(num_checkpoints, B, H, n_state, head_v_dim)
    Sq_cache = S_cache[checkpoints_size:].view(T, B, H, head_v_dim)

    # Backward pass
    print("\n--- Backward Pass ---")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        grads = hasty_pytorch_lib.e88_fla_hybrid_backward(
            k, v, q, decay, S_checkpoints, Sq_cache, d_output, H
        )
    torch.cuda.synchronize()
    bwd_time = (time.perf_counter() - start) * 1000 / 10
    print(f"Backward time: {bwd_time:.2f} ms")

    print(f"\nBackward/Forward ratio: {bwd_time/fwd_time:.1f}x")

    # Memory bandwidth analysis
    print("\n--- Memory Bandwidth Analysis ---")

    # Forward: read k,v,q,decay, write output, checkpoints
    fwd_read = T * B * H * (n_state + head_v_dim + n_state + 1) * 2  # bf16
    fwd_write = T * B * H * head_v_dim * 2 + num_checkpoints * B * H * n_state * head_v_dim * 2
    fwd_total = (fwd_read + fwd_write) / 1e9  # GB

    # Backward: read k,v,q,decay,d_output,checkpoints,segment_cache; write d_k,d_v,d_q,d_decay,segment_cache
    num_segments = (T + checkpoint_interval - 1) // checkpoint_interval
    # Per segment: load checkpoint, replay forward (r/w segment_cache), backward
    # Forward replay: load k,v,decay for 32 steps, write segment_cache
    # Backward: load k,v,q,decay,d_output for 32 steps, read segment_cache, write gradients

    bwd_read_per_seg = (
        B * H * n_state * head_v_dim * 2 +  # checkpoint
        checkpoint_interval * B * H * (n_state + head_v_dim + 1) * 2 +  # k,v,decay for forward
        checkpoint_interval * B * H * (n_state + head_v_dim + n_state + 1 + head_v_dim) * 2 +  # k,v,q,decay,d_output for backward
        checkpoint_interval * B * H * n_state * head_v_dim * 2  # segment_cache read
    )
    bwd_write_per_seg = (
        checkpoint_interval * B * H * n_state * head_v_dim * 2 +  # segment_cache write
        checkpoint_interval * B * H * (n_state + head_v_dim + n_state + 1) * 2  # gradients
    )
    bwd_total = num_segments * (bwd_read_per_seg + bwd_write_per_seg) / 1e9  # GB

    print(f"Forward memory traffic: {fwd_total:.2f} GB")
    print(f"Backward memory traffic: {bwd_total:.2f} GB")
    print(f"Backward/Forward memory ratio: {bwd_total/fwd_total:.1f}x")

    # Achieved bandwidth
    fwd_bw = fwd_total / (fwd_time / 1000)  # GB/s
    bwd_bw = bwd_total / (bwd_time / 1000)  # GB/s
    print(f"\nAchieved forward bandwidth: {fwd_bw:.0f} GB/s")
    print(f"Achieved backward bandwidth: {bwd_bw:.0f} GB/s")
    print(f"(A100 peak: ~2000 GB/s, H100 peak: ~3350 GB/s)")

    # Breakdown by operation (rough estimate)
    print("\n--- Time Breakdown Estimate ---")
    # Each segment: forward replay + backward
    # Forward replay: ~32 timesteps of lightweight ops
    # Backward: ~32 timesteps of heavier ops (2x memory, gradient computation)

    fwd_replay_per_seg = fwd_time * checkpoint_interval / T
    print(f"Forward replay per segment: ~{fwd_replay_per_seg:.2f} ms")
    print(f"Backward per segment: ~{bwd_time / num_segments - fwd_replay_per_seg:.2f} ms")
    print(f"Total segments: {num_segments}")


if __name__ == "__main__":
    profile_e88_backward()
