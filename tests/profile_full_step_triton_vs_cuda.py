"""Torch profiler comparison of Triton vs CUDA E88 at production 1.27B —
identifies which ops dominate outside the recurrence kernel.

Runs ~10 training steps and reports the top time-consuming ops for
each backend. The profile excludes warmup.
"""
from __future__ import absolute_import

import os, sys, time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.profiler

from elman.models.ladder_lm import LadderLM


def make_model_and_data(use_triton, device):
    """Build the production E88 1.27B and a random input."""
    torch.manual_seed(42)
    model = LadderLM(
        vocab_size=256, dim=1408, depth=14, level="E88",
        n_heads=386, n_state=32,
        use_gate=True, gate_activation="silu",
        gradient_checkpointing=True,
        use_triton=use_triton,
    ).to(device).to(torch.bfloat16)
    g = torch.Generator(device=device).manual_seed(0)
    B, T = 8, 512
    x = torch.randint(0, 256, (B, T), generator=g, device=device, dtype=torch.long)
    return model, x


def profile_backend(use_triton, device, label):
    print(f"\n=== Profiling {label} ===")
    model, x = make_model_and_data(use_triton, device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Use return_loss=False + external CE so the recurrence sees T=512
    # (production-aligned with our 16-step ckpt_interval).
    import torch.nn.functional as F
    target = torch.randint(0, 256, (8, 512), device=device, dtype=torch.long)

    def step():
        optim.zero_grad(set_to_none=True)
        logits = model(x, return_loss=False)  # [B, T, V]
        loss = F.cross_entropy(logits.view(-1, 256), target.view(-1))
        loss.backward()
        optim.step()

    # Warmup
    for _ in range(3):
        step()
    torch.cuda.synchronize()

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    n_iters = 5
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(n_iters):
            step()
        torch.cuda.synchronize()

    # Sort by self CUDA time.
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))


def main():
    if not torch.cuda.is_available():
        sys.exit(1)
    device = torch.device("cuda")

    profile_backend(True, device, "TRITON")
    profile_backend(False, device, "CUDA")


if __name__ == "__main__":
    main()
