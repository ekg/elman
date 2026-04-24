"""Detailed profiler showing top CUDA ops by time, CUDA vs HYBRID."""

import sys, os
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if os.environ.get('ELMAN_PARARNN_HYBRID') == '1':
    from install_hybrid import install
    install()

import schedulefree
from elman.models import LadderLM


def run_step(model, opt, x):
    loss = model(x, return_loss=True)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


if __name__ == '__main__':
    n_state, H, depth, dim = 32, 83, 17, 1920  # E88-n32 production
    B, T = 16, 512
    torch.manual_seed(42)
    model = LadderLM(
        vocab_size=256, dim=dim, depth=depth, level=88,
        n_heads=H, n_state=n_state,
        use_gate=True, gate_activation='silu', expansion=1.0,
    ).cuda().to(torch.bfloat16)
    model.train()
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=6.4e-4)
    opt.train()

    x = torch.randint(0, 256, (B, T), dtype=torch.int64, device='cuda')

    # Warmup
    for _ in range(5): run_step(model, opt, x)
    torch.cuda.synchronize()

    # Profile 2 steady-state steps
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                     torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, profile_memory=False, with_stack=False,
    ) as prof:
        for _ in range(2):
            run_step(model, opt, x)

    mode = 'HYBRID' if os.environ.get('ELMAN_PARARNN_HYBRID') == '1' else 'CUDA'
    print(f"\n[{mode}] Top 20 CUDA ops by total time:\n")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
