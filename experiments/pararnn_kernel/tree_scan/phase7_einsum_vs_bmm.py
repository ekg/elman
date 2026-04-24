"""Compare einsum vs bmm vs custom Triton for Sq/dQ ops."""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bench(fn, n_repeat=20):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


def bench_all(H, T, N):
    dt = torch.bfloat16
    S = torch.randn(1, H, T, N, N, dtype=dt, device='cuda')
    Q = torch.randn(1, H, T, N, dtype=dt, device='cuda')

    print(f"  ---- Sq: einsum('bhtpq,bhtq->bhtp', S, Q) (contract last dim) ----")

    # Einsum
    def fn_einsum():
        return torch.einsum('bhtpq,bhtq->bhtp', S, Q)

    # Reshape + bmm
    BHT = 1 * H * T
    def fn_bmm():
        S_m = S.reshape(BHT, N, N)
        Q_m = Q.reshape(BHT, N, 1)
        return torch.bmm(S_m, Q_m).reshape(1, H, T, N)

    # matmul
    def fn_matmul():
        return (S @ Q.unsqueeze(-1)).squeeze(-1)

    # unsqueeze + mul + sum
    def fn_mulsum():
        return (S * Q.unsqueeze(-2)).sum(dim=-1)

    m_einsum = bench(fn_einsum)
    m_bmm = bench(fn_bmm)
    m_matmul = bench(fn_matmul)
    print(f"  H={H} T={T} N={N}  (Sq):")
    print(f"    einsum(last): {m_einsum:>6.2f} ms  bmm: {m_bmm:>6.2f}  matmul: {m_matmul:>6.2f}")

    # dQ-style: contract FIRST spatial dim of S
    # dQ = einsum('bhti,bhtij->bhtj', dL_dout, S) → contracts 'i'
    dL = torch.randn(1, H, T, N, dtype=dt, device='cuda')
    def fn_dQ_einsum():
        return torch.einsum('bhti,bhtij->bhtj', dL, S)

    # Reshape: S.transpose → [1, H, T, N, N] with inner dim swapped, then contract last
    # i.e., S^T(spatial) @ dL
    def fn_dQ_bmm():
        S_t = S.transpose(-1, -2)   # swap spatial dims → [1, H, T, N, N]
        return torch.einsum('bhtpq,bhtq->bhtp', S_t, dL)

    def fn_dQ_matmul():
        # dL [..., N] @ S [..., N, N] = [..., N]
        return (dL.unsqueeze(-2) @ S).squeeze(-2)

    m_dq_e = bench(fn_dQ_einsum)
    m_dq_b = bench(fn_dQ_bmm)
    m_dq_m = bench(fn_dQ_matmul)
    print(f"  H={H} T={T} N={N}  (dQ):")
    print(f"    einsum(first): {m_dq_e:>6.2f} ms  bmm-transposed: {m_dq_b:>6.2f}  matmul: {m_dq_m:>6.2f}")

    r_e = fn_dQ_einsum()
    r_b = fn_dQ_bmm()
    r_m = fn_dQ_matmul()
    print(f"    max |bmm-ref|={((r_b.float()-r_e.float()).abs().max().item()):.2e}")
    print(f"    max |matmul-ref|={((r_m.float()-r_e.float()).abs().max().item()):.2e}")


if __name__ == '__main__':
    for H, T, N in [(141, 16384, 16), (141, 32768, 16), (83, 16384, 32), (83, 32768, 32)]:
        bench_all(H, T, N)
