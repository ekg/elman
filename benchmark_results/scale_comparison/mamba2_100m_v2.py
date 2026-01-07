import torch, torch.nn as nn, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from mamba_ssm import Mamba2
torch.manual_seed(42); np.random.seed(42)
class Mamba2LM(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.embed = nn.Embedding(256, dim)
        self.layers = nn.ModuleList([Mamba2(dim, d_state=128, headdim=64) for _ in range(depth)])
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, 256, bias=False)
        self.head.weight = self.embed.weight
    def forward(self, x, return_loss=False):
        if return_loss: x, targets = x[:, :-1], x[:, 1:]
        h = self.embed(x)
        for layer in self.layers: h = h + layer(h)
        h = self.norm(h)
        logits = self.head(h)
        if return_loss: return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits
    def get_num_params(self): return sum(p.numel() for p in self.parameters())
model = Mamba2LM(dim=1600, depth=6).cuda().bfloat16()
batch_size = 384
print(f'mamba2_100m: dim=1600 depth=6 batch={batch_size} params={model.get_num_params():,}', flush=True)
with open('/home/erikg/elman/data/pile.txt', 'rb') as f: mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)
opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train(); opt.train()
buf = np.zeros((batch_size, 513), dtype=np.uint8)
losses = []; start = time.time(); step = 0; peak_mem = 0
while time.time() - start < 600:
    step += 1
    pos = np.random.randint(0, data_len - 513, size=batch_size)
    for j, p in enumerate(pos): buf[j] = np.frombuffer(mm[p:p+513], dtype=np.uint8)
    batch = torch.from_numpy(buf.astype(np.int64)).cuda()
    opt.zero_grad(); loss = model(batch, return_loss=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    losses.append(loss.item())
    if step % 100 == 0:
        elapsed = time.time() - start; tokens = step * batch_size * 512
        mem_gb = torch.cuda.max_memory_allocated() / 1e9; peak_mem = max(peak_mem, mem_gb)
        print(f'Step {step} | {elapsed:.0f}s | Loss {loss.item():.4f} | Avg100 {np.mean(losses[-100:]):.4f} | {int(tokens/elapsed)} tok/s | Mem {mem_gb:.1f}GB', flush=True)
elapsed = time.time() - start; tokens = step * batch_size * 512
print(f'DONE: mamba2_100m | Steps={step} | Time={elapsed:.0f}s | Last100={np.mean(losses[-100:]):.4f} | {int(tokens/elapsed)} tok/s | PeakMem={peak_mem:.1f}GB', flush=True)
mm.close()
