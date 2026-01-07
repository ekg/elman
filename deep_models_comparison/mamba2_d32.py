
import sys; sys.path.insert(0, '/home/erikg/elman')
import torch, numpy as np, mmap, time
torch.manual_seed(42); np.random.seed(42)
torch.backends.cudnn.benchmark = True

with open('/home/erikg/elman/data/pile.txt', 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

def get_batch(bs, sl):
    pos = np.random.randint(0, data_len - sl - 1, size=bs)
    buf = np.zeros((bs, sl + 1), dtype=np.uint8)
    for i, p in enumerate(pos): buf[i] = np.frombuffer(mm[p:p+sl+1], dtype=np.uint8)
    return torch.from_numpy(buf.astype(np.int64)).cuda()

batch_size, seq_len, num_steps = 16, 512, 1500
from mamba_ssm import Mamba2
import torch.nn as nn
class M2(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(256, 1440)
        self.layers = nn.ModuleList([Mamba2(1440, d_state=128, headdim=64) for _ in range(32)])
        self.norm = nn.RMSNorm(1440)
        self.head = nn.Linear(1440, 256, bias=False)
        self.head.weight = self.embed.weight
    def forward(self, x, return_loss=False):
        if return_loss: x, t = x[:, :-1], x[:, 1:]
        h = self.embed(x)
        for l in self.layers: h = h + l(h)
        h = self.norm(h)
        logits = self.head(h)
        if return_loss: return nn.functional.cross_entropy(logits.view(-1,256), t.reshape(-1))
        return logits
    def get_num_params(self): return sum(p.numel() for p in self.parameters())
name = "Mamba2_d32"
model = M2().cuda().bfloat16()
params = model.get_num_params()
print(f"{name}: {params:,} params", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

for _ in range(3):
    loss = model(get_batch(batch_size, seq_len), return_loss=True)
    loss.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize()

torch.manual_seed(42); np.random.seed(42)
with torch.no_grad():
    init_loss = sum(model(get_batch(batch_size, seq_len), return_loss=True).item() for _ in range(5)) / 5
print(f"Initial: {init_loss:.4f}, Memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)

torch.manual_seed(42); np.random.seed(42)
losses = []
torch.cuda.synchronize()
start = time.time()

for step in range(1, num_steps + 1):
    loss = model(get_batch(batch_size, seq_len), return_loss=True)
    loss.backward(); opt.step(); opt.zero_grad()
    losses.append(loss.item())
    if step % 300 == 0:
        avg = sum(losses[-100:]) / 100
        el = time.time() - start
        print(f"Step {step}: loss={avg:.4f}, {step*batch_size*seq_len/el/1000:.1f}K tok/s", flush=True)

torch.cuda.synchronize()
elapsed = time.time() - start
final_loss = sum(losses[-100:]) / 100
tok_s = (num_steps * batch_size * seq_len) / elapsed
print(f"FINAL: loss={final_loss:.4f}, tok/s={tok_s/1000:.1f}K, time={elapsed:.0f}s", flush=True)
mm.close()
