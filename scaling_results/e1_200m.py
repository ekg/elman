
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

batch_size, seq_len, time_limit = 48, 512, 120
from elman.models import LadderLM
name = "E1"
model = LadderLM(vocab_size=256, dim=1248, depth=26, level=1).cuda().bfloat16()
print(f"{name}: {model.get_num_params():,} params", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for _ in range(5):
    loss = model(get_batch(batch_size, seq_len), return_loss=True)
    loss.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize()
print(f"Memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
losses, step = [], 0
torch.cuda.synchronize()
start = time.time()
while time.time() - start < time_limit:
    loss = model(get_batch(batch_size, seq_len), return_loss=True)
    loss.backward(); opt.step(); opt.zero_grad()
    losses.append(loss.item()); step += 1
    if step % 50 == 0:
        el = time.time() - start
        print(f"Step {step} ({el:.0f}s): loss={sum(losses[-50:])/50:.4f}, {(step*batch_size*seq_len)/el/1000:.1f}K tok/s", flush=True)
torch.cuda.synchronize()
el = time.time() - start
print(f"FINAL: {step} steps, loss={sum(losses[-50:])/min(50,len(losses)):.4f}, {(step*batch_size*seq_len)/el/1000:.1f}K tok/s", flush=True)
mm.close()
