
import sys; sys.path.insert(0, '/home/erikg/elman')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
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
from elman.models import LadderLM
name = "E1_exp2.5"
model = LadderLM(vocab_size=256, dim=1120, depth=26, level=1, expansion=2.5).cuda().bfloat16()
params = model.get_num_params()
print(f"{name}: {params:,} params", flush=True)

opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train()
opt.train()

for _ in range(3):
    loss = model(get_batch(batch_size, seq_len), return_loss=True)
    loss.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize()

torch.manual_seed(42); np.random.seed(42)
opt.eval()
with torch.no_grad():
    init_loss = sum(model(get_batch(batch_size, seq_len), return_loss=True).item() for _ in range(5)) / 5
opt.train()
print(f"Initial: {init_loss:.4f}, Memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)

torch.manual_seed(42); np.random.seed(42)
losses = []
torch.cuda.synchronize()
start = time.time()

for step in range(1, num_steps + 1):
    loss = model(get_batch(batch_size, seq_len), return_loss=True)
    loss.backward()
    opt.step(); opt.zero_grad()
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
