
import sys; sys.path.insert(0, '/home/erikg/elman')
import torch, numpy as np, mmap, time
from schedulefree import AdamWScheduleFree
from elman.models import LadderLM

torch.manual_seed(42); np.random.seed(42)
batch_size, time_limit = 128, 600

model = LadderLM(
    vocab_size=256, dim=512, depth=3, level=10,
    expansion=1.0, n_banks=64, core_ratio=0.5
).cuda().bfloat16()

layer = model.layers[0]
if hasattr(layer, 'n_banks'):
    print(f'e10_k64: dim={model.dim} depth={model.depth} n_banks={layer.n_banks} params={model.get_num_params():,}', flush=True)
elif hasattr(layer, 'core_dim'):
    print(f'e10_k64: dim={model.dim} depth={model.depth} core={layer.core_dim} mem={layer.mem_dim} params={model.get_num_params():,}', flush=True)
else:
    print(f'e10_k64: dim={model.dim} depth={model.depth} params={model.get_num_params():,}', flush=True)

with open('/home/erikg/elman/data/pile.txt', 'rb') as f: mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

opt = AdamWScheduleFree(model.parameters(), lr=3e-4, weight_decay=0.1)
model.train(); opt.train()
buf = np.zeros((batch_size, 513), dtype=np.uint8)
losses = []; start = time.time(); step = 0; peak_mem = 0

while time.time() - start < time_limit:
    step += 1
    pos = np.random.randint(0, data_len - 513, size=batch_size)
    for j, p in enumerate(pos): buf[j] = np.frombuffer(mm[p:p+513], dtype=np.uint8)
    batch = torch.from_numpy(buf.astype(np.int64)).cuda()
    opt.zero_grad(); loss = model(batch, return_loss=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    losses.append(loss.item())
    if step % 100 == 0:
        elapsed = time.time() - start
        tokens = step * batch_size * 512
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        peak_mem = max(peak_mem, mem_gb)
        print(f'Step {step} | {elapsed:.0f}s | Loss {loss.item():.4f} | Avg100 {np.mean(losses[-100:]):.4f} | {int(tokens/elapsed)} tok/s | Mem {mem_gb:.1f}GB', flush=True)

elapsed = time.time() - start
tokens = step * batch_size * 512
print(f'DONE: e10_k64 | Steps={step} | Time={elapsed:.0f}s | Last100={np.mean(losses[-100:]):.4f} | {int(tokens/elapsed)} tok/s | PeakMem={peak_mem:.1f}GB', flush=True)
mm.close()
