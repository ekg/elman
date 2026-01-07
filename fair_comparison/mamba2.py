
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

batch_size, seq_len = 16, 512
from elman.models import create_mamba2_model
name = "Mamba2"
model = create_mamba2_model(target_params="400M", vocab_size=256).cuda().bfloat16()
params = model.get_num_params()
print(f"{name}: {params:,} params", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

# Warmup (don't count these)
torch.manual_seed(42); np.random.seed(42)  # Reset seed after warmup
for _ in range(3):
    loss = model(get_batch(batch_size, seq_len), return_loss=True)
    loss.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize()

# Reset seed so all models see same data
torch.manual_seed(42); np.random.seed(42)

# Initial loss (same batches for all models)
init_losses = []
for _ in range(5):
    with torch.no_grad():
        loss = model(get_batch(batch_size, seq_len), return_loss=True)
        init_losses.append(loss.item())
initial_loss = sum(init_losses) / len(init_losses)
print(f"Initial loss: {initial_loss:.4f}", flush=True)
print(f"Memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)

# Reset seed again for training
torch.manual_seed(42); np.random.seed(42)

# Training - all models see same batches in same order
losses = []
step = 0
torch.cuda.synchronize()
start = time.time()

while time.time() - start < 600:
    loss = model(get_batch(batch_size, seq_len), return_loss=True)
    loss.backward(); opt.step(); opt.zero_grad()
    losses.append(loss.item()); step += 1

    if step % 100 == 0:
        el = time.time() - start
        avg_loss = sum(losses[-100:]) / 100
        tok_s = (step * batch_size * seq_len) / el
        print(f"Step {step} ({el:.0f}s): loss={avg_loss:.4f}, {tok_s/1000:.1f}K tok/s", flush=True)

torch.cuda.synchronize()
elapsed = time.time() - start
final_loss = sum(losses[-100:]) / min(100, len(losses))
tok_s = (step * batch_size * seq_len) / elapsed
total_tokens = step * batch_size * seq_len

print(f"FINAL: steps={step}, tokens={total_tokens/1e6:.1f}M, loss={final_loss:.4f}, tok/s={tok_s/1000:.1f}K", flush=True)
mm.close()
