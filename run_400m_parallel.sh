#!/bin/bash
# E16 400M Parameter Study - Parallel Execution

echo "Starting 400M parameter study on 5 GPUs..."

# E16 state_exp=2
CUDA_VISIBLE_DEVICES=0 python -c "
import torch, time, sys, os, mmap
import numpy as np
sys.path.insert(0, '.')
from elman.models import LadderLM

TRAIN_MINUTES, BATCH_SIZE, CHUNK_SIZE, LR = 10, 32, 512, 1e-4

def load_data():
    with open('data/pile.txt', 'rb') as f:
        return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

def get_batch(mm, device):
    max_start = len(mm) - CHUNK_SIZE - 2
    starts = np.random.randint(0, max_start, size=BATCH_SIZE)
    chunks = [torch.from_numpy(np.frombuffer(mm[s:s+CHUNK_SIZE+1], dtype=np.uint8).copy().astype(np.int64)) for s in starts]
    return torch.stack(chunks).to(device)

def train(model, name):
    device = 'cuda'
    model = model.to(device).bfloat16()
    print(f'{name}: {sum(p.numel() for p in model.parameters()):,} params')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    mm = load_data()
    for _ in range(3):
        loss = model(get_batch(mm, device), return_loss=True)
        if isinstance(loss, tuple): loss = loss[0]
        loss.backward(); optimizer.zero_grad()
    torch.cuda.synchronize()
    print(f'Memory: {torch.cuda.max_memory_allocated()/1e9:.1f}GB')
    train_end, step, total_tokens, losses, start = time.time() + TRAIN_MINUTES*60, 0, 0, [], time.time()
    while time.time() < train_end:
        loss = model(get_batch(mm, device), return_loss=True)
        if isinstance(loss, tuple): loss = loss[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad()
        step += 1; total_tokens += BATCH_SIZE*CHUNK_SIZE; losses.append(loss.item())
        if step % 300 == 0:
            print(f'Step {step}: loss={sum(losses[-100:])/len(losses[-100:]):.4f}, {total_tokens/(time.time()-start)/1000:.1f}K tok/s')
    final_loss = sum(losses[-100:])/len(losses[-100:])
    print(f'FINAL: loss={final_loss:.4f}, tok/s={total_tokens/(time.time()-start)/1000:.1f}K, steps={step}')

model = LadderLM(vocab_size=256, dim=1024, depth=18, level=16, expansion=1.5, state_expansion=2)
train(model, 'E16_state2x')
" 2>&1 | tee /tmp/e16_state2x.log &

# E16 state_exp=4
CUDA_VISIBLE_DEVICES=1 python -c "
import torch, time, sys, os, mmap
import numpy as np
sys.path.insert(0, '.')
from elman.models import LadderLM

TRAIN_MINUTES, BATCH_SIZE, CHUNK_SIZE, LR = 10, 32, 512, 1e-4

def load_data():
    with open('data/pile.txt', 'rb') as f:
        return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

def get_batch(mm, device):
    max_start = len(mm) - CHUNK_SIZE - 2
    starts = np.random.randint(0, max_start, size=BATCH_SIZE)
    chunks = [torch.from_numpy(np.frombuffer(mm[s:s+CHUNK_SIZE+1], dtype=np.uint8).copy().astype(np.int64)) for s in starts]
    return torch.stack(chunks).to(device)

def train(model, name):
    device = 'cuda'
    model = model.to(device).bfloat16()
    print(f'{name}: {sum(p.numel() for p in model.parameters()):,} params')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    mm = load_data()
    for _ in range(3):
        loss = model(get_batch(mm, device), return_loss=True)
        if isinstance(loss, tuple): loss = loss[0]
        loss.backward(); optimizer.zero_grad()
    torch.cuda.synchronize()
    print(f'Memory: {torch.cuda.max_memory_allocated()/1e9:.1f}GB')
    train_end, step, total_tokens, losses, start = time.time() + TRAIN_MINUTES*60, 0, 0, [], time.time()
    while time.time() < train_end:
        loss = model(get_batch(mm, device), return_loss=True)
        if isinstance(loss, tuple): loss = loss[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad()
        step += 1; total_tokens += BATCH_SIZE*CHUNK_SIZE; losses.append(loss.item())
        if step % 300 == 0:
            print(f'Step {step}: loss={sum(losses[-100:])/len(losses[-100:]):.4f}, {total_tokens/(time.time()-start)/1000:.1f}K tok/s')
    final_loss = sum(losses[-100:])/len(losses[-100:])
    print(f'FINAL: loss={final_loss:.4f}, tok/s={total_tokens/(time.time()-start)/1000:.1f}K, steps={step}')

model = LadderLM(vocab_size=256, dim=896, depth=18, level=16, expansion=1.5, state_expansion=4)
train(model, 'E16_state4x')
" 2>&1 | tee /tmp/e16_state4x.log &

# E16 state_exp=8
CUDA_VISIBLE_DEVICES=2 python -c "
import torch, time, sys, os, mmap
import numpy as np
sys.path.insert(0, '.')
from elman.models import LadderLM

TRAIN_MINUTES, BATCH_SIZE, CHUNK_SIZE, LR = 10, 32, 512, 1e-4

def load_data():
    with open('data/pile.txt', 'rb') as f:
        return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

def get_batch(mm, device):
    max_start = len(mm) - CHUNK_SIZE - 2
    starts = np.random.randint(0, max_start, size=BATCH_SIZE)
    chunks = [torch.from_numpy(np.frombuffer(mm[s:s+CHUNK_SIZE+1], dtype=np.uint8).copy().astype(np.int64)) for s in starts]
    return torch.stack(chunks).to(device)

def train(model, name):
    device = 'cuda'
    model = model.to(device).bfloat16()
    print(f'{name}: {sum(p.numel() for p in model.parameters()):,} params')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    mm = load_data()
    for _ in range(3):
        loss = model(get_batch(mm, device), return_loss=True)
        if isinstance(loss, tuple): loss = loss[0]
        loss.backward(); optimizer.zero_grad()
    torch.cuda.synchronize()
    print(f'Memory: {torch.cuda.max_memory_allocated()/1e9:.1f}GB')
    train_end, step, total_tokens, losses, start = time.time() + TRAIN_MINUTES*60, 0, 0, [], time.time()
    while time.time() < train_end:
        loss = model(get_batch(mm, device), return_loss=True)
        if isinstance(loss, tuple): loss = loss[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad()
        step += 1; total_tokens += BATCH_SIZE*CHUNK_SIZE; losses.append(loss.item())
        if step % 300 == 0:
            print(f'Step {step}: loss={sum(losses[-100:])/len(losses[-100:]):.4f}, {total_tokens/(time.time()-start)/1000:.1f}K tok/s')
    final_loss = sum(losses[-100:])/len(losses[-100:])
    print(f'FINAL: loss={final_loss:.4f}, tok/s={total_tokens/(time.time()-start)/1000:.1f}K, steps={step}')

model = LadderLM(vocab_size=256, dim=768, depth=18, level=16, expansion=1.5, state_expansion=8)
train(model, 'E16_state8x')
" 2>&1 | tee /tmp/e16_state8x.log &

# minGRU
CUDA_VISIBLE_DEVICES=3 python -c "
import torch, time, sys, os, mmap
import numpy as np
sys.path.insert(0, '.')
from elman.models.min_rnn_baseline import create_mingru_model

TRAIN_MINUTES, BATCH_SIZE, CHUNK_SIZE, LR = 10, 32, 512, 1e-4

def load_data():
    with open('data/pile.txt', 'rb') as f:
        return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

def get_batch(mm, device):
    max_start = len(mm) - CHUNK_SIZE - 2
    starts = np.random.randint(0, max_start, size=BATCH_SIZE)
    chunks = [torch.from_numpy(np.frombuffer(mm[s:s+CHUNK_SIZE+1], dtype=np.uint8).copy().astype(np.int64)) for s in starts]
    return torch.stack(chunks).to(device)

def train(model, name):
    device = 'cuda'
    model = model.to(device).bfloat16()
    print(f'{name}: {sum(p.numel() for p in model.parameters()):,} params')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    mm = load_data()
    for _ in range(3):
        loss = model(get_batch(mm, device), return_loss=True)
        if isinstance(loss, tuple): loss = loss[0]
        loss.backward(); optimizer.zero_grad()
    torch.cuda.synchronize()
    print(f'Memory: {torch.cuda.max_memory_allocated()/1e9:.1f}GB')
    train_end, step, total_tokens, losses, start = time.time() + TRAIN_MINUTES*60, 0, 0, [], time.time()
    while time.time() < train_end:
        loss = model(get_batch(mm, device), return_loss=True)
        if isinstance(loss, tuple): loss = loss[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad()
        step += 1; total_tokens += BATCH_SIZE*CHUNK_SIZE; losses.append(loss.item())
        if step % 300 == 0:
            print(f'Step {step}: loss={sum(losses[-100:])/len(losses[-100:]):.4f}, {total_tokens/(time.time()-start)/1000:.1f}K tok/s')
    final_loss = sum(losses[-100:])/len(losses[-100:])
    print(f'FINAL: loss={final_loss:.4f}, tok/s={total_tokens/(time.time()-start)/1000:.1f}K, steps={step}')

model = create_mingru_model('400m')
train(model, 'minGRU')
" 2>&1 | tee /tmp/mingru.log &

# minLSTM
CUDA_VISIBLE_DEVICES=4 python -c "
import torch, time, sys, os, mmap
import numpy as np
sys.path.insert(0, '.')
from elman.models.min_rnn_baseline import create_minlstm_model

TRAIN_MINUTES, BATCH_SIZE, CHUNK_SIZE, LR = 10, 32, 512, 1e-4

def load_data():
    with open('data/pile.txt', 'rb') as f:
        return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

def get_batch(mm, device):
    max_start = len(mm) - CHUNK_SIZE - 2
    starts = np.random.randint(0, max_start, size=BATCH_SIZE)
    chunks = [torch.from_numpy(np.frombuffer(mm[s:s+CHUNK_SIZE+1], dtype=np.uint8).copy().astype(np.int64)) for s in starts]
    return torch.stack(chunks).to(device)

def train(model, name):
    device = 'cuda'
    model = model.to(device).bfloat16()
    print(f'{name}: {sum(p.numel() for p in model.parameters()):,} params')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    mm = load_data()
    for _ in range(3):
        loss = model(get_batch(mm, device), return_loss=True)
        if isinstance(loss, tuple): loss = loss[0]
        loss.backward(); optimizer.zero_grad()
    torch.cuda.synchronize()
    print(f'Memory: {torch.cuda.max_memory_allocated()/1e9:.1f}GB')
    train_end, step, total_tokens, losses, start = time.time() + TRAIN_MINUTES*60, 0, 0, [], time.time()
    while time.time() < train_end:
        loss = model(get_batch(mm, device), return_loss=True)
        if isinstance(loss, tuple): loss = loss[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad()
        step += 1; total_tokens += BATCH_SIZE*CHUNK_SIZE; losses.append(loss.item())
        if step % 300 == 0:
            print(f'Step {step}: loss={sum(losses[-100:])/len(losses[-100:]):.4f}, {total_tokens/(time.time()-start)/1000:.1f}K tok/s')
    final_loss = sum(losses[-100:])/len(losses[-100:])
    print(f'FINAL: loss={final_loss:.4f}, tok/s={total_tokens/(time.time()-start)/1000:.1f}K, steps={step}')

model = create_minlstm_model('400m')
train(model, 'minLSTM')
" 2>&1 | tee /tmp/minlstm.log &

echo "All 5 experiments launched on GPUs 0-4. Waiting for completion..."
wait

echo ""
echo "============================================================"
echo "RESULTS SUMMARY"
echo "============================================================"
echo ""
echo "E16 state_exp=2:"
tail -1 /tmp/e16_state2x.log
echo ""
echo "E16 state_exp=4:"
tail -1 /tmp/e16_state4x.log
echo ""
echo "E16 state_exp=8:"
tail -1 /tmp/e16_state8x.log
echo ""
echo "minGRU:"
tail -1 /tmp/mingru.log
echo ""
echo "minLSTM:"
tail -1 /tmp/minlstm.log
echo ""
echo "Prior results for comparison:"
echo "  E1 exp2.5:  loss=1.5355, tok/s=13.9K"
echo "  Mamba2:     loss=1.5028, tok/s=23.4K"
