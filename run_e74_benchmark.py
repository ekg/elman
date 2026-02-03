#!/usr/bin/env python3
"""
E74 Benchmark Runner

Runs all E74 variants and comparison models in the 100M/10min benchmark.
One model per GPU, 8 GPUs available.
"""

import subprocess
import time
import os
from pathlib import Path

# Benchmark settings
TRAIN_MINUTES = 10
BATCH_SIZE = 32
CHUNK_SIZE = 512
SEED = 42
DATA_PATH = "data/pile.txt"
OUTPUT_BASE = "benchmark_results/100m_10min"

# E74 configurations (16 variants) - ALL dims are multiples of 128 for CUDA performance
E74_CONFIGS = {
    # Delta update variants
    "e74-d-tiedkvq-t": {"dim": 1536, "n_state": 96, "update_type": "delta", "proj_type": "tied_kvq", "nonlin_type": "tanh"},
    "e74-d-tiedkvq-l": {"dim": 1536, "n_state": 96, "update_type": "delta", "proj_type": "tied_kvq", "nonlin_type": "linear"},
    "e74-d-tiedkq-t": {"dim": 1408, "n_state": 96, "update_type": "delta", "proj_type": "tied_kq", "nonlin_type": "tanh"},
    "e74-d-tiedkq-l": {"dim": 1408, "n_state": 96, "update_type": "delta", "proj_type": "tied_kq", "nonlin_type": "linear"},
    "e74-d-noz-t": {"dim": 1408, "n_state": 96, "update_type": "delta", "proj_type": "no_z", "nonlin_type": "tanh"},
    "e74-d-noz-l": {"dim": 1408, "n_state": 96, "update_type": "delta", "proj_type": "no_z", "nonlin_type": "linear"},
    "e74-d-full-t": {"dim": 1408, "n_state": 96, "update_type": "delta", "proj_type": "full", "nonlin_type": "tanh"},
    "e74-d-full-l": {"dim": 1408, "n_state": 96, "update_type": "delta", "proj_type": "full", "nonlin_type": "linear"},
    # Simple update variants
    "e74-s-tiedkvq-t": {"dim": 1536, "n_state": 96, "update_type": "simple", "proj_type": "tied_kvq", "nonlin_type": "tanh"},
    "e74-s-tiedkvq-l": {"dim": 1536, "n_state": 96, "update_type": "simple", "proj_type": "tied_kvq", "nonlin_type": "linear"},
    "e74-s-tiedkq-t": {"dim": 1408, "n_state": 96, "update_type": "simple", "proj_type": "tied_kq", "nonlin_type": "tanh"},
    "e74-s-tiedkq-l": {"dim": 1408, "n_state": 96, "update_type": "simple", "proj_type": "tied_kq", "nonlin_type": "linear"},
    "e74-s-noz-t": {"dim": 1408, "n_state": 96, "update_type": "simple", "proj_type": "no_z", "nonlin_type": "tanh"},
    "e74-s-noz-l": {"dim": 1408, "n_state": 96, "update_type": "simple", "proj_type": "no_z", "nonlin_type": "linear"},
    "e74-s-full-t": {"dim": 1408, "n_state": 96, "update_type": "simple", "proj_type": "full", "nonlin_type": "tanh"},
    "e74-s-full-l": {"dim": 1408, "n_state": 96, "update_type": "simple", "proj_type": "full", "nonlin_type": "linear"},
}

# Existing models to compare - ALL dims are multiples of 128 for CUDA performance
EXISTING_CONFIGS = {
    "e1": {"dim": 640, "depth": 20, "expansion": 2.0, "level": 1},  # 128-aligned
    "e42": {"dim": 768, "depth": 20, "expansion": 2.0, "level": 42},
    "e56": {"dim": 640, "depth": 20, "expansion": 2.0, "level": 56},  # 128-aligned
    "e61": {"dim": 640, "depth": 20, "expansion": 2.0, "level": 61},
    "e62": {"dim": 640, "depth": 20, "expansion": 2.0, "level": 62},
    "e63": {"dim": 512, "depth": 20, "expansion": 2.0, "level": 63},
    "e64": {"dim": 640, "depth": 20, "expansion": 2.0, "level": 64},
    "e65": {"dim": 640, "depth": 20, "expansion": 2.0, "level": 65},
    "e66": {"dim": 640, "depth": 20, "expansion": 2.0, "level": 66},
    "e67": {"dim": 640, "depth": 20, "expansion": 2.0, "level": 67},
    "e68": {"dim": 640, "depth": 20, "expansion": 2.0, "level": 68},
    "e70": {"dim": 1408, "depth": 20, "expansion": 2.0, "level": 70, "n_state": 96},
    "e71": {"dim": 1408, "depth": 20, "expansion": 2.0, "level": 71, "n_state": 96},
    "e72": {"dim": 1408, "depth": 20, "expansion": 2.0, "level": 72, "n_state": 96},
    "e73": {"dim": 1408, "depth": 20, "expansion": 2.0, "level": 73, "n_state": 96},
    "cudagru": {"dim": 384, "depth": 20, "expansion": 2.0, "level": "cudagru"},
    "cudalstm": {"dim": 384, "depth": 20, "expansion": 2.0, "level": "cudalstm"},
    "mamba2": {"dim": 896, "depth": 20, "expand": 2, "level": "mamba2"},
    "fla-gdn": {"dim": 768, "depth": 20, "expansion": 2.0, "level": "fla-gdn"},
    "llama": {"dim": 640, "depth": 20, "level": "llama"},
}


def run_e74_model(name, config, gpu_id, output_dir):
    """Run an E74 model on a specific GPU."""
    cmd = [
        "python", "train_e74.py",
        "--data", DATA_PATH,
        "--dim", str(config["dim"]),
        "--depth", "20",
        "--n_state", str(config["n_state"]),
        "--expansion", "2.0",
        "--update_type", config["update_type"],
        "--proj_type", config["proj_type"],
        "--nonlin_type", config["nonlin_type"],
        "--batch_size", str(BATCH_SIZE),
        "--chunk_size", str(CHUNK_SIZE),
        "--train_minutes", str(TRAIN_MINUTES),
        "--use_cuda",
        "--bf16",
        "--seed", str(SEED),
        "--output", str(output_dir / name),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_file = output_dir / f"{name}.log"
    with open(log_file, "w") as f:
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)


def run_existing_model(name, config, gpu_id, output_dir):
    """Run an existing model using run_100m_benchmark.py's approach."""
    level = config["level"]
    dim = config["dim"]
    depth = config.get("depth", 20)
    expansion = config.get("expansion", config.get("expand", 2.0))
    n_state = config.get("n_state", None)

    # Build command based on model type
    if level == "mamba2":
        cmd = [
            "python", "-c", f"""
import torch
import time
import json
from pathlib import Path
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
import mmap
import numpy as np

config = MambaConfig(
    d_model={dim},
    n_layer={depth},
    vocab_size=256,
    ssm_cfg={{"expand": {int(expansion)}}},
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
)
model = MambaLMHeadModel(config).cuda().bfloat16()
n_params = sum(p.numel() for p in model.parameters())
print(f"Mamba2: {{n_params:,}} params")

# Data loading
with open("{DATA_PATH}", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

def get_batch():
    pos = np.random.randint(0, data_len - {CHUNK_SIZE} - 1, size={BATCH_SIZE})
    buf = np.zeros(({BATCH_SIZE}, {CHUNK_SIZE} + 1), dtype=np.int64)
    for i, p in enumerate(pos):
        buf[i] = np.frombuffer(mm[p:p+{CHUNK_SIZE}+1], dtype=np.uint8).astype(np.int64)
    return torch.from_numpy(buf).cuda()

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
model.train()
losses = []
start = time.time()
end_time = start + {TRAIN_MINUTES} * 60
tokens = 0
step = 0

while time.time() < end_time:
    step += 1
    batch = get_batch()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(batch[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            out.logits.reshape(-1, 256), batch[:, 1:].reshape(-1)
        )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    losses.append(loss.item())
    tokens += {BATCH_SIZE} * {CHUNK_SIZE}
    if step % 10 == 0:
        elapsed = time.time() - start
        print(f"step {{step}} | loss {{loss.item():.4f}} | {{tokens/elapsed:,.0f}} tok/s")

elapsed = time.time() - start
final_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
print(f"Final: {{final_loss:.4f}} loss, {{tokens/elapsed:,.0f}} tok/s, {{n_params:,}} params")

# Save results
Path("{output_dir}/{name}").mkdir(parents=True, exist_ok=True)
with open("{output_dir}/{name}/results.json", "w") as f:
    json.dump({{"final_loss": final_loss, "tok_per_sec": tokens/elapsed, "n_params": n_params, "steps": step}}, f)
"""
        ]
    elif level == "fla-gdn":
        cmd = [
            "python", "-c", f"""
import torch
import time
import json
from pathlib import Path
from fla.models import GatedDeltaNetForCausalLM, GatedDeltaNetConfig
import mmap
import numpy as np

config = GatedDeltaNetConfig(
    hidden_size={dim},
    num_hidden_layers={depth},
    vocab_size=256,
    use_short_conv=True,
    conv_size=4,
    expand_k=1.0,
    expand_v={expansion},
)
model = GatedDeltaNetForCausalLM(config).cuda().bfloat16()
n_params = sum(p.numel() for p in model.parameters())
print(f"FLA-GDN: {{n_params:,}} params")

with open("{DATA_PATH}", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

def get_batch():
    pos = np.random.randint(0, data_len - {CHUNK_SIZE} - 1, size={BATCH_SIZE})
    buf = np.zeros(({BATCH_SIZE}, {CHUNK_SIZE} + 1), dtype=np.int64)
    for i, p in enumerate(pos):
        buf[i] = np.frombuffer(mm[p:p+{CHUNK_SIZE}+1], dtype=np.uint8).astype(np.int64)
    return torch.from_numpy(buf).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
model.train()
losses = []
start = time.time()
end_time = start + {TRAIN_MINUTES} * 60
tokens = 0
step = 0

while time.time() < end_time:
    step += 1
    batch = get_batch()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(batch[:, :-1], labels=batch[:, 1:])
        loss = out.loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    losses.append(loss.item())
    tokens += {BATCH_SIZE} * {CHUNK_SIZE}
    if step % 10 == 0:
        elapsed = time.time() - start
        print(f"step {{step}} | loss {{loss.item():.4f}} | {{tokens/elapsed:,.0f}} tok/s")

elapsed = time.time() - start
final_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
print(f"Final: {{final_loss:.4f}} loss, {{tokens/elapsed:,.0f}} tok/s, {{n_params:,}} params")

Path("{output_dir}/{name}").mkdir(parents=True, exist_ok=True)
with open("{output_dir}/{name}/results.json", "w") as f:
    json.dump({{"final_loss": final_loss, "tok_per_sec": tokens/elapsed, "n_params": n_params, "steps": step}}, f)
"""
        ]
    elif level == "llama":
        cmd = [
            "python", "-c", f"""
import torch
import time
import json
from pathlib import Path
from transformers import LlamaConfig, LlamaForCausalLM
import mmap
import numpy as np

config = LlamaConfig(
    hidden_size={dim},
    num_hidden_layers={depth},
    num_attention_heads=8,
    intermediate_size={dim * 4},
    vocab_size=256,
    max_position_embeddings=2048,
)
model = LlamaForCausalLM(config).cuda().bfloat16()
n_params = sum(p.numel() for p in model.parameters())
print(f"Llama: {{n_params:,}} params")

with open("{DATA_PATH}", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

def get_batch():
    pos = np.random.randint(0, data_len - {CHUNK_SIZE} - 1, size={BATCH_SIZE})
    buf = np.zeros(({BATCH_SIZE}, {CHUNK_SIZE} + 1), dtype=np.int64)
    for i, p in enumerate(pos):
        buf[i] = np.frombuffer(mm[p:p+{CHUNK_SIZE}+1], dtype=np.uint8).astype(np.int64)
    return torch.from_numpy(buf).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
model.train()
losses = []
start = time.time()
end_time = start + {TRAIN_MINUTES} * 60
tokens = 0
step = 0

while time.time() < end_time:
    step += 1
    batch = get_batch()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(batch[:, :-1], labels=batch[:, 1:])
        loss = out.loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    losses.append(loss.item())
    tokens += {BATCH_SIZE} * {CHUNK_SIZE}
    if step % 10 == 0:
        elapsed = time.time() - start
        print(f"step {{step}} | loss {{loss.item():.4f}} | {{tokens/elapsed:,.0f}} tok/s")

elapsed = time.time() - start
final_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
print(f"Final: {{final_loss:.4f}} loss, {{tokens/elapsed:,.0f}} tok/s, {{n_params:,}} params")

Path("{output_dir}/{name}").mkdir(parents=True, exist_ok=True)
with open("{output_dir}/{name}/results.json", "w") as f:
    json.dump({{"final_loss": final_loss, "tok_per_sec": tokens/elapsed, "n_params": n_params, "steps": step}}, f)
"""
        ]
    else:
        # E-series models using LadderLM
        n_state_arg = f", n_state={n_state}" if n_state else ""
        cmd = [
            "python", "-c", f"""
import torch
import time
import json
from pathlib import Path
from elman.models import LadderLM
import mmap
import numpy as np

model = LadderLM(
    vocab_size=256,
    dim={dim},
    depth={depth},
    level={level},
    expansion={expansion}{n_state_arg},
).cuda().bfloat16()
n_params = sum(p.numel() for p in model.parameters())
print(f"E{level}: {{n_params:,}} params")

with open("{DATA_PATH}", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_len = len(mm)

def get_batch():
    pos = np.random.randint(0, data_len - {CHUNK_SIZE} - 1, size={BATCH_SIZE})
    buf = np.zeros(({BATCH_SIZE}, {CHUNK_SIZE} + 1), dtype=np.int64)
    for i, p in enumerate(pos):
        buf[i] = np.frombuffer(mm[p:p+{CHUNK_SIZE}+1], dtype=np.uint8).astype(np.int64)
    return torch.from_numpy(buf).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
model.train()
losses = []
start = time.time()
end_time = start + {TRAIN_MINUTES} * 60
tokens = 0
step = 0

while time.time() < end_time:
    step += 1
    batch = get_batch()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = model(batch, return_loss=True)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    losses.append(loss.item())
    tokens += {BATCH_SIZE} * {CHUNK_SIZE}
    if step % 10 == 0:
        elapsed = time.time() - start
        print(f"step {{step}} | loss {{loss.item():.4f}} | {{tokens/elapsed:,.0f}} tok/s")

elapsed = time.time() - start
final_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
print(f"Final: {{final_loss:.4f}} loss, {{tokens/elapsed:,.0f}} tok/s, {{n_params:,}} params")

Path("{output_dir}/{name}").mkdir(parents=True, exist_ok=True)
with open("{output_dir}/{name}/results.json", "w") as f:
    json.dump({{"final_loss": final_loss, "tok_per_sec": tokens/elapsed, "n_params": n_params, "steps": step}}, f)
"""
        ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_file = output_dir / f"{name}.log"
    with open(log_file, "w") as f:
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, shell=False)


def run_batch(models, output_dir, is_e74=False):
    """Run a batch of models in parallel (up to 8)."""
    processes = []
    for i, (name, config) in enumerate(models):
        gpu_id = i % 8
        print(f"  Starting {name} on GPU {gpu_id}...")
        if is_e74:
            proc = run_e74_model(name, config, gpu_id, output_dir)
        else:
            proc = run_existing_model(name, config, gpu_id, output_dir)
        processes.append((name, proc))

    # Wait for all to complete
    for name, proc in processes:
        proc.wait()
        print(f"  {name} completed (return code: {proc.returncode})")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--e74-only", action="store_true", help="Only run E74 variants")
    parser.add_argument("--existing-only", action="store_true", help="Only run existing models")
    parser.add_argument("--batch", type=int, default=None, help="Run specific batch (1-indexed)")
    args = parser.parse_args()

    output_dir = Path(OUTPUT_BASE)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all models to run
    all_models = []

    if not args.e74_only:
        for name, config in EXISTING_CONFIGS.items():
            all_models.append((name, config, False))

    if not args.existing_only:
        for name, config in E74_CONFIGS.items():
            all_models.append((name, config, True))

    # Split into batches of 8
    batches = []
    for i in range(0, len(all_models), 8):
        batches.append(all_models[i:i+8])

    print(f"Total models: {len(all_models)}")
    print(f"Total batches: {len(batches)}")
    print()

    if args.batch:
        # Run specific batch
        batch_idx = args.batch - 1
        if batch_idx < 0 or batch_idx >= len(batches):
            print(f"Invalid batch number. Valid range: 1-{len(batches)}")
            return
        batches = [batches[batch_idx]]
        print(f"Running batch {args.batch} only")

    for batch_idx, batch in enumerate(batches):
        print(f"=" * 60)
        print(f"Batch {batch_idx + 1}/{len(batches)}")
        print(f"=" * 60)

        # Separate E74 and existing models
        e74_models = [(name, config) for name, config, is_e74 in batch if is_e74]
        existing_models = [(name, config) for name, config, is_e74 in batch if not is_e74]

        processes = []
        gpu_id = 0

        # Start E74 models
        for name, config in e74_models:
            print(f"  Starting {name} on GPU {gpu_id}...")
            proc = run_e74_model(name, config, gpu_id, output_dir)
            processes.append((name, proc))
            gpu_id += 1

        # Start existing models
        for name, config in existing_models:
            print(f"  Starting {name} on GPU {gpu_id}...")
            proc = run_existing_model(name, config, gpu_id, output_dir)
            processes.append((name, proc))
            gpu_id += 1

        # Wait for all to complete
        print(f"\nWaiting for batch to complete...")
        start_time = time.time()
        for name, proc in processes:
            proc.wait()
            elapsed = time.time() - start_time
            print(f"  {name} completed after {elapsed/60:.1f}m (return code: {proc.returncode})")

        print(f"\nBatch {batch_idx + 1} complete!")
        print()

    print("=" * 60)
    print("All benchmarks complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
