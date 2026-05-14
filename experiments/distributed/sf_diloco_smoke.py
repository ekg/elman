#!/usr/bin/env python3
"""Small ScheduleFree local-SGD / DiLoCo communication smoke test.

This is intentionally separate from train.py. It lets us test one process per
GPU with periodic model sharing while leaving the production training path
unchanged.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import schedulefree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "elman" / "cuda"))

from elman.data.tokenized_dataset import TokenizedStreamDataset
from elman.models import LadderLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(ROOT / "data" / "pile.txt"))
    p.add_argument("--output", default="/tmp/sf_diloco_smoke")
    p.add_argument("--tokenizer", default="p50k_base")
    p.add_argument("--level", default="E88")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=64)
    p.add_argument("--n_state", type=int, default=16)
    p.add_argument("--expansion", type=float, default=1.0)
    p.add_argument("--use_gate", type=int, default=1)
    p.add_argument("--gate_activation", default="silu")
    p.add_argument("--use_triton", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--max_seconds", type=float, default=0.0)
    p.add_argument("--local_steps", type=int, default=50)
    p.add_argument("--mode", choices=["ddp", "sync_sgd", "local_sgd", "diloco"], default="diloco")
    p.add_argument("--outer_lr", type=float, default=1.0)
    p.add_argument("--outer_beta", type=float, default=0.9)
    p.add_argument("--sync_optimizer_state", action="store_true")
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--bytes_per_token", type=float, default=3.817020)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=0)
    p.add_argument("--eval_batches", type=int, default=0)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--eval_seed", type=int, default=12345)
    p.add_argument("--eval_at_start", action="store_true")
    p.add_argument("--eval_at_end", action="store_true")
    p.add_argument("--eval_schedulefree_average", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--resume_optimizer", action="store_true")
    p.add_argument("--save_final", action="store_true")
    return p.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if "RANK" not in os.environ:
        raise RuntimeError("Launch with torchrun so RANK/WORLD_SIZE are set")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world, local_rank, torch.device("cuda", local_rank)


def make_model(args: argparse.Namespace, vocab_size: int) -> LadderLM:
    return LadderLM(
        vocab_size=vocab_size,
        dim=args.dim,
        depth=args.depth,
        level=args.level,
        expansion=args.expansion,
        n_heads=args.n_heads,
        n_state=args.n_state,
        use_gate=bool(args.use_gate),
        gate_activation=args.gate_activation,
        r_h_mode="none",
        use_conv=False,
        checkpoint_interval=16,
        use_triton=bool(args.use_triton),
    )


@torch.no_grad()
def sync_optimizer_state(optimizer: torch.optim.Optimizer, world: int) -> None:
    """Average floating tensor optimizer state across workers.

    This is deliberately conservative and expensive. For these smoke tests the
    point is to see whether ScheduleFree can survive periodic state sharing.
    """

    for state in optimizer.state.values():
        for value in state.values():
            if not torch.is_tensor(value) or not value.is_floating_point():
                continue
            buf = value.detach()
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            buf.div_(world)


@torch.no_grad()
def sync_model(
    model: torch.nn.Module,
    anchors: list[torch.Tensor],
    velocity: list[torch.Tensor],
    args: argparse.Namespace,
    world: int,
) -> tuple[float, float]:
    """Synchronize model parameters and return drift/communication seconds."""

    t0 = time.time()
    drift_num = torch.zeros((), device=anchors[0].device)
    drift_den = torch.zeros((), device=anchors[0].device)

    for i, p in enumerate(model.parameters()):
        p32 = p.detach().float()
        delta = p32 - anchors[i]
        drift_num += delta.square().sum()
        drift_den += anchors[i].square().sum()

        if args.mode == "local_sgd":
            avg = p32
            dist.all_reduce(avg, op=dist.ReduceOp.SUM)
            avg.div_(world)
            anchors[i].copy_(avg)
        else:
            dist.all_reduce(delta, op=dist.ReduceOp.SUM)
            delta.div_(world)
            velocity[i].mul_(args.outer_beta).add_(delta)
            anchors[i].add_(velocity[i], alpha=args.outer_lr)

        p.copy_(anchors[i].to(dtype=p.dtype))

    dist.all_reduce(drift_num, op=dist.ReduceOp.SUM)
    dist.all_reduce(drift_den, op=dist.ReduceOp.SUM)
    drift = (drift_num / drift_den.clamp_min(1e-30)).sqrt().item()
    torch.cuda.synchronize()
    return drift, time.time() - t0


@torch.no_grad()
def sync_gradients(model: torch.nn.Module, world: int) -> float:
    """Average gradients across ranks, equivalent to DDP all-reduce."""

    t0 = time.time()
    for p in model.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world)
    torch.cuda.synchronize()
    return time.time() - t0


def all_reduce_mean(value: float, device: torch.device, world: int) -> float:
    t = torch.tensor(float(value), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / world).item()


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt.get("step", 0)), float(ckpt.get("loss", float("inf")))


def make_fixed_eval_batches(args: argparse.Namespace) -> list[torch.Tensor]:
    dataset = TokenizedStreamDataset(
        data_path=args.data,
        chunk_size=args.chunk_size + 1,
        rank=0,
        world_size=1,
        seed=args.eval_seed,
        tokenizer_name=args.tokenizer,
    )
    batches = []
    for _ in range(args.eval_batches):
        batch, _, _ = dataset.get_batch(args.eval_batch_size)
        batches.append(batch.clone())
    return batches


@torch.no_grad()
def evaluate_fixed_batches(
    model: torch.nn.Module,
    optimizer: schedulefree.AdamWScheduleFree,
    batches: list[torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    if args.eval_schedulefree_average:
        optimizer.eval()
    model.eval()
    total = 0.0
    for batch_cpu in batches:
        batch = batch_cpu.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
            loss = model(batch, return_loss=True)
        total += float(loss.item())
    model.train()
    if args.eval_schedulefree_average:
        optimizer.train()
    return total / max(1, len(batches))


def main() -> None:
    args = parse_args()
    rank, world, local_rank, device = setup_distributed()

    torch.manual_seed(args.seed)
    out = Path(args.output)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    dataset = TokenizedStreamDataset(
        data_path=args.data,
        chunk_size=args.chunk_size + 1,
        rank=rank,
        world_size=world,
        seed=args.seed,
        tokenizer_name=args.tokenizer,
    )
    eval_batches = make_fixed_eval_batches(args) if rank == 0 and args.eval_batches > 0 else []

    model = make_model(args, dataset.vocab_size).to(device)
    if args.bf16:
        model = model.bfloat16()
    model.train()
    torch.manual_seed(args.seed + rank)
    train_model: torch.nn.Module = model
    if args.mode == "ddp":
        train_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    optimizer.train()

    resume_step = 0
    resume_loss = float("inf")
    if args.resume:
        resume_step, resume_loss = load_checkpoint(
            args.resume,
            model,
            optimizer if args.resume_optimizer else None,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr
        optimizer.train()
        if rank == 0:
            what = "model+optimizer" if args.resume_optimizer else "model"
            print(f"resumed {what} from {args.resume} at step={resume_step} loss={resume_loss}", flush=True)

    anchors = [p.detach().float().clone() for p in model.parameters()]
    velocity = [torch.zeros_like(a) for a in anchors]

    model_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(
            f"mode={args.mode} world={world} model_params={model_params:,} "
            f"local_steps={args.local_steps} sync_optimizer_state={args.sync_optimizer_state}",
            flush=True,
        )

    sync_records = []
    log_records = []
    eval_records = []
    dist.barrier()
    if args.eval_at_start and args.eval_batches > 0:
        eval_t0 = time.time()
        if rank == 0:
            eval_loss = evaluate_fixed_batches(model, optimizer, eval_batches, args, device)
            eval_bpb = eval_loss / math.log(2.0) / args.bytes_per_token
            eval_s = time.time() - eval_t0
            wall = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
            print(
                f"eval {0:6d} | loss {eval_loss:.4f} | bpb {eval_bpb:.4f} | "
                f"eval_s {eval_s:.1f} | time {wall}",
                flush=True,
            )
            eval_records.append(
                {
                    "step": 0,
                    "elapsed_s": 0.0,
                    "loss": eval_loss,
                    "bpb": eval_bpb,
                    "eval_s": eval_s,
                    "batches": args.eval_batches,
                    "batch_size": args.eval_batch_size,
                    "time": wall,
                }
            )
        dist.barrier()

    start = time.time()
    log_start = start
    running_loss = 0.0
    running_tokens = 0
    total_tokens_local = 0
    grad_sync_s_total = 0.0
    grad_sync_count = 0
    completed_steps = 0

    for step in range(1, args.steps + 1):
        if args.max_seconds > 0 and time.time() - start >= args.max_seconds:
            break
        completed_steps = step
        batch, _, lengths = dataset.get_batch(args.batch_size, device=device)
        batch_tokens = int(lengths.sum().item())
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
            loss = train_model(batch, return_loss=True)

        if not torch.isfinite(loss):
            print(f"rank={rank} nonfinite loss at step={step}: {loss.item()}", flush=True)
            break

        loss.backward()
        if args.mode == "sync_sgd":
            grad_sync_s_total += sync_gradients(model, world)
            grad_sync_count += 1
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            grad_norm = torch.tensor(0.0, device=device)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += float(loss.item())
        running_tokens += batch_tokens
        total_tokens_local += batch_tokens

        if args.mode in {"local_sgd", "diloco"} and step % args.local_steps == 0:
            drift, sync_s = sync_model(model, anchors, velocity, args, world)
            if args.sync_optimizer_state:
                sync_optimizer_state(optimizer, world)
            sync_records.append({"step": step, "drift": drift, "sync_s": sync_s})

        if step % args.log_every == 0:
            elapsed = time.time() - log_start
            mean_loss_local = running_loss / args.log_every
            mean_loss = all_reduce_mean(mean_loss_local, device, world)
            total_tokens = torch.tensor(float(running_tokens), device=device)
            dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
            tok_s = total_tokens.item() / elapsed
            if rank == 0:
                bpb = mean_loss / math.log(2.0) / args.bytes_per_token
                wall = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
                print(
                    f"step {step:6d} | loss {mean_loss:.4f} | bpb {bpb:.4f} | "
                    f"grad {float(grad_norm):.2f} | global_tok/s {tok_s:.0f} | time {wall}",
                    flush=True,
                )
                log_records.append(
                    {
                        "step": step,
                        "elapsed_s": time.time() - start,
                        "loss": mean_loss,
                        "bpb": bpb,
                        "global_tok_s": tok_s,
                        "tokens": int(total_tokens.item()),
                        "grad_norm": float(grad_norm),
                        "time": wall,
                    }
                )
            running_loss = 0.0
            running_tokens = 0
            log_start = time.time()

            if args.eval_every > 0 and args.eval_batches > 0 and step % args.eval_every == 0:
                dist.barrier()
                eval_t0 = time.time()
                if rank == 0:
                    eval_loss = evaluate_fixed_batches(model, optimizer, eval_batches, args, device)
                    eval_bpb = eval_loss / math.log(2.0) / args.bytes_per_token
                    eval_s = time.time() - eval_t0
                    wall = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
                    print(
                        f"eval {step:6d} | loss {eval_loss:.4f} | bpb {eval_bpb:.4f} | "
                        f"eval_s {eval_s:.1f} | time {wall}",
                        flush=True,
                    )
                    eval_records.append(
                        {
                            "step": step,
                            "elapsed_s": time.time() - start,
                            "loss": eval_loss,
                            "bpb": eval_bpb,
                            "eval_s": eval_s,
                            "batches": args.eval_batches,
                            "batch_size": args.eval_batch_size,
                            "time": wall,
                        }
                    )
                dist.barrier()
                log_start = time.time()

    dist.barrier()
    partial_steps = completed_steps % args.log_every
    final_loss = all_reduce_mean(running_loss / max(1, partial_steps), device, world) if running_loss else None
    elapsed_total = time.time() - start
    total_tokens = torch.tensor(float(total_tokens_local), device=device)
    dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    if args.eval_at_end and args.eval_batches > 0:
        dist.barrier()
        eval_elapsed = time.time() - start
        eval_t0 = time.time()
        if rank == 0:
            last_eval_step = eval_records[-1]["step"] if eval_records else None
            if last_eval_step != completed_steps:
                eval_loss = evaluate_fixed_batches(model, optimizer, eval_batches, args, device)
                eval_bpb = eval_loss / math.log(2.0) / args.bytes_per_token
                eval_s = time.time() - eval_t0
                wall = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
                print(
                    f"eval {completed_steps:6d} | loss {eval_loss:.4f} | bpb {eval_bpb:.4f} | "
                    f"eval_s {eval_s:.1f} | time {wall}",
                    flush=True,
                )
                eval_records.append(
                    {
                        "step": completed_steps,
                        "elapsed_s": eval_elapsed,
                        "loss": eval_loss,
                        "bpb": eval_bpb,
                        "eval_s": eval_s,
                        "batches": args.eval_batches,
                        "batch_size": args.eval_batch_size,
                        "time": wall,
                    }
                )
        dist.barrier()
    if rank == 0:
        summary = {
            "mode": args.mode,
            "world_size": world,
            "steps": args.steps,
            "max_seconds": args.max_seconds,
            "completed_steps": completed_steps,
            "resume": args.resume,
            "resume_step": resume_step,
            "resume_loss": resume_loss,
            "resume_optimizer": args.resume_optimizer,
            "local_steps": args.local_steps,
            "batch_size": args.batch_size,
            "chunk_size": args.chunk_size,
            "model_params": model_params,
            "sync_optimizer_state": args.sync_optimizer_state,
            "outer_lr": args.outer_lr,
            "outer_beta": args.outer_beta,
            "total_tokens": int(total_tokens.item()),
            "elapsed_s": elapsed_total,
            "effective_tok_s": total_tokens.item() / elapsed_total,
            "grad_sync_s_total": grad_sync_s_total,
            "grad_sync_count": grad_sync_count,
            "sync_records": sync_records,
            "log_records": log_records,
            "eval_records": eval_records,
            "final_partial_loss": final_loss,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if args.save_final:
            torch.save(model.state_dict(), out / "rank0_final_model.pt")
        print(f"complete elapsed_s={elapsed_total:.1f}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
