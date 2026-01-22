# Mega Benchmark Results (Jan 22, 2026)

Training: 10 minutes, batch_size=32, chunk_size=512, lr=3e-4, schedule-free AdamW

## 100M Parameter Scale

| Model | Params | Last100 Loss | Throughput | Dim | Depth | Expansion | Steps | Heads | n_state |
|-------|--------|--------------|------------|-----|-------|-----------|-------|-------|---------|
| mamba2       | 101.9M |       1.4030 |      58.0K |  896 |    20 |       2.0 |  2160 |     - |       - |
| fla-gdn      | 105.7M |       1.4096 |      79.6K | 1024 |    20 |       2.0 |  3000 |     - |       - |
| E88_d12h32   | 100.0M |       1.4862 |      61.9K | 3200 |    20 |       1.0 |  2340 |    12 |      32 |
| E88_h32n16   | 101.8M |       1.4970 |      60.7K | 2432 |    20 |       1.0 |  2290 |    32 |      16 |
| E88_h24n24   | 101.9M |       1.5040 |      55.2K | 2176 |    20 |       1.0 |  2090 |    24 |      24 |
| E88_d20h32   |  99.6M |       1.5213 |      52.8K | 1920 |    20 |       1.0 |  2000 |    20 |      32 |
| E88_h8n64    | 125.6M |       1.5848 |      35.0K | 2432 |    20 |       1.0 |  1320 |     8 |      64 |
| E75h4n32     |  98.8M |       1.5887 |      65.0K | 1920 |    20 |       1.0 |  2450 |     4 |      32 |

## 500M Parameter Scale

| Model | Params | Last100 Loss | Throughput | Dim | Depth | Expansion | Steps | Heads | n_state |
|-------|--------|--------------|------------|-----|-------|-----------|-------|-------|---------|
| fla-gdn      | 533.7M |       1.7449 |      20.9K | 2304 |    20 |       2.0 |   730 |     - |       - |
| mamba2       | 508.4M |       1.8135 |      15.7K | 1600 |    32 |       2.0 |   590 |     - |       - |
| E88_h64n32   | 508.1M |       2.0035 |      13.8K | 3072 |    20 |       1.0 |   520 |    64 |      32 |
| E88_h48n32   | 492.5M |       2.0236 |      15.0K | 3968 |    20 |       1.0 |   570 |    48 |      32 |
| E88_h96n32   | 507.8M |       2.0788 |      10.4K | 2048 |    20 |       1.0 |   390 |    96 |      32 |
| E88_h128n32  | 507.7M |       2.1420 |       8.9K | 1536 |    20 |       1.0 |   330 |   128 |      32 |
| E88_h32n64   | 506.1M |       2.2873 |       8.9K | 3072 |    20 |       1.0 |   330 |    32 |      64 |
| E88_h24n64   | 506.4M |       2.3005 |      10.8K | 4096 |    20 |       1.0 |   410 |    24 |      64 |

## Raw Data (JSON)

```json
[
  {
    "name": "mamba2",
    "scale": "100m",
    "params": 101936528,
    "steps": 2160,
    "last100_loss": 1.403012,
    "throughput": 58008.8932038835,
    "dim": 896,
    "depth": 20,
    "expansion": 2.0,
    "n_heads": "-",
    "n_state": "-"
  },
  {
    "name": "fla-gdn",
    "scale": "100m",
    "params": 105717568,
    "steps": 3000,
    "last100_loss": 1.409607,
    "throughput": 79646.46896551724,
    "dim": 1024,
    "depth": 20,
    "expansion": 2.0,
    "n_heads": "-",
    "n_state": "-"
  },
  {
    "name": "E88_d12h32",
    "scale": "100m",
    "params": 99959520,
    "steps": 2340,
    "last100_loss": 1.4861840000000002,
    "throughput": 61916.84821428572,
    "dim": 3200,
    "depth": 20,
    "expansion": 1.0,
    "n_heads": 12,
    "n_state": 32
  },
  {
    "name": "E88_h32n16",
    "scale": "100m",
    "params": 101846464,
    "steps": 2290,
    "last100_loss": 1.4970269999999999,
    "throughput": 60653.5799086758,
    "dim": 2432,
    "depth": 20,
    "expansion": 1.0,
    "n_heads": 32,
    "n_state": 16
  },
  {
    "name": "E88_h24n24",
    "scale": "100m",
    "params": 101918752,
    "steps": 2090,
    "last100_loss": 1.5039509999999998,
    "throughput": 55221.11055276382,
    "dim": 2176,
    "depth": 20,
    "expansion": 1.0,
    "n_heads": 24,
    "n_state": 24
  },
  {
    "name": "E88_d20h32",
    "scale": "100m",
    "params": 99605280,
    "steps": 2000,
    "last100_loss": 1.521338,
    "throughput": 52838.79473684211,
    "dim": 1920,
    "depth": 20,
    "expansion": 1.0,
    "n_heads": 20,
    "n_state": 32
  },
  {
    "name": "E88_h8n64",
    "scale": "100m",
    "params": 125582784,
    "steps": 1320,
    "last100_loss": 1.584751,
    "throughput": 34976.811475409835,
    "dim": 2432,
    "depth": 20,
    "expansion": 1.0,
    "n_heads": 8,
    "n_state": 64
  },
  {
    "name": "E75h4n32",
    "scale": "100m",
    "params": 98838400,
    "steps": 2450,
    "last100_loss": 1.588742,
    "throughput": 64974.63829787234,
    "dim": 1920,
    "depth": 20,
    "expansion": 1.0,
    "n_heads": 4,
    "n_state": 32
  },
  {
    "name": "fla-gdn",
    "scale": "500m",
    "params": 533694928,
    "steps": 730,
    "last100_loss": 1.7448835616438356,
    "throughput": 20853.444444444445,
    "dim": 2304,
    "depth": 20,
    "expansion": 2.0,
    "n_heads": "-",
    "n_state": "-"
  },
  {
    "name": "mamba2",
    "scale": "500m",
    "params": 508362560,
    "steps": 590,
    "last100_loss": 1.8134915254237287,
    "throughput": 15737.122448979591,
    "dim": 1600,
    "depth": 32,
    "expansion": 2.0,
    "n_heads": "-",
    "n_state": "-"
  },
  {
    "name": "E88_h64n32",
    "scale": "500m",
    "params": 508102784,
    "steps": 520,
    "last100_loss": 2.0035269230769233,
    "throughput": 13789.261904761905,
    "dim": 3072,
    "depth": 20,
    "expansion": 1.0,
    "n_heads": 64,
    "n_state": 32
  },
  {
    "name": "E88_h48n32",
    "scale": "500m",
    "params": 492498816,
    "steps": 570,
    "last100_loss": 2.0235543859649123,
    "throughput": 15040.404255319148,
    "dim": 3968,
    "depth": 20,
    "expansion": 1.0,
    "n_heads": 48,
    "n_state": 32
  },
  {
    "name": "E88_h96n32",
    "scale": "500m",
    "params": 507820416,
    "steps": 390,
    "last100_loss": 2.0787871794871795,
    "throughput": 10447.137931034482,
    "dim": 2048,
    "depth": 20,
    "expansion": 1.0,
    "n_heads": 96,
    "n_state": 32
  },
  {
    "name": "E88_h128n32",
    "scale": "500m",
    "params": 507679872,
    "steps": 330,
    "last100_loss": 2.1420181818181816,
    "throughput": 8893.08695652174,
    "dim": 1536,
    "depth": 20,
    "expansion": 1.0,
    "n_heads": 128,
    "n_state": 32
  },
  {
    "name": "E88_h32n64",
    "scale": "500m",
    "params": 506136064,
    "steps": 330,
    "last100_loss": 2.287281818181818,
    "throughput": 8908.521739130434,
    "dim": 3072,
    "depth": 20,
    "expansion": 1.0,
    "n_heads": 32,
    "n_state": 64
  },
  {
    "name": "E88_h24n64",
    "scale": "500m",
    "params": 506419392,
    "steps": 410,
    "last100_loss": 2.3005,
    "throughput": 10763.967741935483,
    "dim": 4096,
    "depth": 20,
    "expansion": 1.0,
    "n_heads": 24,
    "n_state": 64
  }
]
```
