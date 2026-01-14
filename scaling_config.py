"""
Scaling study configuration for E1, E33, E42, Mamba2 at 5 parameter scales.

All configurations use depth=6 (optimal per CLAUDE.md sweet spot).
Batch sizes determined for ~48GB GPU memory with seq_len=512.

Parameter counts (verified):
- 50M:  E1=49.5M (0.99x), E33=50.2M (1.00x), E42=50.3M (1.01x), Mamba2=49.3M (0.99x)
- 100M: E1=100.3M (1.00x), E33=101.2M (1.01x), E42=100.2M (1.00x), Mamba2=102.1M (1.02x)
- 250M: E1=249.6M (1.00x), E33=246.6M (0.99x), E42=249.0M (1.00x), Mamba2=252.1M (1.01x)
- 500M: E1=504.4M (1.01x), E33=510.8M (1.02x), E42=521.7M (1.04x), Mamba2=502.8M (1.01x)
- 1B:   E1=1041.6M (1.04x), E33=984.8M (0.98x), E42=1063.7M (1.06x), Mamba2=1002.5M (1.00x)

Model architectures:
- E1 (MambaGatedElman): in_proj(dim->2*dim), W_x(dim^2), W_h(dim^2), out_proj(dim->dim)
- E33 (SelfGate): in_proj(dim->dim), W_x(dim^2), W_h(dim^2), out_proj(dim->dim) - self-gating
- E42 (LinearTied): in_proj(dim->dim), W(dim^2) tied, out_proj(dim->dim) - linear recurrence
- Mamba2: SSM with expand=2, d_state=64/128, headdim=64
"""

SCALING_CONFIG = {
    "50M": {  # Target: 50,000,000 params
        "E1": {
            "dim": 1280,
            "depth": 6,
            "batch_size": 256,
            "params": 49496320,
            "expansion": 1.0,
        },
        "E33": {
            "dim": 1440,
            "depth": 6,
            "batch_size": 256,
            "params": 50153760,
            "expansion": 1.0,
        },
        "E42": {
            "dim": 1664,
            "depth": 6,
            "batch_size": 256,
            "params": 50287744,
            "expansion": 1.0,
        },
        "Mamba2": {
            "dim": 1152,
            "depth": 6,
            "batch_size": 256,
            "params": 49307784,
            "d_state": 64,
            "expand": 2,
            "headdim": 64,
        },
    },
    "100M": {  # Target: 100,000,000 params
        "E1": {
            "dim": 1824,
            "depth": 6,
            "batch_size": 160,
            "params": 100299936,
            "expansion": 1.0,
        },
        "E33": {
            "dim": 2048,
            "depth": 6,
            "batch_size": 160,
            "params": 101214208,
            "expansion": 1.0,
        },
        "E42": {
            "dim": 2352,
            "depth": 6,
            "batch_size": 160,
            "params": 100206960,
            "expansion": 1.0,
        },
        "Mamba2": {
            "dim": 1664,
            "depth": 6,
            "batch_size": 160,
            "params": 102051240,
            "d_state": 64,
            "expand": 2,
            "headdim": 64,
        },
    },
    "250M": {  # Target: 250,000,000 params
        "E1": {
            "dim": 2880,
            "depth": 6,
            "batch_size": 96,
            "params": 249606720,
            "expansion": 1.0,
        },
        "E33": {
            "dim": 3200,
            "depth": 6,
            "batch_size": 96,
            "params": 246620800,
            "expansion": 1.0,
        },
        "E42": {
            "dim": 3712,
            "depth": 6,
            "batch_size": 96,
            "params": 249019520,
            "expansion": 1.0,
        },
        "Mamba2": {
            "dim": 2624,
            "depth": 6,
            "batch_size": 96,
            "params": 252082500,
            "d_state": 128,
            "expand": 2,
            "headdim": 64,
        },
    },
    "500M": {  # Target: 500,000,000 params
        "E1": {
            "dim": 4096,
            "depth": 6,
            "batch_size": 64,
            "params": 504418304,
            "expansion": 1.0,
        },
        "E33": {
            "dim": 4608,
            "depth": 6,
            "batch_size": 64,
            "params": 510847488,
            "expansion": 1.0,
        },
        "E42": {
            "dim": 5376,
            "depth": 6,
            "batch_size": 64,
            "params": 521670912,
            "expansion": 1.0,
        },
        "Mamba2": {
            "dim": 3712,
            "depth": 6,
            "batch_size": 64,
            "params": 502751784,
            "d_state": 128,
            "expand": 2,
            "headdim": 64,
        },
    },
    "1B": {  # Target: 1,000,000,000 params
        "E1": {
            "dim": 5888,
            "depth": 6,
            "batch_size": 48,
            "params": 1041640192,
            "expansion": 1.0,
        },
        "E33": {
            "dim": 6400,
            "depth": 6,
            "batch_size": 48,
            "params": 984761600,
            "expansion": 1.0,
        },
        "E42": {
            "dim": 7680,
            "depth": 6,
            "batch_size": 48,
            "params": 1063749120,
            "expansion": 1.0,
        },
        "Mamba2": {
            "dim": 5248,
            "depth": 6,
            "batch_size": 48,
            "params": 1002490248,
            "d_state": 128,
            "expand": 2,
            "headdim": 64,
        },
    },
}


def get_config(scale: str, model: str):
    """Get configuration for a specific scale and model."""
    return SCALING_CONFIG[scale][model]


def get_all_configs():
    """Get all configurations as a flat list of (scale, model, config) tuples."""
    result = []
    for scale, models in SCALING_CONFIG.items():
        for model, cfg in models.items():
            result.append((scale, model, cfg))
    return result


def print_config_table():
    """Print configuration as a formatted table."""
    print("Scaling Study Configuration")
    print("=" * 100)
    print(f"{'Scale':<8} {'Model':<8} {'dim':<6} {'depth':<6} {'batch':<6} {'params':>15} {'ratio':>8}")
    print("-" * 100)
    targets = {'50M': 50e6, '100M': 100e6, '250M': 250e6, '500M': 500e6, '1B': 1e9}
    for scale, models in SCALING_CONFIG.items():
        target = targets[scale]
        for model, cfg in models.items():
            ratio = cfg['params'] / target
            print(f"{scale:<8} {model:<8} {cfg['dim']:<6} {cfg['depth']:<6} "
                  f"{cfg['batch_size']:<6} {cfg['params']:>15,} {ratio:>7.2f}x")
        print()


if __name__ == "__main__":
    print_config_table()

    # Also print in YAML-like format for easy reading
    print("\nConfiguration Summary:")
    print("=" * 60)
    for scale, models in SCALING_CONFIG.items():
        print(f"\n{scale}:")
        for model, cfg in models.items():
            print(f"  {model:8s}: dim={cfg['dim']:4d}, depth={cfg['depth']}, "
                  f"batch={cfg['batch_size']:>3}, params={cfg['params']:,}")
