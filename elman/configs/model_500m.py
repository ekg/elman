"""
500M parameter model configurations for all ladder levels.

All configs target ~500M total params with balanced depth/width:
- dim=1024, expansion≈2.0, layers=16-24
- vocab_size=32000
"""

VOCAB_SIZE = 32000
TARGET_PARAMS = 500_000_000

CONFIGS = {
    # Linear-space levels (0-6)
    0: {'dim': 1024, 'expansion': 2.5, 'n_layers': 24, 'params': 506_000_000},
    1: {'dim': 1024, 'expansion': 2.0, 'n_layers': 24, 'params': 468_000_000},
    2: {'dim': 1024, 'expansion': 2.0, 'n_layers': 20, 'params': 485_000_000},
    3: {'dim': 1024, 'expansion': 2.0, 'n_layers': 24, 'params': 468_000_000},
    4: {'dim': 1024, 'expansion': 2.0, 'n_layers': 20, 'params': 485_000_000},
    5: {'dim': 1024, 'expansion': 2.0, 'n_layers': 16, 'params': 468_000_000},
    6: {'dim': 1024, 'expansion': 2.0, 'n_layers': 20, 'params': 485_000_000},

    # Log-space levels (log_0 to log_5)
    'log_0': {'dim': 1024, 'expansion': 2.0, 'n_layers': 24, 'params': 468_000_000},
    'log_1': {'dim': 1024, 'expansion': 2.0, 'n_layers': 20, 'params': 485_000_000},
    'log_2': {'dim': 1024, 'expansion': 2.0, 'n_layers': 20, 'params': 485_000_000},
    'log_3': {'dim': 1024, 'expansion': 2.0, 'n_layers': 24, 'params': 468_000_000},
    'log_4': {'dim': 1024, 'expansion': 2.0, 'n_layers': 20, 'params': 485_000_000},
    'log_5': {'dim': 1024, 'expansion': 2.0, 'n_layers': 16, 'params': 468_000_000},
}

def get_config(level):
    """Get 500M config for a specific level."""
    if level not in CONFIGS:
        raise ValueError(f"No 500M config for level {level}")
    return CONFIGS[level].copy()

def list_configs():
    """Print all available configs."""
    for level, cfg in CONFIGS.items():
        print(f"Level {level}: dim={cfg['dim']}, exp={cfg['expansion']}, L={cfg['n_layers']} -> {cfg['params']/1e6:.0f}M")
