#!/usr/bin/env python3
"""
Spectral Analysis of E42 Learned Weight Matrices

Tests the eigenspace decomposition hypothesis from E42_Theory.lean:
- Linear tied recurrence decomposes into independent scalar EMAs
- Each eigenvalue λ_i corresponds to a different memory timescale
- Effective memory length = 1/(1-|λ|)

Usage:
    python analysis/spectral_analysis.py --checkpoint output/level42_100m_.../checkpoint_*.pt
"""

import argparse
import torch
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available, skipping plots")


def load_checkpoint(path):
    """Load checkpoint and extract model state dict."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        return ckpt['model_state_dict']
    if 'model' in ckpt:
        return ckpt['model']
    return ckpt


def extract_W_matrices(state_dict):
    """Extract W matrices from E42 layers."""
    W_matrices = []
    for key, value in state_dict.items():
        # E42 uses tied weights stored as 'W' in each layer
        if '.W' in key and value.dim() == 2:
            W_matrices.append((key, value.float().numpy()))
        # Also check for cell.W pattern
        elif 'cell.W' in key and value.dim() == 2:
            W_matrices.append((key, value.float().numpy()))
    return W_matrices


def compute_eigenspectrum(W):
    """Compute eigenvalues and analyze spectrum."""
    eigenvalues = np.linalg.eigvals(W)

    # Sort by magnitude
    magnitudes = np.abs(eigenvalues)
    sorted_idx = np.argsort(magnitudes)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    magnitudes = magnitudes[sorted_idx]

    # Compute statistics
    stats = {
        'max_magnitude': magnitudes[0],
        'min_magnitude': magnitudes[-1],
        'mean_magnitude': np.mean(magnitudes),
        'spectral_radius': magnitudes[0],
        'eigenvalues': eigenvalues,
        'magnitudes': magnitudes,
    }

    # Memory timescales: τ = 1/(1-|λ|) for |λ| < 1
    # Clip to avoid division by zero
    safe_mags = np.clip(magnitudes, 0, 0.9999)
    memory_lengths = 1.0 / (1.0 - safe_mags)
    stats['memory_lengths'] = memory_lengths
    stats['max_memory'] = memory_lengths[0]
    stats['min_memory'] = memory_lengths[-1]
    stats['mean_memory'] = np.mean(memory_lengths)

    return stats


def analyze_checkpoint(checkpoint_path, output_dir=None):
    """Full spectral analysis of a checkpoint."""
    print(f"\n{'='*60}")
    print(f"Spectral Analysis: {checkpoint_path}")
    print(f"{'='*60}\n")

    state_dict = load_checkpoint(checkpoint_path)
    W_matrices = extract_W_matrices(state_dict)

    if not W_matrices:
        print("No W matrices found. Checking all 2D matrices...")
        for key, value in state_dict.items():
            if hasattr(value, 'dim') and value.dim() == 2:
                shape = tuple(value.shape)
                if shape[0] == shape[1] and shape[0] >= 64:  # Square, reasonably sized
                    print(f"  Found: {key} {shape}")
                    W_matrices.append((key, value.numpy()))

    if not W_matrices:
        print("ERROR: No square weight matrices found!")
        return None

    print(f"Found {len(W_matrices)} W matrices:\n")

    all_stats = []
    for name, W in W_matrices:
        print(f"Layer: {name}")
        print(f"  Shape: {W.shape}")

        stats = compute_eigenspectrum(W)
        all_stats.append((name, stats))

        print(f"  Spectral radius: {stats['spectral_radius']:.4f}")
        print(f"  Eigenvalue magnitude range: [{stats['min_magnitude']:.4f}, {stats['max_magnitude']:.4f}]")
        print(f"  Memory length range: [{stats['min_memory']:.1f}, {stats['max_memory']:.1f}] tokens")
        print(f"  Mean memory length: {stats['mean_memory']:.1f} tokens")
        print()

    # Plot eigenvalue spectrum
    if output_dir and HAS_MATPLOTLIB:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: Eigenvalue magnitudes across layers
        ax = axes[0, 0]
        for i, (name, stats) in enumerate(all_stats):
            layer_num = name.split('.')[1] if '.' in name else str(i)
            ax.plot(stats['magnitudes'], label=f'Layer {layer_num}', alpha=0.7)
        ax.set_xlabel('Eigenvalue Index (sorted by magnitude)')
        ax.set_ylabel('|λ|')
        ax.set_title('Eigenvalue Magnitudes by Layer')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Complex plane (first layer)
        ax = axes[0, 1]
        name, stats = all_stats[0]
        eigs = stats['eigenvalues']
        ax.scatter(eigs.real, eigs.imag, alpha=0.5, s=20)
        circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--', label='Unit circle')
        ax.add_patch(circle)
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_title(f'Eigenvalues in Complex Plane (Layer 0)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 3: Memory length distribution
        ax = axes[1, 0]
        for i, (name, stats) in enumerate(all_stats):
            layer_num = name.split('.')[1] if '.' in name else str(i)
            memory = np.clip(stats['memory_lengths'], 0, 1000)  # Clip for visualization
            ax.hist(memory, bins=50, alpha=0.5, label=f'Layer {layer_num}')
        ax.set_xlabel('Effective Memory Length (tokens)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Memory Timescales')
        ax.legend()
        ax.set_xlim(0, 200)
        ax.grid(True, alpha=0.3)

        # Plot 4: Spectral radius by layer
        ax = axes[1, 1]
        layer_nums = range(len(all_stats))
        spectral_radii = [stats['spectral_radius'] for _, stats in all_stats]
        mean_mems = [stats['mean_memory'] for _, stats in all_stats]

        ax.bar(layer_nums, spectral_radii, alpha=0.7, label='Spectral Radius')
        ax.axhline(y=1.0, color='red', linestyle='--', label='Stability boundary')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Spectral Radius')
        ax.set_title('Spectral Radius by Layer')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / 'eigenspectrum.png'
        plt.savefig(plot_path, dpi=150)
        print(f"Saved plot to {plot_path}")
        plt.close()

        # Summary statistics
        summary_path = output_dir / 'spectral_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("E42 Spectral Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Checkpoint: {checkpoint_path}\n\n")

            for name, stats in all_stats:
                f.write(f"Layer: {name}\n")
                f.write(f"  Spectral radius: {stats['spectral_radius']:.6f}\n")
                f.write(f"  Mean |λ|: {stats['mean_magnitude']:.6f}\n")
                f.write(f"  Memory range: [{stats['min_memory']:.1f}, {stats['max_memory']:.1f}]\n")
                f.write(f"  Mean memory: {stats['mean_memory']:.1f} tokens\n\n")

            # Hypothesis validation
            f.write("\nHypothesis Validation:\n")
            f.write("-" * 30 + "\n")

            # Check if spectral radius < 1 (stability)
            all_stable = all(s['spectral_radius'] < 1.0 for _, s in all_stats)
            f.write(f"All layers stable (ρ < 1): {all_stable}\n")

            # Check for spread of timescales
            for name, stats in all_stats:
                spread = stats['max_memory'] / max(stats['min_memory'], 1)
                f.write(f"  {name}: memory spread ratio = {spread:.1f}x\n")

        print(f"Saved summary to {summary_path}")

    return all_stats


def compare_to_random(dim, num_layers=6):
    """Compare learned spectrum to random orthogonal initialization."""
    print("\n" + "="*60)
    print("Random Orthogonal Baseline (E42 init)")
    print("="*60 + "\n")

    spectral_radius = 0.99  # E42 default

    for i in range(num_layers):
        W = np.random.randn(dim, dim)
        Q, _ = np.linalg.qr(W)
        W = Q * spectral_radius

        stats = compute_eigenspectrum(W)
        print(f"Random Layer {i}:")
        print(f"  Spectral radius: {stats['spectral_radius']:.4f}")
        print(f"  Mean memory: {stats['mean_memory']:.1f} tokens")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spectral analysis of E42 weight matrices')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default='analysis/spectral_output',
                        help='Output directory for plots')
    parser.add_argument('--compare-random', action='store_true',
                        help='Also analyze random orthogonal baseline')
    parser.add_argument('--dim', type=int, default=1536,
                        help='Dimension for random baseline comparison')

    args = parser.parse_args()

    # Analyze checkpoint
    stats = analyze_checkpoint(args.checkpoint, args.output)

    # Compare to random if requested
    if args.compare_random and stats:
        dim = stats[0][1]['eigenvalues'].shape[0]
        compare_to_random(dim)
