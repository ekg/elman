#!/usr/bin/env python3
"""
Download a sample of FineWeb-Edu for training experiments.

Usage:
    python download_sample.py [--size 100]  # Download ~100MB sample
"""

import argparse
from pathlib import Path


def download_fineweb_sample(size_mb=100):
    """Download FineWeb-Edu sample using HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return

    print(f"Downloading FineWeb-Edu sample (~{size_mb}MB)...")

    # Load streaming sample
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )

    output_path = Path(__file__).parent / f"fineweb_sample_{size_mb}mb.txt"
    total_bytes = 0
    target_bytes = size_mb * 1024 * 1024

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in ds:
            text = item['text'] + '\n\n'
            f.write(text)
            total_bytes += len(text.encode('utf-8'))

            if total_bytes >= target_bytes:
                break

            if total_bytes % (10 * 1024 * 1024) == 0:
                print(f"  Downloaded {total_bytes / 1024 / 1024:.1f}MB...")

    print(f"Saved to: {output_path}")
    print(f"Total size: {total_bytes / 1024 / 1024:.1f}MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100, help="Target size in MB")
    args = parser.parse_args()

    download_fineweb_sample(args.size)


if __name__ == "__main__":
    main()
