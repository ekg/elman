#!/usr/bin/env python3
"""
Download a small sample from The Pile for benchmarking.

Uses streaming to get just the first ~1MB of text.
"""

import json
import zstandard
import requests
from io import BytesIO

# The Pile validation set (smaller, faster to access)
PILE_URL = "https://the-eye.eu/public/AI/pile/val.jsonl.zst"

def download_pile_sample(output_file="pile_1mb.txt", target_size_mb=1):
    """Download approximately target_size_mb of text from The Pile."""

    target_bytes = target_size_mb * 1024 * 1024
    collected_text = []
    total_bytes = 0

    print(f"Downloading Pile sample (~{target_size_mb}MB)...")

    # Stream download with decompression
    response = requests.get(PILE_URL, stream=True)
    dctx = zstandard.ZstdDecompressor()

    # Create a streaming reader
    stream_reader = dctx.stream_reader(response.raw)
    text_stream = BytesIO()

    buffer = b""
    while total_bytes < target_bytes:
        chunk = stream_reader.read(65536)
        if not chunk:
            break

        buffer += chunk

        # Process complete lines
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            try:
                doc = json.loads(line.decode('utf-8'))
                text = doc.get('text', '')
                if text:
                    collected_text.append(text)
                    total_bytes += len(text.encode('utf-8'))

                    if total_bytes >= target_bytes:
                        break
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

        # Progress
        if len(collected_text) % 100 == 0:
            print(f"  Collected {len(collected_text)} docs, {total_bytes / 1024 / 1024:.2f}MB")

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in collected_text:
            f.write(text)
            f.write("\n\n")  # Document separator

    print(f"Saved {total_bytes / 1024 / 1024:.2f}MB to {output_file}")
    print(f"Total documents: {len(collected_text)}")

    return output_file


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='pile_1mb.txt')
    parser.add_argument('--size', type=float, default=1.0, help='Target size in MB')
    args = parser.parse_args()

    download_pile_sample(args.output, args.size)
