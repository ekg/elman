#!/usr/bin/env python3
"""Download FineWeb-Edu sample from HuggingFace."""

from datasets import load_dataset
import argparse

def download_fineweb(output_file, target_size_mb):
    target_bytes = target_size_mb * 1024 * 1024
    
    print(f"Streaming FineWeb-Edu (~{target_size_mb}MB)...")
    
    # Stream from HuggingFace
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        name="sample-10BT",
        split="train",
        streaming=True
    )
    
    collected_text = []
    total_bytes = 0
    
    for i, doc in enumerate(ds):
        text = doc.get('text', '')
        if text:
            collected_text.append(text)
            total_bytes += len(text.encode('utf-8'))
            
            if i % 1000 == 0:
                print(f"  {i} docs, {total_bytes / 1024 / 1024:.1f}MB")
            
            if total_bytes >= target_bytes:
                break
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in collected_text:
            f.write(text)
            f.write("\n\n")
    
    print(f"Saved {total_bytes / 1024 / 1024:.1f}MB to {output_file}")
    print(f"Documents: {len(collected_text)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='fineweb_500mb.txt')
    parser.add_argument('--size', type=float, default=500, help='Target MB')
    args = parser.parse_args()
    download_fineweb(args.output, args.size)
