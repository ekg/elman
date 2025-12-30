# Training Data

## Available Data

### TinyStories (50MB sample)
Located at `tinystories_50mb.txt` - clean children's stories, good for quick experiments.

Source: `/home/erikg/tinystories/stories.txt` (3.6GB full dataset)

### FineWeb-Edu (download)
High-quality web text filtered for educational content.

```bash
python download_sample.py --size 100  # Download ~100MB sample
```

## Usage

```bash
# Quick test with 50MB TinyStories
./run_500m_comparison.sh data/tinystories_50mb.txt

# Full comparison with larger data
./run_500m_comparison.sh /home/erikg/tinystories/stories.txt
```

## Data Format

Training scripts expect plain UTF-8 text files. Documents should be separated by
blank lines for proper TBPTT (hidden state reset at document boundaries).
