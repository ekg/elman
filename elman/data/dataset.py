"""
Document-aware dataset for byte-level language modeling.

Data format: Raw bytes with 0x1e (ASCII record separator) as document delimiter.

Two modes:
1. DocumentStreamDataset - Single stream, advances on each __getitem__
2. BatchedStreamDataset - batch_size independent streams for TBPTT
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mmap


class DocumentStreamDataset(Dataset):
    """
    Document-aware streaming dataset for training.

    Key features:
    - Respects document boundaries (0x1e delimiter)
    - Byte-level tokenization (vocab size 256)
    - Memory-mapped file access for efficiency
    - Random starting position per rank
    """

    def __init__(self, data_path: str, chunk_size: int, rank: int = 0,
                 world_size: int = 1, seed: int = 42):
        self.chunk_size = chunk_size
        self.rank = rank
        self.world_size = world_size

        # Open the data file with memory mapping
        self.data_file = open(data_path, 'rb')
        self.mmap = mmap.mmap(self.data_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.file_size = len(self.mmap)

        # Random starting position for this rank
        rng = np.random.RandomState(seed + rank)
        self.position = rng.randint(0, max(1, self.file_size - 1000))

        # Track statistics
        self.chunks_served = 0
        self.docs_completed = 0
        self.bytes_processed = 0
        self.wraps = 0

        # Scan forward to next document boundary to start clean
        self._scan_to_next_document()

        # Buffer for accumulating bytes
        self.token_buffer = []

    def __len__(self):
        return 1_000_000_000  # Effectively infinite

    def _scan_to_next_document(self):
        """Scan forward to the start of the next document."""
        while self.position < self.file_size and self.mmap[self.position] != 0x1e:
            self.position += 1

        # Skip the delimiter itself
        if self.position < self.file_size:
            self.position += 1
        else:
            self.position = 0
            self.wraps += 1

    def __getitem__(self, idx):
        """
        Returns: (chunk_tensor, is_final_chunk_in_doc, actual_chunk_length)

        Always returns fixed-size tensors. Partial chunks are padded with zeros.
        """
        while len(self.token_buffer) < self.chunk_size:
            # Check if we need to wrap
            if self.position >= self.file_size:
                self.position = 0
                self.wraps += 1

            byte_val = self.mmap[self.position]
            self.position += 1
            self.bytes_processed += 1

            # Check for document boundary
            if byte_val == 0x1e:
                self.docs_completed += 1

                if len(self.token_buffer) > 0:
                    # Partial chunk at document boundary - pad to maintain fixed size
                    actual_length = len(self.token_buffer)

                    chunk = torch.zeros(self.chunk_size, dtype=torch.long)
                    chunk[:actual_length] = torch.tensor(self.token_buffer, dtype=torch.long)
                    self.token_buffer = []
                    self.chunks_served += 1

                    return chunk, True, actual_length
                else:
                    # Empty buffer at document start, skip delimiter
                    continue
            else:
                self.token_buffer.append(byte_val)

        # Full chunk
        chunk = torch.tensor(self.token_buffer[:self.chunk_size], dtype=torch.long)
        self.token_buffer = self.token_buffer[self.chunk_size:]
        self.chunks_served += 1

        return chunk, False, self.chunk_size

    def get_stats(self):
        """Return current dataset statistics."""
        return {
            'chunks_served': self.chunks_served,
            'docs_completed': self.docs_completed,
            'bytes_processed': self.bytes_processed,
            'wraps': self.wraps,
            'position': self.position
        }

    def __del__(self):
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'data_file'):
            self.data_file.close()


class BatchedStreamDataset:
    """
    Multi-stream dataset for TBPTT training.

    Each batch element has its own independent data stream that persists
    across calls. This is REQUIRED for TBPTT to work correctly - hidden
    states must match up with the data streams they were computed from.

    Usage:
        dataset = BatchedStreamDataset(path, batch_size=16, chunk_size=512)
        for step in range(num_steps):
            chunks, is_doc_end = dataset.get_batch()
            # chunks[i] continues from where chunks[i] left off last step
    """

    def __init__(self, data_path: str, batch_size: int, chunk_size: int,
                 rank: int = 0, world_size: int = 1, seed: int = 42):
        self.batch_size = batch_size
        self.chunk_size = chunk_size

        # Open the data file with memory mapping
        self.data_file = open(data_path, 'rb')
        self.mmap = mmap.mmap(self.data_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.file_size = len(self.mmap)

        # Each batch element has its own stream state
        # Spread starting positions evenly across file
        rng = np.random.RandomState(seed + rank * 1000)
        total_streams = batch_size * world_size
        stream_offset = rank * batch_size

        self.positions = []
        self.buffers = []
        for i in range(batch_size):
            # Evenly spaced + small random offset
            base_pos = (stream_offset + i) * self.file_size // total_streams
            jitter = rng.randint(0, max(1, self.file_size // (total_streams * 10)))
            pos = (base_pos + jitter) % self.file_size
            self.positions.append(pos)
            self.buffers.append([])

        # Scan each stream to next document boundary
        for i in range(batch_size):
            self._scan_to_next_document(i)

    def _scan_to_next_document(self, stream_idx: int):
        """Scan stream to the start of the next document."""
        while self.positions[stream_idx] < self.file_size:
            if self.mmap[self.positions[stream_idx]] == 0x1e:
                self.positions[stream_idx] = (self.positions[stream_idx] + 1) % self.file_size
                return
            self.positions[stream_idx] += 1
        self.positions[stream_idx] = 0

    def _get_chunk(self, stream_idx: int):
        """Get next chunk from a specific stream."""
        buf = self.buffers[stream_idx]

        while len(buf) < self.chunk_size:
            pos = self.positions[stream_idx]
            if pos >= self.file_size:
                self.positions[stream_idx] = 0
                pos = 0

            byte_val = self.mmap[pos]
            self.positions[stream_idx] = pos + 1

            if byte_val == 0x1e:
                # Document boundary
                if len(buf) > 0:
                    # Return partial chunk padded with zeros
                    actual_len = len(buf)
                    chunk = torch.zeros(self.chunk_size, dtype=torch.long)
                    chunk[:actual_len] = torch.tensor(buf, dtype=torch.long)
                    self.buffers[stream_idx] = []
                    return chunk, True, actual_len
                # Empty buffer, skip delimiter
                continue
            else:
                buf.append(byte_val)

        # Full chunk
        chunk = torch.tensor(buf[:self.chunk_size], dtype=torch.long)
        self.buffers[stream_idx] = buf[self.chunk_size:]
        return chunk, False, self.chunk_size

    def get_batch(self, device=None):
        """
        Get one batch where each element continues its own stream.

        Returns:
            chunks: [batch_size, chunk_size] tensor
            is_doc_end: [batch_size] boolean tensor (True if chunk ends at doc boundary)
        """
        chunks = []
        is_doc_ends = []

        for i in range(self.batch_size):
            chunk, is_doc_end, _ = self._get_chunk(i)
            chunks.append(chunk)
            is_doc_ends.append(is_doc_end)

        chunks = torch.stack(chunks)
        is_doc_end = torch.tensor(is_doc_ends, dtype=torch.bool)

        if device is not None:
            chunks = chunks.to(device)
            is_doc_end = is_doc_end.to(device)

        return chunks, is_doc_end

    def __del__(self):
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'data_file'):
            self.data_file.close()


def create_dataloader(data_path: str, batch_size: int, chunk_size: int,
                      device=None, num_workers: int = 0, seed: int = 42):
    """
    Create a dataloader for training.

    Args:
        data_path: Path to training data file
        batch_size: Batch size
        chunk_size: Sequence length
        device: Target device (optional)
        num_workers: Number of data loading workers
        seed: Random seed

    Returns:
        DataLoader that yields (chunks, is_doc_end, actual_lengths)
    """
    dataset = DocumentStreamDataset(
        data_path=data_path,
        chunk_size=chunk_size,
        rank=0,
        world_size=1,
        seed=seed,
    )

    def collate_fn(batch):
        """Collate function that handles doc boundary info."""
        chunks = torch.stack([b[0] for b in batch])
        is_doc_end = torch.tensor([b[1] for b in batch], dtype=torch.bool)
        actual_lengths = torch.tensor([b[2] for b in batch], dtype=torch.long)

        if device is not None:
            chunks = chunks.to(device)
            is_doc_end = is_doc_end.to(device)
            actual_lengths = actual_lengths.to(device)

        return chunks, is_doc_end, actual_lengths

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    return dataloader
