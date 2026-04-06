"""Data loading for MLX NNUE training."""

import random
import struct
import threading
import queue
import numpy as np
import mlx.core as mx
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from src.training.data_format import read_sample


def preload_samples(filepaths: List[str]) -> List[Tuple]:
    """Fast preload of all samples from multiple .bin files into memory.

    Uses memoryview + struct.unpack_from for ~5-10x faster parsing than
    per-read file I/O.  Returns list of (wf, bf, stm, score, result_float).
    """
    all_samples: List[Tuple] = []
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    for fp in filepaths:
        data = Path(fp).read_bytes()
        n = len(data)
        offset = 0

        while offset < n:
            # num_white_features: uint16
            if offset + 2 > n:
                break
            nw = struct.unpack_from("<H", data, offset)[0]
            offset += 2

            # white_feature_indices: uint32[nw]
            wf_end = offset + nw * 4
            if wf_end > n:
                break
            wf = list(struct.unpack_from(f"<{nw}I", data, offset))
            offset = wf_end

            # num_black_features: uint16
            if offset + 2 > n:
                break
            nb = struct.unpack_from("<H", data, offset)[0]
            offset += 2

            # black_feature_indices: uint32[nb]
            bf_end = offset + nb * 4
            if bf_end > n:
                break
            bf = list(struct.unpack_from(f"<{nb}I", data, offset))
            offset = bf_end

            # side_to_move(uint8) + score(int16) + result(int8)
            if offset + 4 > n:
                break
            stm = data[offset]
            score = struct.unpack_from("<h", data, offset + 1)[0]
            result = struct.unpack_from("<b", data, offset + 3)[0]
            offset += 4

            result_float = (result + 1.0) / 2.0
            all_samples.append((wf, bf, stm, float(score), result_float))

    return all_samples


def load_batches_from_samples(
    samples: List[Tuple],
    batch_size: int = 2048,
    max_active: int = 32,
    shuffle: bool = True,
    prefetch: int = 8,
) -> Iterator[Dict[str, mx.array]]:
    """Yield batches from pre-loaded samples with optional shuffling.

    Uses threaded prefetch for CPU/GPU overlap.
    """
    if shuffle:
        indices = list(range(len(samples)))
        random.shuffle(indices)
        samples_ordered = [samples[i] for i in indices]
    else:
        samples_ordered = samples

    q: queue.Queue = queue.Queue(maxsize=prefetch)

    def _producer():
        for i in range(0, len(samples_ordered), batch_size):
            batch_samples = samples_ordered[i:i + batch_size]
            if batch_samples:
                q.put(_collate_batch(batch_samples, max_active))
        q.put(None)

    t = threading.Thread(target=_producer, daemon=True)
    t.start()

    while True:
        batch = q.get()
        if batch is None:
            break
        yield batch

    t.join()


def load_batches(filepath: str, batch_size: int = 2048,
                 max_active: int = 32,
                 prefetch: int = 8) -> Iterator[Dict[str, mx.array]]:
    """Yield batches of training data as MLX arrays with threaded prefetch.

    A background thread reads and collates batches while the main thread
    runs GPU computation. Since MLX releases the GIL during Metal compute,
    this gives genuine CPU/GPU overlap.

    Args:
        filepath: Path to .bin training data file.
        batch_size: Samples per batch.
        max_active: Maximum active features per perspective (for padding).
        prefetch: Number of batches to buffer ahead.
    """
    q: queue.Queue = queue.Queue(maxsize=prefetch)

    def _reader():
        buffer: List[Tuple] = []
        with open(filepath, "rb") as f:
            while True:
                sample = read_sample(f)
                if sample is None:
                    break
                wf, bf, stm, score, result = sample
                result_float = (result + 1.0) / 2.0
                buffer.append((wf, bf, stm, float(score), result_float))

                if len(buffer) >= batch_size:
                    q.put(_collate_batch(buffer[:batch_size], max_active))
                    buffer = buffer[batch_size:]

        # Final partial batch
        if buffer:
            q.put(_collate_batch(buffer, max_active))
        q.put(None)  # sentinel

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    while True:
        batch = q.get()
        if batch is None:
            break
        yield batch

    t.join()


def _collate_batch(samples, max_active: int) -> Dict[str, mx.array]:
    """Pad and collate a list of samples into MLX arrays."""
    n = len(samples)
    # Pre-allocate numpy arrays, then convert once to mx.array
    wf_np = np.zeros((n, max_active), dtype=np.int32)
    bf_np = np.zeros((n, max_active), dtype=np.int32)
    wm_np = np.zeros((n, max_active), dtype=np.float32)
    bm_np = np.zeros((n, max_active), dtype=np.float32)
    scores = np.empty(n, dtype=np.float32)
    results = np.empty(n, dtype=np.float32)
    stms = np.empty(n, dtype=np.int32)

    for i, (wf, bf, stm, score, result) in enumerate(samples):
        wlen = min(len(wf), max_active)
        wf_np[i, :wlen] = wf[:wlen]
        wm_np[i, :wlen] = 1.0
        blen = min(len(bf), max_active)
        bf_np[i, :blen] = bf[:blen]
        bm_np[i, :blen] = 1.0
        scores[i] = score
        results[i] = result
        stms[i] = stm

    return {
        "white_features": mx.array(wf_np),
        "black_features": mx.array(bf_np),
        "white_mask": mx.array(wm_np),
        "black_mask": mx.array(bm_np),
        "score": mx.array(scores),
        "result": mx.array(results),
        "side_to_move": mx.array(stms),
    }
