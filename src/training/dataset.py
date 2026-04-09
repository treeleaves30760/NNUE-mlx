"""Data loading for MLX NNUE training.

Optimized for Apple Silicon: pre-pads into contiguous numpy arrays during
loading so that batching is pure array slicing (no Python per-sample loops).
"""

import struct
import threading
import queue
import numpy as np
import mlx.core as mx
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from src.training.data_format import read_sample


def preload_samples(filepaths: List[str],
                    max_active: int = 32) -> Dict[str, np.ndarray]:
    """Fast preload of all samples from .bin files into padded numpy arrays.

    Returns a dict of numpy arrays ready for batch slicing:
        white_features: (N, max_active) int32
        black_features: (N, max_active) int32
        white_mask:     (N, max_active) float32
        black_mask:     (N, max_active) float32
        score:          (N,) float32
        result:         (N,) float32
        side_to_move:   (N,) int32
    """
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    # First pass: collect raw tuples (variable-length features)
    raw: List[Tuple] = []
    for fp in filepaths:
        data = Path(fp).read_bytes()
        n = len(data)
        offset = 0

        while offset < n:
            if offset + 2 > n:
                break
            nw = struct.unpack_from("<H", data, offset)[0]
            offset += 2

            wf_end = offset + nw * 4
            if wf_end > n:
                break
            wf = struct.unpack_from(f"<{nw}I", data, offset)
            offset = wf_end

            if offset + 2 > n:
                break
            nb = struct.unpack_from("<H", data, offset)[0]
            offset += 2

            bf_end = offset + nb * 4
            if bf_end > n:
                break
            bf = struct.unpack_from(f"<{nb}I", data, offset)
            offset = bf_end

            if offset + 4 > n:
                break
            stm = data[offset]
            score = struct.unpack_from("<h", data, offset + 1)[0]
            result = struct.unpack_from("<b", data, offset + 3)[0]
            offset += 4

            raw.append((wf, bf, stm, float(score), (result + 1.0) / 2.0))

    # Second pass: pack into contiguous padded numpy arrays
    total = len(raw)
    wf_arr = np.zeros((total, max_active), dtype=np.int32)
    bf_arr = np.zeros((total, max_active), dtype=np.int32)
    wm_arr = np.zeros((total, max_active), dtype=np.float32)
    bm_arr = np.zeros((total, max_active), dtype=np.float32)
    score_arr = np.empty(total, dtype=np.float32)
    result_arr = np.empty(total, dtype=np.float32)
    stm_arr = np.empty(total, dtype=np.int32)

    for i, (wf, bf, stm, score, result) in enumerate(raw):
        wlen = min(len(wf), max_active)
        wf_arr[i, :wlen] = wf[:wlen]
        wm_arr[i, :wlen] = 1.0
        blen = min(len(bf), max_active)
        bf_arr[i, :blen] = bf[:blen]
        bm_arr[i, :blen] = 1.0
        score_arr[i] = score
        result_arr[i] = result
        stm_arr[i] = stm

    return {
        "white_features": wf_arr,
        "black_features": bf_arr,
        "white_mask": wm_arr,
        "black_mask": bm_arr,
        "score": score_arr,
        "result": result_arr,
        "side_to_move": stm_arr,
    }


def load_batches_from_samples(
    samples,
    batch_size: int = 16384,
    max_active: int = 32,
    shuffle: bool = True,
    prefetch: int = 8,
    mirror_table: np.ndarray = None,
) -> Iterator[Dict[str, mx.array]]:
    """Yield batches from pre-loaded numpy arrays with optional shuffling.

    Accepts either:
    - Dict of numpy arrays (from preload_samples) -> fast array slicing
    - List of tuples (legacy) -> falls back to per-sample collation

    Uses threaded prefetch for CPU/GPU overlap.
    If mirror_table is provided, each batch has a 50% chance of being mirrored.
    """
    if isinstance(samples, dict):
        yield from _batches_from_arrays(samples, batch_size, shuffle, prefetch,
                                         mirror_table)
    else:
        yield from _batches_from_tuples(samples, batch_size, max_active,
                                         shuffle, prefetch)


def _batches_from_arrays(
    arrays: Dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
    prefetch: int,
    mirror_table: np.ndarray = None,
) -> Iterator[Dict[str, mx.array]]:
    """Fast path: batch via numpy fancy indexing, no per-sample Python loop.

    If mirror_table is provided, each batch has a 50% chance of having all
    feature indices remapped via the mirror lookup table (horizontal flip).
    """
    n = len(arrays["score"])
    indices = np.arange(n, dtype=np.int64)
    if shuffle:
        np.random.shuffle(indices)  # In-place, no copy

    q: queue.Queue = queue.Queue(maxsize=prefetch)

    def _producer():
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            batch_np = {k: v[idx] for k, v in arrays.items()}
            # Data augmentation: mirror with 50% probability
            if mirror_table is not None and np.random.random() < 0.5:
                wf = batch_np["white_features"]
                bf = batch_np["black_features"]
                wm = batch_np["white_mask"]
                bm = batch_np["black_mask"]
                # Only mirror non-padding indices (mask > 0)
                wf_mir = np.where(wm > 0, mirror_table[wf], 0)
                bf_mir = np.where(bm > 0, mirror_table[bf], 0)
                batch_np["white_features"] = wf_mir
                batch_np["black_features"] = bf_mir
            batch = {k: mx.array(v) for k, v in batch_np.items()}
            q.put(batch)
        q.put(None)

    t = threading.Thread(target=_producer, daemon=True)
    t.start()

    while True:
        batch = q.get()
        if batch is None:
            break
        yield batch

    t.join()


def _batches_from_tuples(
    samples: List[Tuple],
    batch_size: int,
    max_active: int,
    shuffle: bool,
    prefetch: int,
) -> Iterator[Dict[str, mx.array]]:
    """Legacy fallback for tuple-based samples."""
    if shuffle:
        indices = np.arange(len(samples), dtype=np.int64)
        np.random.shuffle(indices)
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


def load_batches(filepath: str, batch_size: int = 16384,
                 max_active: int = 32,
                 prefetch: int = 8) -> Iterator[Dict[str, mx.array]]:
    """Yield batches of training data as MLX arrays with threaded prefetch.

    A background thread reads and collates batches while the main thread
    runs GPU computation. Since MLX releases the GIL during Metal compute,
    this gives genuine CPU/GPU overlap.
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
