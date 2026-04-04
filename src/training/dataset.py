"""Data loading for MLX NNUE training."""

import threading
import queue
import numpy as np
import mlx.core as mx
from typing import Dict, Iterator, List, Tuple

from src.training.data_format import read_sample


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
