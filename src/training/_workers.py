"""Worker functions and pool utilities for parallel self-play generation."""

import atexit
import io
import multiprocessing
import os
import signal
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.training.data_format import write_sample


# ---------------------------------------------------------------------------
# Shared memory weight manager (parent process) and worker-side loader
# ---------------------------------------------------------------------------

class SharedWeightsManager:
    """Manages shared memory segments for NNUE model weights.

    Loads weights once in the parent process and shares the underlying
    buffers with worker processes via multiprocessing.shared_memory.
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self._shm_blocks: List[SharedMemory] = []
        self._shm_info: Dict[str, Tuple[str, Tuple[int, ...], str]] = {}

        for key in data.files:
            arr = data[key]
            # Ensure contiguous float32
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            shm = SharedMemory(create=True, size=arr.nbytes)
            # Copy data into shared memory
            buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            np.copyto(buf, arr)
            self._shm_blocks.append(shm)
            self._shm_info[key] = (shm.name, arr.shape, str(arr.dtype))

        # Safety: register cleanup on exit
        atexit.register(self.cleanup)

    def get_shm_info(self) -> Dict[str, Tuple[str, Tuple[int, ...], str]]:
        """Return picklable descriptor for worker processes."""
        return dict(self._shm_info)

    def cleanup(self):
        """Close and unlink all shared memory segments."""
        for shm in self._shm_blocks:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
        self._shm_blocks.clear()
        self._shm_info.clear()


def _load_weights_from_shm(
    shm_info: Dict[str, Tuple[str, Tuple[int, ...], str]],
) -> Dict[str, np.ndarray]:
    """Worker-side: attach to shared memory and copy weights.

    Copies data so shared memory handles can be immediately closed.
    The AcceleratedAccumulator C extension also copies into aligned
    buffers, so this adds no extra overhead vs numpy views.
    """
    weights = {}
    for key, (shm_name, shape, dtype_str) in shm_info.items():
        shm = SharedMemory(name=shm_name, create=False)
        view = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        weights[key] = view.copy()  # Copy so we can close shm immediately
        shm.close()
    return weights


# ---------------------------------------------------------------------------
# Feature set / game factory helpers (top-level for picklability)
# ---------------------------------------------------------------------------

def _create_feature_set(game_name: str):
    """Create a feature set by game name (used in worker processes)."""
    if game_name == "chess":
        from src.features.halfkp import chess_features
        return chess_features()
    elif game_name == "minichess":
        from src.features.halfkp import minichess_features
        return minichess_features()
    elif game_name == "shogi":
        from src.features.halfkp_shogi import shogi_features
        return shogi_features()
    elif game_name == "minishogi":
        from src.features.halfkp_shogi import minishogi_features
        return minishogi_features()
    raise ValueError(f"Unknown game: {game_name}")


def _create_game(game_name: str):
    """Create initial game state by name (used in worker processes)."""
    from src.utils.config import create_game
    return create_game(game_name)


# ---------------------------------------------------------------------------
# Pool management utilities
# ---------------------------------------------------------------------------

def _worker_ignore_sigint():
    """Worker initializer: ignore SIGINT so only the parent handles it."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _get_pool_pids(pool: ProcessPoolExecutor) -> List[int]:
    """Extract PIDs from a ProcessPoolExecutor's internal process map."""
    pids = []
    # _processes is a dict {pid: process} in CPython's implementation
    processes = getattr(pool, '_processes', None)
    if processes is not None:
        for proc in processes.values():
            if proc.is_alive():
                pids.append(proc.pid)
    return pids


def _split_work(num_games: int, workers: int) -> List[int]:
    """Split num_games into chunks for workers."""
    chunk_size = max(1, num_games // (workers * 4))
    chunks = []
    remaining = num_games
    while remaining > 0:
        n = min(chunk_size, remaining)
        chunks.append(n)
        remaining -= n
    return chunks


class _managed_pool:
    """ProcessPoolExecutor wrapper that kills workers on unexpected exit.

    Workers are started with SIGINT ignored so only the parent handles
    KeyboardInterrupt.  On context-manager exit (normal or via exception /
    signal), all worker processes are terminated and joined.
    """

    def __init__(self, max_workers: int):
        self._max_workers = max_workers
        self._pool: Optional[ProcessPoolExecutor] = None
        self._prev_sigint = None
        self._prev_sigterm = None

    def __enter__(self) -> ProcessPoolExecutor:
        self._pool = ProcessPoolExecutor(
            max_workers=self._max_workers,
            initializer=_worker_ignore_sigint,
            mp_context=multiprocessing.get_context("spawn"),
        )
        # Install signal handlers so that SIGINT/SIGTERM clean up workers
        self._prev_sigint = signal.getsignal(signal.SIGINT)
        self._prev_sigterm = signal.getsignal(signal.SIGTERM)

        def _shutdown(signum, frame):
            self._kill_workers()
            # Re-raise so the caller sees the interruption
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            raise SystemExit(1)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
        return self._pool

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original signal handlers
        if self._prev_sigint is not None:
            signal.signal(signal.SIGINT, self._prev_sigint)
        if self._prev_sigterm is not None:
            signal.signal(signal.SIGTERM, self._prev_sigterm)
        self._kill_workers()
        return False

    def _kill_workers(self):
        if self._pool is None:
            return
        # Kill each worker process directly
        for pid in _get_pool_pids(self._pool):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._pool = None


# ---------------------------------------------------------------------------
# Top-level worker functions (must be module-level for multiprocessing pickle)
# ---------------------------------------------------------------------------

def _worker_play_games(
    game_name: str,
    num_games: int,
    max_moves: int,
    random_play_prob: float,
    random_opening_plies: int = 0,
) -> bytes:
    """Worker function for multiprocessing (no model). Returns bytes."""
    # Import here to avoid circular import; SelfPlayEngine lives in selfplay.py
    from src.training.selfplay import SelfPlayEngine

    fs = _create_feature_set(game_name)
    engine = SelfPlayEngine(
        feature_set=fs,
        evaluator=None,
        random_play_prob=random_play_prob,
        random_opening_plies=random_opening_plies,
    )
    buf = io.BytesIO()
    for _ in range(num_games):
        state = _create_game(game_name)
        positions = engine.play_game(state, max_moves)
        for wf, bf, stm, score, result in positions:
            write_sample(buf, wf, bf, stm, score, result)
    return buf.getvalue()


def _worker_play_games_rule_eval(
    game_name: str,
    search_depth: int,
    time_limit_ms: int,
    num_games: int,
    max_moves: int,
    random_play_prob: float,
    random_opening_plies: int = 0,
) -> bytes:
    """Worker function with RuleBasedEvaluator for bootstrap. Returns bytes."""
    from src.search.evaluator import RuleBasedEvaluator
    from src.search.alphabeta import AlphaBetaSearch
    from src.training.selfplay import SelfPlayEngine

    fs = _create_feature_set(game_name)
    rule_eval = RuleBasedEvaluator()
    searcher = AlphaBetaSearch(
        rule_eval, max_depth=search_depth, time_limit_ms=time_limit_ms,
    )
    engine = SelfPlayEngine(
        feature_set=fs,
        evaluator=searcher,
        search_depth=search_depth,
        random_play_prob=random_play_prob,
        random_opening_plies=random_opening_plies,
    )
    buf = io.BytesIO()
    for _ in range(num_games):
        state = _create_game(game_name)
        positions = engine.play_game(state, max_moves)
        for wf, bf, stm, score, result in positions:
            write_sample(buf, wf, bf, stm, score, result)
    return buf.getvalue()


def _worker_play_games_with_model(
    game_name: str,
    model_path: Optional[str],
    search_depth: int,
    time_limit_ms: int,
    num_games: int,
    max_moves: int,
    random_play_prob: float,
    shm_info: Optional[Dict] = None,
    random_opening_plies: int = 0,
) -> bytes:
    """Worker function with NNUE model. Loads from shared memory or file."""
    from src.search.evaluator import NNUEEvaluator
    from src.search.alphabeta import AlphaBetaSearch
    from src.training.selfplay import SelfPlayEngine

    fs = _create_feature_set(game_name)
    if shm_info is not None:
        weights = _load_weights_from_shm(shm_info)
        evaluator = NNUEEvaluator.from_weights_dict(weights, fs)
    else:
        evaluator = NNUEEvaluator.from_numpy(model_path, fs)

    searcher = AlphaBetaSearch(
        evaluator, max_depth=search_depth, time_limit_ms=time_limit_ms,
    )
    engine = SelfPlayEngine(
        feature_set=fs,
        evaluator=searcher,
        search_depth=search_depth,
        random_play_prob=random_play_prob,
        random_opening_plies=random_opening_plies,
    )
    buf = io.BytesIO()
    for _ in range(num_games):
        state = _create_game(game_name)
        positions = engine.play_game(state, max_moves)
        for wf, bf, stm, score, result in positions:
            write_sample(buf, wf, bf, stm, score, result)
    return buf.getvalue()
