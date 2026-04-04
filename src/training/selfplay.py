"""Self-play engine for generating NNUE training data."""

import atexit
import io
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.features.base import FeatureSet
from src.games.base import GameState
from src.training.data_format import write_sample


def _create_feature_set(game_name: str) -> FeatureSet:
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


def _create_game(game_name: str) -> GameState:
    """Create initial game state by name (used in worker processes)."""
    from src.utils.config import create_game
    return create_game(game_name)


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


class SelfPlayEngine:
    """Generates training data through self-play games.

    Uses either a simple evaluation function (material counting) for
    bootstrapping, or an NNUE evaluator for later iterations.
    """

    def __init__(self, feature_set: FeatureSet, evaluator=None,
                 search_depth: int = 4, random_play_prob: float = 0.1,
                 game_name: Optional[str] = None,
                 model_path: Optional[str] = None,
                 time_limit_ms: int = 2000):
        """
        Args:
            feature_set: Feature extractor for the game.
            evaluator: Optional evaluator with a search(state, depth) method.
                       If None, uses random moves for bootstrapping.
            search_depth: Search depth for generating evaluations.
            random_play_prob: Probability of making a random move (for diversity).
            game_name: Game variant name (for parallel worker model loading).
            model_path: Path to .npz model (for parallel worker model loading).
            time_limit_ms: Time limit per move for NNUE search.
        """
        self.feature_set = feature_set
        self.evaluator = evaluator
        self.search_depth = search_depth
        self.random_play_prob = random_play_prob
        self.game_name = game_name
        self.model_path = model_path
        self.time_limit_ms = time_limit_ms

    def play_game(self, initial_state: GameState,
                  max_moves: int = 512) -> List[Tuple]:
        """Play one self-play game, collecting training positions.

        Returns:
            List of (white_features, black_features, side_to_move, score, result)
            for each position in the game.
        """
        positions = []
        state = initial_state

        for _ in range(max_moves):
            if state.is_terminal():
                break

            # Extract features
            wf = self.feature_set.active_features(state, 0)
            bf = self.feature_set.active_features(state, 1)
            stm = state.side_to_move()

            # Get evaluation score
            moves = state.legal_moves()
            if not moves:
                break

            if self.evaluator and random.random() > self.random_play_prob:
                # Use evaluator to pick move and get score
                best_move, score = self.evaluator.search(state, self.search_depth)
            else:
                # Random move, score = 0
                best_move = random.choice(moves)
                score = 0

            positions.append((wf, bf, stm, score))
            state = state.make_move(best_move)

        # Determine game result
        result_val = state.result()
        if result_val is None:
            game_result = 0
        elif result_val == 1.0:
            game_result = 1
        elif result_val == 0.0:
            game_result = -1
        else:
            game_result = 0

        # Attach result relative to each position's side to move
        final_stm = state.side_to_move()
        results = []
        for wf, bf, stm, score in positions:
            r = game_result if stm == final_stm else -game_result
            results.append((wf, bf, stm, score, r))

        return results

    def _play_game_to_bytes(self, initial_state: GameState,
                            max_moves: int = 512) -> bytes:
        """Play a game and return serialized training data as bytes."""
        positions = self.play_game(initial_state, max_moves)
        buf = io.BytesIO()
        for wf, bf, stm, score, result in positions:
            write_sample(buf, wf, bf, stm, score, result)
        return buf.getvalue()

    def generate_data(self, initial_state_fn: Callable[[], GameState],
                      output_path: str, num_games: int = 1000,
                      max_moves: int = 512, num_workers: int = 0):
        """Generate training data from multiple self-play games.

        Args:
            initial_state_fn: Callable that returns a fresh initial state.
            output_path: Path to output .bin file.
            num_games: Number of games to play.
            max_moves: Maximum moves per game.
            num_workers: Number of parallel workers. 0 = auto (cpu count).
                         1 = single-process (no multiprocessing overhead).
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        if num_workers == 1:
            self._generate_single(initial_state_fn, output_path,
                                  num_games, max_moves)
        elif self.model_path and self.game_name:
            # Parallel with per-worker model loading
            workers = num_workers or min(os.cpu_count() or 4, 8)
            self._generate_parallel_with_model(
                output_path, num_games, max_moves, workers)
        elif self.evaluator is not None:
            # Has in-process evaluator but no model_path for workers
            self._generate_single(initial_state_fn, output_path,
                                  num_games, max_moves)
        else:
            # No evaluator, parallel random games
            workers = num_workers or min(os.cpu_count() or 4, 8)
            self._generate_parallel(output_path, num_games, max_moves,
                                    workers)

        elapsed = time.time() - t0
        print(f"Completed in {elapsed:.1f}s "
              f"({num_games / elapsed:.1f} games/s)")

    def _generate_single(self, initial_state_fn, output_path: str,
                          num_games: int, max_moves: int):
        """Single-process generation."""
        with open(output_path, "wb") as f:
            for game_num in range(num_games):
                state = initial_state_fn()
                data = self._play_game_to_bytes(state, max_moves)
                f.write(data)
                if (game_num + 1) % 100 == 0:
                    print(f"Game {game_num + 1}/{num_games}")

        print(f"Generated data from {num_games} games")
        print(f"Saved to {output_path}")

    def _generate_parallel(self, output_path: str,
                            num_games: int, max_moves: int, workers: int):
        """Multi-process generation for bootstrapping (no evaluator)."""
        print(f"Using {workers} worker processes")
        chunks = _split_work(num_games, workers)

        with open(output_path, "wb") as f:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        _worker_play_games,
                        self.game_name or "chess",
                        n, max_moves, self.random_play_prob,
                    )
                    for n in chunks
                ]
                for i, future in enumerate(futures):
                    data = future.result()
                    f.write(data)
                    print(f"Chunk {i + 1}/{len(chunks)} done")

        print(f"Generated data from {num_games} games")
        print(f"Saved to {output_path}")

    def _generate_parallel_with_model(self, output_path: str,
                                       num_games: int, max_moves: int,
                                       workers: int):
        """Multi-process generation with shared NNUE model weights."""
        print(f"Using {workers} worker processes (with NNUE model)")
        chunks = _split_work(num_games, workers)

        # Load weights once, share via shared memory
        shm_mgr = SharedWeightsManager(self.model_path)
        try:
            shm_info = shm_mgr.get_shm_info()
            with open(output_path, "wb") as f:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = [
                        pool.submit(
                            _worker_play_games_with_model,
                            self.game_name, None,
                            self.search_depth, self.time_limit_ms,
                            n, max_moves, self.random_play_prob,
                            shm_info,
                        )
                        for n in chunks
                    ]
                    for i, future in enumerate(futures):
                        data = future.result()
                        f.write(data)
                        print(f"Chunk {i + 1}/{len(chunks)} done")
        finally:
            shm_mgr.cleanup()

        print(f"Generated data from {num_games} games")
        print(f"Saved to {output_path}")


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


def _worker_play_games(
    game_name: str,
    num_games: int,
    max_moves: int,
    random_play_prob: float,
) -> bytes:
    """Worker function for multiprocessing (no model). Returns bytes."""
    fs = _create_feature_set(game_name)
    engine = SelfPlayEngine(
        feature_set=fs,
        evaluator=None,
        random_play_prob=random_play_prob,
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
) -> bytes:
    """Worker function with NNUE model. Loads from shared memory or file."""
    from src.search.evaluator import NNUEEvaluator
    from src.search.alphabeta import AlphaBetaSearch

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
    )
    buf = io.BytesIO()
    for _ in range(num_games):
        state = _create_game(game_name)
        positions = engine.play_game(state, max_moves)
        for wf, bf, stm, score, result in positions:
            write_sample(buf, wf, bf, stm, score, result)
    return buf.getvalue()
