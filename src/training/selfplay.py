"""Self-play engine for generating NNUE training data."""

import io
import os
import random
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from src.features.base import FeatureSet
from src.games.base import GameState
from src.training.data_format import write_sample
from src.training._workers import (
    SharedWeightsManager,
    _managed_pool,
    _split_work,
    _worker_play_games,
    _worker_play_games_rule_eval,
    _worker_play_games_with_model,
)


class SelfPlayEngine:
    """Generates training data through self-play games.

    Uses either a simple evaluation function (material counting) for
    bootstrapping, or an NNUE evaluator for later iterations.
    """

    def __init__(self, feature_set: FeatureSet, evaluator=None,
                 search_depth: int = 4, random_play_prob: float = 0.1,
                 game_name: Optional[str] = None,
                 model_path: Optional[str] = None,
                 time_limit_ms: int = 2000,
                 use_rule_eval: bool = False):
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
            use_rule_eval: Use RuleBasedEvaluator (for parallel bootstrap).
        """
        self.feature_set = feature_set
        self.evaluator = evaluator
        self.search_depth = search_depth
        self.random_play_prob = random_play_prob
        self.game_name = game_name
        self.model_path = model_path
        self.time_limit_ms = time_limit_ms
        self.use_rule_eval = use_rule_eval

    # Quiet position filter thresholds
    SCORE_CLIP = 10000   # Skip positions with |score| > this (won/lost)
    SKIP_CHECK = True    # Skip positions where side to move is in check

    def play_game(self, initial_state: GameState,
                  max_moves: int = 512) -> List[Tuple]:
        """Play one self-play game, collecting training positions.

        Applies quiet position filtering:
        - Skip positions where the side to move is in check
        - Skip positions with extreme evaluations (|score| > 10000)

        Returns:
            List of (white_features, black_features, side_to_move, score, result)
            for each position in the game.
        """
        positions = []
        state = initial_state

        for _ in range(max_moves):
            if state.is_terminal():
                break

            # Get evaluation score and pick move
            moves = state.legal_moves()
            if not moves:
                break

            if self.evaluator and random.random() > self.random_play_prob:
                best_move, score = self.evaluator.search(state, self.search_depth)
            else:
                best_move = random.choice(moves)
                score = 0

            # Quiet position filtering: only collect "quiet" positions
            is_quiet = True
            if self.SKIP_CHECK and state.is_check():
                is_quiet = False
            if abs(score) > self.SCORE_CLIP:
                is_quiet = False

            if is_quiet:
                wf = self.feature_set.active_features(state, 0)
                bf = self.feature_set.active_features(state, 1)
                stm = state.side_to_move()
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
        elif self.use_rule_eval and self.game_name:
            # Parallel bootstrap with RuleBasedEvaluator
            workers = num_workers or min(os.cpu_count() or 4, 8)
            self._generate_parallel_rule_eval(
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
            with _managed_pool(workers) as pool:
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

    def _generate_parallel_rule_eval(self, output_path: str,
                                       num_games: int, max_moves: int,
                                       workers: int):
        """Multi-process generation with RuleBasedEvaluator (bootstrap)."""
        print(f"Using {workers} worker processes (Rule-Based AI)")
        chunks = _split_work(num_games, workers)

        with open(output_path, "wb") as f:
            with _managed_pool(workers) as pool:
                futures = [
                    pool.submit(
                        _worker_play_games_rule_eval,
                        self.game_name,
                        self.search_depth, self.time_limit_ms,
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
                with _managed_pool(workers) as pool:
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
