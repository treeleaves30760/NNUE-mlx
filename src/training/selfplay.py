"""Self-play engine for generating NNUE training data."""

import io
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from src.features.base import FeatureSet
from src.games.base import GameState
from src.training.data_format import write_sample


class SelfPlayEngine:
    """Generates training data through self-play games.

    Uses either a simple evaluation function (material counting) for
    bootstrapping, or an NNUE evaluator for later iterations.
    """

    def __init__(self, feature_set: FeatureSet, evaluator=None,
                 search_depth: int = 4, random_play_prob: float = 0.1):
        """
        Args:
            feature_set: Feature extractor for the game.
            evaluator: Optional evaluator with a search(state, depth) method.
                       If None, uses random moves for bootstrapping.
            search_depth: Search depth for generating evaluations.
            random_play_prob: Probability of making a random move (for diversity).
        """
        self.feature_set = feature_set
        self.evaluator = evaluator
        self.search_depth = search_depth
        self.random_play_prob = random_play_prob

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

        if num_workers == 1 or self.evaluator is not None:
            # Single-process: evaluator objects are not picklable
            self._generate_single(initial_state_fn, output_path,
                                  num_games, max_moves)
        else:
            workers = num_workers or min(os.cpu_count() or 4, 8)
            self._generate_parallel(initial_state_fn, output_path,
                                    num_games, max_moves, workers)

        elapsed = time.time() - t0
        print(f"Completed in {elapsed:.1f}s "
              f"({num_games / elapsed:.1f} games/s)")

    def _generate_single(self, initial_state_fn, output_path: str,
                          num_games: int, max_moves: int):
        """Single-process generation (used when evaluator is present)."""
        total_positions = 0
        with open(output_path, "wb") as f:
            for game_num in range(num_games):
                state = initial_state_fn()
                data = self._play_game_to_bytes(state, max_moves)
                f.write(data)
                # Count positions: each sample has at least 7 bytes header
                total_positions += data.count(b'') - 1  # rough
                if (game_num + 1) % 100 == 0:
                    print(f"Game {game_num + 1}/{num_games}")

        print(f"Generated data from {num_games} games")
        print(f"Saved to {output_path}")

    def _generate_parallel(self, initial_state_fn, output_path: str,
                            num_games: int, max_moves: int, workers: int):
        """Multi-process generation for bootstrapping (no evaluator)."""
        print(f"Using {workers} worker processes")
        # Split games into chunks for each worker
        chunk_size = max(1, num_games // (workers * 4))
        chunks = []
        remaining = num_games
        while remaining > 0:
            n = min(chunk_size, remaining)
            chunks.append(n)
            remaining -= n

        completed = 0
        with open(output_path, "wb") as f:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        _worker_play_games,
                        initial_state_fn, self.feature_set,
                        n, max_moves, self.random_play_prob,
                    )
                    for n in chunks
                ]
                for future in futures:
                    data = future.result()
                    f.write(data)
                    completed += chunks[futures.index(future)]
                    print(f"Progress: {completed}/{num_games} games")

        print(f"Generated data from {num_games} games")
        print(f"Saved to {output_path}")


def _worker_play_games(
    initial_state_fn: Callable[[], GameState],
    feature_set: FeatureSet,
    num_games: int,
    max_moves: int,
    random_play_prob: float,
) -> bytes:
    """Worker function for multiprocessing. Plays games and returns bytes."""
    engine = SelfPlayEngine(
        feature_set=feature_set,
        evaluator=None,
        random_play_prob=random_play_prob,
    )
    buf = io.BytesIO()
    for _ in range(num_games):
        state = initial_state_fn()
        positions = engine.play_game(state, max_moves)
        for wf, bf, stm, score, result in positions:
            write_sample(buf, wf, bf, stm, score, result)
    return buf.getvalue()
