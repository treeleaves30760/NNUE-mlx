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
from src.training.openings import sample_opening
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
                 use_rule_eval: bool = False,
                 random_opening_plies: int = 0):
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
            random_opening_plies: Number of random moves at the start of each
                game to diversify openings (0 = disabled).
        """
        self.feature_set = feature_set
        self.evaluator = evaluator
        self.search_depth = search_depth
        self.random_play_prob = random_play_prob
        self.game_name = game_name
        self.model_path = model_path
        self.time_limit_ms = time_limit_ms
        self.use_rule_eval = use_rule_eval
        self.random_opening_plies = random_opening_plies

    # Quiet position filter thresholds
    SCORE_CLIP = 10000   # Skip positions with |score| > this (won/lost)
    SKIP_CHECK = True    # Skip positions where side to move is in check

    # Adjudication thresholds
    DRAW_SCORE_THRESHOLD = 15    # |score| below this = "drawish"
    DRAW_PLY_COUNT = 12          # consecutive drawish plies to adjudicate draw
    WIN_SCORE_THRESHOLD = 1000   # |score| above this = "winning"
    WIN_PLY_COUNT = 4            # consecutive winning plies to adjudicate win

    # Temperature for move selection (exploration)
    TEMP_PLIES = 20              # apply temperature for first N plies
    TEMP_START = 200.0           # initial temperature (in centipawns)
    TEMP_END = 0.0               # final temperature (greedy)
    TEMP_TOP_N = 5               # consider top N moves

    def play_game(self, initial_state: GameState,
                  max_moves: int = 512) -> List[Tuple]:
        """Play one self-play game, collecting training positions.

        Features:
        - Opening book or random opening plies for diversity
        - Temperature-based move selection for exploration (first N plies)
        - Adjudication rules: draw (low scores), win (high scores)
        - Quiet position filtering

        Returns:
            List of (white_features, black_features, side_to_move, score, result)
            for each position in the game.
        """
        import math

        positions = []
        state = initial_state

        # Opening book: use FEN opening if available, else random plies
        if self.game_name:
            fen = sample_opening(self.game_name)
            if fen:
                try:
                    from src.games.chess_pc import from_fen
                    state = from_fen(fen)
                except Exception:
                    pass  # fallback to initial_state

        # Random opening plies as fallback (for games without opening book)
        if self.random_opening_plies > 0 and state is initial_state:
            n_plies = random.randint(
                max(2, self.random_opening_plies - 2),
                self.random_opening_plies + 2,
            )
            for _ in range(n_plies):
                if state.is_terminal():
                    break
                moves = state.legal_moves()
                if not moves:
                    break
                state = state.make_move(random.choice(moves))

        # Adjudication state
        draw_streak = 0
        win_streak = 0
        win_side = 0
        ply = 0
        adjudicated_result = None  # None=game continues, 0=draw, 1=white, -1=black

        for _ in range(max_moves):
            if state.is_terminal():
                break

            moves = state.legal_moves()
            if not moves:
                break

            # Get evaluation and pick move
            if self.evaluator:
                # Temperature-based move selection for early plies
                if ply < self.TEMP_PLIES and hasattr(self.evaluator, 'search_top_n'):
                    t = ply / max(self.TEMP_PLIES, 1)
                    temp = self.TEMP_START * (1.0 - t) + self.TEMP_END * t
                    if temp > 1.0:
                        top_moves = self.evaluator.search_top_n(
                            state, n=min(self.TEMP_TOP_N, len(moves)))
                        if top_moves:
                            # Softmax over scores with temperature
                            scores_raw = [s for _, s in top_moves]
                            max_s = max(scores_raw)
                            weights = [math.exp((s - max_s) / temp) for s in scores_raw]
                            total = sum(weights)
                            weights = [w / total for w in weights]
                            chosen = random.choices(top_moves, weights=weights, k=1)[0]
                            best_move, score = chosen
                        else:
                            best_move, score = self.evaluator.search(state, self.search_depth)
                    else:
                        best_move, score = self.evaluator.search(state, self.search_depth)
                else:
                    best_move, score = self.evaluator.search(state, self.search_depth)
            else:
                best_move = random.choice(moves)
                score = 0

            # --- Adjudication checks ---
            abs_score = abs(score)
            if abs_score < self.DRAW_SCORE_THRESHOLD:
                draw_streak += 1
            else:
                draw_streak = 0

            stm = state.side_to_move()
            # Score from side-to-move perspective; convert to white perspective
            white_score = score if stm == 0 else -score
            if abs_score > self.WIN_SCORE_THRESHOLD:
                current_win_side = 1 if white_score > 0 else -1
                if current_win_side == win_side:
                    win_streak += 1
                else:
                    win_side = current_win_side
                    win_streak = 1
            else:
                win_streak = 0
                win_side = 0

            # Draw adjudication
            if draw_streak >= self.DRAW_PLY_COUNT and ply >= 40:
                adjudicated_result = 0
            # Win adjudication
            elif win_streak >= self.WIN_PLY_COUNT:
                adjudicated_result = win_side

            # --- Collect training position ---
            is_quiet = True
            if self.SKIP_CHECK and state.is_check():
                is_quiet = False
            if abs_score > self.SCORE_CLIP:
                is_quiet = False

            if is_quiet:
                wf = self.feature_set.active_features(state, 0)
                bf = self.feature_set.active_features(state, 1)
                positions.append((wf, bf, stm, score))

            state = state.make_move(best_move)
            ply += 1

            if adjudicated_result is not None:
                break

        # Determine game result
        if adjudicated_result is not None:
            game_result = adjudicated_result  # 1=white, -1=black, 0=draw
        else:
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
        results = []
        for wf, bf, stm, score in positions:
            # game_result: 1=white wins, -1=black wins, 0=draw
            # For white (stm=0): r = game_result
            # For black (stm=1): r = -game_result
            r = game_result if stm == 0 else -game_result
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
                        self.random_opening_plies,
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
                        self.random_opening_plies,
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
                            self.random_opening_plies,
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
