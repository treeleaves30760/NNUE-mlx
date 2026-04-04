"""Self-play engine for generating NNUE training data."""

import random
from pathlib import Path
from typing import List, Optional, Tuple

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
            evaluator: Optional NNUE evaluator. If None, uses random/material eval.
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
            List of (white_features, black_features, side_to_move, score)
            for each position in the game. Game result is not yet attached.
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
            # Game didn't finish (max moves reached), treat as draw
            game_result = 0
        elif result_val == 1.0:
            game_result = 1  # Side to move at end won
        elif result_val == 0.0:
            game_result = -1  # Side to move at end lost
        else:
            game_result = 0  # Draw

        # Attach result relative to each position's side to move
        final_stm = state.side_to_move()
        results = []
        for wf, bf, stm, score in positions:
            if stm == final_stm:
                r = game_result
            else:
                r = -game_result
            results.append((wf, bf, stm, score, r))

        return results

    def generate_data(self, initial_state_fn, output_path: str,
                      num_games: int = 1000, max_moves: int = 512):
        """Generate training data from multiple self-play games.

        Args:
            initial_state_fn: Callable that returns a fresh initial state.
            output_path: Path to output .bin file.
            num_games: Number of games to play.
            max_moves: Maximum moves per game.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        total_positions = 0

        with open(output_path, "wb") as f:
            for game_num in range(num_games):
                state = initial_state_fn()
                positions = self.play_game(state, max_moves)
                for wf, bf, stm, score, result in positions:
                    write_sample(f, wf, bf, stm, score, result)
                    total_positions += 1

                if (game_num + 1) % 100 == 0:
                    print(f"Game {game_num + 1}/{num_games}, "
                          f"total positions: {total_positions}")

        print(f"Generated {total_positions} positions from {num_games} games")
        print(f"Saved to {output_path}")
