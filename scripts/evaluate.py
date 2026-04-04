"""Evaluate model strength by playing model vs model matches."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.halfkp import chess_features, minichess_features
from src.features.halfkp_shogi import shogi_features, minishogi_features
from src.search.alphabeta import AlphaBetaSearch
from src.search.evaluator import NNUEEvaluator, MaterialEvaluator
from src.utils.config import create_game

FEATURE_SETS = {
    "chess": chess_features,
    "minichess": minichess_features,
    "shogi": shogi_features,
    "minishogi": minishogi_features,
}


def play_match(game_name, eval1, eval2, num_games=100, depth=4, max_moves=200):
    """Play a match between two evaluators."""
    wins = [0, 0]
    draws = 0

    for game_num in range(num_games):
        state = create_game(game_name)
        searchers = [
            AlphaBetaSearch(eval1, max_depth=depth, time_limit_ms=2000),
            AlphaBetaSearch(eval2, max_depth=depth, time_limit_ms=2000),
        ]

        # Alternate colors each game
        if game_num % 2 == 1:
            searchers = [searchers[1], searchers[0]]

        for move_num in range(max_moves):
            if state.is_terminal():
                break
            player = state.side_to_move()
            move, _ = searchers[player].search(state)
            if move is None:
                break
            state = state.make_move(move)

        result = state.result()
        if result is None:
            draws += 1
        elif result == 1.0:
            winner = state.side_to_move()
            if game_num % 2 == 1:
                winner = 1 - winner
            wins[winner] += 1
        elif result == 0.0:
            loser = state.side_to_move()
            winner = 1 - loser
            if game_num % 2 == 1:
                winner = 1 - winner
            wins[winner] += 1
        else:
            draws += 1

        if (game_num + 1) % 10 == 0:
            print(f"Game {game_num + 1}: Model1 {wins[0]} - Model2 {wins[1]} - Draws {draws}")

    return wins[0], wins[1], draws


def main():
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument("--game", required=True, choices=FEATURE_SETS.keys())
    parser.add_argument("--model1", required=True, help="First model .npz")
    parser.add_argument("--model2", default=None, help="Second model .npz (or 'material')")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--depth", type=int, default=4)
    args = parser.parse_args()

    fs = FEATURE_SETS[args.game]()

    eval1 = NNUEEvaluator.from_numpy(args.model1, fs)

    if args.model2 and args.model2 != "material":
        eval2 = NNUEEvaluator.from_numpy(args.model2, fs)
    else:
        eval2 = MaterialEvaluator()

    w1, w2, d = play_match(args.game, eval1, eval2, args.games, args.depth)
    print(f"\nFinal: Model1 {w1} - Model2 {w2} - Draws {d}")
    print(f"Model1 win rate: {w1 / max(w1 + w2 + d, 1) * 100:.1f}%")


if __name__ == "__main__":
    main()
