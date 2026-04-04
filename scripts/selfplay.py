"""Generate training data through self-play."""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.halfkp import chess_features, minichess_features
from src.features.halfkp_shogi import shogi_features, minishogi_features
from src.training.selfplay import SelfPlayEngine
from src.utils.config import create_game

FEATURE_SETS = {
    "chess": chess_features,
    "minichess": minichess_features,
    "shogi": shogi_features,
    "minishogi": minishogi_features,
}


def main():
    parser = argparse.ArgumentParser(description="Generate self-play training data")
    parser.add_argument("--game", required=True, choices=FEATURE_SETS.keys())
    parser.add_argument("--games", type=int, default=1000, help="Number of games")
    parser.add_argument("--depth", type=int, default=4, help="Search depth for evaluation")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--model", default=None, help="NNUE model .npz for evaluation")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0=auto, 1=single-process)")
    args = parser.parse_args()

    fs = FEATURE_SETS[args.game]()

    evaluator = None
    if args.model:
        from src.search.evaluator import NNUEEvaluator
        from src.search.alphabeta import AlphaBetaSearch
        nnue_eval = NNUEEvaluator.from_numpy(args.model, fs)
        evaluator = AlphaBetaSearch(
            nnue_eval, max_depth=args.depth, time_limit_ms=2000,
        )

    engine = SelfPlayEngine(
        feature_set=fs,
        evaluator=evaluator,
        search_depth=args.depth,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"{args.game}_{timestamp}.bin"

    engine.generate_data(
        initial_state_fn=lambda: create_game(args.game),
        output_path=str(output_path),
        num_games=args.games,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
