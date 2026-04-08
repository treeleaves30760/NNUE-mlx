"""Iterative self-play training pipeline.

Runs a loop of: generate data -> train model -> evaluate -> repeat.
Each iteration produces a stronger model that generates better training data.

Models are saved to models/<YYYY-MM-DD-HH-MM-SS>/ directories.
Pipeline state is saved for resume support.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline_config import FEATURE_SETS, GAME_DEFAULTS
from pipeline_core import IterativePipeline


def main():
    parser = argparse.ArgumentParser(
        description="Iterative self-play training pipeline")
    parser.add_argument("--game", required=True, choices=FEATURE_SETS.keys())
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of iterations")
    parser.add_argument("--bootstrap-games", type=int, default=None,
                        help="Games for bootstrap iteration (override default)")
    parser.add_argument("--games-per-iter", type=int, default=None,
                        help="Games per non-bootstrap iteration (override)")
    parser.add_argument("--epochs-bootstrap", type=int, default=None)
    parser.add_argument("--epochs-per-iter", type=int, default=None)
    parser.add_argument("--eval-games", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (0=auto)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda", type=float, default=None, dest="lambda_")
    parser.add_argument("--lambda-end", type=float, default=None, dest="lambda_end",
                        help="Lambda end value (decays from --lambda to this)")
    parser.add_argument("--lr-gamma", type=float, default=None, dest="lr_gamma",
                        help="LR exponential decay gamma (default 0.992)")
    parser.add_argument("--data-window", type=int, default=None,
                        help="Rolling window of iterations to keep data from")
    parser.add_argument("--bootstrap-depth", type=int, default=None,
                        dest="bootstrap_depth",
                        help="Search depth for bootstrap games")
    parser.add_argument("--early-stop-patience", type=int, default=None,
                        dest="early_stop_patience",
                        help="Epochs without val improvement before stopping")
    parser.add_argument("--eval-depth", type=int, default=None,
                        help="Search depth for evaluation games")
    parser.add_argument("--eval-time-limit", type=int, default=None,
                        dest="eval_time_limit_ms",
                        help="Time limit (ms) per move for evaluation games")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous pipeline state")
    parser.add_argument("--bootstrap-data", default=None,
                        help="Existing .bin data file to use for bootstrap "
                             "(skip random game generation)")
    parser.add_argument("--bootstrap-model", default=None,
                        help="Existing .npz model to warm-start from "
                             "(skip bootstrap training entirely)")
    args = parser.parse_args()

    # Build config from defaults + overrides
    config = dict(GAME_DEFAULTS.get(args.game, GAME_DEFAULTS["chess"]))
    for key in ["bootstrap_games", "bootstrap_depth", "games_per_iter",
                "epochs_bootstrap", "epochs_per_iter",
                "early_stop_patience", "eval_games", "workers",
                "batch_size", "lr", "lambda_", "lambda_end", "lr_gamma",
                "data_window", "eval_depth", "eval_time_limit_ms"]:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    print(f"Game: {args.game}")
    print(f"Iterations: {args.iterations}")
    print(f"Config: {json.dumps(config, indent=2)}")

    pipeline = IterativePipeline(
        game=args.game,
        num_iterations=args.iterations,
        config=config,
        bootstrap_data=args.bootstrap_data,
        bootstrap_model=args.bootstrap_model,
    )

    if args.resume:
        print("Resume mode enabled")

    pipeline.run()


if __name__ == "__main__":
    main()
