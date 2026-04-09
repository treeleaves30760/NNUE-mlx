"""Pipeline configuration: feature set registry and per-game hyperparameters."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.halfkp import chess_features, minichess_features
from src.features.halfkp_shogi import shogi_features, minishogi_features

FEATURE_SETS = {
    "chess": chess_features,
    "minichess": minichess_features,
    "shogi": shogi_features,
    "minishogi": minishogi_features,
}

# Default configuration per game
GAME_DEFAULTS = {
    "minichess": {
        "bootstrap_games": 20000,
        "bootstrap_depth": 4,
        "games_per_iter": 8000,
        "epochs_bootstrap": 80,
        "epochs_per_iter": 80,
        "early_stop_patience": 15,
        "eval_games": 50,
        "eval_depth": 6,
        "eval_time_limit_ms": 2000,
        "max_depth_schedule": [3, 4, 4, 5, 5, 6],
        "time_limit_schedule": [300, 500, 500, 800, 1000, 1500],
        "workers": 8,
        "batch_size": 16384,
        "lr": 1e-3,
        "l1_size": 128,
        "lambda_": 1.0,
        "lambda_end": 0.70,
        "lr_gamma": 0.992,
        "data_window": 5,
        "random_opening_plies": 6,
    },
    "chess": {
        "bootstrap_games": 30000,
        "bootstrap_depth": 4,
        "games_per_iter": 1500,
        "epochs_bootstrap": 80,
        "epochs_per_iter": 60,
        "early_stop_patience": 12,
        "eval_games": 50,
        "eval_depth": 5,
        "eval_time_limit_ms": 3000,
        # Progressive depth: deeper search as model gets stronger.
        # Adjudication keeps games short despite higher depth.
        "max_depth_schedule": [4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8],
        "time_limit_schedule": [500, 500, 500, 600, 600, 800, 800, 800, 1000, 1000, 1200, 1200, 1500, 1500, 1800, 1800, 2000, 2000, 2500, 2500],
        "workers": 8,
        "batch_size": 16384,
        "lr": 1e-3,
        "accumulator_size": 256,
        "l1_size": 128,
        # Favor game outcomes over self-referential scores
        "lambda_": 0.75,
        "lambda_end": 0.50,
        "lr_gamma": 0.992,
        "data_window": 5,
        "random_opening_plies": 8,
        "max_moves": 250,
        "eval_max_moves": 500,
        # Phase out bootstrap after iter 3
        "bootstrap_window": 3,
    },
    "shogi": {
        "bootstrap_games": 20000,
        "games_per_iter": 2000,
        "epochs_bootstrap": 200,
        "epochs_per_iter": 150,
        "eval_games": 30,
        "max_depth_schedule": [3, 3, 4, 4, 5, 5],
        "time_limit_schedule": [1000, 1000, 1500, 2000, 3000, 5000],
        "workers": 0,
        "batch_size": 16384,
        "lr": 8.75e-4,
        "lambda_": 1.0,
        "lambda_end": 0.75,
        "lr_gamma": 0.992,
        "data_window": 3,
    },
    "minishogi": {
        "bootstrap_games": 30000,
        "games_per_iter": 3000,
        "epochs_bootstrap": 200,
        "epochs_per_iter": 150,
        "eval_games": 50,
        "max_depth_schedule": [3, 4, 4, 5, 5, 6],
        "time_limit_schedule": [500, 1000, 1000, 1500, 2000, 3000],
        "workers": 0,
        "batch_size": 16384,
        "lr": 8.75e-4,
        "lambda_": 1.0,
        "lambda_end": 0.75,
        "lr_gamma": 0.992,
        "data_window": 3,
    },
}
