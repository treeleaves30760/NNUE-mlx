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
        "bootstrap_games": 10000,
        "bootstrap_depth": 3,
        "games_per_iter": 3000,
        "epochs_bootstrap": 100,
        "epochs_per_iter": 150,
        "early_stop_patience": 15,
        "eval_games": 30,
        "eval_depth": 6,
        "eval_time_limit_ms": 3000,
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
    "chess": {
        "bootstrap_games": 2000,
        "bootstrap_depth": 2,
        "games_per_iter": 500,
        "epochs_bootstrap": 100,
        "epochs_per_iter": 200,
        "early_stop_patience": 15,
        "eval_games": 10,
        "eval_depth": 5,
        "eval_time_limit_ms": 3000,
        "max_depth_schedule": [3, 3, 4, 4, 4, 5, 5, 5, 6, 6],
        "time_limit_schedule": [1000, 1000, 1500, 1500, 2000, 2000, 3000, 3000, 5000, 5000],
        "workers": 8,
        "batch_size": 16384,
        "lr": 8.75e-4,
        "lambda_": 1.0,
        "lambda_end": 0.75,
        "lr_gamma": 0.992,
        "data_window": 5,
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
