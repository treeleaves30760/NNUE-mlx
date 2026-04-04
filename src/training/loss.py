"""NNUE training loss function.

Blends evaluation accuracy with game-result prediction using sigmoid scaling.
"""

import mlx.core as mx

# Standard scaling factor to convert centipawns to win probability
EVAL_SCALE = 410.0


def nnue_loss(predicted: mx.array, score: mx.array,
              result: mx.array, lambda_: float = 0.5) -> mx.array:
    """Blended loss between search score fitting and game result fitting.

    Both the model output and search score are passed through sigmoid to
    map centipawn values into [0, 1] win-probability space.

    Args:
        predicted: Model output, shape (batch, 1).
        score: Search evaluation in centipawns, shape (batch,).
        result: Game result (0.0=loss, 0.5=draw, 1.0=win), shape (batch,).
        lambda_: Interpolation weight (1.0=pure eval, 0.0=pure result).

    Returns:
        Scalar loss value.
    """
    p = mx.sigmoid(mx.squeeze(predicted, axis=-1) / EVAL_SCALE)
    t_eval = mx.sigmoid(score / EVAL_SCALE)
    target = lambda_ * t_eval + (1.0 - lambda_) * result
    return mx.mean((p - target) ** 2)
