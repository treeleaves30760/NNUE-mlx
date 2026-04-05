"""NNUE training loss function.

Blends evaluation accuracy with game-result prediction using sigmoid scaling.
"""

import mlx.core as mx

# Scaling factor to convert centipawns to win probability.
# The model output is multiplied by OUTPUT_SCALE before sigmoid to allow
# the small-magnitude NNUE output (~1-50) to span the full [0, 1] range.
EVAL_SCALE = 410.0
OUTPUT_SCALE = 32.0  # Model output × OUTPUT_SCALE → centipawn-like range


def nnue_loss(predicted: mx.array, score: mx.array,
              result: mx.array, lambda_: float = 0.5) -> mx.array:
    """Blended loss between search score fitting and game result fitting.

    Both the model output and search score are passed through sigmoid to
    map into [0, 1] win-probability space. Model output is scaled up by
    OUTPUT_SCALE to compensate for the small magnitude of NNUE outputs.

    Args:
        predicted: Model output, shape (batch, 1).
        score: Search evaluation in centipawns, shape (batch,).
        result: Game result (0.0=loss, 0.5=draw, 1.0=win), shape (batch,).
        lambda_: Interpolation weight (1.0=pure eval, 0.0=pure result).

    Returns:
        Scalar loss value.
    """
    p = mx.sigmoid(mx.squeeze(predicted, axis=-1) * OUTPUT_SCALE / EVAL_SCALE)
    t_eval = mx.sigmoid(score / EVAL_SCALE)
    target = lambda_ * t_eval + (1.0 - lambda_) * result
    return mx.mean((p - target) ** 2)
