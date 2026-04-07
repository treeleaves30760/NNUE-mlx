"""Activation functions for NNUE: ClippedReLU and SCReLU."""

import mlx.core as mx
import mlx.nn as nn


class ClippedReLU(nn.Module):
    """Clamp activation to [0, 1].

    Used for the feature transformer where simple clamping is needed
    for efficient incremental accumulator updates.
    Maps to [0, 127] in quantized int8 inference.
    """

    def __call__(self, x: mx.array) -> mx.array:
        return mx.clip(x, 0.0, 1.0)


class SCReLU(nn.Module):
    """Squared Clipped ReLU: clamp(x, 0, 1)^2.

    Modern NNUE activation that outperforms ClippedReLU in hidden layers,
    providing ~50% effective capacity increase. Used post-feature-transformer
    in Stockfish-style networks.
    """

    def __call__(self, x: mx.array) -> mx.array:
        return mx.clip(x, 0.0, 1.0) ** 2
