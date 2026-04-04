"""ClippedReLU activation: clamp to [0, 1]."""

import mlx.core as mx
import mlx.nn as nn


class ClippedReLU(nn.Module):
    """Clamp activation to [0, 1].

    Maps to [0, 127] in quantized int8 inference.
    """

    def __call__(self, x: mx.array) -> mx.array:
        return mx.clip(x, 0.0, 1.0)
