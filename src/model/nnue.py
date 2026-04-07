"""NNUE (Efficiently Updatable Neural Network) model for board game evaluation.

Architecture:
    HalfKP features -> Feature Transformer (Embedding + sum -> 256)
    -> ClippedReLU -> Concat perspectives (512)
    -> Linear(512, 32) -> SCReLU
    -> Linear(32, 32)  -> SCReLU
    -> Linear(32, 1)   -> evaluation score

Uses Apple MLX for native Apple Silicon training with unified memory.
SCReLU (Squared Clipped ReLU) provides ~50% effective capacity increase
over ClippedReLU in hidden layers.
"""

import mlx.core as mx
import mlx.nn as nn

from src.model.clipped_relu import ClippedReLU, SCReLU


class NNUEModel(nn.Module):
    """NNUE evaluation network with dual-perspective feature transformer."""

    def __init__(self, num_features: int, accumulator_size: int = 256,
                 l1_size: int = 32, l2_size: int = 32):
        """
        Args:
            num_features: Total HalfKP feature space size (e.g. 40960 for chess).
            accumulator_size: Feature transformer output dimension.
            l1_size: First hidden layer size.
            l2_size: Second hidden layer size.
        """
        super().__init__()
        self.num_features = num_features
        self.accumulator_size = accumulator_size

        # Feature transformer: shared between white and black perspectives.
        self.feature_table = nn.Embedding(num_features, accumulator_size)
        self.ft_bias = mx.zeros((accumulator_size,))

        # ClippedReLU for feature transformer (needed for incremental accumulator)
        self.clipped_relu = ClippedReLU()
        # SCReLU for hidden layers (better capacity than ClippedReLU)
        self.screlu = SCReLU()
        self.l1 = nn.Linear(accumulator_size * 2, l1_size)
        self.l2 = nn.Linear(l1_size, l2_size)
        self.output = nn.Linear(l2_size, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        self.feature_table.weight = mx.random.normal(
            shape=self.feature_table.weight.shape) * 0.01

    def _accumulate(self, features: mx.array,
                    mask: mx.array) -> mx.array:
        """Compute accumulator output from sparse feature indices.

        Args:
            features: (batch, max_active) int32 feature indices.
            mask: (batch, max_active) float32, 1.0 for real features, 0.0 for padding.

        Returns:
            (batch, accumulator_size) accumulated feature vectors.
        """
        # Gather embedding rows: (batch, max_active, accumulator_size)
        gathered = self.feature_table(features)
        # Mask out padding and sum: (batch, accumulator_size)
        return (gathered * mx.expand_dims(mask, axis=-1)).sum(axis=1) + self.ft_bias

    def __call__(self, white_features: mx.array, black_features: mx.array,
                 white_mask: mx.array, black_mask: mx.array,
                 side_to_move: mx.array) -> mx.array:
        """Forward pass through the NNUE network.

        Args:
            white_features: (batch, max_active) int32 indices for white perspective.
            black_features: (batch, max_active) int32 indices for black perspective.
            white_mask: (batch, max_active) float32 mask for white features.
            black_mask: (batch, max_active) float32 mask for black features.
            side_to_move: (batch,) int, 0=white to move, 1=black to move.

        Returns:
            (batch, 1) evaluation score.
        """
        # Compute accumulators for both perspectives
        w_acc = self.clipped_relu(self._accumulate(white_features, white_mask))
        b_acc = self.clipped_relu(self._accumulate(black_features, black_mask))

        # Order by side-to-move: [stm_perspective, opponent_perspective]
        stm = mx.expand_dims(side_to_move.astype(mx.float32), axis=1)
        first = w_acc * (1.0 - stm) + b_acc * stm
        second = b_acc * (1.0 - stm) + w_acc * stm
        x = mx.concatenate([first, second], axis=1)

        # Hidden layers with SCReLU
        x = self.screlu(self.l1(x))
        x = self.screlu(self.l2(x))
        return self.output(x)
