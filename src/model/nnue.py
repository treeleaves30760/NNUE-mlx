"""NNUE (Efficiently Updatable Neural Network) model for board game evaluation.

Architecture:
    HalfKP features -> Feature Transformer (Embedding + sum -> accumulator_size)
    -> ClippedReLU -> Concat perspectives (accumulator_size * 2)
    -> Linear(accumulator_size * 2, l1_size) -> SCReLU
    -> Linear(l1_size, l2_size)              -> SCReLU
    -> Linear(l2_size, num_buckets)          -> evaluation score(s)

Uses Apple MLX for native Apple Silicon training with unified memory.
SCReLU (Squared Clipped ReLU) provides ~50% effective capacity increase
over ClippedReLU in hidden layers.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from src.model.clipped_relu import ClippedReLU, SCReLU

DEFAULT_NUM_BUCKETS = 8


class NNUEModel(nn.Module):
    """NNUE evaluation network with dual-perspective feature transformer."""

    def __init__(self, num_features: int, accumulator_size: int = 256,
                 l1_size: int = 128, l2_size: int = 32,
                 num_output_buckets: int = 1, use_wdl_head: bool = False):
        """
        Args:
            num_features: Total HalfKP feature space size (e.g. 40960 for chess).
            accumulator_size: Feature transformer output dimension.
            l1_size: First hidden layer size.
            l2_size: Second hidden layer size.
            num_output_buckets: Number of output heads selected by piece count.
            use_wdl_head: If True, add a second WDL logit output head.
        """
        super().__init__()
        self.num_features = num_features
        self.accumulator_size = accumulator_size
        self.num_output_buckets = num_output_buckets
        self.use_wdl_head = use_wdl_head

        # Feature transformer: shared between white and black perspectives.
        self.feature_table = nn.Embedding(num_features, accumulator_size)
        self.ft_bias = mx.zeros((accumulator_size,))

        # ClippedReLU for feature transformer (needed for incremental accumulator)
        self.clipped_relu = ClippedReLU()
        # SCReLU for hidden layers (better capacity than ClippedReLU)
        self.screlu = SCReLU()
        self.l1 = nn.Linear(accumulator_size * 2, l1_size)
        self.l2 = nn.Linear(l1_size, l2_size)
        self.output = nn.Linear(l2_size, num_output_buckets)

        if use_wdl_head:
            self.wdl_output = nn.Linear(l2_size, num_output_buckets)

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
                 side_to_move: mx.array,
                 bucket_idx: Optional[mx.array] = None):
        """Forward pass through the NNUE network.

        Args:
            white_features: (batch, max_active) int32 indices for white perspective.
            black_features: (batch, max_active) int32 indices for black perspective.
            white_mask: (batch, max_active) float32 mask for white features.
            black_mask: (batch, max_active) float32 mask for black features.
            side_to_move: (batch,) int, 0=white to move, 1=black to move.
            bucket_idx: (batch,) int, output bucket per sample. None uses bucket 0.

        Returns:
            (batch, 1) evaluation score, or tuple((batch, 1), (batch, 1)) when
            use_wdl_head=True.
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

        # Output: (batch, num_output_buckets)
        out = self.output(x)

        if self.num_output_buckets == 1 and bucket_idx is None:
            score = out
        elif bucket_idx is None:
            score = out[:, :1]
        else:
            score = mx.take_along_axis(
                out,
                mx.expand_dims(bucket_idx.astype(mx.int32), axis=1),
                axis=1,
            )

        if not self.use_wdl_head:
            return score

        wdl_out = self.wdl_output(x)
        if self.num_output_buckets == 1 and bucket_idx is None:
            wdl = wdl_out
        elif bucket_idx is None:
            wdl = wdl_out[:, :1]
        else:
            wdl = mx.take_along_axis(
                wdl_out,
                mx.expand_dims(bucket_idx.astype(mx.int32), axis=1),
                axis=1,
            )

        return score, wdl
