"""Incremental accumulator for efficient NNUE evaluation during search.

During alpha-beta search, positions change by one move at a time. Instead of
recomputing the full accumulator from scratch, we add/subtract weight rows
for changed features. This runs on CPU with numpy for minimal latency.
"""

import numpy as np
from typing import List, Tuple


class IncrementalAccumulator:
    """Maintains incrementally updatable feature accumulators for search."""

    def __init__(self, ft_weight: np.ndarray, ft_bias: np.ndarray,
                 l1_weight: np.ndarray, l1_bias: np.ndarray,
                 l2_weight: np.ndarray, l2_bias: np.ndarray,
                 out_weight: np.ndarray, out_bias: np.ndarray):
        """Initialize with numpy weights extracted from trained model.

        Args:
            ft_weight: (num_features, accumulator_size) feature transformer weights.
            ft_bias: (accumulator_size,) feature transformer bias.
            l1_weight: (l1_size, accumulator_size * 2) first hidden layer weights.
            l1_bias: (l1_size,) first hidden layer bias.
            l2_weight: (l2_size, l1_size) second hidden layer weights.
            l2_bias: (l2_size,) second hidden layer bias.
            out_weight: (1, l2_size) output layer weights.
            out_bias: (1,) output layer bias.
        """
        self.ft_weight = ft_weight
        self.ft_bias = ft_bias
        self.l1_weight = l1_weight
        self.l1_bias = l1_bias
        self.l2_weight = l2_weight
        self.l2_bias = l2_bias
        self.out_weight = out_weight.flatten()
        self.out_bias = float(out_bias.flatten()[0])

        # Current accumulator state for each perspective
        acc_size = len(ft_bias)
        self.white_acc = np.copy(ft_bias).astype(np.float32)
        self.black_acc = np.copy(ft_bias).astype(np.float32)

        # Pre-allocated stack for save/restore during search tree traversal
        max_depth = 64
        self._stack_w = np.zeros((max_depth, acc_size), dtype=np.float32)
        self._stack_b = np.zeros((max_depth, acc_size), dtype=np.float32)
        self._sp = 0

    def refresh(self, white_features: List[int], black_features: List[int]):
        """Full recomputation from feature lists.

        Called at the root position and after king moves (which change all
        feature indices since they include king square).
        """
        np.copyto(self.white_acc, self.ft_bias)
        for idx in white_features:
            self.white_acc += self.ft_weight[idx]

        np.copyto(self.black_acc, self.ft_bias)
        for idx in black_features:
            self.black_acc += self.ft_weight[idx]

    def refresh_perspective(self, perspective: int, features: List[int]):
        """Full recompute for a single perspective (after king moves)."""
        acc = self.white_acc if perspective == 0 else self.black_acc
        np.copyto(acc, self.ft_bias)
        for idx in features:
            acc += self.ft_weight[idx]

    def update(self, perspective: int,
               added: List[int], removed: List[int]):
        """Incremental update: add new features, remove old ones.

        Args:
            perspective: 0=white, 1=black.
            added: Feature indices that became active.
            removed: Feature indices that became inactive.
        """
        acc = self.white_acc if perspective == 0 else self.black_acc
        for idx in removed:
            acc -= self.ft_weight[idx]
        for idx in added:
            acc += self.ft_weight[idx]

    def push(self):
        """Save current accumulator state before making a move in search."""
        sp = self._sp
        self._stack_w[sp] = self.white_acc
        self._stack_b[sp] = self.black_acc
        self._sp = sp + 1

    def pop(self):
        """Restore accumulator state after unmaking a move in search."""
        self._sp -= 1
        sp = self._sp
        np.copyto(self.white_acc, self._stack_w[sp])
        np.copyto(self.black_acc, self._stack_b[sp])

    def evaluate(self, side_to_move: int) -> float:
        """Run forward pass through the small hidden layers.

        This is extremely fast on CPU (~microseconds) because the layers
        are tiny: 512 -> 32 -> 32 -> 1.

        Args:
            side_to_move: 0=white, 1=black.

        Returns:
            Evaluation score (positive = good for side to move).
        """
        # ClippedReLU on accumulators (feature transformer)
        w = np.clip(self.white_acc, 0.0, 1.0)
        b = np.clip(self.black_acc, 0.0, 1.0)

        # Concat with side-to-move ordering
        if side_to_move == 0:
            x = np.concatenate([w, b])
        else:
            x = np.concatenate([b, w])

        # L1 with SCReLU: clamp(x, 0, 1)^2
        x = np.clip(self.l1_weight @ x + self.l1_bias, 0.0, 1.0)
        x = x * x
        # L2 with SCReLU
        x = np.clip(self.l2_weight @ x + self.l2_bias, 0.0, 1.0)
        x = x * x
        # Output
        return float(self.out_weight @ x + self.out_bias)

    @classmethod
    def from_model(cls, model) -> "IncrementalAccumulator":
        """Create an accumulator from a trained MLX NNUEModel.

        Args:
            model: An NNUEModel instance (MLX).
        """
        from mlx.utils import tree_flatten
        state = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
        return cls(
            ft_weight=state["feature_table.weight"],
            ft_bias=state["ft_bias"],
            l1_weight=state["l1.weight"],
            l1_bias=state["l1.bias"],
            l2_weight=state["l2.weight"],
            l2_bias=state["l2.bias"],
            out_weight=state["output.weight"],
            out_bias=state["output.bias"],
        )
