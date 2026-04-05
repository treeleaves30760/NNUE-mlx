"""NNUE evaluator wrapper for search integration."""

import numpy as np
from typing import List, Optional

from src.features.base import FeatureSet
from src.games.base import GameState, Move
from src.model.accumulator import IncrementalAccumulator

try:
    from src.accel import AcceleratedAccumulator as _AccelAccum
    _HAS_ACCEL = True
except ImportError:
    _HAS_ACCEL = False


class NNUEEvaluator:
    """Wraps the incremental accumulator for use in alpha-beta search."""

    def __init__(self, accumulator: IncrementalAccumulator,
                 feature_set: FeatureSet):
        self.accumulator = accumulator
        self.feature_set = feature_set

    def set_position(self, state: GameState):
        """Initialize accumulator for a root position (full recompute)."""
        wf = self.feature_set.active_features(state, 0)
        bf = self.feature_set.active_features(state, 1)
        self.accumulator.refresh(wf, bf)

    def evaluate(self, state: GameState) -> float:
        """Evaluate current position using the accumulator.

        Returns score from side-to-move's perspective.
        """
        return self.accumulator.evaluate(state.side_to_move())

    def push_move(self, state_before: GameState, move: Move,
                  state_after: GameState):
        """Update accumulator for a new move (incremental or full refresh)."""
        self.accumulator.push()

        for perspective in [0, 1]:
            delta = self.feature_set.feature_delta(
                state_before, move, state_after, perspective
            )
            if delta is None:
                # King moved, need full refresh for this perspective
                features = self.feature_set.active_features(state_after, perspective)
                self.accumulator.refresh_perspective(perspective, features)
            else:
                added, removed = delta
                self.accumulator.update(perspective, added, removed)

    def push_move_refresh(self, state_after: GameState):
        """Full accumulator refresh for use with make_move_inplace."""
        self.accumulator.push()
        wf = self.feature_set.active_features(state_after, 0)
        bf = self.feature_set.active_features(state_after, 1)
        self.accumulator.refresh(wf, bf)

    def pop_move(self):
        """Restore accumulator state after unmaking a move."""
        self.accumulator.pop()

    @classmethod
    def from_numpy(cls, npz_path: str, feature_set: FeatureSet) -> "NNUEEvaluator":
        """Create evaluator from exported numpy weights.

        Uses the accelerated C extension (NEON SIMD + Accelerate) when
        available, falling back to the pure-numpy IncrementalAccumulator.
        """
        data = np.load(npz_path)
        AccumClass = _AccelAccum if _HAS_ACCEL else IncrementalAccumulator
        accumulator = AccumClass(
            ft_weight=data["feature_table.weight"],
            ft_bias=data["ft_bias"],
            l1_weight=data["l1.weight"],
            l1_bias=data["l1.bias"],
            l2_weight=data["l2.weight"],
            l2_bias=data["l2.bias"],
            out_weight=data["output.weight"],
            out_bias=data["output.bias"],
        )
        return cls(accumulator, feature_set)

    @classmethod
    def from_weights_dict(cls, weights: dict, feature_set: FeatureSet) -> "NNUEEvaluator":
        """Create evaluator from a pre-loaded dict of numpy arrays.

        Keys must match: feature_table.weight, ft_bias, l1.weight, l1.bias,
        l2.weight, l2.bias, output.weight, output.bias.
        """
        AccumClass = _AccelAccum if _HAS_ACCEL else IncrementalAccumulator
        accumulator = AccumClass(
            ft_weight=weights["feature_table.weight"],
            ft_bias=weights["ft_bias"],
            l1_weight=weights["l1.weight"],
            l1_bias=weights["l1.bias"],
            l2_weight=weights["l2.weight"],
            l2_bias=weights["l2.bias"],
            out_weight=weights["output.weight"],
            out_bias=weights["output.bias"],
        )
        return cls(accumulator, feature_set)


class MaterialEvaluator:
    """Simple material-counting evaluator for bootstrapping (no NNUE needed)."""

    PIECE_VALUES = {
        # Chess piece codes
        1: 100,   # Pawn
        2: 320,   # Knight
        3: 330,   # Bishop
        4: 500,   # Rook
        5: 900,   # Queen
    }

    def evaluate(self, state: GameState) -> float:
        """Evaluate by material balance from side-to-move's perspective."""
        board = state.board_array() if hasattr(state, "board_array") else None
        if board is None:
            return 0.0

        score = 0.0
        for sq in range(len(board)):
            piece = board[sq]
            if piece == 0:
                continue
            abs_piece = abs(piece)
            value = self.PIECE_VALUES.get(abs_piece, 100)
            if piece > 0:
                score += value  # White piece
            else:
                score -= value  # Black piece

        # Return from side-to-move's perspective
        if state.side_to_move() == 1:
            score = -score
        return score

    def set_position(self, state: GameState):
        pass

    def push_move(self, state_before, move, state_after):
        pass

    def pop_move(self):
        pass
