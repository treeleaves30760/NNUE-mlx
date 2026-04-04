"""HalfKP feature extraction for shogi-family games with hand pieces.

Extends HalfKP with additional features for pieces in hand (captured pieces
that can be dropped). Hand piece features are appended after board features.

Board features: king_sq * num_piece_sq_combos + piece_type * num_squares + piece_sq
Hand features:  board_feature_count + king_sq * hand_feature_combos + hand_index

For full shogi 9x9: board features + hand features
For mini shogi 5x5: board features + hand features
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.features.base import FeatureSet
from src.games.base import GameState, Move

try:
    from src.accel import halfkp_shogi_active_features as _c_shogi_features, _HAS_ACCEL
except ImportError:
    _c_shogi_features = None
    _HAS_ACCEL = False


class HalfKPShogi(FeatureSet):
    """HalfKP feature set for shogi-family games with drops."""

    def __init__(self, num_squares: int, num_board_piece_types: int,
                 num_hand_piece_types: int, max_hand_count: int,
                 board_val_to_type: Optional[Dict[int, int]] = None,
                 king_board_val: int = 8):
        """
        Args:
            num_squares: Board size (81 for shogi, 25 for mini shogi).
            num_board_piece_types: Non-king piece types on board including promoted
                                  (13 for shogi, 9 for mini shogi).
            num_hand_piece_types: Unpromoted piece types that can be in hand
                                 (7 for shogi: P/L/N/S/G/B/R,
                                  5 for mini shogi: P/S/G/B/R).
            max_hand_count: Maximum count of one piece type in hand
                           (18 for shogi pawns, 2 for mini shogi).
        """
        self._num_squares = num_squares
        self._num_board_piece_types = num_board_piece_types
        self._num_hand_piece_types = num_hand_piece_types
        self._max_hand_count = max_hand_count
        self._board_val_to_type = board_val_to_type or {}
        self._king_board_val = king_board_val

        # Board features: king_sq * (piece_types * 2 colors * squares)
        self._piece_sq_combos = num_board_piece_types * 2 * num_squares
        self._board_features = num_squares * self._piece_sq_combos

        # Hand features: king_sq * (hand_piece_types * 2 sides * max_count)
        self._hand_combos = num_hand_piece_types * 2 * max_hand_count
        self._hand_features = num_squares * self._hand_combos

        self._total = self._board_features + self._hand_features

        # Pre-compute bv2type array for C extension
        if board_val_to_type:
            arr_len = max(board_val_to_type.keys()) + 1 if board_val_to_type else 0
            self._bv2type_arr = np.full(arr_len, -1, dtype=np.int8)
            for bv, pt in board_val_to_type.items():
                if bv < arr_len:
                    self._bv2type_arr[bv] = pt
        else:
            self._bv2type_arr = None

    def num_features(self) -> int:
        return self._total

    def max_active_features(self) -> int:
        # Generous upper bound: all pieces on board + hand pieces
        return 40 if self._num_squares == 81 else 12

    def _board_feature_index(self, king_sq: int, piece_type: int,
                             piece_color: int, piece_sq: int,
                             perspective: int) -> int:
        relative_color = 0 if piece_color == perspective else 1
        color_type = relative_color * self._num_board_piece_types + piece_type
        return (king_sq * self._piece_sq_combos
                + color_type * self._num_squares
                + piece_sq)

    def _hand_feature_indices(self, king_sq: int,
                              hand: Dict[int, int],
                              side: int, perspective: int) -> List[int]:
        """Generate feature indices for hand pieces.

        Each hand piece with count N generates N features (one for each count
        level 1..N), encoding "this side has at least k of this piece type".
        """
        relative_color = 0 if side == perspective else 1
        indices = []
        for piece_type, count in hand.items():
            for k in range(min(count, self._max_hand_count)):
                idx = (self._board_features
                       + king_sq * self._hand_combos
                       + (relative_color * self._num_hand_piece_types + piece_type)
                       * self._max_hand_count
                       + k)
                indices.append(idx)
        return indices

    def active_features(self, state: GameState, perspective: int) -> List[int]:
        king_sq = state.king_square(perspective)

        # C accelerated path
        if _HAS_ACCEL and self._bv2type_arr is not None:
            board = state.board_array()
            hand0 = state.hand_pieces(0)
            hand1 = state.hand_pieces(1)
            h0t = np.array(list(hand0.keys()), dtype=np.int32) if hand0 else np.array([], dtype=np.int32)
            h0c = np.array(list(hand0.values()), dtype=np.int32) if hand0 else np.array([], dtype=np.int32)
            h1t = np.array(list(hand1.keys()), dtype=np.int32) if hand1 else np.array([], dtype=np.int32)
            h1c = np.array(list(hand1.values()), dtype=np.int32) if hand1 else np.array([], dtype=np.int32)
            return _c_shogi_features(
                bytes(board), king_sq, perspective,
                self._num_squares, self._num_board_piece_types,
                self._num_hand_piece_types, self._max_hand_count,
                self._king_board_val, bytes(self._bv2type_arr),
                self._board_features,
                bytes(h0t), bytes(h0c), len(h0t),
                bytes(h1t), bytes(h1c), len(h1t),
            )

        # Python fallback
        features = []
        for piece_type, color, sq in state.pieces_on_board():
            idx = self._board_feature_index(king_sq, piece_type, color, sq, perspective)
            features.append(idx)
        for side in [0, 1]:
            hand = state.hand_pieces(side)
            features.extend(self._hand_feature_indices(king_sq, hand, side, perspective))
        return features

    def feature_delta(
        self, state_before: GameState, move: Move,
        state_after: GameState, perspective: int
    ) -> Optional[Tuple[List[int], List[int]]]:
        king_before = state_before.king_square(perspective)
        king_after = state_after.king_square(perspective)
        if king_before != king_after:
            return None  # Full refresh needed

        # Fast path: derive delta from the move
        if not self._board_val_to_type:
            return self._fallback_delta(state_before, state_after, perspective)

        king_sq = king_before
        added: List[int] = []
        removed: List[int] = []
        side_to_move = state_before.side_to_move()

        if move.from_sq is None:
            # Drop move: piece placed on board + hand count decreases
            to_sq = move.to_sq
            board_after = state_after.board_array()
            placed_val = int(board_after[to_sq])
            placed_abs = abs(placed_val)

            if placed_abs != self._king_board_val and placed_abs in self._board_val_to_type:
                placed_type = self._board_val_to_type[placed_abs]
                placed_color = 0 if placed_val > 0 else 1
                added.append(self._board_feature_index(
                    king_sq, placed_type, placed_color, to_sq, perspective))

            # Hand features diff for the dropping side
            self._diff_hand_features(
                king_sq, state_before, state_after, side_to_move,
                perspective, added, removed)
        else:
            from_sq = move.from_sq
            to_sq = move.to_sq
            board_before = state_before.board_array()
            mover_val = int(board_before[from_sq])
            mover_abs = abs(mover_val)
            mover_color = 0 if mover_val > 0 else 1
            captured_val = int(board_before[to_sq])

            # Board piece changes
            if mover_abs != self._king_board_val and mover_abs in self._board_val_to_type:
                mover_type = self._board_val_to_type[mover_abs]
                # Remove mover from source
                removed.append(self._board_feature_index(
                    king_sq, mover_type, mover_color, from_sq, perspective))
                # Add piece at destination (possibly promoted)
                if move.promotion is not None:
                    board_after = state_after.board_array()
                    placed_abs = abs(int(board_after[to_sq]))
                    placed_type = self._board_val_to_type.get(placed_abs, mover_type)
                else:
                    placed_type = mover_type
                added.append(self._board_feature_index(
                    king_sq, placed_type, mover_color, to_sq, perspective))

            # Remove captured piece
            if captured_val != 0:
                cap_abs = abs(captured_val)
                if cap_abs != self._king_board_val and cap_abs in self._board_val_to_type:
                    cap_color = 0 if captured_val > 0 else 1
                    cap_type = self._board_val_to_type[cap_abs]
                    removed.append(self._board_feature_index(
                        king_sq, cap_type, cap_color, to_sq, perspective))

                # Hand features diff for the capturing side
                self._diff_hand_features(
                    king_sq, state_before, state_after, side_to_move,
                    perspective, added, removed)

        return (added, removed)

    def _diff_hand_features(
        self, king_sq: int,
        state_before: GameState, state_after: GameState,
        side: int, perspective: int,
        added: List[int], removed: List[int],
    ) -> None:
        """Compute hand feature diffs for a single side."""
        hand_before = state_before.hand_pieces(side)
        hand_after = state_after.hand_pieces(side)
        if hand_before != hand_after:
            old_idx = set(self._hand_feature_indices(king_sq, hand_before, side, perspective))
            new_idx = set(self._hand_feature_indices(king_sq, hand_after, side, perspective))
            added.extend(new_idx - old_idx)
            removed.extend(old_idx - new_idx)

    def _fallback_delta(
        self, state_before: GameState, state_after: GameState, perspective: int
    ) -> Tuple[List[int], List[int]]:
        """Compute delta by set difference (slow fallback)."""
        before_set = set(self.active_features(state_before, perspective))
        after_set = set(self.active_features(state_after, perspective))
        return (list(after_set - before_set), list(before_set - after_set))


def shogi_features() -> HalfKPShogi:
    """Create feature set for standard 9x9 shogi."""
    # Board values: 1=P,2=L,3=N,4=S,5=G,6=B,7=R,8=K,9=+P,...,14=+R
    return HalfKPShogi(
        num_squares=81,
        num_board_piece_types=13,
        num_hand_piece_types=7,
        max_hand_count=18,
        board_val_to_type={
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
            9: 7, 10: 8, 11: 9, 12: 10, 13: 11, 14: 12,
        },
        king_board_val=8,
    )


def minishogi_features() -> HalfKPShogi:
    """Create feature set for 5x5 mini shogi."""
    # Board values: 1=P,2=S,3=G,4=B,5=R,6=K,7=+P,8=+S,9=+B,10=+R
    return HalfKPShogi(
        num_squares=25,
        num_board_piece_types=9,
        num_hand_piece_types=5,
        max_hand_count=2,
        board_val_to_type={
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
            7: 5, 8: 6, 9: 7, 10: 8,
        },
        king_board_val=6,
    )
