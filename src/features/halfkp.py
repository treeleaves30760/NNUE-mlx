"""HalfKP feature extraction for chess-family games.

HalfKP encodes (King position, Piece position) pairs from each perspective.
Feature index = king_sq * num_piece_sq_combos + piece_type * num_squares + piece_sq

For standard chess: 64 king squares * (10 piece types * 64 squares) = 40,960 features
For Los Alamos 6x6: 36 king squares * (8 piece types * 36 squares) = 10,368 features
"""

from typing import Dict, List, Optional, Tuple

from src.features.base import FeatureSet
from src.games.base import GameState, Move


class HalfKP(FeatureSet):
    """HalfKP feature set for chess-family games (no drops)."""

    def __init__(self, num_squares: int, num_piece_types: int,
                 board_val_to_type: Optional[Dict[int, int]] = None,
                 king_board_val: int = 6):
        """
        Args:
            num_squares: Board size (64 for chess, 36 for Los Alamos).
            num_piece_types: Number of non-king piece types per side
                            (5 for chess: P/N/B/R/Q, 4 for LA: P/N/R/Q).
            board_val_to_type: Mapping from abs board value to feature piece type.
            king_board_val: The abs board value representing the king.
        """
        self._num_squares = num_squares
        self._num_piece_types = num_piece_types
        self._king_board_val = king_board_val
        self._board_val_to_type = board_val_to_type or {}
        # Each piece type has two colors (own/opponent from perspective)
        self._piece_sq_combos = num_piece_types * 2 * num_squares
        self._total = num_squares * self._piece_sq_combos

    def num_features(self) -> int:
        return self._total

    def max_active_features(self) -> int:
        # Maximum pieces on board minus kings: 30 for chess, 22 for LA
        return 30 if self._num_squares == 64 else 22

    def _feature_index(self, king_sq: int, piece_type: int,
                       piece_color: int, piece_sq: int,
                       perspective: int) -> int:
        """Compute a single HalfKP feature index.

        Args:
            king_sq: King square from perspective's view.
            piece_type: 0-indexed piece type (excluding king).
            piece_color: 0=white, 1=black.
            piece_sq: Square the piece is on.
            perspective: 0=white, 1=black.
        """
        # Relative color: 0=own piece, 1=opponent piece
        relative_color = 0 if piece_color == perspective else 1
        color_type = relative_color * self._num_piece_types + piece_type
        return (king_sq * self._piece_sq_combos
                + color_type * self._num_squares
                + piece_sq)

    def active_features(self, state: GameState, perspective: int) -> List[int]:
        king_sq = state.king_square(perspective)
        features = []
        for piece_type, color, sq in state.pieces_on_board():
            idx = self._feature_index(king_sq, piece_type, color, sq, perspective)
            features.append(idx)
        return features

    def feature_delta(
        self, state_before: GameState, move: Move,
        state_after: GameState, perspective: int
    ) -> Optional[Tuple[List[int], List[int]]]:
        # If the king of this perspective moved, need full refresh
        king_before = state_before.king_square(perspective)
        king_after = state_after.king_square(perspective)
        if king_before != king_after:
            return None  # Full refresh needed

        # Fast path: derive delta directly from the move
        if not self._board_val_to_type:
            return self._fallback_delta(state_before, state_after, perspective)

        king_sq = king_before
        added: List[int] = []
        removed: List[int] = []

        from_sq = move.from_sq
        to_sq = move.to_sq

        if from_sq is None:
            # Drop moves shouldn't happen in chess/minichess
            return self._fallback_delta(state_before, state_after, perspective)

        board_before = state_before.board_array()
        mover_val = int(board_before[from_sq])
        mover_abs = abs(mover_val)
        mover_color = 0 if mover_val > 0 else 1

        # Skip if mover is a king (handled by king_before != king_after above,
        # but this covers the OTHER perspective where the king didn't change)
        if mover_abs != self._king_board_val:
            mover_type = self._board_val_to_type[mover_abs]
            # Remove mover from source square
            removed.append(self._feature_index(
                king_sq, mover_type, mover_color, from_sq, perspective))
            # Add piece at destination (possibly promoted)
            if move.promotion is not None:
                board_after = state_after.board_array()
                placed_abs = abs(int(board_after[to_sq]))
                placed_type = self._board_val_to_type[placed_abs]
            else:
                placed_type = mover_type
            added.append(self._feature_index(
                king_sq, placed_type, mover_color, to_sq, perspective))

        # Remove captured piece at destination
        captured_val = int(board_before[to_sq])
        if captured_val != 0:
            cap_abs = abs(captured_val)
            if cap_abs != self._king_board_val:
                cap_color = 0 if captured_val > 0 else 1
                cap_type = self._board_val_to_type[cap_abs]
                removed.append(self._feature_index(
                    king_sq, cap_type, cap_color, to_sq, perspective))

        # En-passant: detect diagonal pawn move to empty square (chess only)
        if (mover_abs == 1 and captured_val == 0
                and self._num_squares == 64):
            file_diff = abs((to_sq % 8) - (from_sq % 8))
            if file_diff == 1:  # diagonal pawn move to empty = en passant
                ep_victim_sq = (from_sq // 8) * 8 + (to_sq % 8)
                victim_val = int(board_before[ep_victim_sq])
                if victim_val != 0:
                    v_color = 0 if victim_val > 0 else 1
                    v_type = self._board_val_to_type[abs(victim_val)]
                    removed.append(self._feature_index(
                        king_sq, v_type, v_color, ep_victim_sq, perspective))

        return (added, removed)

    def _fallback_delta(
        self, state_before: GameState, state_after: GameState, perspective: int
    ) -> Tuple[List[int], List[int]]:
        """Compute delta by set difference (slow fallback)."""
        before_set = set(self.active_features(state_before, perspective))
        after_set = set(self.active_features(state_after, perspective))
        return (list(after_set - before_set), list(before_set - after_set))


def chess_features() -> HalfKP:
    """Create HalfKP feature set for standard 8x8 chess."""
    # Board values: 1=Pawn, 2=Knight, 3=Bishop, 4=Rook, 5=Queen, 6=King
    return HalfKP(
        num_squares=64, num_piece_types=5,
        board_val_to_type={1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
        king_board_val=6,
    )


def minichess_features() -> HalfKP:
    """Create HalfKP feature set for Los Alamos 6x6 chess."""
    # Board values: 1=Pawn, 2=Knight, 3=Rook, 4=Queen, 5=King
    return HalfKP(
        num_squares=36, num_piece_types=4,
        board_val_to_type={1: 0, 2: 1, 3: 2, 4: 3},
        king_board_val=5,
    )
