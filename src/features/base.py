"""Abstract base for NNUE feature extraction."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from src.games.base import GameState, Move


# Sentinel value indicating the accumulator must do a full refresh
# (returned by feature_delta when king moves)
REFRESH_NEEDED = None


class FeatureSet(ABC):
    """Converts game states to NNUE feature indices."""

    _max_material: int = 40
    _max_feature_count: int = 40

    @abstractmethod
    def num_features(self) -> int:
        """Total feature space size (e.g. 40960 for chess HalfKP)."""
        ...

    @abstractmethod
    def max_active_features(self) -> int:
        """Upper bound on active features per perspective per position."""
        ...

    @abstractmethod
    def active_features(self, state: GameState, perspective: int) -> List[int]:
        """Return active feature indices for the given perspective.

        Args:
            state: Current game state.
            perspective: 0=white/sente, 1=black/gote.

        Returns:
            List of active feature indices (length varies per position).
        """
        ...

    @abstractmethod
    def feature_delta(
        self, state_before: GameState, move: Move,
        state_after: GameState, perspective: int
    ) -> Optional[Tuple[List[int], List[int]]]:
        """Compute incremental feature changes for a move.

        Args:
            state_before: Position before the move.
            move: The move being made.
            state_after: Position after the move.
            perspective: 0=white/sente, 1=black/gote.

        Returns:
            (added_indices, removed_indices) for incremental update,
            or None if a full refresh is needed (king moved).
        """
        ...

    def _is_king(self, piece_type: int) -> bool:
        """Return True if piece_type is a king. Subclasses may override."""
        return False

    def material_bucket(self, state: GameState, num_buckets: int = 8) -> int:
        """Map the current material count on the board to an output bucket index.

        Bucket 0 = fewest pieces (late endgame), num_buckets-1 = most (opening).
        """
        count = sum(
            1 for pt, _color, _sq in state.pieces_on_board()
            if not self._is_king(pt)
        )
        return min((count * num_buckets) // (self._max_material + 1), num_buckets - 1)

    def bucket_from_feature_counts(
        self, num_white: int, num_black: int, num_buckets: int = 8
    ) -> int:
        """Approximate bucket from feature list lengths (used at training data load time)."""
        total = num_white + num_black
        denom = self._max_feature_count * 2 + 1
        return min((total * num_buckets) // denom, num_buckets - 1)

    def mirror_table(self) -> Optional[np.ndarray]:
        """Return a feature-index mirror table, or None if not supported."""
        return None
