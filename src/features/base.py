"""Abstract base for NNUE feature extraction."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from src.games.base import GameState, Move


# Sentinel value indicating the accumulator must do a full refresh
# (returned by feature_delta when king moves)
REFRESH_NEEDED = None


class FeatureSet(ABC):
    """Converts game states to NNUE feature indices."""

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
