"""Transposition table for caching search results."""

from dataclasses import dataclass
from typing import Optional

from src.games.base import Move


# Entry type flags
EXACT = 0   # Exact score
ALPHA = 1   # Upper bound (failed low)
BETA = 2    # Lower bound (failed high)


@dataclass
class TTEntry:
    """A single transposition table entry."""
    key: int        # Full Zobrist hash for verification
    depth: int      # Search depth
    score: float    # Evaluation score
    flag: int       # EXACT, ALPHA, or BETA
    best_move: Optional[Move]  # Best move found


class TranspositionTable:
    """Fixed-size hash table for caching search results."""

    def __init__(self, size: int = 1 << 20):
        """
        Args:
            size: Number of entries (should be power of 2). Default 1M entries.
        """
        self.size = size
        self.mask = size - 1
        self.table = [None] * size
        self.hits = 0
        self.misses = 0

    def probe(self, key: int) -> Optional[TTEntry]:
        """Look up a position in the table."""
        index = key & self.mask
        entry = self.table[index]
        if entry is not None and entry.key == key:
            self.hits += 1
            return entry
        self.misses += 1
        return None

    def store(self, key: int, depth: int, score: float,
              flag: int, best_move: Optional[Move]):
        """Store a search result. Always-replace strategy."""
        index = key & self.mask
        self.table[index] = TTEntry(
            key=key, depth=depth, score=score,
            flag=flag, best_move=best_move,
        )

    def clear(self):
        """Clear all entries."""
        self.table = [None] * self.size
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
