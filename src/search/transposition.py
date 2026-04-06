"""Transposition table for caching search results."""

from dataclasses import dataclass, field
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
    age: int = 0    # Search generation (for replacement policy)


class TranspositionTable:
    """Fixed-size hash table for caching search results.

    Replacement policy: depth-preferred with age-based eviction.
    Entries from stale search generations are replaced by any new entry.
    Same-generation entries are replaced only by deeper or same-position entries.
    """

    def __init__(self, size: int = 1 << 20):
        self.size = size
        self.mask = size - 1
        self.table = [None] * size
        self.hits = 0
        self.misses = 0
        self._age = 0

    def new_search(self):
        """Bump the age counter at the start of each new search call."""
        self._age = (self._age + 1) & 0xFF

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
        """Store a search result with depth-preferred + age replacement."""
        index = key & self.mask
        existing = self.table[index]
        if (existing is None
                or existing.key == key
                or depth >= existing.depth
                or existing.age != self._age):
            self.table[index] = TTEntry(
                key=key, depth=depth, score=score,
                flag=flag, best_move=best_move, age=self._age,
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
