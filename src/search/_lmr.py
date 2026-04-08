"""Constants and LMR table shared across search modules."""

import math

# Large value representing a won/lost position
INF = 1_000_000
MATE_SCORE = 100_000

# Quiescence search depth limit
MAX_QDEPTH = 8

# Futility pruning margins (indexed by depth, in centipawn-like units)
_FUTILITY_MARGINS = [0, 200, 500]


def _build_lmr_table(max_depth: int = 64, max_moves: int = 64):
    """Pre-compute Late Move Reduction table."""
    table = [[0] * max_moves for _ in range(max_depth)]
    for d in range(1, max_depth):
        for m in range(1, max_moves):
            table[d][m] = max(0, int(1 + math.log(d) * math.log(m) / 2.0))
    return table


_LMR_TABLE = _build_lmr_table()
