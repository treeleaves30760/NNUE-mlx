"""Zobrist hash key generation for transposition tables."""

import random
from typing import Dict, Tuple


class ZobristKeys:
    """Pre-generated random numbers for Zobrist hashing."""

    def __init__(self, num_squares: int, num_piece_types: int,
                 num_colors: int = 2, seed: int = 42):
        """
        Args:
            num_squares: Board size.
            num_piece_types: Number of distinct piece types (including promoted).
            num_colors: Number of sides (2).
            seed: Random seed for reproducibility.
        """
        rng = random.Random(seed)

        # piece_keys[color][piece_type][square]
        self.piece_keys: Dict[Tuple[int, int, int], int] = {}
        for color in range(num_colors):
            for pt in range(num_piece_types):
                for sq in range(num_squares):
                    self.piece_keys[(color, pt, sq)] = rng.getrandbits(64)

        # Side to move key
        self.side_key = rng.getrandbits(64)

        # Hand piece keys (for shogi): hand_keys[color][piece_type][count]
        self.hand_keys: Dict[Tuple[int, int, int], int] = {}
        for color in range(num_colors):
            for pt in range(num_piece_types):
                for count in range(20):  # max 18 pawns in hand
                    self.hand_keys[(color, pt, count)] = rng.getrandbits(64)

        # Castling rights keys (for chess)
        self.castling_keys = [rng.getrandbits(64) for _ in range(16)]

        # En passant file keys (for chess)
        self.ep_keys = [rng.getrandbits(64) for _ in range(num_squares)]
