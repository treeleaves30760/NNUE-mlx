"""Abstract base classes for all board games."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class GameConfig:
    """Static configuration for a game variant."""

    name: str
    board_height: int
    board_width: int
    num_piece_types: int  # per side, excluding king
    has_drops: bool  # True for shogi family
    has_promotion: bool

    @property
    def num_squares(self) -> int:
        return self.board_height * self.board_width


@dataclass(frozen=True)
class Move:
    """A move on the board.

    For normal moves: from_sq and to_sq are set.
    For drops (shogi): from_sq is None, drop_piece is set.
    For promotions: promotion is the piece type to promote to.
    """

    from_sq: Optional[int]
    to_sq: int
    promotion: Optional[int] = None
    drop_piece: Optional[int] = None

    def __str__(self) -> str:
        if self.drop_piece is not None:
            return f"*{self.drop_piece}@{self.to_sq}"
        s = f"{self.from_sq}->{self.to_sq}"
        if self.promotion is not None:
            s += f"={self.promotion}"
        return s


# Piece color constants
WHITE = 0  # Sente in shogi
BLACK = 1  # Gote in shogi


class GameState(ABC):
    """Abstract game position. All game variants implement this interface."""

    @abstractmethod
    def config(self) -> GameConfig:
        """Return the game configuration."""
        ...

    @abstractmethod
    def legal_moves(self) -> List[Move]:
        """Return all legal moves for the side to move."""
        ...

    @abstractmethod
    def make_move(self, move: Move) -> "GameState":
        """Apply a move and return a NEW game state."""
        ...

    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if the game is over."""
        ...

    @abstractmethod
    def result(self) -> Optional[float]:
        """Return game result from side-to-move's perspective.

        1.0 = side to move wins
        0.0 = side to move loses
        0.5 = draw
        None = game not over
        """
        ...

    @abstractmethod
    def side_to_move(self) -> int:
        """Return current side to move: WHITE (0) or BLACK (1)."""
        ...

    @abstractmethod
    def king_square(self, side: int) -> int:
        """Return the square index of the given side's king."""
        ...

    @abstractmethod
    def pieces_on_board(self) -> List[Tuple[int, int, int]]:
        """Return list of (piece_type, color, square) for all pieces on board.

        piece_type does not include king (king is implicit via king_square).
        """
        ...

    @abstractmethod
    def hand_pieces(self, side: int) -> Dict[int, int]:
        """Return captured pieces in hand: {piece_type: count}.

        Returns empty dict for games without drops (chess).
        """
        ...

    @abstractmethod
    def zobrist_hash(self) -> int:
        """Return a Zobrist hash of the current position."""
        ...

    @abstractmethod
    def board_array(self) -> np.ndarray:
        """Return the raw board as a flat numpy array.

        Positive values = white/sente pieces, negative = black/gote.
        0 = empty square.
        """
        ...

    @abstractmethod
    def copy(self) -> "GameState":
        """Return a deep copy of this state."""
        ...

    def is_check(self) -> bool:
        """Return True if the side to move is in check. Override if needed."""
        return False
