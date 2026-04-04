"""Move ordering heuristics for alpha-beta search."""

from typing import List, Optional

from src.games.base import GameState, Move
from src.search.transposition import TTEntry

# Piece values for MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
# Higher index = more valuable. Used as generic ordering across games.
DEFAULT_PIECE_VALUES = {
    0: 100,    # Pawn
    1: 300,    # Knight / Lance
    2: 300,    # Bishop / Silver
    3: 500,    # Rook / Gold
    4: 900,    # Queen / Bishop (shogi)
    5: 900,    # Rook (shogi)
    6: 0,      # King (shouldn't be captured normally)
}


class MoveOrdering:
    """Orders moves to improve alpha-beta pruning efficiency."""

    def __init__(self, piece_values=None):
        self.piece_values = piece_values or DEFAULT_PIECE_VALUES
        # Killer moves: 2 slots per depth
        self.killers: List[List[Optional[Move]]] = [
            [None, None] for _ in range(64)
        ]
        # History heuristic: history[from_sq][to_sq]
        self.history = {}

    def order_moves(self, state: GameState, moves: List[Move],
                    depth: int, tt_entry: Optional[TTEntry] = None) -> List[Move]:
        """Sort moves by estimated quality (best first).

        Priority: TT move > Captures (MVV-LVA) > Killers > History > Rest
        """
        scored = []
        tt_move = tt_entry.best_move if tt_entry else None
        board = state.board_array()

        for move in moves:
            score = 0

            # TT move gets highest priority
            if tt_move and move == tt_move:
                score = 100000
            # Captures scored by MVV-LVA
            elif self._is_capture(board, move):
                victim_val = self._piece_value_at(board, move.to_sq)
                attacker_val = self._piece_value_at(board, move.from_sq) if move.from_sq is not None else 0
                score = 10000 + victim_val * 10 - attacker_val
            # Killer moves
            elif depth < len(self.killers):
                if move in self.killers[depth]:
                    score = 5000
            # History heuristic
            if move.from_sq is not None:
                key = (move.from_sq, move.to_sq)
                score += self.history.get(key, 0)

            # Promotion bonus
            if move.promotion is not None:
                score += 3000

            scored.append((score, move))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def update_killers(self, move: Move, depth: int):
        """Store a killer move (non-capture that caused a beta cutoff)."""
        if depth < len(self.killers):
            if self.killers[depth][0] != move:
                self.killers[depth][1] = self.killers[depth][0]
                self.killers[depth][0] = move

    def update_history(self, move: Move, depth: int):
        """Increase history score for a move that caused a cutoff."""
        if move.from_sq is not None:
            key = (move.from_sq, move.to_sq)
            self.history[key] = self.history.get(key, 0) + depth * depth

    def _is_capture(self, board, move: Move) -> bool:
        """Check if a move is a capture. Uses board array."""
        if move.drop_piece is not None:
            return False
        return board[move.to_sq] != 0

    def _piece_value_at(self, board, sq: int) -> int:
        """Get approximate piece value at a square."""
        piece = abs(board[sq])
        if piece == 0:
            return 0
        # Map piece code to value (rough approximation)
        return self.piece_values.get(piece - 1, 100)
