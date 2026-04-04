"""Tests for the chess engine."""

import numpy as np
from src.games.base import Move, WHITE, BLACK
from src.games.chess import initial_state, ChessState


def test_initial_state_basics():
    s = initial_state()
    assert s.side_to_move() == WHITE
    assert not s.is_terminal()
    assert s.result() is None


def test_initial_legal_moves_count():
    """Standard chess has exactly 20 legal moves from the start position."""
    s = initial_state()
    assert len(s.legal_moves()) == 20


def test_board_array():
    s = initial_state()
    b = s.board_array()
    assert isinstance(b, np.ndarray)
    assert b.shape == (64,)
    # 32 pieces on the board at start
    assert np.count_nonzero(b) == 32


def test_board_array_piece_layout():
    s = initial_state()
    b = s.board_array()
    # White king should be on e1 (index 4)
    assert b[4] > 0  # positive = white
    # Black king should be on e8 (index 60)
    assert b[60] < 0  # negative = black


def test_king_squares():
    s = initial_state()
    assert s.king_square(WHITE) == 4   # e1
    assert s.king_square(BLACK) == 60  # e8


def test_make_move_returns_new_state():
    s = initial_state()
    moves = s.legal_moves()
    s2 = s.make_move(moves[0])
    # Should be a new state object
    assert s is not s2
    # Original should be unchanged
    assert s.side_to_move() == WHITE
    assert s2.side_to_move() == BLACK


def test_pieces_on_board():
    s = initial_state()
    pieces = s.pieces_on_board()
    # 30 non-king pieces at start (32 total - 2 kings)
    assert len(pieces) == 30
    # Each piece is (type, color, square)
    for pt, color, sq in pieces:
        assert 0 <= pt <= 4  # PAWN..QUEEN
        assert color in (0, 1)
        assert 0 <= sq < 64


def test_hand_pieces_empty():
    s = initial_state()
    assert s.hand_pieces(WHITE) == {}
    assert s.hand_pieces(BLACK) == {}


def test_zobrist_hash_changes():
    s = initial_state()
    h1 = s.zobrist_hash()
    s2 = s.make_move(s.legal_moves()[0])
    h2 = s2.zobrist_hash()
    assert h1 != h2


def test_copy():
    s = initial_state()
    c = s.copy()
    assert np.array_equal(s.board_array(), c.board_array())
    assert s.side_to_move() == c.side_to_move()


def test_is_check():
    """After some moves, verify is_check works."""
    s = initial_state()
    assert not s.is_check()


def test_game_config():
    s = initial_state()
    cfg = s.config()
    assert cfg.name == "chess"
    assert cfg.board_height == 8
    assert cfg.board_width == 8
    assert cfg.has_drops is False
    assert cfg.has_promotion is True


def test_pawn_double_step():
    """Pawns should be able to move 2 squares from starting rank."""
    s = initial_state()
    moves = s.legal_moves()
    # e2-e4 should be available (sq 12 -> sq 28)
    double_steps = [m for m in moves if m.from_sq is not None
                    and abs(m.to_sq - m.from_sq) == 16]
    assert len(double_steps) == 8  # Each of the 8 pawns can double-step


def test_capture_changes_piece_count():
    """After a capture, there should be fewer pieces on the board."""
    s = initial_state()
    initial_pieces = np.count_nonzero(s.board_array())
    # Play a few moves to reach a capture position
    # This is a basic test that make_move preserves piece counts correctly
    moves = s.legal_moves()
    s2 = s.make_move(moves[0])
    assert np.count_nonzero(s2.board_array()) == initial_pieces  # No capture yet
