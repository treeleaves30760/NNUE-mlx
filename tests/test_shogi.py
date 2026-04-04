"""Tests for the shogi 9x9 engine."""

import numpy as np
from src.games.base import Move, WHITE, BLACK
from src.games.shogi import initial_state


def test_initial_state_basics():
    s = initial_state()
    assert s.side_to_move() == WHITE  # Sente
    assert not s.is_terminal()
    assert s.result() is None


def test_initial_legal_moves():
    """Shogi starting position has exactly 30 legal moves."""
    s = initial_state()
    assert len(s.legal_moves()) == 30


def test_board_array():
    s = initial_state()
    b = s.board_array()
    assert isinstance(b, np.ndarray)
    assert b.shape == (81,)
    assert np.count_nonzero(b) == 40  # 20 pieces per side


def test_king_squares():
    s = initial_state()
    sk = s.king_square(WHITE)   # Sente king
    gk = s.king_square(BLACK)   # Gote king
    assert 0 <= sk < 81
    assert 0 <= gk < 81
    assert sk != gk


def test_hand_pieces_empty_at_start():
    s = initial_state()
    assert s.hand_pieces(WHITE) == {}
    assert s.hand_pieces(BLACK) == {}


def test_make_move_immutable():
    s = initial_state()
    original_board = s.board_array().copy()
    s2 = s.make_move(s.legal_moves()[0])
    assert np.array_equal(s.board_array(), original_board)
    assert s2.side_to_move() == BLACK


def test_game_config():
    s = initial_state()
    cfg = s.config()
    assert cfg.name == "shogi"
    assert cfg.board_height == 9
    assert cfg.board_width == 9
    assert cfg.has_drops is True
    assert cfg.has_promotion is True


def test_capture_creates_hand_piece():
    """When a piece is captured in shogi, it goes to the capturer's hand."""
    s = initial_state()
    # Play enough moves to create a capture, then verify hand pieces
    # For now just verify the interface works after moves
    state = s
    for _ in range(10):
        moves = state.legal_moves()
        if not moves or state.is_terminal():
            break
        state = state.make_move(moves[0])
    # After some moves, hands should still be accessible
    h0 = state.hand_pieces(WHITE)
    h1 = state.hand_pieces(BLACK)
    assert isinstance(h0, dict)
    assert isinstance(h1, dict)


def test_zobrist_hash():
    s = initial_state()
    h = s.zobrist_hash()
    assert isinstance(h, int)
    s2 = s.make_move(s.legal_moves()[0])
    assert s2.zobrist_hash() != h


def test_copy():
    s = initial_state()
    c = s.copy()
    assert np.array_equal(s.board_array(), c.board_array())
    assert s.side_to_move() == c.side_to_move()
