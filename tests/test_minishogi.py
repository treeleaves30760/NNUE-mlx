"""Tests for the mini shogi 5x5 engine."""

import numpy as np
from src.games.base import Move, WHITE, BLACK
from src.games.minishogi import initial_state


def test_initial_state_basics():
    s = initial_state()
    assert s.side_to_move() == WHITE
    assert not s.is_terminal()
    assert s.result() is None


def test_initial_legal_moves():
    s = initial_state()
    moves = s.legal_moves()
    assert len(moves) > 0
    assert len(moves) == 12


def test_board_array():
    s = initial_state()
    b = s.board_array()
    assert isinstance(b, np.ndarray)
    assert b.shape == (25,)
    assert np.count_nonzero(b) == 12  # 6 pieces per side


def test_king_squares():
    s = initial_state()
    sk = s.king_square(WHITE)
    gk = s.king_square(BLACK)
    assert 0 <= sk < 25
    assert 0 <= gk < 25
    assert sk != gk


def test_hand_pieces_at_start():
    s = initial_state()
    # Hands should be empty or zero-count at start
    for side in [WHITE, BLACK]:
        hand = s.hand_pieces(side)
        assert isinstance(hand, dict)
        for count in hand.values():
            assert count == 0


def test_make_move_immutable():
    s = initial_state()
    original_board = s.board_array().copy()
    s2 = s.make_move(s.legal_moves()[0])
    assert np.array_equal(s.board_array(), original_board)
    assert s2.side_to_move() == BLACK


def test_game_config():
    s = initial_state()
    cfg = s.config()
    assert cfg.name == "minishogi"
    assert cfg.board_height == 5
    assert cfg.board_width == 5
    assert cfg.has_drops is True
    assert cfg.has_promotion is True


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


def test_play_multiple_moves():
    """Play several moves and verify state stays consistent."""
    state = initial_state()
    for i in range(20):
        if state.is_terminal():
            result = state.result()
            assert result in (0.0, 0.5, 1.0)
            break
        moves = state.legal_moves()
        assert len(moves) > 0
        state = state.make_move(moves[0])
        assert state.board_array().shape == (25,)
        assert 0 <= state.king_square(WHITE) < 25
        assert 0 <= state.king_square(BLACK) < 25
