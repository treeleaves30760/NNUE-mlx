"""Tests for Los Alamos 6x6 mini chess engine."""

import numpy as np
from src.games.base import Move, WHITE, BLACK
from src.games.minichess import initial_state


def test_initial_state_basics():
    s = initial_state()
    assert s.side_to_move() == WHITE
    assert not s.is_terminal()
    assert s.result() is None


def test_initial_legal_moves():
    s = initial_state()
    moves = s.legal_moves()
    assert len(moves) > 0
    # Los Alamos: 6 pawns move 1 forward + 4 knight moves = 10
    assert len(moves) == 10


def test_board_array():
    s = initial_state()
    b = s.board_array()
    assert isinstance(b, np.ndarray)
    assert b.shape == (36,)
    assert np.count_nonzero(b) == 24  # 12 pieces per side


def test_king_squares():
    s = initial_state()
    wk = s.king_square(WHITE)
    bk = s.king_square(BLACK)
    assert 0 <= wk < 36
    assert 0 <= bk < 36
    assert wk != bk


def test_make_move_immutable():
    s = initial_state()
    original_board = s.board_array().copy()
    s2 = s.make_move(s.legal_moves()[0])
    # Original unchanged
    assert np.array_equal(s.board_array(), original_board)
    assert s.side_to_move() == WHITE
    assert s2.side_to_move() == BLACK


def test_no_bishops():
    """Los Alamos chess has no bishops."""
    s = initial_state()
    b = s.board_array()
    # In our encoding: Pawn=1, Knight=2, Rook=3, Queen=4, King=5
    # No piece with value 3 that would be a bishop (if bishop existed)
    # Actually in LA chess encoding: no bishop exists at all
    pieces = s.pieces_on_board()
    piece_types = set(pt for pt, _, _ in pieces)
    # Should only have types 0 (pawn), 1 (knight), 2 (rook), 3 (queen)
    assert all(pt <= 3 for pt in piece_types)


def test_game_config():
    s = initial_state()
    cfg = s.config()
    assert cfg.name == "minichess"
    assert cfg.board_height == 6
    assert cfg.board_width == 6
    assert cfg.num_piece_types == 4
    assert cfg.has_drops is False


def test_no_pawn_double_step():
    """Los Alamos pawns can only move 1 square forward."""
    s = initial_state()
    moves = s.legal_moves()
    pawn_moves = [m for m in moves if m.from_sq is not None
                  and abs(m.to_sq - m.from_sq) == 12]  # 2 ranks = 12 squares
    assert len(pawn_moves) == 0


def test_hand_pieces_empty():
    s = initial_state()
    assert s.hand_pieces(WHITE) == {}
    assert s.hand_pieces(BLACK) == {}


def test_zobrist_hash():
    s = initial_state()
    h = s.zobrist_hash()
    assert isinstance(h, int)
    s2 = s.make_move(s.legal_moves()[0])
    assert s2.zobrist_hash() != h
