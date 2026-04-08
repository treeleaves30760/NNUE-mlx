"""Alpha-beta search with iterative deepening and NNUE evaluation.

Enhancements: quiescence search, check extensions, null-move pruning,
LMR, PVS, aspiration windows, futility pruning, root move ordering.
"""

import threading
import time
from typing import Dict, List, Optional, Tuple

from src.games.base import GameState, Move
from src.search.transposition import TranspositionTable, EXACT, ALPHA, BETA
from src.search.move_ordering import MoveOrdering
from src.search._lmr import INF, MATE_SCORE
from src.search._negamax import _NegamaxMixin

try:
    from src.accel import CSearch as _CSearch, AcceleratedAccumulator as _AccelAccum
except ImportError:
    _CSearch = None
    _AccelAccum = None


class AlphaBetaSearch(_NegamaxMixin):
    """Iterative deepening alpha-beta search engine."""

    def __init__(self, evaluator, max_depth: int = 6,
                 time_limit_ms: int = 5000):
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.time_limit_ms = time_limit_ms
        self.tt = TranspositionTable(size=1 << 20)
        self.move_ordering = MoveOrdering()
        self.nodes_searched = 0
        self._start_time = 0.0
        self._time_up = False
        self._stop_event: Optional[threading.Event] = None
        self._use_inplace = False
        self._root_move_scores: Dict[Move, float] = {}

        self._csearch = None
        if (_CSearch is not None and _AccelAccum is not None
                and hasattr(evaluator, 'accumulator')
                and isinstance(evaluator.accumulator, _AccelAccum)):
            try:
                cfg = getattr(evaluator.feature_set, '_num_squares', 64)
                self._csearch = _CSearch(
                    accumulator=evaluator.accumulator,
                    feature_set=evaluator.feature_set,
                    tt_size=1 << 20,
                    eval_scale=getattr(evaluator, 'EVAL_OUTPUT_SCALE', 128.0),
                    max_sq=cfg,
                )
            except Exception:
                self._csearch = None

    def search(self, state: GameState,
               depth_override: Optional[int] = None) -> Tuple[Optional[Move], float]:
        """Find the best move using iterative deepening alpha-beta."""
        max_d = depth_override or self.max_depth

        # C fast path
        if self._csearch is not None and hasattr(state, 'make_move_inplace'):
            self.evaluator.set_position(state)
            result = self._csearch.search(state, max_d, float(self.time_limit_ms))
            if result is not None:
                (from_sq, to_sq, promo, drop), score, nodes = result
                self.nodes_searched = nodes
                return Move(from_sq=from_sq, to_sq=to_sq,
                            promotion=promo, drop_piece=drop), score

        self._start_time = time.time()
        self._time_up = False
        self.nodes_searched = 0
        self._use_inplace = hasattr(state, 'make_move_inplace')
        self.evaluator.set_position(state)
        self.tt.new_search()
        best_move = None
        best_score = -INF
        prev_score = 0
        prev_move_scores: Dict[Move, float] = {}

        for depth in range(1, max_d + 1):
            # Aspiration windows (skip for shallow depths)
            if depth <= 2:
                alpha, beta = -INF, INF
            else:
                delta = 25
                alpha = prev_score - delta
                beta = prev_score + delta

            move, score = None, None
            while True:
                m, s = self._search_root(state, depth, alpha, beta,
                                         prev_move_scores)
                if self._time_up:
                    break
                if s <= alpha:
                    alpha = max(alpha - delta * 2, -INF)
                    delta *= 2
                elif s >= beta:
                    beta = min(beta + delta * 2, INF)
                    delta *= 2
                else:
                    move, score = m, s
                    break

            if self._time_up and depth > 1:
                break
            if move is not None:
                best_move = move
                best_score = score
                prev_score = score
                prev_move_scores = dict(self._root_move_scores)

        return best_move, best_score

    def search_top_n(self, state: GameState,
                     n: int = 3) -> List[Tuple[Move, float]]:
        """Return the top *n* moves sorted by score (best first)."""
        self._start_time = time.time()
        self._time_up = False
        self.nodes_searched = 0
        self._use_inplace = hasattr(state, 'make_move_inplace')
        self.evaluator.set_position(state)
        moves = state.legal_moves()
        if not moves:
            return []
        best_scores: List[Tuple[Move, float]] = [(m, -INF) for m in moves]
        for depth in range(1, self.max_depth + 1):
            iteration_scores: List[Tuple[Move, float]] = []
            tt_entry = self.tt.probe(state.zobrist_hash())
            ordered = self.move_ordering.order_moves(state, moves, depth, tt_entry)
            for move in ordered:
                score = self._do_move_and_search(state, move, depth - 1, -INF, INF)
                if self._time_up:
                    break
                iteration_scores.append((move, score))
            if self._time_up and depth > 1:
                break
            if not self._time_up:
                best_scores = iteration_scores
        best_scores.sort(key=lambda x: x[1], reverse=True)
        return best_scores[:n]

    def search_top_n_live(self, state: GameState, n: int = 3,
                          live_ref: Optional[list] = None,
                          stop_event: Optional[threading.Event] = None,
                          ) -> List[Tuple[Move, float]]:
        """Iterative deepening that publishes results after each depth."""
        self._stop_event = stop_event
        self._start_time = time.time()
        self._time_up = False
        self.nodes_searched = 0
        self._use_inplace = hasattr(state, 'make_move_inplace')
        self.evaluator.set_position(state)
        moves = state.legal_moves()
        if not moves:
            if live_ref is not None:
                live_ref[0] = (0, self.max_depth, [], True)
            self._stop_event = None
            return []
        best_scores: List[Tuple[Move, float]] = [(m, -INF) for m in moves]
        final_depth = 0
        for depth in range(1, self.max_depth + 1):
            if stop_event and stop_event.is_set():
                break

            iteration_scores: List[Tuple[Move, float]] = []
            tt_entry = self.tt.probe(state.zobrist_hash())
            ordered = self.move_ordering.order_moves(state, moves, depth, tt_entry)

            aborted = False
            for move in ordered:
                if stop_event and stop_event.is_set():
                    aborted = True
                    break
                score = self._do_move_and_search(state, move, depth - 1, -INF, INF)
                if self._time_up or (stop_event and stop_event.is_set()):
                    aborted = True
                    break
                iteration_scores.append((move, score))

            if aborted:
                break
            best_scores = iteration_scores
            final_depth = depth

            top_n = sorted(best_scores, key=lambda x: x[1], reverse=True)[:n]
            done = (depth == self.max_depth)
            if live_ref is not None:
                live_ref[0] = (depth, self.max_depth, top_n, done)

        top_n = sorted(best_scores, key=lambda x: x[1], reverse=True)[:n]
        if live_ref is not None:
            live_ref[0] = (final_depth, self.max_depth, top_n, True)
        self._stop_event = None
        return top_n

    @staticmethod
    def _is_capture(board, move: Move) -> bool:
        if move.drop_piece is not None:
            return False
        return board[move.to_sq] != 0

    def _do_move_and_search(self, state: GameState, move: Move,
                            depth: int, alpha: float, beta: float) -> float:
        """Apply move, search recursively, then undo. Returns -negamax score."""
        if self._use_inplace:
            undo = state.make_move_inplace(move)
            if hasattr(self.evaluator, 'push_move_refresh'):
                self.evaluator.push_move_refresh(state)
            else:
                self.evaluator.push_move(state, move, state)
            score = -self._alphabeta(state, depth, -beta, -alpha)
            self.evaluator.pop_move()
            state.unmake_move(undo)
        else:
            new_state = state.make_move(move)
            self.evaluator.push_move(state, move, new_state)
            score = -self._alphabeta(new_state, depth, -beta, -alpha)
            self.evaluator.pop_move()
        return score

    def _do_move_and_qsearch(self, state: GameState, move: Move,
                              alpha: float, beta: float,
                              qdepth: int) -> float:
        """Apply move, qsearch recursively, then undo."""
        if self._use_inplace:
            undo = state.make_move_inplace(move)
            if hasattr(self.evaluator, 'push_move_refresh'):
                self.evaluator.push_move_refresh(state)
            else:
                self.evaluator.push_move(state, move, state)
            score = -self._quiescence(state, -beta, -alpha, qdepth)
            self.evaluator.pop_move()
            state.unmake_move(undo)
        else:
            new_state = state.make_move(move)
            self.evaluator.push_move(state, move, new_state)
            score = -self._quiescence(new_state, -beta, -alpha, qdepth)
            self.evaluator.pop_move()
        return score

    def _search_root(self, state: GameState, depth: int,
                     alpha: float, beta: float,
                     prev_scores: Dict[Move, float],
                     ) -> Tuple[Optional[Move], float]:
        """Search at the root node with PVS."""
        moves = state.legal_moves()
        if not moves:
            return None, -MATE_SCORE

        tt_entry = self.tt.probe(state.zobrist_hash())
        if prev_scores:
            moves.sort(key=lambda m: prev_scores.get(m, -INF), reverse=True)
        else:
            moves = self.move_ordering.order_moves(state, moves, depth, tt_entry)
        if tt_entry and tt_entry.best_move and tt_entry.best_move in moves:
            moves.remove(tt_entry.best_move)
            moves.insert(0, tt_entry.best_move)

        orig_alpha = alpha
        best_move = moves[0]
        best_score = -INF
        move_scores: Dict[Move, float] = {}

        for i, move in enumerate(moves):
            if i == 0:
                score = self._do_move_and_search(state, move, depth - 1,
                                                  alpha, beta)
            else:
                score = self._do_move_and_search(state, move, depth - 1,
                                                  alpha, alpha + 1)
                if alpha < score < beta and not self._time_up:
                    score = self._do_move_and_search(state, move, depth - 1,
                                                      alpha, beta)
            if self._time_up:
                break
            move_scores[move] = score

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if score >= beta:
                break

        self._root_move_scores = move_scores

        if best_score <= orig_alpha:
            flag = ALPHA
        elif best_score >= beta:
            flag = BETA
        else:
            flag = EXACT
        self.tt.store(state.zobrist_hash(), depth, best_score, flag, best_move)
        return best_move, best_score
