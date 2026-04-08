"""Negamax mixin providing _alphabeta and _quiescence search methods.

Intended to be mixed into AlphaBetaSearch. Expects the host class to provide:
  self.evaluator, self.tt, self.move_ordering, self.nodes_searched,
  self._start_time, self._time_up, self._stop_event, self.time_limit_ms,
  self.max_depth, self._is_capture(), self._do_move_and_search(),
  self._do_move_and_qsearch()
"""

import time

from src.games.base import GameState
from src.search.transposition import EXACT, ALPHA, BETA
from src.search._lmr import INF, MATE_SCORE, MAX_QDEPTH, _FUTILITY_MARGINS, _LMR_TABLE


class _NegamaxMixin:
    """Mixin that provides the core recursive search methods."""

    def _alphabeta(self, state: GameState, depth: int,
                   alpha: float, beta: float,
                   allow_null: bool = True) -> float:
        """Recursive alpha-beta with QSearch, NMP, LMR, PVS, futility."""
        self.nodes_searched += 1

        # Time / stop check every 4096 nodes
        if self.nodes_searched & 4095 == 0:
            if self._stop_event and self._stop_event.is_set():
                self._time_up = True
                return 0
            elapsed = (time.time() - self._start_time) * 1000
            if elapsed >= self.time_limit_ms:
                self._time_up = True
                return 0

        # Terminal node
        if state.is_terminal():
            result = state.result()
            if result is None:
                return 0
            if result == 1.0:
                return MATE_SCORE - (self.max_depth - depth)
            if result == 0.0:
                return -MATE_SCORE + (self.max_depth - depth)
            return 0

        # Check extension: extend search when in check
        in_check = state.is_check()
        if in_check:
            depth += 1

        # Leaf node -> quiescence search
        if depth <= 0:
            return self._quiescence(state, alpha, beta)

        # TT probe
        key = state.zobrist_hash()
        tt_entry = self.tt.probe(key)
        if tt_entry is not None and tt_entry.depth >= depth:
            if tt_entry.flag == EXACT:
                return tt_entry.score
            elif tt_entry.flag == ALPHA and tt_entry.score <= alpha:
                return alpha
            elif tt_entry.flag == BETA and tt_entry.score >= beta:
                return beta

        # Null Move Pruning
        if (allow_null and depth > 2 and not in_check
                and hasattr(state, 'make_null_move')
                and hasattr(self.evaluator, 'accumulator')):
            R = 2 + (1 if depth > 6 else 0)
            null_state = state.make_null_move()
            self.evaluator.accumulator.push()
            score = -self._alphabeta(null_state, depth - 1 - R,
                                      -beta, -beta + 1, allow_null=False)
            self.evaluator.accumulator.pop()
            if self._time_up:
                return 0
            if score >= beta:
                return beta

        # Generate and order moves
        moves = state.legal_moves()
        if not moves:
            if in_check:
                return -MATE_SCORE + (self.max_depth - depth)
            return 0  # stalemate

        moves = self.move_ordering.order_moves(state, moves, depth, tt_entry)
        board = state.board_array()

        # Futility pruning decision
        futile = False
        if depth <= 2 and not in_check:
            static_eval = self.evaluator.evaluate(state)
            if static_eval + _FUTILITY_MARGINS[depth] <= alpha:
                futile = True

        orig_alpha = alpha
        best_score = -INF
        best_move = None

        for i, move in enumerate(moves):
            is_cap = self._is_capture(board, move)
            is_promo = move.promotion is not None

            # Futility: skip quiet moves that can't raise alpha
            if futile and not is_cap and not is_promo and i > 0:
                continue

            # LMR: reduce late quiet moves
            reduction = 0
            if (i >= 3 and depth >= 3 and not in_check
                    and not is_cap and not is_promo):
                reduction = _LMR_TABLE[min(depth, 63)][min(i, 63)]
                reduction = min(reduction, depth - 2)

            if i == 0:
                # PVS: full window on first move
                score = self._do_move_and_search(state, move,
                                                  depth - 1, alpha, beta)
            else:
                # Zero-width search with possible LMR
                score = self._do_move_and_search(state, move,
                                                  depth - 1 - reduction,
                                                  alpha, alpha + 1)
                # Re-search without LMR if it raised alpha
                if reduction > 0 and score > alpha and not self._time_up:
                    score = self._do_move_and_search(state, move,
                                                      depth - 1,
                                                      alpha, alpha + 1)
                # Re-search with full window if PVS raised alpha
                if alpha < score < beta and not self._time_up:
                    score = self._do_move_and_search(state, move,
                                                      depth - 1, alpha, beta)

            if self._time_up:
                return 0

            if score > best_score:
                best_score = score
                best_move = move

            if score >= beta:
                if not is_cap:
                    self.move_ordering.update_killers(move, depth)
                    self.move_ordering.update_history(move, depth)
                self.tt.store(key, depth, beta, BETA, move)
                return beta

            if score > alpha:
                alpha = score

        flag = EXACT if alpha > orig_alpha else ALPHA
        self.tt.store(key, depth, alpha, flag, best_move)
        return alpha

    def _quiescence(self, state: GameState, alpha: float,
                    beta: float, qdepth: int = 0) -> float:
        """Search captures until the position is quiet."""
        self.nodes_searched += 1

        if self.nodes_searched & 4095 == 0:
            if self._stop_event and self._stop_event.is_set():
                self._time_up = True
                return 0
            elapsed = (time.time() - self._start_time) * 1000
            if elapsed >= self.time_limit_ms:
                self._time_up = True
                return 0

        if state.is_terminal():
            result = state.result()
            if result is None:
                return 0
            if result == 1.0:
                return MATE_SCORE
            if result == 0.0:
                return -MATE_SCORE
            return 0

        # Stand-pat: static eval as lower bound
        stand_pat = self.evaluator.evaluate(state)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        if qdepth >= MAX_QDEPTH:
            return alpha

        # Generate only captures and promotions
        moves = state.legal_moves()
        board = state.board_array()
        captures = [m for m in moves
                    if self._is_capture(board, m) or m.promotion is not None]

        if not captures:
            return alpha

        # Order by MVV-LVA
        captures = self.move_ordering.order_moves(state, captures, 0, None)

        for move in captures:
            score = self._do_move_and_qsearch(state, move, alpha, beta,
                                               qdepth + 1)
            if self._time_up:
                return 0
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha
