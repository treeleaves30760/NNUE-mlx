"""Alpha-beta search with iterative deepening and NNUE evaluation."""

import threading
import time
from typing import List, Optional, Tuple

from src.games.base import GameState, Move
from src.search.transposition import TranspositionTable, EXACT, ALPHA, BETA
from src.search.move_ordering import MoveOrdering

try:
    from src.accel import CSearch as _CSearch, AcceleratedAccumulator as _AccelAccum
except ImportError:
    _CSearch = None
    _AccelAccum = None

# Large value representing a won/lost position
INF = 1_000_000
MATE_SCORE = 100_000


class AlphaBetaSearch:
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

        # Try to create C search engine for maximum speed
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

        # C fast path: use CSearch when available and state supports inplace
        if self._csearch is not None and hasattr(state, 'make_move_inplace'):
            self.evaluator.set_position(state)
            result = self._csearch.search(state, max_d, float(self.time_limit_ms))
            if result is not None:
                (from_sq, to_sq, promo, drop), score, nodes = result
                self.nodes_searched = nodes
                move = Move(
                    from_sq=from_sq,
                    to_sq=to_sq,
                    promotion=promo,
                    drop_piece=drop,
                )
                return move, score

        # Python fallback
        self._start_time = time.time()
        self._time_up = False
        self.nodes_searched = 0
        self._use_inplace = hasattr(state, 'make_move_inplace')
        self.evaluator.set_position(state)

        best_move = None
        best_score = -INF

        for depth in range(1, max_d + 1):
            move, score = self._search_root(state, depth)
            if self._time_up and depth > 1:
                break
            if move is not None:
                best_move = move
                best_score = score

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

    def _search_root(self, state: GameState,
                     depth: int) -> Tuple[Optional[Move], float]:
        """Search at the root node."""
        moves = state.legal_moves()
        if not moves:
            return None, -MATE_SCORE

        tt_entry = self.tt.probe(state.zobrist_hash())
        moves = self.move_ordering.order_moves(state, moves, depth, tt_entry)

        best_move = moves[0]
        best_score = -INF
        alpha = -INF
        beta = INF

        for move in moves:
            score = self._do_move_and_search(state, move, depth - 1, alpha, beta)
            if self._time_up:
                break
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score

        self.tt.store(state.zobrist_hash(), depth, best_score, EXACT, best_move)
        return best_move, best_score

    def _alphabeta(self, state: GameState, depth: int,
                   alpha: float, beta: float) -> float:
        """Recursive alpha-beta search."""
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

        # Leaf node: evaluate
        if depth <= 0:
            return self.evaluator.evaluate(state)

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

        # Generate and order moves
        moves = state.legal_moves()
        if not moves:
            if state.is_check():
                return -MATE_SCORE + (self.max_depth - depth)
            return 0

        moves = self.move_ordering.order_moves(state, moves, depth, tt_entry)

        best_score = -INF
        best_move = None
        flag = ALPHA

        for move in moves:
            score = self._do_move_and_search(state, move, depth - 1, alpha, beta)

            if self._time_up:
                return 0

            if score > best_score:
                best_score = score
                best_move = move

            if score >= beta:
                self.move_ordering.update_killers(move, depth)
                self.move_ordering.update_history(move, depth)
                self.tt.store(key, depth, beta, BETA, move)
                return beta

            if score > alpha:
                alpha = score
                flag = EXACT

        self.tt.store(key, depth, alpha, flag, best_move)
        return alpha
