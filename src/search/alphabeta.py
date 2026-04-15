"""Alpha-beta search with iterative deepening and NNUE evaluation.

Enhancements: quiescence search, check extensions, null-move pruning,
LMR, PVS, aspiration windows, futility pruning, root move ordering.

Fast path: for chess and shogi (which have full C movegen/make/unmake
under src/accel/), AlphaBetaSearch dispatches to the C-native
multi-threaded Lazy SMP search in ``_csearch_cnative.c`` instead of
the Python alphabeta loop. That gives roughly 10-20× faster
think-time with identical search features (LMR, PVS, null move, TT,
killer/history). Mini variants (minichess, minishogi) still use the
Python path.
"""

import os
import struct
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


def _default_n_threads() -> int:
    """Default worker count for Lazy SMP: performance cores on M-series
    Macs, generally cpu_count() - 1 with a floor of 1 and cap of 8."""
    n = os.cpu_count() or 4
    n = max(1, min(8, n - 1))
    return n


try:
    from src.games.chess.state import ChessState as _ChessState
except Exception:
    _ChessState = None
try:
    from src.games.shogi.state import ShogiState as _ShogiState
except Exception:
    _ShogiState = None


def _cnative_game_kind(state) -> Optional[str]:
    """Return 'chess' or 'shogi' if the state supports the C-native fast
    path, otherwise None. Mini variants (minichess, minishogi) lack the
    full C movegen so we skip them here and fall through to the Python
    alphabeta path. Uses isinstance rather than module-path substring
    matching so 'games.minichess' isn't confused with 'games.chess'."""
    if _ChessState is not None and isinstance(state, _ChessState):
        return 'chess'
    if _ShogiState is not None and isinstance(state, _ShogiState):
        return 'shogi'
    return None


def _pack_chess_history(state) -> bytes:
    hist = getattr(state, "_history", ()) or ()
    if not hist:
        return b""
    hist = tuple(hist)[-64:]
    return struct.pack(f"<{len(hist)}Q", *hist)


def _pack_shogi_history(state) -> bytes:
    hist = getattr(state, "_history", ()) or ()
    if not hist:
        return b""
    hist = tuple(hist)[-64:]
    return struct.pack(f"<{len(hist)}Q", *hist)

try:
    from src.accel._nnue_accel import shogi_rule_search as _c_shogi_rule_search
    from src.accel._nnue_accel import shogi_rule_search_live as _c_shogi_rule_search_live
except ImportError:
    _c_shogi_rule_search = None
    _c_shogi_rule_search_live = None

try:
    from src.accel._nnue_accel import chess_c_rule_search as _c_chess_rule_search
except ImportError:
    _c_chess_rule_search = None


class ShogiCRuleSearch:
    """Drop-in replacement for AlphaBetaSearch on shogi rule-based bootstrap.

    Exposes the same ``search(state, depth_override=None)`` API but delegates
    the entire alpha-beta loop to a self-contained C implementation that
    operates directly on a ShogiPosition struct (no Python callbacks in the
    hot loop). Typical speedup over the Python path is 100-500x at equal
    depth, which is what makes rule-based bootstrap feasible past depth 4.

    Construct via :func:`create_rule_based_search("shogi", ...)` rather than
    instantiating directly.
    """

    def __init__(self, max_depth: int = 6, time_limit_ms: int = 5000):
        self.max_depth = max_depth
        self.time_limit_ms = time_limit_ms
        self.nodes_searched = 0

    @staticmethod
    def _pack_position(state):
        board = bytes(state.board_array())
        sh = tuple(state.hand_pieces(0).get(i, 0) for i in range(7))
        gh = tuple(state.hand_pieces(1).get(i, 0) for i in range(7))
        return board, sh, gh, int(state.side_to_move())

    def search(self, state, depth_override: Optional[int] = None):
        """Same always-has-best-move guarantee as AlphaBetaSearch.search.

        The C search can return ``None`` on time budget = 0 or when its
        internal setup fails before producing any root candidate. In
        those cases we fall back to the first legal move so the caller
        (training bootstrap, GUI analysis) always has something to
        play. Terminal positions legitimately return ``(None, 0.0)``.
        """
        legal = state.legal_moves()
        if not legal:
            return None, 0.0

        board, sh, gh, side = self._pack_position(state)
        depth = depth_override if depth_override is not None else self.max_depth
        result = _c_shogi_rule_search(
            board, sh, gh, side,
            int(depth), float(self.time_limit_ms),
        )
        if result is None:
            # Give the caller a real move so training bootstrap never
            # loses a ply to a None return from the fast search.
            return legal[0], 0.0
        (from_sq, to_sq, promo, drop), score, nodes = result
        self.nodes_searched = int(nodes)
        mv = Move(from_sq=from_sq, to_sq=to_sq, promotion=promo, drop_piece=drop)
        return mv, float(score)

    def search_top_n_live(self, state, n: int = 3,
                           live_ref: Optional[list] = None,
                           stop_event: Optional[threading.Event] = None):
        """Iterative-deepening C search with live progress updates.

        Signature matches :meth:`AlphaBetaSearch.search_top_n_live` so the
        GUI's analysis worker thread can use this class as a drop-in
        replacement. After each completed iteration (and roughly every
        500k nodes during long iterations) the provided ``live_ref`` list is
        updated with a ``(depth, max_depth, top_n_moves, done)`` tuple and
        ``stop_event`` is polled so the user can abort analysis.

        Passing ``self.max_depth <= 0`` activates the C search's infinite
        mode (capped internally at a ply count so high it never finishes
        in practice), which lets the GUI run an analysis that keeps
        deepening until the user cancels.
        """
        if _c_shogi_rule_search_live is None:
            return []

        board, sh, gh, side = self._pack_position(state)
        max_depth_sentinel = self.max_depth if self.max_depth > 0 else 0
        time_ms_sentinel = (float(self.time_limit_ms)
                            if self.time_limit_ms > 0 else 0.0)

        # Keep the latest snapshot locally so we can return it after C finishes
        latest: list = [[]]

        def _cb(depth, max_depth_reported, top_moves, done):
            py_moves = []
            for (m_tup, sc) in top_moves[:n]:
                from_sq, to_sq, promo, drop = m_tup
                mv = Move(from_sq=from_sq, to_sq=to_sq,
                          promotion=promo, drop_piece=drop)
                py_moves.append((mv, float(sc)))
            latest[0] = py_moves
            if live_ref is not None:
                # Pass through C-reported max_depth. When the user asked
                # for infinite mode (self.max_depth <= 0) the C side sets
                # its internal cap to ~64, which the GUI panel treats as
                # the "∞" sentinel (anything > DEPTH_MAX).
                live_ref[0] = (depth, max_depth_reported, py_moves, bool(done))
            if stop_event is not None and stop_event.is_set():
                return True
            return False

        result = _c_shogi_rule_search_live(
            board, sh, gh, side,
            int(max_depth_sentinel), float(time_ms_sentinel),
            _cb, 500_000,
        )
        if result is None:
            return latest[0]
        (_from_sq, _to_sq, _promo, _drop), _score, nodes = result
        self.nodes_searched = int(nodes)
        return latest[0]

    def search_top_n(self, state, n: int = 3):
        """Return the top ``n`` moves sorted by score (best first).

        Used by :class:`SelfPlayEngine` for temperature-based opening
        exploration. Implemented by searching each child position at
        depth-1: correct but costs roughly N * search_cost. Since opening
        exploration runs only for the first few plies of bootstrap games,
        this is acceptable — the bulk of bootstrap is still one C search
        per ply.
        """
        legal = state.legal_moves()
        if not legal:
            return []
        child_depth = max(1, self.max_depth - 1)
        per_child_budget = max(50.0, self.time_limit_ms / max(len(legal), 1))
        scored = []
        total_nodes = 0
        for mv in legal:
            child = state.make_move(mv)
            if child.is_terminal():
                res = child.result()
                # child.result() is from child's side-to-move perspective.
                # Flip so the score is from *state*'s perspective.
                if res is None:
                    s = 0.0
                else:
                    s = -(100000.0 * (2.0 * res - 1.0))
                scored.append((mv, s))
                continue
            board, sh, gh, side = self._pack_position(child)
            result = _c_shogi_rule_search(
                board, sh, gh, side,
                int(child_depth), float(per_child_budget),
            )
            if result is None:
                continue
            _m, child_score, nodes = result
            total_nodes += int(nodes)
            # child_score is from the child's (opponent's) side-to-move view,
            # so negate to get the score from the parent's perspective.
            scored.append((mv, -float(child_score)))
        self.nodes_searched = total_nodes
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]


class ChessCRuleSearch:
    """C-backed drop-in for chess rule-based bootstrap.

    Mirrors :class:`ShogiCRuleSearch` — all the search work happens in
    ``_chess_rule_search.c`` via a single C call per ``search()``.
    Typical speedup over the Python path is ~80× at equal depth, which
    is what makes deeper chess bootstrap practical.

    Use :func:`create_rule_based_search("chess", ...)` to construct.
    """

    def __init__(self, max_depth: int = 6, time_limit_ms: int = 5000):
        self.max_depth = max_depth
        self.time_limit_ms = time_limit_ms
        self.nodes_searched = 0

    @staticmethod
    def _pack_history(state) -> bytes:
        """Serialise the state's repetition history as packed uint64.

        The C search uses this for threefold-draw detection inside the
        tree. Cap at 64 entries; search rarely needs more.
        """
        import struct
        hist = getattr(state, "_history", ())
        if not hist:
            return b""
        hist = tuple(hist)[-64:]
        return struct.pack(f"<{len(hist)}Q", *hist)

    def search(self, state, depth_override: Optional[int] = None):
        """Same always-has-best-move contract as the Python path.

        Refuses to return None unless the position has no legal moves;
        on any C-side failure it falls back to the first legal move so
        training bootstrap never loses a ply.
        """
        legal = state.legal_moves()
        if not legal:
            return None, 0.0

        depth = depth_override if depth_override is not None else self.max_depth
        try:
            result = _c_chess_rule_search(
                state.board_array(),
                int(state.side_to_move()),
                int(getattr(state, "_castling", 0)),
                int(getattr(state, "_ep_square", -1)),
                int(getattr(state, "_halfmove", 0)),
                int(state.king_square(0)),
                int(state.king_square(1)),
                self._pack_history(state),
                int(depth),
                float(self.time_limit_ms),
            )
        except Exception:
            return legal[0], 0.0
        if result is None:
            return legal[0], 0.0
        (from_sq, to_sq, promo), score, nodes = result
        self.nodes_searched = int(nodes)
        mv = Move(from_sq=from_sq, to_sq=to_sq, promotion=promo)
        return mv, float(score)


def create_rule_based_search(game_name: str, max_depth: int = 6,
                             time_limit_ms: int = 5000):
    """Build a rule-based search engine appropriate for ``game_name``.

    * ``shogi`` → ``ShogiCRuleSearch`` when C accel available (100-500× over
      Python alpha-beta).
    * ``chess`` → ``ChessCRuleSearch`` when C accel available (~80× over
      Python alpha-beta). Falls back to Python for mini variants and when
      the C extension is not built.
    """
    from src.search.evaluator import RuleBasedEvaluator, SHOGI_MVV_LVA_VALUES

    if game_name == "shogi" and _c_shogi_rule_search is not None:
        return ShogiCRuleSearch(
            max_depth=max_depth, time_limit_ms=time_limit_ms,
        )

    if game_name == "chess" and _c_chess_rule_search is not None:
        return ChessCRuleSearch(
            max_depth=max_depth, time_limit_ms=time_limit_ms,
        )

    evaluator = RuleBasedEvaluator()
    search = AlphaBetaSearch(
        evaluator, max_depth=max_depth, time_limit_ms=time_limit_ms,
    )
    if game_name == "shogi":
        search.move_ordering = MoveOrdering(piece_values=SHOGI_MVV_LVA_VALUES)
    return search


class AlphaBetaSearch(_NegamaxMixin):
    """Iterative deepening alpha-beta search engine."""

    def __init__(self, evaluator, max_depth: int = 6,
                 time_limit_ms: int = 5000,
                 n_threads: Optional[int] = None):
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.time_limit_ms = time_limit_ms
        self.n_threads = n_threads if n_threads is not None else _default_n_threads()
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
                # Prefer the per-instance scale (loaded from npz metadata
                # with old-checkpoint remapping applied) over the class
                # default, so warm-loaded models carry their corrected
                # scale into the C search.
                scale = getattr(
                    evaluator, 'output_eval_scale',
                    getattr(evaluator, 'EVAL_OUTPUT_SCALE', 32.0),
                )
                self._csearch = _CSearch(
                    accumulator=evaluator.accumulator,
                    feature_set=evaluator.feature_set,
                    tt_size=1 << 20,
                    eval_scale=scale,
                    max_sq=cfg,
                )
            except Exception:
                self._csearch = None

    # ----------------------------------------------------------------
    # C-native multi-threaded (Lazy SMP) fast path
    # ----------------------------------------------------------------

    def _try_cnative_search(self, state: GameState, max_d: int
                             ) -> Optional[Tuple[Optional[Move], float]]:
        """Attempt a single C-native Lazy SMP search. Returns (move, score)
        on success or None if this state / evaluator combo can't use the
        fast path (caller then falls through to the Python alphabeta)."""
        if self._csearch is None:
            return None
        kind = _cnative_game_kind(state)
        if kind is None:
            return None

        try:
            self.evaluator.set_position(state)
            if kind == 'chess':
                result = self._csearch.search_cnative_chess(
                    bytes(state.board_array()),
                    int(state.side_to_move()),
                    int(state._castling),
                    int(state._ep_square),
                    int(state._halfmove),
                    int(state.king_square(0)),
                    int(state.king_square(1)),
                    _pack_chess_history(state),
                    int(max_d),
                    float(self.time_limit_ms),
                    int(self.n_threads),
                )
                if result is None:
                    return None
                (from_sq, to_sq, promo), score, nodes = result
                self.nodes_searched = int(nodes)
                mv = Move(from_sq=from_sq, to_sq=to_sq,
                          promotion=promo, drop_piece=None)
                return mv, float(score)

            # shogi
            hand0 = state.hand_pieces(0)
            hand1 = state.hand_pieces(1)
            sh = tuple(hand0.get(i, 0) for i in range(7))
            gh = tuple(hand1.get(i, 0) for i in range(7))
            result = self._csearch.search_cnative_shogi(
                bytes(state.board_array()),
                sh, gh,
                int(state.side_to_move()),
                _pack_shogi_history(state),
                int(max_d),
                float(self.time_limit_ms),
                int(self.n_threads),
            )
            if result is None:
                return None
            (from_sq, to_sq, promo, drop), score, nodes = result
            self.nodes_searched = int(nodes)
            mv = Move(from_sq=from_sq, to_sq=to_sq,
                      promotion=promo, drop_piece=drop)
            return mv, float(score)
        except Exception:
            return None

    def _try_cnative_live(self, state: GameState, n: int,
                           live_ref: Optional[list],
                           stop_event: Optional[threading.Event],
                           ) -> Optional[List[Tuple[Move, float]]]:
        """Live-search fast path that mirrors shogi_rule_search_live but
        for the C-native NNUE search. Returns a list of (Move, score)
        pairs, or None if this state can't use the fast path."""
        if self._csearch is None:
            return None
        kind = _cnative_game_kind(state)
        if kind is None:
            return None

        try:
            self.evaluator.set_position(state)
            # For "infinite think" (max_depth <= 0) we pass 0 to the C
            # side which interprets it as "search until time limit or
            # abort". Same for time_limit_ms.
            max_d_sentinel = self.max_depth if self.max_depth > 0 else 0
            time_ms_sentinel = (float(self.time_limit_ms)
                                if self.time_limit_ms > 0 else 0.0)

            if kind == 'chess':
                result = self._csearch.search_cnative_live_chess(
                    bytes(state.board_array()),
                    int(state.side_to_move()),
                    int(state._castling),
                    int(state._ep_square),
                    int(state._halfmove),
                    int(state.king_square(0)),
                    int(state.king_square(1)),
                    _pack_chess_history(state),
                    int(max_d_sentinel),
                    float(time_ms_sentinel),
                    int(self.n_threads),
                    live_ref if live_ref is not None else [None],
                    stop_event,
                )
            else:
                hand0 = state.hand_pieces(0)
                hand1 = state.hand_pieces(1)
                sh = tuple(hand0.get(i, 0) for i in range(7))
                gh = tuple(hand1.get(i, 0) for i in range(7))
                result = self._csearch.search_cnative_live_shogi(
                    bytes(state.board_array()),
                    sh, gh,
                    int(state.side_to_move()),
                    _pack_shogi_history(state),
                    int(max_d_sentinel),
                    float(time_ms_sentinel),
                    int(self.n_threads),
                    live_ref if live_ref is not None else [None],
                    stop_event,
                )

            # Derive top-N from the live_ref snapshot — the C side
            # publishes (depth, max_depth, top_moves, done). Before
            # returning we normalise the tuples back into Python Move
            # objects.
            if live_ref is not None and live_ref[0] is not None:
                _d, _md, snapshot_top, _done = live_ref[0]
                out: List[Tuple[Move, float]] = []
                for (m_tup, sc) in snapshot_top[:n]:
                    from_sq, to_sq, promo, drop = m_tup
                    out.append((
                        Move(from_sq=from_sq, to_sq=to_sq,
                             promotion=promo, drop_piece=drop),
                        float(sc),
                    ))
                if result is not None:
                    (from_sq, to_sq, promo, drop), score, nodes = result
                    self.nodes_searched = int(nodes)
                return out

            if result is not None:
                (from_sq, to_sq, promo, drop), score, nodes = result
                self.nodes_searched = int(nodes)
                return [(
                    Move(from_sq=from_sq, to_sq=to_sq,
                         promotion=promo, drop_piece=drop),
                    float(score),
                )]
            return []
        except Exception:
            return None

    def search(self, state: GameState,
               depth_override: Optional[int] = None) -> Tuple[Optional[Move], float]:
        """Iterative-deepening alpha-beta with a firm "always has a best
        move" guarantee.

        Whenever the position has any legal moves, this method returns a
        legal move — even if the time budget is zero, a stop_event was
        set before entry, or the first iteration aborts mid-way. The
        guarantees in order:

          1. If ``legal_moves()`` is empty, return ``(None, 0.0)``.
          2. Seed ``best_move`` to the first legal move so a pathological
             early abort still produces *something* playable.
          3. Any iteration that even partially searches root moves
             adopts its best-scored move — we never discard a partial
             iteration's work.
          4. The deepest completed iteration overrides shallower fallbacks.
        """
        max_d = depth_override or self.max_depth

        legal = state.legal_moves()
        if not legal:
            return None, 0.0

        # C-native Lazy SMP fast path for chess/shogi: replaces the old
        # Python-callback CSearch.search. ~10-20× faster per node at
        # n_threads=1, plus ~3-4× from multi-threading.
        fast = self._try_cnative_search(state, max_d)
        if fast is not None:
            return fast

        # Legacy CSearch Python-callback path (kept for states without
        # the C-native fast path, e.g. minichess/minishogi if their
        # evaluator still exposes an AcceleratedAccumulator).
        if self._csearch is not None and hasattr(state, 'make_move_inplace'):
            self.evaluator.set_position(state)
            result = self._csearch.search(state, max_d, float(self.time_limit_ms))
            if result is not None:
                (from_sq, to_sq, promo, drop), score, nodes = result
                self.nodes_searched = nodes
                return Move(from_sq=from_sq, to_sq=to_sq,
                            promotion=promo, drop_piece=drop), score
            # C returned None — fall through to Python so we still return
            # at least a legal move.

        self._start_time = time.time()
        self._time_up = False
        self.nodes_searched = 0
        self._use_inplace = hasattr(state, 'make_move_inplace')
        self.evaluator.set_position(state)
        self.tt.new_search()

        # Seed the best move with a legal fallback. Any completed iteration
        # below will replace it; if none complete we still ship a legal
        # move to the caller instead of None.
        best_move: Optional[Move] = legal[0]
        best_score = -INF
        prev_score = 0
        prev_move_scores: Dict[Move, float] = {}

        for depth in range(1, max_d + 1):
            # Aspiration windows (skip for shallow depths)
            if depth <= 2:
                alpha, beta = -INF, INF
                delta = 25
            else:
                delta = 25
                alpha = prev_score - delta
                beta = prev_score + delta

            iter_move: Optional[Move] = None
            iter_score: float = -INF
            while True:
                m, s = self._search_root(state, depth, alpha, beta,
                                         prev_move_scores)
                if self._time_up:
                    # Partial iteration: still accept whatever _search_root
                    # found. _search_root returns (fallback_move, -INF) if
                    # interrupted before any move completed, and
                    # (best_so_far, real_score) otherwise. Adopt only if
                    # we have a real score — don't overwrite a previous
                    # iteration's result with a bogus -INF.
                    if s > -INF and m is not None:
                        iter_move, iter_score = m, s
                    break
                if s <= alpha:
                    alpha = max(alpha - delta * 2, -INF)
                    delta *= 2
                elif s >= beta:
                    beta = min(beta + delta * 2, INF)
                    delta *= 2
                else:
                    iter_move, iter_score = m, s
                    break

            if iter_move is not None and iter_score > -INF:
                best_move = iter_move
                best_score = iter_score
                prev_score = iter_score
                prev_move_scores = dict(self._root_move_scores)

            if self._time_up:
                break

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
        """Iterative-deepening MultiPV PVS analysis.

        For each depth, performs up to N sequential root searches to compute
        exact scores for the top-N moves:

          1. PVS search with aspiration window => find PV1
          2. Exclude PV1 from candidates, search again => find PV2
          3. Exclude PV1..PV(N-1), search again => find PVN

        Publishes to ``live_ref`` *after each completed PV within an
        iteration*, so a freshly-found PV1 is immediately visible even
        if PV2/PV3 haven't finished yet. That gives the controller
        (and ultimately any caller asking "what's the best move right
        now?") a current answer at every moment during the search.

        Scores remain stable across depths because aspiration windows
        bound the search around the previous iteration's score, the TT
        keeps ordering honest, and the downstream commit logic in the
        AnalysisController filters per-iteration noise before touching
        the display.
        """
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

        # C-native Lazy SMP live-analysis fast path: chess + shogi. It
        # publishes (depth, max_depth, top_moves, done) to live_ref[0]
        # directly from C after every completed iteration, so the GUI
        # picks up updates without needing the Python loop below.
        cn_live = self._try_cnative_live(state, n, live_ref, stop_event)
        if cn_live is not None:
            self._stop_event = None
            return cn_live

        multipv = min(n, len(moves))
        prev_pv_scores: List[Tuple[Move, float]] = []
        final_depth = 0

        for depth in range(1, self.max_depth + 1):
            if stop_event and stop_event.is_set():
                break
            if self._time_up:
                break

            iter_pvs: List[Tuple[Move, float]] = []
            iter_excluded: List[Move] = []
            iter_aborted = False

            for pv_idx in range(multipv):
                candidates = [m for m in moves if m not in iter_excluded]
                if not candidates:
                    break

                # Aspiration window: narrow only when we have a reliable
                # previous score that isn't near mate territory.
                use_aspiration = (
                    depth > 2
                    and pv_idx < len(prev_pv_scores)
                    and abs(prev_pv_scores[pv_idx][1]) < MATE_SCORE - 1000
                )
                if use_aspiration:
                    center = prev_pv_scores[pv_idx][1]
                    delta = 25
                    alpha_w = center - delta
                    beta_w = center + delta
                else:
                    alpha_w, beta_w = -INF, INF
                    delta = 25

                best_move: Optional[Move] = None
                best_score: float = -INF
                while True:
                    m, s = self._search_multipv_root(
                        state, depth, alpha_w, beta_w,
                        candidates, prev_pv_scores, pv_idx,
                    )
                    if (self._time_up
                            or (stop_event and stop_event.is_set())):
                        iter_aborted = True
                        break
                    if s <= alpha_w and alpha_w > -INF:
                        alpha_w = max(alpha_w - delta * 2, -INF)
                        delta *= 2
                        continue
                    if s >= beta_w and beta_w < INF:
                        beta_w = min(beta_w + delta * 2, INF)
                        delta *= 2
                        continue
                    best_move, best_score = m, s
                    break

                if iter_aborted or best_move is None:
                    break

                iter_pvs.append((best_move, best_score))
                iter_excluded.append(best_move)

                # Partial publish: as soon as a PV finishes we surface it
                # to the live ref so the caller always has the latest
                # best move available. The AnalysisController is already
                # duplicate-aware (sig deduplication + history-replace at
                # same depth) so the extra publishes don't create churn
                # downstream.
                if live_ref is not None:
                    live_ref[0] = (
                        depth, self.max_depth, list(iter_pvs), False,
                    )

            # A full MultiPV iteration that completed updates the
            # primary "best known set" used for aspiration in the next
            # depth. If the iteration aborted mid-PV we still keep
            # iter_pvs in live_ref (partial publish above) but do not
            # promote it to prev_pv_scores — the aspiration window in
            # the next depth prefers a fully-searched baseline.
            if iter_aborted or not iter_pvs:
                break

            prev_pv_scores = iter_pvs
            final_depth = depth
            done = (depth == self.max_depth)
            if live_ref is not None:
                live_ref[0] = (
                    depth, self.max_depth, list(prev_pv_scores), done,
                )

        if live_ref is not None:
            live_ref[0] = (
                final_depth, self.max_depth, list(prev_pv_scores), True,
            )
        self._stop_event = None
        return list(prev_pv_scores)

    def _search_multipv_root(self, state: GameState, depth: int,
                              alpha: float, beta: float,
                              candidates: List[Move],
                              prev_pv_scores: List[Tuple[Move, float]],
                              pv_idx: int,
                              ) -> Tuple[Optional[Move], float]:
        """PVS root search restricted to a candidate move set.

        Mirrors :meth:`_search_root` but only considers ``candidates``
        (callers exclude prior PVs for MultiPV). The previous iteration's
        PV at ``pv_idx`` is pinned as the first move to search so the PVS
        first-move full-window hit lands on the expected score, maximising
        the chance of a tight aspiration window. Stores to the TT only when
        ``pv_idx == 0`` to avoid overwriting the root entry with a
        secondary PV's score.
        """
        if not candidates:
            return None, -INF

        if pv_idx == 0:
            tt_entry = self.tt.probe(state.zobrist_hash())
            ordered = self.move_ordering.order_moves(
                state, candidates, depth, tt_entry,
            )
        else:
            # For non-primary PVs the TT best_move is PV1 (excluded from
            # candidates), so passing it would waste the TT-move ordering
            # slot. Fall back to heuristic ordering.
            ordered = self.move_ordering.order_moves(
                state, candidates, depth, None,
            )

        # Pin the previous iteration's PV for this slot as the first move.
        if pv_idx < len(prev_pv_scores):
            preferred = prev_pv_scores[pv_idx][0]
            if preferred in ordered:
                ordered.remove(preferred)
                ordered.insert(0, preferred)

        orig_alpha = alpha
        best_move = ordered[0]
        best_score = -INF

        for i, move in enumerate(ordered):
            if self._time_up:
                break
            if self._stop_event and self._stop_event.is_set():
                self._time_up = True
                break

            if i == 0:
                score = self._do_move_and_search(
                    state, move, depth - 1, alpha, beta,
                )
            else:
                score = self._do_move_and_search(
                    state, move, depth - 1, alpha, alpha + 1,
                )
                if alpha < score < beta and not self._time_up:
                    score = self._do_move_and_search(
                        state, move, depth - 1, alpha, beta,
                    )

            if self._time_up:
                break

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if score >= beta:
                break

        if pv_idx == 0 and not self._time_up:
            if best_score <= orig_alpha:
                flag = ALPHA
            elif best_score >= beta:
                flag = BETA
            else:
                flag = EXACT
            self.tt.store(
                state.zobrist_hash(), depth, best_score, flag, best_move,
            )

        return best_move, best_score

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
