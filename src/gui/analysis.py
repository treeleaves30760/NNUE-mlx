"""Background analysis controller with stability-based commit.

Problem this module solves
--------------------------
Iterative-deepening MultiPV search publishes a fresh top-N snapshot after
every completed depth. Showing every one of those snapshots on screen is
what causes the hint arrows, score text, and chart to flicker: at shallow
depths the best move can legitimately change from one iteration to the
next, and every change triggers a redraw.

The user's primary goal is *finding the best next move reliably*, not
showing every intermediate search state. So the controller gates display
updates on a stability criterion instead of raw depth progress:

  A snapshot is **committed** to the display when either
    (1) ``done == True`` (the search has finished), or
    (2) the iteration depth has reached ``MIN_COMMIT_DEPTH`` *and* the
        top-1 move has been chosen by ``STABILITY_COUNT`` consecutive
        completed iterations.

Once committed, subsequent iterations either
  * refine the committed snapshot in place if the top-1 move is unchanged
    (panel scores update, no arrow movement), or
  * supersede the committed snapshot if a *different* top-1 move proves
    stable across ``STABILITY_COUNT`` iterations.

Mid-iteration publishes (same depth as last, not done) are discarded so
the controller never ingests partial progress.

Lifecycle
---------
The controller owns the worker thread and its stop event, so callers do
not need to juggle threading primitives. Typical usage:

    controller = AnalysisController(create_search_fn=my_factory)
    controller.launch(state)                # start a worker
    while True:
        if not controller.is_for(state):
            controller.launch(state)        # position changed
        fresh = controller.update(state)    # process latest snapshot
        if fresh is not None:
            seed_score_chart(fresh)         # fresh commit this frame
        draw(controller.committed)          # current display state

``update()`` returns non-None *only on the first commit per ply*, which
is exactly the moment that deserves side effects like writing the score
chart's current point. Refinements after the first commit update
``.committed`` but do not re-fire the signal.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import threading

from src.games.base import GameState, Move


@dataclass(frozen=True)
class HintSnapshot:
    """An immutable top-N hint set that has passed the commit criteria."""
    depth: int
    max_depth: int
    moves: List[Tuple[Move, float]]
    done: bool

    @property
    def best_move(self) -> Optional[Move]:
        return self.moves[0][0] if self.moves else None

    @property
    def best_score(self) -> float:
        return self.moves[0][1] if self.moves else 0.0


# Type alias: a factory takes a state and returns a search engine exposing
# ``search_top_n_live(state, n, live_ref, stop_event)``, or ``None`` when
# analysis should be suppressed for this state (e.g. AI's turn in H-vs-AI).
SearchFactory = Callable[[GameState], Optional[object]]


class AnalysisController:
    """Owns the analysis worker thread and the commit-decision logic."""

    #: Minimum depth an iteration must reach before it can be committed.
    #: Tuned so the first commit happens within a second or two on chess
    #: rule-based at typical hardware, while still giving the search enough
    #: room to correct shallow-depth mistakes.
    MIN_COMMIT_DEPTH: int = 4

    #: Number of consecutive iterations whose top-1 move must agree before
    #: a commit fires. 2 means "depth D and depth D-1 both picked this
    #: same move", which is a mild but effective flicker filter.
    STABILITY_COUNT: int = 2

    def __init__(self, search_factory: SearchFactory):
        self._search_factory = search_factory
        self._live: list = [None]
        self._stop: threading.Event = threading.Event()
        self._stop.set()
        self._position_id: Optional[int] = None
        self._iter_history: List[Tuple[int, Move, float]] = []
        self._committed: Optional[HintSnapshot] = None
        self._latest: Optional[HintSnapshot] = None
        self._last_sig: Optional[tuple] = None

    # ------------------------------------------------------------------ API

    @property
    def committed(self) -> Optional[HintSnapshot]:
        """Current hint snapshot to display. ``None`` before first commit.

        This is the *stable* view — gated on depth + move stability so
        the GUI never flickers. Use :attr:`latest` instead when you
        need the raw latest-computed best without the stability filter
        (e.g. programmatic "what's the best move right now?" queries).
        """
        return self._committed

    @property
    def latest(self) -> Optional[HintSnapshot]:
        """Most recent ingested snapshot, regardless of commit state.

        Unlike :attr:`committed`, this reflects every partial publish
        the search worker has made — including PV1 mid-iteration. Use
        it when the caller needs an always-current best move and can
        tolerate intra-iteration reordering (programmatic clients,
        bootstrap fallbacks, debug overlays). Display code should
        almost always prefer :attr:`committed` for stability.
        """
        return self._latest

    @property
    def latest_best_move(self) -> Optional[Move]:
        """Raw best-move shortcut. Prefers latest, falls back to committed."""
        if self._latest is not None and self._latest.moves:
            return self._latest.moves[0][0]
        if self._committed is not None and self._committed.moves:
            return self._committed.moves[0][0]
        return None

    def is_for(self, state: GameState) -> bool:
        """Does the running worker belong to this specific state object?"""
        return self._position_id is not None and id(state) == self._position_id

    def launch(self, state: GameState) -> None:
        """Cancel any prior analysis and start a new worker for ``state``.

        If the factory returns ``None`` for this state (e.g. it's the AI's
        turn and we don't want analysis running), the controller is left
        in an empty-committed state and no worker thread is spawned.
        """
        self.cancel()
        self._position_id = id(state)
        self._iter_history = []
        self._committed = None
        self._latest = None
        self._last_sig = None
        self._live = [None]
        self._stop = threading.Event()

        if state.is_terminal():
            return

        search = self._search_factory(state)
        if search is None:
            return

        state_copy = state.copy()
        live = self._live
        stop = self._stop

        def _worker():
            search.search_top_n_live(
                state_copy, n=3, live_ref=live, stop_event=stop,
            )

        threading.Thread(target=_worker, daemon=True).start()

    def cancel(self) -> None:
        """Signal the worker to stop and forget its output."""
        self._stop.set()
        self._live = [None]
        self._position_id = None
        self._iter_history = []
        self._committed = None
        self._latest = None
        self._last_sig = None

    def update(self, current_state: GameState) -> Optional[HintSnapshot]:
        """Ingest the latest live snapshot and apply commit logic.

        Call every frame. Returns a :class:`HintSnapshot` **only** when a
        brand-new commit fires on this call — i.e. on the first commit per
        ply. Refinements and no-op ingests return ``None``. Use this return
        value to seed side effects (score chart, sound cues, …) that
        should only happen once per ply. For the current display state,
        read :attr:`committed` instead.
        """
        if not self.is_for(current_state):
            return None

        raw = self._live[0]
        if raw is None:
            return None

        try:
            d, md, moves, done = raw
        except (TypeError, ValueError):
            return None
        if d <= 0 or not moves:
            return None

        sig = (d, bool(done), tuple((m, round(s, 3)) for m, s in moves))
        if sig == self._last_sig:
            return None
        self._last_sig = sig

        best_move, best_score = moves[0]

        # Always refresh the "latest" snapshot — this tracks whatever
        # the search has most recently produced, including partial
        # MultiPV sets where only PV1 is known so far. Callers asking
        # "what's the best move right now?" read this via .latest.
        self._latest = HintSnapshot(
            depth=d,
            max_depth=md,
            moves=list(moves),
            done=bool(done),
        )

        # History: one entry per distinct depth, later publishes at the
        # same depth replace the previous entry (the done=True signal
        # supersedes the earlier not-done).
        if self._iter_history and self._iter_history[-1][0] >= d:
            if self._iter_history[-1][0] > d:
                # Depth regressed (shouldn't happen for a sane search) —
                # ignore rather than corrupt history.
                return None
            self._iter_history[-1] = (d, best_move, best_score)
        else:
            self._iter_history.append((d, best_move, best_score))

        was_uncommitted = self._committed is None
        should_commit = self._evaluate_commit_criteria(
            done=done, best_move=best_move,
        )

        if not should_commit:
            return None

        self._committed = self._latest
        return self._committed if was_uncommitted else None

    # ------------------------------------------------------------------ impl

    def _evaluate_commit_criteria(
        self, *, done: bool, best_move: Move,
    ) -> bool:
        """Decide whether the latest history entry warrants a commit.

        Three independent acceptance paths:
          1. The search is done — always commit the final state.
          2. Nothing committed yet — need ``MIN_COMMIT_DEPTH`` + stability.
          3. Already committed — refine if the top move is unchanged, or
             supersede if a *different* top move proves stable.
        """
        if done:
            return True

        latest_depth = self._iter_history[-1][0]

        if self._committed is None:
            return (
                latest_depth >= self.MIN_COMMIT_DEPTH
                and self._is_top_move_stable(best_move)
            )

        if best_move == self._committed.best_move:
            # Same move across iterations → safe refinement.
            return True

        # A different move claims the top slot: require stability before
        # we swap it in, so one noisy iteration can't move the arrow.
        return (
            latest_depth >= self.MIN_COMMIT_DEPTH
            and self._is_top_move_stable(best_move)
        )

    def _is_top_move_stable(self, move: Move) -> bool:
        recent = self._iter_history[-self.STABILITY_COUNT:]
        if len(recent) < self.STABILITY_COUNT:
            return False
        return all(entry[1] == move for entry in recent)
