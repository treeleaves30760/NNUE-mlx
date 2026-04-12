"""Game session: manages per-game state and the main game loop."""

import threading
from typing import Dict, List, Optional

import pygame

from src.games.base import GameState, WHITE, BLACK
from src.gui._board_interaction import _BoardInteractionMixin
from src.gui.analysis import AnalysisController
from src.gui.constants import BG, DEPTH_MAX
from src.gui.panel import draw_panel, eval_white_pov
from src.utils.config import create_game


class GameSession(_BoardInteractionMixin):
    """Encapsulates all state and logic for a single game session."""

    def __init__(self, app):
        self.app = app
        self.state = create_game(app.game_name)
        self.renderer = app._create_renderer(app.game_name)
        self.has_hand = self.state.config().has_drops

        # AI
        self.ai_searcher = None
        self.ai_searcher_2 = None
        self.ai_side = BLACK
        self.ai_thinking = False
        self.ai_result: list = [None]

        # Board interaction
        self.selected_sq: Optional[int] = None
        self.legal_targets: List[int] = []
        self.selected_hand_piece: Optional[int] = None
        self.status_text = ""
        self.score_history: List[float] = [eval_white_pov(self.state)]

        # Analysis: the controller owns worker lifecycle and commit logic.
        # hint_* fields are the display-facing mirror of controller.committed,
        # refreshed once per frame in _update_analysis.
        self.analysis = AnalysisController(
            search_factory=self._build_analysis_search,
        )
        self.hint_moves: list = []
        self.hint_depth = 0
        self.hint_max_depth = 0
        self.hint_done = False

        # Panel clickable rects (populated after first draw)
        self.panel_rects: Dict[str, pygame.Rect] = {}

        # Layout
        self.board_x = 30
        if self.has_hand:
            hand_h, hand_gap = 50, 5
            self.gote_hand_rect = pygame.Rect(
                self.board_x, 30, self.renderer.board_pixel_w, hand_h)
            self.board_y = 30 + hand_h + hand_gap
            self.sente_hand_rect = pygame.Rect(
                self.board_x,
                self.board_y + self.renderer.board_pixel_h + hand_gap,
                self.renderer.board_pixel_w, hand_h)
        else:
            self.board_y = 30
            self.gote_hand_rect = None
            self.sente_hand_rect = None
        self.panel_x = self.board_x + self.renderer.board_pixel_w + 20
        self.panel_w = app.W - self.panel_x - 10

        if app.mode in ("human-vs-ai", "ai-vs-ai"):
            self._ensure_ai()

    # -------------------------------------------------- AI management
    def _ensure_ai(self):
        if self.ai_searcher is None:
            _, self.ai_searcher = self.app._create_ai(self.app.model_path)
        self.ai_searcher.max_depth = self.app.ai_depth
        self.ai_searcher.time_limit_ms = self.app.ai_time_sec * 1000
        if self.app.mode == "ai-vs-ai":
            if self.ai_searcher_2 is None:
                _, self.ai_searcher_2 = self.app._create_ai(
                    self.app.model_path_2)
            self.ai_searcher_2.max_depth = self.app.ai_depth
            self.ai_searcher_2.time_limit_ms = self.app.ai_time_sec * 1000

    # -------------------------------------------------- Analysis
    def _cancel_analysis(self):
        """Stop any running analysis worker and forget its output."""
        self.analysis.cancel()

    def _launch_analysis(self):
        """Kick off a new analysis run for the current position."""
        self.analysis.launch(self.state)

    def _build_analysis_search(self, state: GameState):
        """Factory used by AnalysisController to build a search engine.

        Returns ``None`` when analysis should be suppressed for the given
        state — e.g. when it's the AI's turn in human-vs-AI mode, where
        the AI searcher already owns the thinking budget. The controller
        treats ``None`` as "leave committed state empty, spawn no worker".
        """
        if (self.app.mode == "human-vs-ai"
                and state.side_to_move() == self.ai_side):
            return None

        # Shogi without an NNUE model: use the C-backed rule search in
        # infinite-depth mode so the GUI keeps deepening until cancelled.
        # Everything else goes through the Python AlphaBetaSearch with
        # whichever evaluator the app configured.
        if self.app.game_name == "shogi" and not self.app.model_path:
            from src.search.alphabeta import create_rule_based_search
            return create_rule_based_search(
                "shogi", max_depth=0, time_limit_ms=0,
            )

        evaluator = self.app._make_evaluator(
            self.app.game_name, self.app.model_path,
        )
        from src.search.alphabeta import AlphaBetaSearch
        return AlphaBetaSearch(
            evaluator, max_depth=DEPTH_MAX, time_limit_ms=600_000,
        )

    # -------------------------------------------------- Restart
    def _restart(self):
        self.state = create_game(self.app.game_name)
        self.score_history = [eval_white_pov(self.state)]
        self.selected_sq = None
        self.legal_targets = []
        self.selected_hand_piece = None
        self.hint_moves = []
        self.hint_depth = 0
        self.hint_done = False
        self.status_text = ""
        if self.app.analysis_on:
            self._launch_analysis()

    # -------------------------------------------------- Events
    def _handle_events(self) -> Optional[str]:
        """Process pygame events. Returns 'quit', 'menu', or None."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._cancel_analysis()
                return "quit"

            if event.type == pygame.KEYDOWN:
                r = self._handle_key(event.key)
                if r:
                    return r

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                r = self._handle_click(event.pos)
                if r:
                    return r
        return None

    def _handle_key(self, key) -> Optional[str]:
        if key in (pygame.K_ESCAPE, pygame.K_m):
            self._cancel_analysis()
            return "menu"
        if key == pygame.K_r:
            self._restart()
        if key == pygame.K_h:
            # Flip the toggle; _update_analysis handles both launch and
            # teardown based on app.analysis_on in the same frame.
            self.app.analysis_on = not self.app.analysis_on
        return None

    # -------------------------------------------------- Update

    def _update_ai(self):
        # Pick the active searcher
        if self.app.mode == "ai-vs-ai":
            _active = (self.ai_searcher
                       if self.state.side_to_move() == WHITE
                       else (self.ai_searcher_2 or self.ai_searcher))
        else:
            _active = self.ai_searcher

        # Launch AI search if needed
        if (not self.state.is_terminal() and not self.ai_thinking
                and _active is not None):
            should = (
                (self.app.mode == "human-vs-ai"
                 and self.state.side_to_move() == self.ai_side)
                or self.app.mode == "ai-vs-ai"
            )
            if should:
                self.ai_thinking = True
                if self.app.mode == "ai-vs-ai":
                    sn = ("White" if self.state.side_to_move() == WHITE
                          else "Black")
                    self.status_text = f"{sn} AI thinking..."
                else:
                    self.status_text = "AI thinking..."
                _st2 = self.state.copy()
                _s = _active

                def _ai():
                    self.ai_result[0] = _s.search(_st2)
                threading.Thread(target=_ai, daemon=True).start()
        # Collect AI result
        if self.ai_thinking and self.ai_result[0] is not None:
            move, score = self.ai_result[0]
            self.ai_result[0] = None
            self.ai_thinking = False
            still_ai = (
                (self.app.mode == "human-vs-ai"
                 and self.state.side_to_move() == self.ai_side)
                or self.app.mode == "ai-vs-ai"
            )
            if still_ai and move is not None:
                self.state = self.state.make_move(move)
                self.score_history.append(eval_white_pov(self.state))
                self.hint_moves = []
                self.status_text = f"AI: {move} (score: {score:.0f})"
            elif move is None:
                self.status_text = "AI: no move found"

    def _update_analysis(self):
        """Drive the analysis controller and mirror its committed state.

        Runs every frame. Responsibilities (in order):
          1. If analysis is off, tear down any running worker and clear
             the display mirror so stale hints don't linger.
          2. If the controller is running for a different state object
             (because the player or AI just moved), relaunch it for the
             current position.
          3. Ingest the latest live snapshot; if that produces a *fresh*
             first commit for this ply, seed the score chart once.
          4. Mirror ``controller.committed`` onto ``hint_*`` fields so
             the draw path has simple, stable data to render.
        """
        if not self.app.analysis_on:
            if self.analysis.committed is not None or self.hint_moves:
                self.analysis.cancel()
                self._clear_hint_mirror()
            return

        if not self.analysis.is_for(self.state):
            self.analysis.launch(self.state)
            self._clear_hint_mirror()
            return

        fresh_commit = self.analysis.update(self.state)
        if fresh_commit is not None and self.score_history:
            # First commit of this ply: replace the material-only seed
            # at score_history[-1] with the deep-search evaluation.
            white_score = (fresh_commit.best_score
                           if self.state.side_to_move() == WHITE
                           else -fresh_commit.best_score)
            self.score_history[-1] = white_score / 100.0

        committed = self.analysis.committed
        if committed is None:
            self._clear_hint_mirror()
        else:
            self.hint_moves = committed.moves
            self.hint_depth = committed.depth
            self.hint_max_depth = committed.max_depth
            self.hint_done = committed.done

    def _clear_hint_mirror(self):
        self.hint_moves = []
        self.hint_depth = 0
        self.hint_max_depth = 0
        self.hint_done = False

    def _check_terminal(self):
        if not self.state.is_terminal():
            return
        result = self.state.result()
        if result == 1.0:
            stm = ("White" if self.state.side_to_move() == WHITE
                   else "Black")
            self.status_text = f"{stm} wins!"
        elif result == 0.0:
            stm = ("White" if self.state.side_to_move() == WHITE
                   else "Black")
            other = "Black" if stm == "White" else "White"
            self.status_text = f"{other} wins!"
        else:
            self.status_text = "Draw!"

    # -------------------------------------------------- Draw

    def _draw(self):
        screen = self.app.screen
        screen.fill(BG)

        if self.has_hand:
            self.renderer.draw_hand(
                screen, self.state, BLACK, self.gote_hand_rect)
            self.renderer.draw_hand(
                screen, self.state, WHITE, self.sente_hand_rect)

        bsurf = pygame.Surface(
            (self.renderer.board_pixel_w, self.renderer.board_pixel_h))
        self.renderer.draw_board(
            bsurf, self.state, self.selected_sq, self.legal_targets)
        if self.hint_moves:
            self.renderer.draw_hints(bsurf, self.hint_moves)
        screen.blit(bsurf, (self.board_x, self.board_y))

        if self.hint_moves and self.has_hand:
            self.renderer.draw_drop_hints(
                screen, self.hint_moves, self.state,
                (self.board_x, self.board_y),
                self.sente_hand_rect, self.gote_hand_rect)

        self.panel_rects = draw_panel(
            self.app, self.panel_x, 30, self.panel_w,
            self.state, self.hint_moves, self.ai_thinking,
            self.status_text, self.app.analysis_on,
            self.hint_depth, self.hint_done, self.score_history,
            hint_max_depth=self.hint_max_depth,
        )
        pygame.display.flip()

    # -------------------------------------------------- Main loop

    def run(self) -> str:
        """Run the game loop. Returns 'menu' or 'quit'."""
        while True:
            result = self._handle_events()
            if result:
                return result
            self._update_ai()
            self._update_analysis()
            self._check_terminal()
            self._draw()
            self.app.clock.tick(60)
