"""Game session: manages per-game state and the main game loop."""

import threading
from typing import Dict, List, Optional

import pygame

from src.games.base import WHITE, BLACK
from src.gui.constants import BG, DEPTH_MAX
from src.gui.panel import draw_panel, eval_white_pov
from src.utils.config import create_game


class GameSession:
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

        # Analysis
        self.analysis_live: list = [None]
        self.analysis_stop = threading.Event()
        self.analysis_state_id: Optional[int] = None
        self.hint_moves: list = []
        self.hint_depth = 0
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
        self.analysis_stop.set()
        self.analysis_live = [None]
        self.analysis_state_id = None
        self.analysis_stop = threading.Event()

    def _launch_analysis(self):
        self._cancel_analysis()
        self.analysis_state_id = id(self.state)
        if self.state.is_terminal():
            return
        if (self.app.mode == "human-vs-ai"
                and self.state.side_to_move() == self.ai_side):
            return
        fresh_live = [None]
        self.analysis_live = fresh_live
        _st = self.state.copy()
        _stop = self.analysis_stop
        _ev = self.app._make_evaluator(
            self.app.game_name, self.app.model_path)
        from src.search.alphabeta import AlphaBetaSearch
        _hs = AlphaBetaSearch(_ev, max_depth=DEPTH_MAX,
                              time_limit_ms=600_000)

        def _worker():
            _hs.search_top_n_live(_st, n=3, live_ref=fresh_live,
                                  stop_event=_stop)
        threading.Thread(target=_worker, daemon=True).start()

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
            self.app.analysis_on = not self.app.analysis_on
            if self.app.analysis_on:
                self._launch_analysis()
            else:
                self._cancel_analysis()
                self.hint_moves = []
                self.hint_depth = 0
                self.hint_done = False
        return None

    def _handle_click(self, pos) -> Optional[str]:
        mx, my = pos

        # Panel clicks
        for key, rect in self.panel_rects.items():
            if not rect.collidepoint(mx, my):
                continue
            if key == "hint":
                self.app.analysis_on = not self.app.analysis_on
                if self.app.analysis_on:
                    self._launch_analysis()
                else:
                    self._cancel_analysis()
                    self.hint_moves = []
                    self.hint_depth = 0
                    self.hint_done = False
            elif key == "restart":
                self._restart()
            elif key == "menu":
                self._cancel_analysis()
                return "menu"
            return None

        # Guard: not interactable
        if self.state.is_terminal() or self.ai_thinking:
            return None
        is_human = (
            self.app.mode == "human-vs-human"
            or (self.app.mode == "human-vs-ai"
                and self.state.side_to_move() != self.ai_side)
        )
        if not is_human:
            return None

        self._handle_board_click(mx, my)
        return None

    def _handle_board_click(self, mx, my):
        # Hand piece click (shogi)
        if self.has_hand and self.selected_sq is None:
            side = self.state.side_to_move()
            hr = (self.sente_hand_rect if side == WHITE
                  else self.gote_hand_rect)
            if hr and hr.collidepoint(mx, my):
                hp = self.renderer.hand_piece_at(
                    mx - hr.x, my - hr.y, self.state, side, hr)
                if hp is not None:
                    self.selected_hand_piece = hp
                    self.selected_sq = None
                    self.legal_targets = [
                        m.to_sq for m in self.state.legal_moves()
                        if m.drop_piece == hp
                    ]
                    return

        bx, by = mx - self.board_x, my - self.board_y
        clicked_sq = self.renderer.pixel_to_sq(bx, by)
        if clicked_sq is None:
            return

        if self.selected_hand_piece is not None:
            if clicked_sq in self.legal_targets:
                for m in self.state.legal_moves():
                    if (m.drop_piece == self.selected_hand_piece
                            and m.to_sq == clicked_sq):
                        self.state = self.state.make_move(m)
                        self.score_history.append(
                            eval_white_pov(self.state))
                        self.hint_moves = []
                        break
            self.selected_hand_piece = None
            self.selected_sq = None
            self.legal_targets = []

        elif self.selected_sq is None:
            board = self.state.board_array()
            piece = board[clicked_sq]
            if piece != 0:
                own = ((piece > 0 and self.state.side_to_move() == WHITE)
                       or (piece < 0
                           and self.state.side_to_move() == BLACK))
                if own:
                    self.selected_sq = clicked_sq
                    self.legal_targets = [
                        m.to_sq for m in self.state.legal_moves()
                        if m.from_sq == clicked_sq
                    ]

        elif clicked_sq == self.selected_sq:
            self.selected_sq = None
            self.legal_targets = []

        elif clicked_sq in self.legal_targets:
            matching = [m for m in self.state.legal_moves()
                        if m.from_sq == self.selected_sq
                        and m.to_sq == clicked_sq]
            if len(matching) == 1:
                self.state = self.state.make_move(matching[0])
                self.score_history.append(eval_white_pov(self.state))
            elif len(matching) > 1:
                promo = [m for m in matching if m.promotion is not None]
                self.state = self.state.make_move(
                    promo[0] if promo else matching[0])
                self.score_history.append(eval_white_pov(self.state))
            self.hint_moves = []
            self.selected_sq = None
            self.legal_targets = []
        else:
            self.selected_sq = clicked_sq
            self.legal_targets = [
                m.to_sq for m in self.state.legal_moves()
                if m.from_sq == clicked_sq
            ]

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
        if not self.app.analysis_on:
            return

        snap = self.analysis_live[0]
        if snap is not None and self.analysis_state_id == id(self.state):
            d, md, moves, done = snap
            self.hint_moves = moves
            self.hint_depth = d
            self.hint_done = done

        if self.analysis_state_id != id(self.state):
            self.hint_moves = []
            self.hint_depth = 0
            self.hint_done = False
            self._launch_analysis()

        # Live-update chart with analysis eval
        if self.hint_moves and self.score_history:
            best = self.hint_moves[0][1]
            white_s = (best if self.state.side_to_move() == WHITE
                       else -best)
            self.score_history[-1] = white_s / 100.0

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
