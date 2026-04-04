"""Main Pygame application: settings menu, game loop, and in-game panel."""

import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pygame

from src.games.base import GameState, Move, WHITE, BLACK
from src.utils.config import create_game

# ------------------------------------------------------------------ palette
_BG = (49, 46, 43)
_PANEL_BG = (58, 55, 52)
_TEXT = (220, 220, 220)
_TEXT_DIM = (140, 140, 140)
_TEXT_MUTED = (100, 100, 100)
_ACCENT = (100, 160, 100)
_ACCENT_HI = (120, 180, 120)
_BTN = (70, 90, 70)
_BTN_HI = (90, 120, 90)
_BTN_SEL = (60, 130, 60)
_BTN_BORDER = (100, 140, 100)
_SEP = (80, 75, 70)

# ------------------------------------------------------------------ options
_GAMES = [
    ("Chess (8\u00d78)", "chess"),
    ("Los Alamos (6\u00d76)", "minichess"),
    ("Shogi (9\u00d79)", "shogi"),
    ("Mini Shogi (5\u00d75)", "minishogi"),
]
_MODES = [
    ("Human vs Human", "human-vs-human"),
    ("Human vs AI", "human-vs-ai"),
    ("AI vs AI", "ai-vs-ai"),
]
_MODE_SHORT = {"human-vs-human": "HvH", "human-vs-ai": "HvAI", "ai-vs-ai": "AvA"}
_DEPTH_MIN, _DEPTH_MAX = 1, 10
_TIME_STEPS = [1, 2, 3, 5, 10, 15, 30]


def _time_idx(sec: int) -> int:
    if sec in _TIME_STEPS:
        return _TIME_STEPS.index(sec)
    return 3  # default to 5s


class GameApp:
    """Main application with settings menu and in-game control panel."""

    W = 1100
    H = 780

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("NNUE-mps Board Games (MLX)")
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.clock = pygame.time.Clock()

        self.title_font = pygame.font.SysFont("Arial", 44, bold=True)
        self.heading_font = pygame.font.SysFont("Arial", 22, bold=True)
        self.font = pygame.font.SysFont("Arial", 20)
        self.small_font = pygame.font.SysFont("Arial", 16)

        # Persistent settings
        self.game_name = "chess"
        self.mode = "human-vs-human"
        self.ai_depth = 4
        self.ai_time_sec = 5
        self.model_path: Optional[str] = None
        self.analysis_on = False  # persistent analysis toggle

    # ============================================================== public
    def run(self, game_name=None, mode=None, model_path=None,
            ai_depth=None, ai_time_limit=None):
        if game_name:
            self.game_name = game_name
        if mode:
            self.mode = mode
        if model_path:
            self.model_path = model_path
        if ai_depth is not None:
            self.ai_depth = ai_depth
        if ai_time_limit is not None:
            self.ai_time_sec = max(1, ai_time_limit // 1000)

        while True:
            action = self._run_menu()
            if action == "quit":
                break
            result = self._run_game()
            if result == "quit":
                break
        pygame.quit()

    # ============================================================== MENU
    def _run_menu(self) -> str:
        """Settings menu. Returns 'start' or 'quit'."""
        models = self._scan_models()

        while True:
            mouse = pygame.mouse.get_pos()
            self.screen.fill(_BG)
            cx = self.W // 2

            # Title
            self._blit_centered(self.title_font, "NNUE-mps", cx, 45, _TEXT)
            self._blit_centered(self.small_font,
                                "Board Game AI with Apple MLX Training",
                                cx, 88, _TEXT_DIM)

            y = 130
            # ---- SELECT GAME ----
            self._section_label("SELECT GAME", cx, y); y += 28
            game_rects: List[Tuple[pygame.Rect, str]] = []
            for i, (label, key) in enumerate(_GAMES):
                col, row = i % 2, i // 2
                r = pygame.Rect(cx - 270 + col * 280, y + row * 50, 260, 42)
                game_rects.append((r, key))
                self._menu_btn(r, label, key == self.game_name, r.collidepoint(mouse))
            y += 112

            # ---- GAME MODE ----
            self._section_label("GAME MODE", cx, y); y += 28
            mode_rects: List[Tuple[pygame.Rect, str]] = []
            bw = 172
            mx_start = cx - (bw * 3 + 16) // 2
            for i, (label, key) in enumerate(_MODES):
                r = pygame.Rect(mx_start + i * (bw + 8), y, bw, 38)
                mode_rects.append((r, key))
                self._menu_btn(r, label, key == self.mode, r.collidepoint(mouse))
            y += 56

            # ---- AI SETTINGS ----
            self._section_label("AI SETTINGS", cx, y); y += 30
            d_minus, d_plus = self._stepper(cx - 240, y, "Search Depth",
                                            str(self.ai_depth), 210)
            t_minus, t_plus = self._stepper(cx + 30, y, "Time Limit",
                                            f"{self.ai_time_sec}s", 210)
            y += 52

            # ---- ANALYSIS ----
            analysis_label = "Analysis: ON" if self.analysis_on else "Analysis: OFF"
            analysis_r = pygame.Rect(cx - 100, y, 200, 38)
            self._menu_btn(analysis_r, analysis_label,
                           self.analysis_on, analysis_r.collidepoint(mouse))
            y += 56

            # ---- MODEL ----
            self._section_label("MODEL", cx, y); y += 28
            model_opts: List[Tuple[str, Optional[str]]] = [
                ("None (Material Only)", None)]
            for p in models:
                model_opts.append((os.path.basename(p), p))
            # Find current index
            model_idx = 0
            for i, (_, p) in enumerate(model_opts):
                if p == self.model_path:
                    model_idx = i
                    break
            cur_label = model_opts[model_idx][0]
            m_minus, m_plus = self._stepper(cx - 240, y, "Model File",
                                            cur_label, 480)
            y += 52

            # ---- START ----
            start_r = pygame.Rect(cx - 150, y, 300, 54)
            hover = start_r.collidepoint(mouse)
            pygame.draw.rect(self.screen, _ACCENT_HI if hover else _ACCENT,
                             start_r, border_radius=10)
            lbl = self.heading_font.render(">>  START GAME", True, (255, 255, 255))
            self.screen.blit(lbl, lbl.get_rect(center=start_r.center))

            self._blit_centered(self.small_font, "Press ENTER to start \u2022 ESC to quit",
                                cx, self.H - 28, _TEXT_MUTED)
            pygame.display.flip()

            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "quit"
                    if event.key == pygame.K_RETURN:
                        return "start"
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    for r, k in game_rects:
                        if r.collidepoint(pos):
                            self.game_name = k
                    for r, k in mode_rects:
                        if r.collidepoint(pos):
                            self.mode = k
                    if analysis_r.collidepoint(pos):
                        self.analysis_on = not self.analysis_on
                    if m_minus.collidepoint(pos):
                        model_idx = max(0, model_idx - 1)
                        self.model_path = model_opts[model_idx][1]
                    if m_plus.collidepoint(pos):
                        model_idx = min(len(model_opts) - 1, model_idx + 1)
                        self.model_path = model_opts[model_idx][1]
                    if d_minus.collidepoint(pos):
                        self.ai_depth = max(_DEPTH_MIN, self.ai_depth - 1)
                    if d_plus.collidepoint(pos):
                        self.ai_depth = min(_DEPTH_MAX, self.ai_depth + 1)
                    if t_minus.collidepoint(pos):
                        i = _time_idx(self.ai_time_sec)
                        self.ai_time_sec = _TIME_STEPS[max(0, i - 1)]
                    if t_plus.collidepoint(pos):
                        i = _time_idx(self.ai_time_sec)
                        self.ai_time_sec = _TIME_STEPS[min(len(_TIME_STEPS) - 1, i + 1)]
                    if start_r.collidepoint(pos):
                        return "start"
            self.clock.tick(60)

    # ============================================================== GAME
    def _run_game(self) -> str:
        """Game loop. Returns 'menu' or 'quit'."""
        state = create_game(self.game_name)
        renderer = self._create_renderer(self.game_name)
        has_hand = state.config().has_drops

        ai_searcher = None
        ai_side = BLACK

        def _ensure_ai():
            nonlocal ai_searcher
            if ai_searcher is None:
                _, ai_searcher = self._create_ai()
            ai_searcher.max_depth = self.ai_depth
            ai_searcher.time_limit_ms = self.ai_time_sec * 1000

        if self.mode in ("human-vs-ai", "ai-vs-ai"):
            _ensure_ai()

        selected_sq: Optional[int] = None
        legal_targets: List[int] = []
        selected_hand_piece: Optional[int] = None
        status_text = ""
        ai_thinking = False
        ai_result: list = [None]
        score_history: List[float] = [self._eval_white_pov(state)]

        # Live analysis state
        current_analysis_live: list = [None]
        analysis_stop = threading.Event()
        analysis_state_id: Optional[int] = None
        hint_moves: list = []
        hint_depth = 0
        hint_done = False

        def _cancel_analysis():
            nonlocal analysis_stop, analysis_state_id, current_analysis_live
            analysis_stop.set()
            current_analysis_live = [None]
            analysis_state_id = None
            analysis_stop = threading.Event()

        def _launch_analysis():
            nonlocal analysis_state_id, current_analysis_live
            _cancel_analysis()
            analysis_state_id = id(state)
            if state.is_terminal():
                return
            if (self.mode == "human-vs-ai"
                    and state.side_to_move() == ai_side):
                return
            fresh_live = [None]
            current_analysis_live = fresh_live
            _st, _stop = state, analysis_stop
            _ev = self._make_evaluator(self.game_name, self.model_path)
            from src.search.alphabeta import AlphaBetaSearch
            _hs = AlphaBetaSearch(_ev, max_depth=_DEPTH_MAX,
                                  time_limit_ms=600_000)
            def _worker():
                _hs.search_top_n_live(_st, n=3, live_ref=fresh_live,
                                      stop_event=_stop)
            threading.Thread(target=_worker, daemon=True).start()

        # Layout
        board_x = 30
        if has_hand:
            hand_h = 50
            hand_gap = 5
            gote_hand_rect = pygame.Rect(
                board_x, 30, renderer.board_pixel_w, hand_h)
            board_y = 30 + hand_h + hand_gap
            sente_hand_rect = pygame.Rect(
                board_x, board_y + renderer.board_pixel_h + hand_gap,
                renderer.board_pixel_w, hand_h)
        else:
            board_y = 30
            gote_hand_rect = None
            sente_hand_rect = None
        panel_x = board_x + renderer.board_pixel_w + 20
        panel_w = self.W - panel_x - 10

        # Clickable rects from panel (populated after first draw)
        panel_rects: Dict[str, pygame.Rect] = {}

        while True:
            # ============================================ EVENTS
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    _cancel_analysis()
                    return "quit"

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        _cancel_analysis()
                        return "menu"
                    if event.key == pygame.K_m:
                        _cancel_analysis()
                        return "menu"
                    if event.key == pygame.K_r:
                        state = create_game(self.game_name)
                        score_history = [self._eval_white_pov(state)]
                        selected_sq = None; legal_targets = []
                        selected_hand_piece = None
                        hint_moves = []; hint_depth = 0; hint_done = False
                        status_text = ""
                        if self.analysis_on:
                            _launch_analysis()
                    if event.key == pygame.K_h:
                        self.analysis_on = not self.analysis_on
                        if self.analysis_on:
                            _launch_analysis()
                        else:
                            _cancel_analysis()
                            hint_moves = []; hint_depth = 0; hint_done = False

                if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
                    continue

                mx, my = event.pos
                handled = False

                # ---- panel clicks ----
                for key, rect in panel_rects.items():
                    if not rect.collidepoint(mx, my):
                        continue
                    handled = True
                    if key.startswith("mode:"):
                        self.mode = key[5:]
                        if self.mode in ("human-vs-ai", "ai-vs-ai"):
                            _ensure_ai()
                    elif key == "depth-":
                        self.ai_depth = max(_DEPTH_MIN, self.ai_depth - 1)
                        if ai_searcher:
                            ai_searcher.max_depth = self.ai_depth
                    elif key == "depth+":
                        self.ai_depth = min(_DEPTH_MAX, self.ai_depth + 1)
                        if ai_searcher:
                            ai_searcher.max_depth = self.ai_depth
                    elif key == "time-":
                        i = _time_idx(self.ai_time_sec)
                        self.ai_time_sec = _TIME_STEPS[max(0, i - 1)]
                        if ai_searcher:
                            ai_searcher.time_limit_ms = self.ai_time_sec * 1000
                    elif key == "time+":
                        i = _time_idx(self.ai_time_sec)
                        self.ai_time_sec = _TIME_STEPS[min(len(_TIME_STEPS) - 1, i + 1)]
                        if ai_searcher:
                            ai_searcher.time_limit_ms = self.ai_time_sec * 1000
                    elif key == "hint":
                        self.analysis_on = not self.analysis_on
                        if self.analysis_on:
                            _launch_analysis()
                        else:
                            _cancel_analysis()
                            hint_moves = []; hint_depth = 0; hint_done = False
                    elif key == "restart":
                        state = create_game(self.game_name)
                        score_history = [self._eval_white_pov(state)]
                        selected_sq = None; legal_targets = []
                        selected_hand_piece = None
                        hint_moves = []; hint_depth = 0; hint_done = False
                        status_text = ""
                        if self.analysis_on:
                            _launch_analysis()
                    elif key == "menu":
                        _cancel_analysis()
                        return "menu"
                    break

                if handled:
                    continue

                # ---- guard: not interactable ----
                if state.is_terminal() or ai_thinking:
                    continue
                is_human = (
                    self.mode == "human-vs-human"
                    or (self.mode == "human-vs-ai"
                        and state.side_to_move() != ai_side)
                )
                if not is_human:
                    continue

                # ---- hand piece click (shogi) ----
                if has_hand and selected_sq is None:
                    side = state.side_to_move()
                    hr = sente_hand_rect if side == WHITE else gote_hand_rect
                    if hr and hr.collidepoint(mx, my):
                        hp = renderer.hand_piece_at(
                            mx - hr.x, my - hr.y, state, side, hr)
                        if hp is not None:
                            selected_hand_piece = hp
                            selected_sq = None
                            legal_targets = [
                                m.to_sq for m in state.legal_moves()
                                if m.drop_piece == hp
                            ]
                            continue

                # ---- board click ----
                bx, by = mx - board_x, my - board_y
                clicked_sq = renderer.pixel_to_sq(bx, by)
                if clicked_sq is None:
                    continue

                if selected_hand_piece is not None:
                    if clicked_sq in legal_targets:
                        for m in state.legal_moves():
                            if (m.drop_piece == selected_hand_piece
                                    and m.to_sq == clicked_sq):
                                state = state.make_move(m)
                                score_history.append(self._eval_white_pov(state))
                                hint_moves = []
                                break
                    selected_hand_piece = None
                    selected_sq = None; legal_targets = []

                elif selected_sq is None:
                    board = state.board_array()
                    piece = board[clicked_sq]
                    if piece != 0:
                        own = ((piece > 0 and state.side_to_move() == WHITE)
                               or (piece < 0 and state.side_to_move() == BLACK))
                        if own:
                            selected_sq = clicked_sq
                            legal_targets = [
                                m.to_sq for m in state.legal_moves()
                                if m.from_sq == clicked_sq
                            ]
                elif clicked_sq == selected_sq:
                    selected_sq = None; legal_targets = []

                elif clicked_sq in legal_targets:
                    matching = [m for m in state.legal_moves()
                                if m.from_sq == selected_sq
                                and m.to_sq == clicked_sq]
                    if len(matching) == 1:
                        state = state.make_move(matching[0])
                        score_history.append(self._eval_white_pov(state))
                    elif len(matching) > 1:
                        promo = [m for m in matching if m.promotion is not None]
                        state = state.make_move(promo[0] if promo else matching[0])
                        score_history.append(self._eval_white_pov(state))
                    hint_moves = []; selected_sq = None; legal_targets = []
                else:
                    selected_sq = clicked_sq
                    legal_targets = [
                        m.to_sq for m in state.legal_moves()
                        if m.from_sq == clicked_sq
                    ]

            # ============================================ AI
            if (not state.is_terminal() and not ai_thinking
                    and ai_searcher is not None):
                should = (
                    (self.mode == "human-vs-ai"
                     and state.side_to_move() == ai_side)
                    or self.mode == "ai-vs-ai"
                )
                if should:
                    ai_thinking = True
                    status_text = "AI thinking..."
                    _st2 = state
                    def _ai():
                        ai_result[0] = ai_searcher.search(_st2)
                    threading.Thread(target=_ai, daemon=True).start()

            if ai_thinking and ai_result[0] is not None:
                move, score = ai_result[0]
                ai_result[0] = None
                ai_thinking = False
                still_ai = (
                    (self.mode == "human-vs-ai"
                     and state.side_to_move() == ai_side)
                    or self.mode == "ai-vs-ai"
                )
                if still_ai and move is not None:
                    state = state.make_move(move)
                    score_history.append(self._eval_white_pov(state))
                    hint_moves = []
                    status_text = f"AI: {move} (score: {score:.0f})"
                elif move is None:
                    status_text = "AI: no move found"

            # ============================================ ANALYSIS (live)
            if self.analysis_on:
                # Read latest live result
                snap = current_analysis_live[0]
                if snap is not None and analysis_state_id == id(state):
                    d, md, moves, done = snap
                    hint_moves = moves
                    hint_depth = d
                    hint_done = done
                    if moves:
                        parts = [f"#{i+1} {m}({s:+.0f})"
                                 for i, (m, s) in enumerate(moves)]
                        tag = f"d{d}/{md}" + (" done" if done else "")
                        status_text = f"Analysis ({tag}): " + "  ".join(parts)

                # Auto-relaunch when state changed (new move made)
                if analysis_state_id != id(state):
                    hint_moves = []; hint_depth = 0; hint_done = False
                    _launch_analysis()

            # ============================================ TERMINAL
            if state.is_terminal():
                result = state.result()
                if result == 1.0:
                    stm = "White" if state.side_to_move() == WHITE else "Black"
                    status_text = f"{stm} wins!"
                elif result == 0.0:
                    stm = "White" if state.side_to_move() == WHITE else "Black"
                    other = "Black" if stm == "White" else "White"
                    status_text = f"{other} wins!"
                else:
                    status_text = "Draw!"

            # ============================================ DRAW
            self.screen.fill(_BG)

            # Hand bars (shogi) - above and below board
            if has_hand:
                renderer.draw_hand(self.screen, state, BLACK, gote_hand_rect)
                renderer.draw_hand(self.screen, state, WHITE, sente_hand_rect)

            # Board
            bsurf = pygame.Surface(
                (renderer.board_pixel_w, renderer.board_pixel_h))
            renderer.draw_board(bsurf, state, selected_sq, legal_targets)
            if hint_moves:
                renderer.draw_hints(bsurf, hint_moves)
            self.screen.blit(bsurf, (board_x, board_y))

            # Drop arrows (hand → board) on main screen
            if hint_moves and has_hand:
                renderer.draw_drop_hints(
                    self.screen, hint_moves, state,
                    (board_x, board_y), sente_hand_rect, gote_hand_rect)

            # Panel
            panel_rects = self._draw_panel(
                panel_x, 30, panel_w,
                state, renderer, has_hand,
                hint_moves, ai_thinking, status_text,
                self.analysis_on, hint_depth, hint_done,
                score_history,
            )

            pygame.display.flip()
            self.clock.tick(60)

    # ============================================================== PANEL
    def _draw_panel(self, px: int, py: int, pw: int,
                    state: GameState, renderer, has_hand: bool,
                    hint_moves, ai_thinking, status_text,
                    analysis_on: bool = False,
                    hint_depth: int = 0, hint_done: bool = False,
                    score_history: Optional[List[float]] = None,
                    ) -> Dict[str, pygame.Rect]:
        rects: Dict[str, pygame.Rect] = {}
        scr = self.screen
        mouse = pygame.mouse.get_pos()

        # Background
        bg_rect = pygame.Rect(px, py, pw, self.H - py * 2)
        pygame.draw.rect(scr, _PANEL_BG, bg_rect, border_radius=8)

        x = px + 14
        w = pw - 28
        y = py + 14

        # ---- title + turn ----
        titles = {"chess": "Chess 8\u00d78", "minichess": "Los Alamos 6\u00d76",
                  "shogi": "Shogi 9\u00d79", "minishogi": "Mini Shogi 5\u00d75"}
        t = self.heading_font.render(titles.get(self.game_name, ""), True, _TEXT)
        scr.blit(t, (x, y)); y += 28

        side_name = "White" if state.side_to_move() == WHITE else "Black"
        side_col = (255, 255, 255) if state.side_to_move() == WHITE else (40, 40, 40)
        st = self.font.render(f"Turn: {side_name}", True, _TEXT)
        scr.blit(st, (x, y))
        cx_dot = x + st.get_width() + 14
        pygame.draw.circle(scr, side_col, (cx_dot, y + 10), 7)
        pygame.draw.circle(scr, _TEXT_DIM, (cx_dot, y + 10), 7, 1)
        y += 28

        self._sep(x, y, w); y += 8

        # ---- MODE ----
        lbl = self.small_font.render("MODE", True, _TEXT_DIM)
        scr.blit(lbl, (x, y)); y += 20
        bw = (w - 8) // 3
        for i, (_, key) in enumerate(_MODES):
            r = pygame.Rect(x + i * (bw + 4), y, bw, 28)
            rects[f"mode:{key}"] = r
            self._panel_btn(r, _MODE_SHORT[key],
                            key == self.mode, r.collidepoint(mouse))
        y += 38

        # ---- SEARCH ----
        lbl = self.small_font.render("SEARCH", True, _TEXT_DIM)
        scr.blit(lbl, (x, y)); y += 20
        dm, dp = self._stepper_inline(x, y, w, "Depth", str(self.ai_depth))
        rects["depth-"] = dm; rects["depth+"] = dp; y += 32
        tm, tp = self._stepper_inline(x, y, w, "Time", f"{self.ai_time_sec}s")
        rects["time-"] = tm; rects["time+"] = tp; y += 38

        self._sep(x, y, w); y += 8

        # ---- analysis ----
        lbl = self.small_font.render("ANALYSIS", True, _TEXT_DIM)
        scr.blit(lbl, (x, y)); y += 20

        if analysis_on:
            if hint_done:
                hint_label = f"ON  -  depth {hint_depth} (done)"
            elif hint_depth > 0:
                hint_label = f"ON  -  depth {hint_depth}/{_DEPTH_MAX}..."
            else:
                hint_label = "ON  -  starting..."
        else:
            hint_label = "OFF  (H)"
        r = pygame.Rect(x, y, w, 30)
        rects["hint"] = r
        self._panel_btn(r, hint_label, analysis_on, r.collidepoint(mouse))
        y += 36

        r = pygame.Rect(x, y, w, 30)
        rects["restart"] = r
        self._panel_btn(r, "Restart  (R)", False, r.collidepoint(mouse))
        y += 36

        r = pygame.Rect(x, y, w, 30)
        rects["menu"] = r
        self._panel_btn(r, "Back to Menu  (ESC)", False, r.collidepoint(mouse))
        y += 40

        self._sep(x, y, w); y += 8

        # ---- status ----
        chart_h = 130
        chart_top = bg_rect.bottom - chart_h - 28
        max_status_y = chart_top - 6

        lbl = self.small_font.render("STATUS", True, _TEXT_DIM)
        scr.blit(lbl, (x, y)); y += 18
        if status_text:
            for line in self._wrap(status_text, w, self.small_font):
                if y >= max_status_y:
                    break
                s = self.small_font.render(line, True, _TEXT)
                scr.blit(s, (x, y)); y += 17

        # ---- score chart (fixed at panel bottom) ----
        self._sep(x, chart_top - 8, w)
        lbl = self.small_font.render("SCORE", True, _TEXT_DIM)
        scr.blit(lbl, (x, chart_top - 6))
        self._draw_score_chart(x, chart_top + 12, w, chart_h - 12,
                               score_history or [])

        return rects

    # ============================================================== WIDGETS
    def _menu_btn(self, rect, text, selected, hover, font=None):
        font = font or self.font
        bg = _BTN_SEL if selected else (_BTN_HI if hover else _BTN)
        pygame.draw.rect(self.screen, bg, rect, border_radius=6)
        pygame.draw.rect(self.screen, _BTN_BORDER, rect, 1, border_radius=6)
        lbl = font.render(text, True, _TEXT)
        self.screen.blit(lbl, lbl.get_rect(center=rect.center))

    def _panel_btn(self, rect, text, selected, hover):
        bg = _BTN_SEL if selected else (_BTN_HI if hover else _BTN)
        pygame.draw.rect(self.screen, bg, rect, border_radius=5)
        pygame.draw.rect(self.screen, _BTN_BORDER, rect, 1, border_radius=5)
        lbl = self.small_font.render(text, True, _TEXT)
        self.screen.blit(lbl, lbl.get_rect(center=rect.center))

    def _stepper(self, x, y, label, value, width):
        """Menu stepper: label above, ◄ value ► below. Returns (minus, plus)."""
        lbl = self.font.render(label, True, _TEXT_DIM)
        self.screen.blit(lbl, (x, y))
        y2 = y + 24
        btn_w = 36
        minus = pygame.Rect(x, y2, btn_w, 32)
        plus = pygame.Rect(x + width - btn_w, y2, btn_w, 32)
        mouse = pygame.mouse.get_pos()
        self._menu_btn(minus, "\u25c4", False, minus.collidepoint(mouse))
        self._menu_btn(plus, "\u25ba", False, plus.collidepoint(mouse))
        # Truncate value if it doesn't fit between arrows
        max_val_w = width - 2 * btn_w - 12
        display = value
        while self.font.size(display)[0] > max_val_w and len(display) > 4:
            display = display[:-4] + "..."
        val = self.font.render(display, True, _TEXT)
        vx = x + btn_w + (width - 2 * btn_w - val.get_width()) // 2
        self.screen.blit(val, (vx, y2 + 4))
        return minus, plus

    def _stepper_inline(self, x, y, total_w, label, value):
        """Panel stepper: label left, ◄ value ► right. Returns (minus, plus)."""
        lbl = self.font.render(label, True, _TEXT)
        self.screen.blit(lbl, (x, y + 1))
        sw, btn_w = 110, 28
        sx = x + total_w - sw
        minus = pygame.Rect(sx, y, btn_w, 26)
        plus = pygame.Rect(sx + sw - btn_w, y, btn_w, 26)
        mouse = pygame.mouse.get_pos()
        self._panel_btn(minus, "\u25c4", False, minus.collidepoint(mouse))
        self._panel_btn(plus, "\u25ba", False, plus.collidepoint(mouse))
        val = self.small_font.render(value, True, _TEXT)
        vx = sx + btn_w + (sw - 2 * btn_w - val.get_width()) // 2
        self.screen.blit(val, (vx, y + 4))
        return minus, plus

    def _section_label(self, text, cx, y):
        lbl = self.small_font.render(text, True, _TEXT_DIM)
        self.screen.blit(lbl, lbl.get_rect(center=(cx, y)))

    def _blit_centered(self, font, text, cx, y, color):
        s = font.render(text, True, color)
        self.screen.blit(s, s.get_rect(center=(cx, y)))

    def _sep(self, x, y, w):
        pygame.draw.line(self.screen, _SEP, (x, y), (x + w, y), 1)

    @staticmethod
    def _wrap(text: str, max_w: int, font) -> List[str]:
        words = text.split()
        lines: List[str] = []
        line = ""
        for word in words:
            test = f"{line} {word}".strip()
            if font.size(test)[0] > max_w and line:
                lines.append(line)
                line = word
            else:
                line = test
        if line:
            lines.append(line)
        return lines

    # ============================================================== chart
    @staticmethod
    def _eval_white_pov(state) -> float:
        """Material score from White's perspective (pawn units)."""
        board = state.board_array()
        vals = {1: 1.0, 2: 3.2, 3: 3.3, 4: 5.0, 5: 9.0}
        score = 0.0
        for sq in range(len(board)):
            p = board[sq]
            if p != 0:
                v = vals.get(abs(p), 1.0)
                score += v if p > 0 else -v
        return score

    def _draw_score_chart(self, x: int, y: int, w: int, h: int,
                          scores: List[float]):
        """Draw evaluation history chart with green/red fill."""
        scr = self.screen
        pygame.draw.rect(scr, (38, 36, 34), (x, y, w, h), border_radius=4)

        pad = 6
        cx, cy = x + pad, y + pad
        cw, ch = w - 2 * pad, h - 2 * pad
        if cw < 10 or ch < 10:
            return

        if not scores:
            return

        max_abs = max((abs(s) for s in scores), default=0)
        y_range = max(max_abs * 1.2, 2.0)
        center_y = cy + ch / 2.0

        # Center line (score = 0)
        pygame.draw.line(scr, (70, 70, 70),
                         (cx, int(center_y)), (cx + cw, int(center_y)), 1)

        if len(scores) < 2:
            py_ = center_y - (scores[0] / y_range) * (ch / 2)
            py_ = max(cy, min(cy + ch, py_))
            pygame.draw.circle(scr, _TEXT, (cx + cw // 2, int(py_)), 3)
            return

        step = cw / (len(scores) - 1)
        pts = []
        for i, s in enumerate(scores):
            px = cx + i * step
            py_ = center_y - (s / y_range) * (ch / 2)
            py_ = max(cy, min(cy + ch, py_))
            pts.append((px, py_))

        # Filled segments
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            avg = (scores[i] + scores[i + 1]) / 2
            color = (50, 110, 50) if avg >= 0 else (110, 45, 45)
            poly = [(int(x1), int(center_y)), (int(x1), int(y1)),
                    (int(x2), int(y2)), (int(x2), int(center_y))]
            pygame.draw.polygon(scr, color, poly)

        # Line
        int_pts = [(int(px), int(py_)) for px, py_ in pts]
        pygame.draw.lines(scr, _TEXT, False, int_pts, 2)

        # Latest point highlight
        pygame.draw.circle(scr, (255, 255, 255), int_pts[-1], 3)

        # Scale labels
        top_lbl = self.small_font.render(f"+{y_range:.0f}", True, (80, 140, 80))
        bot_lbl = self.small_font.render(f"-{y_range:.0f}", True, (140, 70, 70))
        scr.blit(top_lbl, (x + w - top_lbl.get_width() - 4, y + 2))
        scr.blit(bot_lbl, (x + w - bot_lbl.get_width() - 4, y + h - 15))

    # ============================================================== factories
    def _create_renderer(self, game_name: str):
        from src.gui.themes import DEFAULT_CHESS_THEME, DEFAULT_SHOGI_THEME
        if game_name == "chess":
            from src.gui.chess_gui import ChessRenderer
            return ChessRenderer(theme=DEFAULT_CHESS_THEME, square_size=80)
        elif game_name == "minichess":
            from src.gui.minichess_gui import MiniChessRenderer
            return MiniChessRenderer(theme=DEFAULT_CHESS_THEME, square_size=90)
        elif game_name == "shogi":
            from src.gui.shogi_gui import ShogiRenderer
            return ShogiRenderer(theme=DEFAULT_SHOGI_THEME, square_size=70)
        elif game_name == "minishogi":
            from src.gui.minishogi_gui import MiniShogiRenderer
            return MiniShogiRenderer(theme=DEFAULT_SHOGI_THEME, square_size=90)
        raise ValueError(f"Unknown game: {game_name}")

    def _create_ai(self):
        from src.search.alphabeta import AlphaBetaSearch
        evaluator = self._make_evaluator(self.game_name, self.model_path)
        searcher = AlphaBetaSearch(
            evaluator, max_depth=self.ai_depth,
            time_limit_ms=self.ai_time_sec * 1000)
        return evaluator, searcher

    @staticmethod
    def _make_evaluator(game_name, model_path):
        if model_path:
            from src.search.evaluator import NNUEEvaluator
            from src.features.halfkp import chess_features, minichess_features
            from src.features.halfkp_shogi import shogi_features, minishogi_features
            fs_map = {
                "chess": chess_features, "minichess": minichess_features,
                "shogi": shogi_features, "minishogi": minishogi_features,
            }
            return NNUEEvaluator.from_numpy(model_path, fs_map[game_name]())
        from src.search.evaluator import MaterialEvaluator
        return MaterialEvaluator()

    @staticmethod
    def _scan_models() -> List[str]:
        d = Path(__file__).resolve().parent.parent.parent / "models"
        if not d.exists():
            return []
        return sorted(str(f) for f in d.glob("*.npz"))
