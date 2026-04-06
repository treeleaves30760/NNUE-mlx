"""Main Pygame application: settings menu and game orchestration."""

import os
from pathlib import Path
from typing import Optional

import pygame

from src.gui.constants import (
    BG, TEXT, TEXT_DIM, TEXT_MUTED,
    ACCENT, ACCENT_HI,
    BTN, BTN_HI, BTN_BORDER,
    GAMES, MODES,
    DEPTH_MIN, DEPTH_MAX, TIME_STEPS, time_idx,
)
from src.gui.widgets import menu_btn, stepper, section_label, blit_centered, trunc
from src.gui.game_session import GameSession


class GameApp:
    """Main application with settings menu and in-game control panel."""

    W = 1100
    H = 780

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("NNUE-mlx Board Games")
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
        self.model_path_2: Optional[str] = None  # AI vs AI black model
        self.analysis_on = False

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
            result = GameSession(self).run()
            if result == "quit":
                break
        pygame.quit()

    # ============================================================== MENU

    def _run_menu(self) -> str:
        """Settings menu. Returns 'start' or 'quit'."""
        while True:
            mouse = pygame.mouse.get_pos()
            self.screen.fill(BG)
            cx = self.W // 2

            # Title
            blit_centered(self.screen, self.title_font,
                          "NNUE-mlx", cx, 45, TEXT)
            blit_centered(self.screen, self.small_font,
                          "Board Game AI with Apple MLX Training",
                          cx, 88, TEXT_DIM)

            y = 130
            # ---- SELECT GAME ----
            section_label(self.screen, self.small_font,
                          "SELECT GAME", cx, y)
            y += 28
            game_rects = []
            for i, (label, key) in enumerate(GAMES):
                col, row = i % 2, i // 2
                r = pygame.Rect(cx - 270 + col * 280, y + row * 50, 260, 42)
                game_rects.append((r, key))
                menu_btn(self.screen, r, label,
                         key == self.game_name,
                         r.collidepoint(mouse), self.font)
            y += 112

            # ---- GAME MODE ----
            section_label(self.screen, self.small_font,
                          "GAME MODE", cx, y)
            y += 28
            mode_rects = []
            bw = 172
            mx_start = cx - (bw * 3 + 16) // 2
            for i, (label, key) in enumerate(MODES):
                r = pygame.Rect(mx_start + i * (bw + 8), y, bw, 38)
                mode_rects.append((r, key))
                menu_btn(self.screen, r, label,
                         key == self.mode,
                         r.collidepoint(mouse), self.font)
            y += 56

            # ---- AI SETTINGS ----
            section_label(self.screen, self.small_font,
                          "AI SETTINGS", cx, y)
            y += 30
            d_minus, d_plus = stepper(
                self.screen, self.font, cx - 240, y,
                "Search Depth", str(self.ai_depth), 210)
            t_minus, t_plus = stepper(
                self.screen, self.font, cx + 30, y,
                "Time Limit", f"{self.ai_time_sec}s", 210)
            y += 68

            # ---- ANALYSIS ----
            analysis_label = ("Analysis: ON" if self.analysis_on
                              else "Analysis: OFF")
            analysis_r = pygame.Rect(cx - 100, y, 200, 38)
            menu_btn(self.screen, analysis_r, analysis_label,
                     self.analysis_on, analysis_r.collidepoint(mouse),
                     self.font)
            y += 56

            # ---- MODEL ----
            is_ava = self.mode == "ai-vs-ai"
            section_label(self.screen, self.small_font,
                          "WHITE AI MODEL" if is_ava else "MODEL", cx, y)
            y += 24
            browse_w, none_w, gap = 400, 70, 10
            bx = cx - (browse_w + gap + none_w) // 2
            m1_name = (os.path.basename(self.model_path)
                       if self.model_path else "None (Material Only)")
            m1_browse = pygame.Rect(bx, y, browse_w, 38)
            m1_none = pygame.Rect(bx + browse_w + gap, y, none_w, 38)
            menu_btn(self.screen, m1_browse,
                     trunc(m1_name, browse_w - 20, self.font),
                     self.model_path is not None,
                     m1_browse.collidepoint(mouse), self.font)
            menu_btn(self.screen, m1_none, "None", False,
                     m1_none.collidepoint(mouse), self.small_font)
            y += 48

            m2_browse = m2_none = None
            if is_ava:
                section_label(self.screen, self.small_font,
                              "BLACK AI MODEL", cx, y)
                y += 24
                m2_name = (os.path.basename(self.model_path_2)
                           if self.model_path_2
                           else "None (Material Only)")
                m2_browse = pygame.Rect(bx, y, browse_w, 38)
                m2_none = pygame.Rect(
                    bx + browse_w + gap, y, none_w, 38)
                menu_btn(self.screen, m2_browse,
                         trunc(m2_name, browse_w - 20, self.font),
                         self.model_path_2 is not None,
                         m2_browse.collidepoint(mouse), self.font)
                menu_btn(self.screen, m2_none, "None", False,
                         m2_none.collidepoint(mouse), self.small_font)
                y += 48

            # ---- START ----
            start_r = pygame.Rect(cx - 150, y, 300, 54)
            hover = start_r.collidepoint(mouse)
            pygame.draw.rect(self.screen,
                             ACCENT_HI if hover else ACCENT,
                             start_r, border_radius=10)
            lbl = self.heading_font.render(
                ">>  START GAME", True, (255, 255, 255))
            self.screen.blit(lbl, lbl.get_rect(center=start_r.center))

            blit_centered(self.screen, self.small_font,
                          "Press ENTER to start \u2022 ESC to quit",
                          cx, self.H - 28, TEXT_MUTED)
            pygame.display.flip()

            # ---- Events ----
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "quit"
                    if event.key == pygame.K_RETURN:
                        return "start"
                if (event.type == pygame.MOUSEBUTTONDOWN
                        and event.button == 1):
                    pos = event.pos
                    for r, k in game_rects:
                        if r.collidepoint(pos):
                            self.game_name = k
                    for r, k in mode_rects:
                        if r.collidepoint(pos):
                            self.mode = k
                    if analysis_r.collidepoint(pos):
                        self.analysis_on = not self.analysis_on
                    if m1_browse.collidepoint(pos):
                        p = self._browse_model()
                        if p:
                            self.model_path = p
                    if m1_none.collidepoint(pos):
                        self.model_path = None
                    if (m2_browse is not None
                            and m2_browse.collidepoint(pos)):
                        p = self._browse_model()
                        if p:
                            self.model_path_2 = p
                    if (m2_none is not None
                            and m2_none.collidepoint(pos)):
                        self.model_path_2 = None
                    if d_minus.collidepoint(pos):
                        self.ai_depth = max(DEPTH_MIN,
                                            self.ai_depth - 1)
                    if d_plus.collidepoint(pos):
                        self.ai_depth = min(DEPTH_MAX,
                                            self.ai_depth + 1)
                    if t_minus.collidepoint(pos):
                        i = time_idx(self.ai_time_sec)
                        self.ai_time_sec = TIME_STEPS[max(0, i - 1)]
                    if t_plus.collidepoint(pos):
                        i = time_idx(self.ai_time_sec)
                        self.ai_time_sec = TIME_STEPS[
                            min(len(TIME_STEPS) - 1, i + 1)]
                    if start_r.collidepoint(pos):
                        return "start"
            self.clock.tick(60)

    # ============================================================== factories

    def _create_renderer(self, game_name: str):
        from src.gui.themes import DEFAULT_CHESS_THEME, DEFAULT_SHOGI_THEME
        if game_name == "chess":
            from src.gui.chess_gui import ChessRenderer
            return ChessRenderer(theme=DEFAULT_CHESS_THEME, square_size=80)
        elif game_name == "minichess":
            from src.gui.minichess_gui import MiniChessRenderer
            return MiniChessRenderer(theme=DEFAULT_CHESS_THEME,
                                     square_size=90)
        elif game_name == "shogi":
            from src.gui.shogi_gui import ShogiRenderer
            return ShogiRenderer(theme=DEFAULT_SHOGI_THEME, square_size=70)
        elif game_name == "minishogi":
            from src.gui.minishogi_gui import MiniShogiRenderer
            return MiniShogiRenderer(theme=DEFAULT_SHOGI_THEME,
                                     square_size=90)
        raise ValueError(f"Unknown game: {game_name}")

    def _create_ai(self, model_path=None):
        from src.search.alphabeta import AlphaBetaSearch
        if model_path is None:
            model_path = self.model_path
        evaluator = self._make_evaluator(self.game_name, model_path)
        searcher = AlphaBetaSearch(
            evaluator, max_depth=self.ai_depth,
            time_limit_ms=self.ai_time_sec * 1000)
        return evaluator, searcher

    @staticmethod
    def _make_evaluator(game_name, model_path):
        if model_path:
            from src.search.evaluator import NNUEEvaluator
            from src.features.halfkp import (
                chess_features, minichess_features)
            from src.features.halfkp_shogi import (
                shogi_features, minishogi_features)
            fs_map = {
                "chess": chess_features,
                "minichess": minichess_features,
                "shogi": shogi_features,
                "minishogi": minishogi_features,
            }
            return NNUEEvaluator.from_numpy(
                model_path, fs_map[game_name]())
        from src.search.evaluator import MaterialEvaluator
        return MaterialEvaluator()

    def _browse_model(self) -> Optional[str]:
        """Pygame-based file browser for .npz model files."""
        models_root = Path(__file__).resolve().parent.parent.parent / "models"
        cur_dir = models_root if models_root.is_dir() else Path.cwd()

        font = self.font
        small = self.small_font
        scroll_off = 0
        ROW_H = 36
        PAD = 20
        # usable area
        list_x, list_y = PAD, 80
        list_w = self.W - 2 * PAD
        list_h = self.H - 160

        def entries(d: Path):
            """Return sorted (name, path, is_dir) list for directory."""
            items = []
            try:
                for p in sorted(d.iterdir()):
                    if p.name.startswith('.'):
                        continue
                    if p.is_dir():
                        items.append((p.name + "/", p, True))
                    elif p.suffix == '.npz':
                        items.append((p.name, p, False))
            except PermissionError:
                pass
            return items

        while True:
            items = entries(cur_dir)
            max_scroll = max(0, len(items) * ROW_H - list_h)
            scroll_off = min(scroll_off, max_scroll)
            mouse = pygame.mouse.get_pos()

            self.screen.fill(BG)
            # header
            blit_centered(self.screen, self.heading_font,
                          "Select NNUE Model (.npz)", self.W // 2, 30, TEXT)
            # path breadcrumb
            path_str = str(cur_dir)
            blit_centered(self.screen, small, trunc(path_str, self.W - 40,
                          small), self.W // 2, 58, TEXT_DIM)

            # clip region for list
            clip = pygame.Rect(list_x, list_y, list_w, list_h)
            self.screen.set_clip(clip)
            # ".." entry
            all_rows = [(".. (up)", cur_dir.parent, True)] + items
            hovered_idx = -1
            for i, (name, path, is_dir) in enumerate(all_rows):
                ry = list_y + i * ROW_H - scroll_off
                if ry + ROW_H < list_y or ry > list_y + list_h:
                    continue
                row_rect = pygame.Rect(list_x, ry, list_w, ROW_H - 2)
                hover = row_rect.collidepoint(mouse)
                if hover:
                    hovered_idx = i
                bg = BTN_HI if hover else BTN
                pygame.draw.rect(self.screen, bg, row_rect, border_radius=4)
                prefix = "\U0001f4c1 " if is_dir else "\U0001f4c4 "
                lbl = font.render(prefix + name, True, TEXT)
                self.screen.blit(lbl, (list_x + 10, ry + 6))
            self.screen.set_clip(None)

            # border around list
            pygame.draw.rect(self.screen, BTN_BORDER, clip, 1, border_radius=4)

            # bottom bar
            cancel_r = pygame.Rect(self.W // 2 - 80,
                                   self.H - 58, 160, 40)
            menu_btn(self.screen, cancel_r, "Cancel", False,
                     cancel_r.collidepoint(mouse), font)

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                if event.type == pygame.MOUSEWHEEL:
                    scroll_off = max(0, min(max_scroll,
                                            scroll_off - event.y * ROW_H))
                if (event.type == pygame.MOUSEBUTTONDOWN
                        and event.button == 1):
                    if cancel_r.collidepoint(event.pos):
                        return None
                    if hovered_idx >= 0:
                        _, path, is_dir = all_rows[hovered_idx]
                        if is_dir:
                            cur_dir = path
                            scroll_off = 0
                        else:
                            return str(path)
            self.clock.tick(30)
