"""Side panel drawing, score chart, and notation helpers."""

from typing import Dict, List, Optional

import pygame

from src.games.base import GameState, Move, WHITE
from src.gui.constants import (
    PANEL_BG, TEXT, TEXT_DIM, TEXT_MUTED,
    MODE_SHORT, DEPTH_MAX,
)
from src.gui.widgets import panel_btn, sep, wrap


# ------------------------------------------------------------------ notation

def eval_white_pov(state) -> float:
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


def sq_to_coord(game: str, sq: int) -> str:
    if game == "chess":
        return f"{chr(97 + sq % 8)}{sq // 8 + 1}"
    if game == "minichess":
        return f"{chr(97 + sq % 6)}{sq // 6 + 1}"
    if game == "shogi":
        return f"{9 - sq % 9}{chr(97 + sq // 9)}"
    if game == "minishogi":
        return f"{5 - sq % 5}{chr(97 + sq // 5)}"
    return str(sq)


def piece_name(game: str, board_val: int) -> str:
    if game == "chess":
        return {1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"}.get(
            board_val, "?")
    if game == "minichess":
        return {1: "P", 2: "N", 3: "R", 4: "Q", 5: "K"}.get(
            board_val, "?")
    if game == "shogi":
        from src.gui.pieces import SHOGI_KANJI
        return SHOGI_KANJI.get(board_val, "?")
    if game == "minishogi":
        from src.gui.pieces import MINISHOGI_KANJI
        return MINISHOGI_KANJI.get(board_val, "?")
    return "?"


def move_notation(game_name: str, move: Move, state) -> str:
    """Format move as (Piece-from-to) for display."""
    board = state.board_array()
    if move.drop_piece is not None:
        pname = piece_name(game_name, move.drop_piece + 1)
    elif move.from_sq is not None:
        pname = piece_name(game_name, abs(board[move.from_sq]))
    else:
        pname = "?"
    from_s = sq_to_coord(game_name, move.from_sq) \
        if move.from_sq is not None else "00"
    to_s = sq_to_coord(game_name, move.to_sq)
    return f"({pname}-{from_s}-{to_s})"


# ------------------------------------------------------------------ chart

def draw_score_chart(screen, small_font, x: int, y: int, w: int, h: int,
                     scores: List[float]):
    """Draw evaluation history chart with green/red fill."""
    pygame.draw.rect(screen, (38, 36, 34), (x, y, w, h), border_radius=4)

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

    pygame.draw.line(screen, (70, 70, 70),
                     (cx, int(center_y)), (cx + cw, int(center_y)), 1)

    if len(scores) < 2:
        py_ = center_y - (scores[0] / y_range) * (ch / 2)
        py_ = max(cy, min(cy + ch, py_))
        pygame.draw.circle(screen, TEXT, (cx + cw // 2, int(py_)), 3)
        return

    step = cw / (len(scores) - 1)
    pts = []
    for i, s in enumerate(scores):
        px = cx + i * step
        py_ = center_y - (s / y_range) * (ch / 2)
        py_ = max(cy, min(cy + ch, py_))
        pts.append((px, py_))

    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        avg = (scores[i] + scores[i + 1]) / 2
        color = (50, 110, 50) if avg >= 0 else (110, 45, 45)
        poly = [(int(x1), int(center_y)), (int(x1), int(y1)),
                (int(x2), int(y2)), (int(x2), int(center_y))]
        pygame.draw.polygon(screen, color, poly)

    int_pts = [(int(px), int(py_)) for px, py_ in pts]
    pygame.draw.lines(screen, TEXT, False, int_pts, 2)
    pygame.draw.circle(screen, (255, 255, 255), int_pts[-1], 3)

    top_lbl = small_font.render(f"+{y_range:.0f}", True, (80, 140, 80))
    bot_lbl = small_font.render(f"-{y_range:.0f}", True, (140, 70, 70))
    screen.blit(top_lbl, (x + w - top_lbl.get_width() - 4, y + 2))
    screen.blit(bot_lbl, (x + w - bot_lbl.get_width() - 4, y + h - 15))


# ------------------------------------------------------------------ panel

def draw_panel(app, px: int, py: int, pw: int,
               state: GameState, hint_moves, ai_thinking, status_text,
               analysis_on: bool = False,
               hint_depth: int = 0, hint_done: bool = False,
               score_history: Optional[List[float]] = None,
               hint_max_depth: int = 0,
               ) -> Dict[str, pygame.Rect]:
    """Draw the in-game side panel. Returns dict of clickable rects."""
    rects: Dict[str, pygame.Rect] = {}
    screen = app.screen
    mouse = pygame.mouse.get_pos()

    bg_rect = pygame.Rect(px, py, pw, app.H - py * 2)
    pygame.draw.rect(screen, PANEL_BG, bg_rect, border_radius=8)

    x = px + 14
    w = pw - 28
    y = py + 14

    # Restart + Menu
    btn_w = (w - 8) // 2
    r = pygame.Rect(x, y, btn_w, 26)
    rects["restart"] = r
    panel_btn(screen, r, "Restart (R)", False, r.collidepoint(mouse),
              app.small_font)
    r = pygame.Rect(x + btn_w + 8, y, btn_w, 26)
    rects["menu"] = r
    panel_btn(screen, r, "Menu (ESC)", False, r.collidepoint(mouse),
              app.small_font)
    y += 34

    sep(screen, x, y, w); y += 8

    # Title + turn
    titles = {"chess": "Chess 8\u00d78", "minichess": "Los Alamos 6\u00d76",
              "shogi": "Shogi 9\u00d79", "minishogi": "Mini Shogi 5\u00d75"}
    t = app.heading_font.render(titles.get(app.game_name, ""), True, TEXT)
    screen.blit(t, (x, y)); y += 28

    side_name = "White" if state.side_to_move() == WHITE else "Black"
    side_col = (255, 255, 255) if state.side_to_move() == WHITE else (40, 40, 40)
    st = app.font.render(f"Turn: {side_name}", True, TEXT)
    screen.blit(st, (x, y))
    cx_dot = x + st.get_width() + 14
    pygame.draw.circle(screen, side_col, (cx_dot, y + 10), 7)
    pygame.draw.circle(screen, TEXT_DIM, (cx_dot, y + 10), 7, 1)
    y += 28

    # Info line
    mode_short = MODE_SHORT.get(app.mode, app.mode)
    info = (f"{mode_short}  \u00b7  Depth {app.ai_depth}"
            f"  \u00b7  Time {app.ai_time_sec}s")
    lbl = app.small_font.render(info, True, TEXT_MUTED)
    screen.blit(lbl, (x, y)); y += 22

    sep(screen, x, y, w); y += 8

    # Analysis
    lbl = app.small_font.render("ANALYSIS", True, TEXT_DIM)
    screen.blit(lbl, (x, y)); y += 20

    if analysis_on:
        # Infinite-analysis mode reports a sentinel max depth far beyond
        # the configured DEPTH_MAX (the C shogi rule search uses ~64 as its
        # safety cap). Show "depth D/∞" in that case.
        infinite = hint_max_depth > DEPTH_MAX
        if hint_done:
            toggle_lbl = f"ON  \u2014  depth {hint_depth} (done)"
        elif hint_depth > 0:
            cap = "\u221e" if infinite else str(hint_max_depth or DEPTH_MAX)
            toggle_lbl = f"ON  \u2014  depth {hint_depth}/{cap}..."
        else:
            toggle_lbl = "ON  \u2014  starting..."
    else:
        toggle_lbl = "OFF  (H)"
    r = pygame.Rect(x, y, w, 28)
    rects["hint"] = r
    panel_btn(screen, r, toggle_lbl, analysis_on, r.collidepoint(mouse),
              app.small_font)
    y += 34

    # Analysis move results
    _rank_colors = [(255, 200, 40), (180, 180, 190), (200, 140, 80)]
    if analysis_on and hint_moves:
        for i, (mv, score) in enumerate(hint_moves):
            notation = move_notation(app.game_name, mv, state)
            line = f"#{i+1}  {notation}  {score:+.0f}"
            c = _rank_colors[i] if i < len(_rank_colors) else TEXT_DIM
            s = app.small_font.render(line, True, c)
            screen.blit(s, (x + 4, y)); y += 18
        y += 4

    sep(screen, x, y, w); y += 8

    # Status
    chart_h = 130
    chart_top = bg_rect.bottom - chart_h - 28
    max_status_y = chart_top - 6

    lbl = app.small_font.render("STATUS", True, TEXT_DIM)
    screen.blit(lbl, (x, y)); y += 18
    if status_text:
        for line in wrap(status_text, w, app.small_font):
            if y >= max_status_y:
                break
            s = app.small_font.render(line, True, TEXT)
            screen.blit(s, (x, y)); y += 17

    # Score chart (fixed at panel bottom)
    sep(screen, x, chart_top - 8, w)
    lbl = app.small_font.render("SCORE", True, TEXT_DIM)
    screen.blit(lbl, (x, chart_top - 6))
    draw_score_chart(screen, app.small_font, x, chart_top + 12, w,
                     chart_h - 12, score_history or [])

    return rects
