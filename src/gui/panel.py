"""Side panel drawing, score chart, and notation helpers."""

from typing import Dict, List, Optional

import pygame

from src.games.base import GameState, Move, WHITE, BLACK
from src.gui.constants import (
    PANEL_BG, PANEL_BG_ALT, PANEL_BORDER,
    TEXT, TEXT_DIM, TEXT_MUTED,
    MODE_SHORT, DEPTH_MAX,
    EVAL_GOOD, EVAL_BAD, EVAL_NEUTRAL,
    WHITE_ADV, BLACK_ADV,
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

# The chart uses a fixed y-range so the line doesn't visually jump as new
# extreme scores arrive. 6 pawns covers the vast majority of real positions;
# beyond that we just clip so the shape stays readable.
_CHART_CLAMP = 6.0


def draw_score_chart(screen, small_font, x: int, y: int, w: int, h: int,
                     scores: List[Optional[float]]):
    """Draw evaluation history chart with green/red fill.

    ``scores`` is a list of floats-or-None entries. ``None`` entries are
    plies without a committed eval yet — we skip them when connecting
    segments so a pending ply doesn't produce a zig-zag artifact.
    """
    pygame.draw.rect(screen, (22, 24, 30), (x, y, w, h), border_radius=6)
    pygame.draw.rect(screen, PANEL_BORDER, (x, y, w, h), 1, border_radius=6)

    pad = 8
    cx, cy = x + pad, y + pad
    cw, ch = w - 2 * pad, h - 2 * pad
    if cw < 10 or ch < 10:
        return
    filled = [s for s in scores if s is not None]
    if not filled:
        # Zero line with muted label.
        center_y = int(cy + ch / 2)
        pygame.draw.line(screen, (60, 64, 74),
                         (cx, center_y), (cx + cw, center_y), 1)
        lbl = small_font.render("no data yet", True, TEXT_MUTED)
        screen.blit(lbl, (cx + (cw - lbl.get_width()) // 2,
                          center_y - lbl.get_height() // 2))
        return

    y_range = _CHART_CLAMP
    center_y = cy + ch / 2.0

    # Horizontal zero line
    pygame.draw.line(screen, (68, 72, 86),
                     (cx, int(center_y)), (cx + cw, int(center_y)), 1)

    # Pre-compute filled points (clamp + x coordinate)
    n = len(scores)
    if n == 1:
        s = filled[0]
        s_clamped = max(-y_range, min(y_range, s))
        py = center_y - (s_clamped / y_range) * (ch / 2)
        pygame.draw.circle(
            screen, (255, 255, 255), (cx + cw // 2, int(py)), 4)
        return

    step = cw / (n - 1)
    pts = []
    for i, s in enumerate(scores):
        if s is None:
            continue
        s_clamped = max(-y_range, min(y_range, s))
        px = cx + i * step
        py = center_y - (s_clamped / y_range) * (ch / 2)
        pts.append((px, py, s))

    if not pts:
        return

    # Filled area polygons (one per consecutive segment)
    for i in range(len(pts) - 1):
        x1, y1, s1 = pts[i]
        x2, y2, s2 = pts[i + 1]
        avg = (s1 + s2) / 2
        if avg >= 0:
            color = (50, 110, 70, 190)
        else:
            color = (140, 60, 60, 190)
        poly = [(int(x1), int(center_y)), (int(x1), int(y1)),
                (int(x2), int(y2)), (int(x2), int(center_y))]
        overlay = pygame.Surface((cw + 4 * pad, ch + 4 * pad),
                                 pygame.SRCALPHA)
        shifted = [(px - (x - 2 * pad), py_ - (y - 2 * pad))
                   for (px, py_) in poly]
        pygame.draw.polygon(overlay, color, shifted)
        screen.blit(overlay, (x - 2 * pad, y - 2 * pad))

    # Line path through committed points
    int_pts = [(int(px), int(py_)) for px, py_, _ in pts]
    if len(int_pts) >= 2:
        pygame.draw.lines(screen, TEXT, False, int_pts, 2)
    # Latest point marker
    pygame.draw.circle(screen, (255, 255, 255), int_pts[-1], 4)
    pygame.draw.circle(screen, (0, 0, 0), int_pts[-1], 4, 1)

    # Axis labels (fixed ± range)
    top_lbl = small_font.render(
        f"+{y_range:.0f}", True, (120, 200, 140))
    bot_lbl = small_font.render(
        f"-{y_range:.0f}", True, (220, 110, 110))
    screen.blit(top_lbl, (x + w - top_lbl.get_width() - 6, y + 4))
    screen.blit(bot_lbl, (x + w - bot_lbl.get_width() - 6,
                          y + h - bot_lbl.get_height() - 4))


# ------------------------------------------------------------------ helpers

def _format_eval(pawns: Optional[float]) -> str:
    if pawns is None:
        return "–"
    if abs(pawns) < 0.05:
        return "0.0"
    if pawns > 0:
        return f"+{pawns:.1f}"
    return f"{pawns:.1f}"


def _eval_color(pawns: Optional[float]):
    if pawns is None or abs(pawns) < 0.1:
        return EVAL_NEUTRAL
    return EVAL_GOOD if pawns > 0 else EVAL_BAD


def _winner_label(pawns: Optional[float]) -> str:
    if pawns is None:
        return "analysis pending"
    if abs(pawns) < 0.1:
        return "equal position"
    if pawns > 0:
        return "White advantage"
    return "Black advantage"


def _side_dot(screen, cx: int, cy: int, side_int: int):
    """Circle marker for the side-to-move (white or black)."""
    col = (240, 240, 240) if side_int == WHITE else (20, 20, 20)
    pygame.draw.circle(screen, col, (cx, cy), 8)
    pygame.draw.circle(screen, (140, 150, 170), (cx, cy), 8, 1)


# ------------------------------------------------------------------ panel

def draw_panel(app, px: int, py: int, pw: int,
               state: GameState, hint_moves, ai_thinking, status_text,
               analysis_on: bool = False,
               hint_depth: int = 0, hint_done: bool = False,
               score_history: Optional[List[Optional[float]]] = None,
               hint_max_depth: int = 0,
               white_eval: Optional[float] = None,
               flipped: bool = False,
               can_flip: bool = False,
               player_side_int: int = WHITE,
               mode: str = "human-vs-human",
               ) -> Dict[str, pygame.Rect]:
    """Draw the in-game side panel. Returns dict of clickable rects."""
    rects: Dict[str, pygame.Rect] = {}
    screen = app.screen
    mouse = pygame.mouse.get_pos()

    bg_rect = pygame.Rect(px, py, pw, app.H - py * 2)
    pygame.draw.rect(screen, PANEL_BG, bg_rect, border_radius=12)
    pygame.draw.rect(screen, PANEL_BORDER, bg_rect, 1, border_radius=12)

    x = px + 14
    w = pw - 28
    y = py + 14

    # ---- Header row: Restart | Flip | Menu ----
    if can_flip:
        btn_w = (w - 16) // 3
        restart_r = pygame.Rect(x, y, btn_w, 28)
        flip_r = pygame.Rect(x + btn_w + 8, y, btn_w, 28)
        menu_r = pygame.Rect(x + (btn_w + 8) * 2, y, btn_w, 28)
        rects["restart"] = restart_r
        rects["flip"] = flip_r
        rects["menu"] = menu_r
        panel_btn(screen, restart_r, "Restart (R)", False,
                  restart_r.collidepoint(mouse), app.small_font)
        panel_btn(screen, flip_r, "Flip (F)", flipped,
                  flip_r.collidepoint(mouse), app.small_font)
        panel_btn(screen, menu_r, "Menu (ESC)", False,
                  menu_r.collidepoint(mouse), app.small_font)
    else:
        btn_w = (w - 8) // 2
        restart_r = pygame.Rect(x, y, btn_w, 28)
        menu_r = pygame.Rect(x + btn_w + 8, y, btn_w, 28)
        rects["restart"] = restart_r
        rects["menu"] = menu_r
        panel_btn(screen, restart_r, "Restart (R)", False,
                  restart_r.collidepoint(mouse), app.small_font)
        panel_btn(screen, menu_r, "Menu (ESC)", False,
                  menu_r.collidepoint(mouse), app.small_font)
    y += 36

    sep(screen, x, y, w); y += 10

    # ---- Title + subtitle ----
    titles = {"chess": "Chess 8\u00d78", "minichess": "Los Alamos 6\u00d76",
              "shogi": "Shogi 9\u00d79", "minishogi": "Mini Shogi 5\u00d75"}
    t = app.heading_font.render(titles.get(app.game_name, ""), True, TEXT)
    screen.blit(t, (x, y)); y += 28

    # Mode-aware subtitle: "Human (White) vs AI" / "Human vs Human" etc.
    if mode == "human-vs-ai":
        player_txt = "White" if player_side_int == WHITE else "Black"
        subtitle = f"You play {player_txt} vs AI"
    elif mode == "ai-vs-ai":
        subtitle = "AI vs AI"
    else:
        subtitle = "Human vs Human"
    st = app.small_font.render(subtitle, True, TEXT_DIM)
    screen.blit(st, (x, y)); y += 22

    # ---- Turn + info line ----
    side = state.side_to_move()
    side_name = "White" if side == WHITE else "Black"
    turn_lbl = app.font.render(f"Turn: {side_name}", True, TEXT)
    screen.blit(turn_lbl, (x, y))
    _side_dot(screen, x + turn_lbl.get_width() + 14, y + 10, side)
    y += 28

    mode_short = MODE_SHORT.get(app.mode, app.mode)
    info = (f"{mode_short}  \u00b7  Depth {app.ai_depth}"
            f"  \u00b7  Time {app.ai_time_sec}s")
    lbl = app.small_font.render(info, True, TEXT_MUTED)
    screen.blit(lbl, (x, y)); y += 22

    sep(screen, x, y, w); y += 10

    # ---- EVAL SUMMARY ----
    # A prominent read-out of "who's winning and by how much". This is the
    # user's primary answer to "is the position good or bad?" and sits at
    # the top of the panel where it's easy to glance at.
    eval_box = pygame.Rect(x, y, w, 70)
    pygame.draw.rect(screen, PANEL_BG_ALT, eval_box, border_radius=8)
    pygame.draw.rect(screen, PANEL_BORDER, eval_box, 1, border_radius=8)

    eval_txt = _format_eval(white_eval)
    eval_color = _eval_color(white_eval)
    eval_surf = app.score_font.render(eval_txt, True, eval_color)
    screen.blit(eval_surf,
                (eval_box.x + 16,
                 eval_box.y + (eval_box.height - eval_surf.get_height()) // 2))

    winner_lbl = _winner_label(white_eval)
    winner_surf = app.small_font.render(winner_lbl, True, TEXT_DIM)
    screen.blit(winner_surf,
                (eval_box.x + 16 + eval_surf.get_width() + 14,
                 eval_box.y + 14))

    # Tiny note: pawn-units caption below the winner label
    cap = app.tiny_font.render(
        "pawn units  \u00b7  White POV", True, TEXT_MUTED)
    screen.blit(cap,
                (eval_box.x + 16 + eval_surf.get_width() + 14,
                 eval_box.y + 14 + winner_surf.get_height() + 2))

    y += eval_box.height + 10

    sep(screen, x, y, w); y += 10

    # ---- ANALYSIS ----
    lbl = app.small_font.render("ANALYSIS", True, TEXT_DIM)
    screen.blit(lbl, (x, y)); y += 20

    if analysis_on:
        infinite = hint_max_depth > DEPTH_MAX
        if hint_done:
            toggle_lbl = f"ON  \u2014  depth {hint_depth} (done)"
        elif hint_depth > 0:
            cap_depth = "\u221e" if infinite else str(hint_max_depth or DEPTH_MAX)
            toggle_lbl = f"ON  \u2014  depth {hint_depth}/{cap_depth}..."
        else:
            toggle_lbl = "ON  \u2014  starting..."
    else:
        toggle_lbl = "OFF  (H)"
    r = pygame.Rect(x, y, w, 30)
    rects["hint"] = r
    panel_btn(screen, r, toggle_lbl, analysis_on, r.collidepoint(mouse),
              app.small_font)
    y += 36

    # Analysis move results
    _rank_colors = [(255, 200, 40), (200, 200, 210), (210, 150, 90)]
    if analysis_on and hint_moves:
        for i, (mv, score) in enumerate(hint_moves):
            notation = move_notation(app.game_name, mv, state)
            # Format as pawn units for consistency with the eval box.
            cp = score
            pawns = cp / 100.0
            sign = "+" if pawns >= 0 else ""
            line = f"#{i+1}  {notation}  {sign}{pawns:.2f}"
            c = _rank_colors[i] if i < len(_rank_colors) else TEXT_DIM
            s = app.small_font.render(line, True, c)
            screen.blit(s, (x + 4, y)); y += 18
        y += 4

    sep(screen, x, y, w); y += 10

    # ---- Status ----
    chart_h = 130
    chart_top = bg_rect.bottom - chart_h - 32
    max_status_y = chart_top - 6

    lbl = app.small_font.render("STATUS", True, TEXT_DIM)
    screen.blit(lbl, (x, y)); y += 18
    if status_text:
        for line in wrap(status_text, w, app.small_font):
            if y >= max_status_y:
                break
            s = app.small_font.render(line, True, TEXT)
            screen.blit(s, (x, y)); y += 17

    # ---- Score chart (fixed at panel bottom) ----
    sep(screen, x, chart_top - 10, w)
    lbl = app.small_font.render("SCORE HISTORY", True, TEXT_DIM)
    screen.blit(lbl, (x, chart_top - 8))
    draw_score_chart(screen, app.small_font, x, chart_top + 14, w,
                     chart_h - 14, score_history or [])

    return rects
