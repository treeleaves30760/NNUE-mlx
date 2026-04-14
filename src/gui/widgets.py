"""Reusable Pygame widget drawing functions."""

from typing import List, Tuple

import pygame

from src.gui.constants import (
    TEXT, TEXT_DIM, BTN, BTN_HI, BTN_SEL, BTN_BORDER, SEP,
    EVAL_GOOD, EVAL_BAD, EVAL_NEUTRAL, WHITE_ADV, BLACK_ADV,
    PANEL_BORDER,
)


def menu_btn(screen, rect, text, selected, hover, font):
    bg = BTN_SEL if selected else (BTN_HI if hover else BTN)
    pygame.draw.rect(screen, bg, rect, border_radius=8)
    border = (140, 180, 240) if selected else BTN_BORDER
    pygame.draw.rect(screen, border, rect, 1, border_radius=8)
    lbl = font.render(text, True, TEXT)
    screen.blit(lbl, lbl.get_rect(center=rect.center))


def panel_btn(screen, rect, text, selected, hover, font):
    bg = BTN_SEL if selected else (BTN_HI if hover else BTN)
    pygame.draw.rect(screen, bg, rect, border_radius=6)
    border = (140, 180, 240) if selected else BTN_BORDER
    pygame.draw.rect(screen, border, rect, 1, border_radius=6)
    lbl = font.render(text, True, TEXT)
    screen.blit(lbl, lbl.get_rect(center=rect.center))


def stepper(screen, font, x, y, label, value, width):
    """Menu stepper: label above, \u25c4 value \u25ba below. Returns (minus, plus)."""
    lbl = font.render(label, True, TEXT_DIM)
    screen.blit(lbl, (x, y))
    y2 = y + 24
    btn_w = 36
    minus = pygame.Rect(x, y2, btn_w, 32)
    plus = pygame.Rect(x + width - btn_w, y2, btn_w, 32)
    mouse = pygame.mouse.get_pos()
    menu_btn(screen, minus, "-", False, minus.collidepoint(mouse), font)
    menu_btn(screen, plus, "+", False, plus.collidepoint(mouse), font)
    max_val_w = width - 2 * btn_w - 12
    display = value
    while font.size(display)[0] > max_val_w and len(display) > 4:
        display = display[:-4] + "..."
    val = font.render(display, True, TEXT)
    vx = x + btn_w + (width - 2 * btn_w - val.get_width()) // 2
    screen.blit(val, (vx, y2 + 4))
    return minus, plus


def section_label(screen, font, text, cx, y):
    lbl = font.render(text, True, TEXT_DIM)
    screen.blit(lbl, lbl.get_rect(center=(cx, y)))


def blit_centered(screen, font, text, cx, y, color):
    s = font.render(text, True, color)
    screen.blit(s, s.get_rect(center=(cx, y)))


def sep(screen, x, y, w):
    pygame.draw.line(screen, SEP, (x, y), (x + w, y), 1)


def wrap(text: str, max_w: int, font) -> List[str]:
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


def trunc(text: str, max_w: int, font) -> str:
    if font.size(text)[0] <= max_w:
        return text
    while len(text) > 4 and font.size(text + "...")[0] > max_w:
        text = text[:-1]
    return text + "..."


# ---------------------------------------------------------------- eval bar

def draw_eval_bar(screen, rect: pygame.Rect, pawns: float,
                  flipped: bool, font) -> None:
    """Vertical eval bar showing which side is winning.

    ``pawns`` is the score in pawn units from White's perspective
    (positive = White ahead, negative = Black ahead). ``flipped``
    means the player is viewing the board from Black's side, so the
    bar's orientation should mirror — White's share sits at the top
    instead of the bottom.

    The bar uses a logistic mapping so large advantages taper off
    smoothly (reaching ~95% at +6 pawns), preventing the split from
    slamming all the way to an edge during a sharp tactical swing.
    """
    import math
    # Logistic: 0.5 at 0, ~0.92 at +4, ~0.98 at +6, saturates.
    white_frac = 1.0 / (1.0 + math.exp(-pawns / 1.8))
    white_frac = max(0.02, min(0.98, white_frac))

    x, y, w, h = rect.x, rect.y, rect.width, rect.height

    # Background
    pygame.draw.rect(screen, (20, 22, 28), rect, border_radius=6)
    pygame.draw.rect(screen, PANEL_BORDER, rect, 1, border_radius=6)

    inset = pygame.Rect(x + 2, y + 2, w - 4, h - 4)
    ix, iy, iw, ih = inset.x, inset.y, inset.width, inset.height

    if flipped:
        # White at top, Black at bottom.
        white_h = int(ih * white_frac)
        black_h = ih - white_h
        pygame.draw.rect(screen, WHITE_ADV,
                         (ix, iy, iw, white_h), border_radius=4)
        pygame.draw.rect(screen, BLACK_ADV,
                         (ix, iy + white_h, iw, black_h), border_radius=4)
        split_y = iy + white_h
    else:
        # Black at top, White at bottom (standard lichess orientation).
        white_h = int(ih * white_frac)
        black_h = ih - white_h
        pygame.draw.rect(screen, BLACK_ADV,
                         (ix, iy, iw, black_h), border_radius=4)
        pygame.draw.rect(screen, WHITE_ADV,
                         (ix, iy + black_h, iw, white_h), border_radius=4)
        split_y = iy + black_h

    # Split line
    pygame.draw.line(screen, (140, 150, 170),
                     (ix, split_y), (ix + iw, split_y), 1)

    # Numeric overlay: the leading side's eval.
    if abs(pawns) < 0.05:
        text_str = "0.0"
        text_color = EVAL_NEUTRAL
        text_bg_white = True  # center neutral label
    elif pawns > 0:
        text_str = f"+{pawns:.1f}"
        text_color = (30, 32, 40)
        text_bg_white = True
    else:
        text_str = f"{pawns:.1f}"
        text_color = (230, 232, 240)
        text_bg_white = False

    lbl = font.render(text_str, True, text_color)
    lx = x + (w - lbl.get_width()) // 2
    if text_bg_white:
        # Label in the white region
        if flipped:
            ly = iy + 6
        else:
            ly = iy + ih - lbl.get_height() - 6
    else:
        # Label in the black region
        if flipped:
            ly = iy + ih - lbl.get_height() - 6
        else:
            ly = iy + 6
    screen.blit(lbl, (lx, ly))


def draw_toggle(screen, rect, label, on, hover, font):
    """Small horizontal toggle button with on/off state."""
    bg = BTN_SEL if on else (BTN_HI if hover else BTN)
    pygame.draw.rect(screen, bg, rect, border_radius=6)
    border = (140, 180, 240) if on else BTN_BORDER
    pygame.draw.rect(screen, border, rect, 1, border_radius=6)
    lbl = font.render(label, True, TEXT)
    screen.blit(lbl, lbl.get_rect(center=rect.center))
