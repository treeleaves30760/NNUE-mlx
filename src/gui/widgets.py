"""Reusable Pygame widget drawing functions."""

from typing import List

import pygame

from src.gui.constants import (
    TEXT, TEXT_DIM, BTN, BTN_HI, BTN_SEL, BTN_BORDER, SEP,
)


def menu_btn(screen, rect, text, selected, hover, font):
    bg = BTN_SEL if selected else (BTN_HI if hover else BTN)
    pygame.draw.rect(screen, bg, rect, border_radius=6)
    pygame.draw.rect(screen, BTN_BORDER, rect, 1, border_radius=6)
    lbl = font.render(text, True, TEXT)
    screen.blit(lbl, lbl.get_rect(center=rect.center))


def panel_btn(screen, rect, text, selected, hover, font):
    bg = BTN_SEL if selected else (BTN_HI if hover else BTN)
    pygame.draw.rect(screen, bg, rect, border_radius=5)
    pygame.draw.rect(screen, BTN_BORDER, rect, 1, border_radius=5)
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
    menu_btn(screen, minus, "\u25c4", False, minus.collidepoint(mouse), font)
    menu_btn(screen, plus, "\u25ba", False, plus.collidepoint(mouse), font)
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
