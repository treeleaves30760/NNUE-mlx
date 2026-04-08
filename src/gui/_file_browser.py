"""Standalone Pygame file browser for selecting .npz model files."""

import os
from pathlib import Path
from typing import Optional

import pygame

from src.gui.constants import BG, TEXT, TEXT_DIM, BTN, BTN_HI, BTN_BORDER
from src.gui.widgets import blit_centered, trunc, menu_btn


def browse_model(
    screen: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    heading_font: pygame.font.Font,
    W: int,
    H: int,
    clock: pygame.time.Clock,
) -> Optional[str]:
    """Pygame-based file browser for .npz model files.

    Returns the selected file path as a string, or None if cancelled.
    """
    models_root = Path(__file__).resolve().parent.parent.parent / "models"
    cur_dir = models_root if models_root.is_dir() else Path.cwd()

    scroll_off = 0
    ROW_H = 36
    PAD = 20
    # usable area
    list_x, list_y = PAD, 80
    list_w = W - 2 * PAD
    list_h = H - 160

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

        screen.fill(BG)
        # header
        blit_centered(screen, heading_font,
                      "Select NNUE Model (.npz)", W // 2, 30, TEXT)
        # path breadcrumb
        path_str = str(cur_dir)
        blit_centered(screen, small_font, trunc(path_str, W - 40, small_font),
                      W // 2, 58, TEXT_DIM)

        # clip region for list
        clip = pygame.Rect(list_x, list_y, list_w, list_h)
        screen.set_clip(clip)
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
            pygame.draw.rect(screen, bg, row_rect, border_radius=4)
            prefix = "\U0001f4c1 " if is_dir else "\U0001f4c4 "
            lbl = font.render(prefix + name, True, TEXT)
            screen.blit(lbl, (list_x + 10, ry + 6))
        screen.set_clip(None)

        # border around list
        pygame.draw.rect(screen, BTN_BORDER, clip, 1, border_radius=4)

        # bottom bar
        cancel_r = pygame.Rect(W // 2 - 80, H - 58, 160, 40)
        menu_btn(screen, cancel_r, "Cancel", False,
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
        clock.tick(30)
