"""Abstract board renderer for Pygame GUIs."""

from abc import ABC, abstractmethod
import math
from typing import List, Optional, Tuple

import pygame

from src.games.base import GameConfig, GameState, Move
from src.gui.themes import Theme

# Colors for top-3 hint arrows: gold, silver, bronze
_HINT_COLORS = [
    (255, 200, 40, 180),   # #1 gold
    (180, 180, 190, 160),  # #2 silver
    (200, 140, 80, 150),   # #3 bronze
]


class BoardRenderer(ABC):
    """Base class for rendering a game board in pygame.

    ``flipped`` rotates the display 180° — used to let the player view
    the board from Black/gote's side without changing the underlying
    game state. Subclasses use ``_display_rank``/``_display_file`` to
    express their natural orientation; the base class composes that
    with the flip flag in ``sq_to_pixel``/``pixel_to_sq``.
    """

    def __init__(self, config: GameConfig, theme: Theme, square_size: int = 80):
        self.config = config
        self.theme = theme
        self.square_size = square_size
        self.board_pixel_w = config.board_width * square_size
        self.board_pixel_h = config.board_height * square_size
        self.flipped = False

    def set_flipped(self, flipped: bool) -> None:
        self.flipped = bool(flipped)

    def sq_to_pixel(self, sq: int) -> tuple:
        """Convert board square index to pixel coordinates (top-left of square)."""
        rank = sq // self.config.board_width
        file = sq % self.config.board_width
        display_rank = self._display_rank(rank)
        display_file = self._display_file(file)
        if self.flipped:
            display_rank = self.config.board_height - 1 - display_rank
            display_file = self.config.board_width - 1 - display_file
        x = display_file * self.square_size
        y = display_rank * self.square_size
        return (x, y)

    def pixel_to_sq(self, x: int, y: int) -> Optional[int]:
        """Convert pixel coordinates to board square index, or None if outside board."""
        if x < 0 or y < 0 or x >= self.board_pixel_w or y >= self.board_pixel_h:
            return None
        display_file = x // self.square_size
        display_rank = y // self.square_size
        if self.flipped:
            display_rank = self.config.board_height - 1 - display_rank
            display_file = self.config.board_width - 1 - display_file
        file = self._file_from_display(display_file)
        rank = self._rank_from_display(display_rank)
        if 0 <= rank < self.config.board_height and 0 <= file < self.config.board_width:
            return rank * self.config.board_width + file
        return None

    def draw_board(self, surface: pygame.Surface, state: GameState,
                   selected_sq: Optional[int] = None,
                   legal_targets: Optional[List[int]] = None):
        """Draw the board background, highlighting, and pieces."""
        self._draw_squares(surface)
        if selected_sq is not None:
            self._draw_selected(surface, selected_sq)
        if legal_targets:
            self._draw_legal_targets(surface, legal_targets)
        self._draw_pieces(surface, state)
        self._draw_coordinates(surface)

    def _draw_squares(self, surface: pygame.Surface):
        """Draw the checkerboard pattern."""
        for sq in range(self.config.num_squares):
            rank = sq // self.config.board_width
            file = sq % self.config.board_width
            is_light = (rank + file) % 2 == 0
            color = self.theme.light_sq if is_light else self.theme.dark_sq
            x, y = self.sq_to_pixel(sq)
            pygame.draw.rect(surface, color,
                             (x, y, self.square_size, self.square_size))

    def _draw_selected(self, surface: pygame.Surface, sq: int):
        """Highlight the selected square."""
        x, y = self.sq_to_pixel(sq)
        highlight = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
        highlight.fill(self.theme.selected)
        surface.blit(highlight, (x, y))

    def _draw_legal_targets(self, surface: pygame.Surface, targets: List[int]):
        """Draw legal move indicators."""
        for sq in targets:
            x, y = self.sq_to_pixel(sq)
            center = (x + self.square_size // 2, y + self.square_size // 2)
            radius = self.square_size // 6
            dot = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            pygame.draw.circle(dot, self.theme.highlight,
                               (self.square_size // 2, self.square_size // 2), radius)
            surface.blit(dot, (x, y))

    def _draw_coordinates(self, surface: pygame.Surface):
        """Draw file/rank labels. Override for game-specific labels."""
        pass

    @abstractmethod
    def _draw_pieces(self, surface: pygame.Surface, state: GameState):
        """Draw all pieces on the board."""
        ...

    @abstractmethod
    def _display_rank(self, rank: int) -> int:
        """Convert internal rank to display rank (for drawing)."""
        ...

    @abstractmethod
    def _display_file(self, file: int) -> int:
        """Convert internal file to display file (for drawing)."""
        ...

    def _rank_from_display(self, display_rank: int) -> int:
        """Inverse of _display_rank. Override if not simple inverse."""
        # Default assumes _display_rank is its own inverse
        return self._display_rank(display_rank)

    def _file_from_display(self, display_file: int) -> int:
        """Inverse of _display_file. Override if not simple inverse."""
        return self._display_file(display_file)

    def draw_hand(self, surface: pygame.Surface, state: GameState,
                  side: int, area_rect: pygame.Rect):
        """Draw hand pieces for shogi variants. No-op by default."""
        pass

    def hand_piece_at(self, x: int, y: int, state: GameState,
                      side: int, area_rect: pygame.Rect) -> Optional[int]:
        """Return piece type if click is on a hand piece, else None."""
        return None

    def hand_piece_center(self, state: GameState, side: int,
                          piece_type: int,
                          area_rect: pygame.Rect) -> Optional[Tuple[int, int]]:
        """Return screen-coords center of a hand-piece tile, or None."""
        return None

    # ----------------------------------------------------------------- hints

    def draw_hints(self, surface: pygame.Surface,
                   hints: List[Tuple[Move, float]]):
        """Draw ranked arrows for the top-N move hints."""
        font = pygame.font.SysFont("Arial", 14, bold=True)
        for rank, (move, score) in enumerate(hints):
            color = _HINT_COLORS[rank] if rank < len(_HINT_COLORS) else _HINT_COLORS[-1]

            if move.from_sq is not None:
                x1, y1 = self.sq_to_pixel(move.from_sq)
                x2, y2 = self.sq_to_pixel(move.to_sq)
                self._draw_arrow(surface, x1, y1, x2, y2, color)
            else:
                # Drop move: highlight target square
                x2, y2 = self.sq_to_pixel(move.to_sq)

            # Rank badge on destination square
            bx = x2 + self.square_size - 18
            by = y2 + 2
            badge_bg = (color[0], color[1], color[2])
            pygame.draw.circle(surface, badge_bg, (bx + 8, by + 8), 10)
            pygame.draw.circle(surface, (0, 0, 0), (bx + 8, by + 8), 10, 1)
            label = font.render(str(rank + 1), True, (0, 0, 0))
            surface.blit(label, (bx + 8 - label.get_width() // 2,
                                 by + 8 - label.get_height() // 2))

            # Score text next to badge
            score_label = font.render(f"{score:+.0f}", True, badge_bg)
            surface.blit(score_label, (bx - score_label.get_width() - 2, by + 2))

    def _draw_arrow(self, surface: pygame.Surface,
                    x1: int, y1: int, x2: int, y2: int,
                    color: Tuple[int, ...]):
        """Draw a thick arrow from center of one square to another."""
        half = self.square_size // 2
        sx, sy = x1 + half, y1 + half
        ex, ey = x2 + half, y2 + half
        dx, dy = ex - sx, ey - sy
        dist = math.hypot(dx, dy)
        if dist < 1:
            return

        # Unit vectors
        ux, uy = dx / dist, dy / dist
        # Perpendicular
        px, py = -uy, ux

        # Shorten arrow tip a bit
        head_len = min(self.square_size * 0.35, dist * 0.4)
        shaft_w = self.square_size * 0.12
        head_w = self.square_size * 0.28

        # Arrow tip point
        tip = (ex, ey)
        # Base of arrowhead
        base_x, base_y = ex - ux * head_len, ey - uy * head_len
        # Shaft end (same as arrowhead base)
        shaft_end = (base_x, base_y)

        # Shaft polygon
        shaft = [
            (sx + px * shaft_w, sy + py * shaft_w),
            (base_x + px * shaft_w, base_y + py * shaft_w),
            (base_x - px * shaft_w, base_y - py * shaft_w),
            (sx - px * shaft_w, sy - py * shaft_w),
        ]
        # Arrowhead triangle
        head = [
            tip,
            (base_x + px * head_w, base_y + py * head_w),
            (base_x - px * head_w, base_y - py * head_w),
        ]

        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(overlay, color, shaft)
        pygame.draw.polygon(overlay, color, head)
        # Outline
        outline = (0, 0, 0, color[3] if len(color) > 3 else 180)
        pygame.draw.polygon(overlay, outline, shaft, 1)
        pygame.draw.polygon(overlay, outline, head, 1)
        surface.blit(overlay, (0, 0))

    def draw_drop_hints(self, surface: pygame.Surface,
                        hints: List[Tuple[Move, float]],
                        state: GameState,
                        board_offset: Tuple[int, int],
                        sente_rect: Optional[pygame.Rect],
                        gote_rect: Optional[pygame.Rect]):
        """Draw arrows for drop-move hints from hand area to board."""
        from src.games.base import WHITE
        side = state.side_to_move()
        hand_rect = sente_rect if side == WHITE else gote_rect
        if hand_rect is None:
            return

        drops = [(r, m, s) for r, (m, s) in enumerate(hints)
                 if m.from_sq is None]
        if not drops:
            return

        bx, by = board_offset

        for rank, move, score in drops:
            color = _HINT_COLORS[rank] if rank < len(_HINT_COLORS) \
                else _HINT_COLORS[-1]
            rgb = (color[0], color[1], color[2])
            src = self.hand_piece_center(state, side,
                                         move.drop_piece, hand_rect)
            if src is None:
                continue

            tx_rel, ty_rel = self.sq_to_pixel(move.to_sq)
            half = self.square_size // 2
            ex, ey = bx + tx_rel + half, by + ty_rel + half
            sx, sy = src

            dx, dy = ex - sx, ey - sy
            dist = math.hypot(dx, dy)
            if dist < 1:
                continue
            ux, uy = dx / dist, dy / dist
            px, py = -uy, ux

            head_len = min(self.square_size * 0.35, dist * 0.4)
            shaft_w = self.square_size * 0.12
            head_w = self.square_size * 0.28
            base_x, base_y = ex - ux * head_len, ey - uy * head_len

            shaft = [
                (sx + px * shaft_w, sy + py * shaft_w),
                (base_x + px * shaft_w, base_y + py * shaft_w),
                (base_x - px * shaft_w, base_y - py * shaft_w),
                (sx - px * shaft_w, sy - py * shaft_w),
            ]
            head_poly = [
                (ex, ey),
                (base_x + px * head_w, base_y + py * head_w),
                (base_x - px * head_w, base_y - py * head_w),
            ]
            pygame.draw.polygon(surface, rgb, shaft)
            pygame.draw.polygon(surface, rgb, head_poly)
            pygame.draw.polygon(surface, (0, 0, 0), shaft, 1)
            pygame.draw.polygon(surface, (0, 0, 0), head_poly, 1)
