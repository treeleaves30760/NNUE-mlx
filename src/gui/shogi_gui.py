"""Shogi (9x9) board renderer for Pygame."""

from typing import Optional

import pygame

from src.games.base import GameConfig, GameState, WHITE, BLACK
from src.gui.board_renderer import BoardRenderer
from src.gui.pieces import PieceRenderer, SHOGI_KANJI
from src.gui.themes import Theme


SHOGI_CONFIG = GameConfig(
    name="shogi", board_height=9, board_width=9,
    num_piece_types=13, has_drops=True, has_promotion=True,
)


class ShogiRenderer(BoardRenderer):
    """Renderer for standard 9x9 shogi."""

    def __init__(self, theme: Theme, square_size: int = 70):
        super().__init__(SHOGI_CONFIG, theme, square_size)
        self.piece_renderer = PieceRenderer(square_size)
        self.hand_renderer = PieceRenderer(42)

    def _display_rank(self, rank: int) -> int:
        # Shogi: rank 0 at top, no flip needed
        return rank

    def _display_file(self, file: int) -> int:
        # Shogi: files numbered 9-1 right to left, but display left to right
        return self.config.board_width - 1 - file

    def _rank_from_display(self, display_rank: int) -> int:
        return display_rank

    def _file_from_display(self, display_file: int) -> int:
        return self.config.board_width - 1 - display_file

    def _draw_squares(self, surface: pygame.Surface):
        # Shogi uses uniform colored board with grid lines
        surface.fill(self.theme.light_sq)
        for i in range(self.config.board_width + 1):
            x = i * self.square_size
            pygame.draw.line(surface, (0, 0, 0), (x, 0),
                             (x, self.board_pixel_h), 1)
        for i in range(self.config.board_height + 1):
            y = i * self.square_size
            pygame.draw.line(surface, (0, 0, 0), (0, y),
                             (self.board_pixel_w, y), 1)
        # Star points (hoshi) - use sq_to_pixel so flipping is handled
        for r, f in [(2, 2), (2, 5), (2, 8), (5, 2), (5, 5), (5, 8),
                     (8, 2), (8, 5), (8, 8)]:
            if r < 9 and f < 9:
                # The hoshi sit on grid intersections between ranks/files.
                # Use sq_to_pixel to find the square's top-left, then the
                # grid intersection is at the square's top-left corner.
                # Internal (r, f) hoshi sits at intersection between squares
                # (r, f) and (r-1, f+1) (for right-to-left files).
                x, y = self.sq_to_pixel(r * 9 + f)
                if self.flipped:
                    cx = x + self.square_size
                    cy = y + self.square_size
                else:
                    cx = x + self.square_size
                    cy = y + self.square_size
                pygame.draw.circle(surface, (0, 0, 0), (cx, cy), 3)

    def _draw_pieces(self, surface: pygame.Surface, state: GameState):
        board = state.board_array()
        for sq in range(81):
            piece = board[sq]
            if piece == 0:
                continue
            x, y = self.sq_to_pixel(sq)
            # When the board is flipped, rotate the piece tile too so
            # the player sees their own pieces "right side up" from their
            # side of the board.
            piece_surf = self.piece_renderer.render_shogi_piece(
                piece, is_mini=False, flipped_board=self.flipped)
            px = x + (self.square_size - piece_surf.get_width()) // 2
            py = y + (self.square_size - piece_surf.get_height()) // 2
            surface.blit(piece_surf, (px, py))

    def _draw_coordinates(self, surface: pygame.Surface):
        font = pygame.font.SysFont("Arial", 12)
        # File numbers: when not flipped, 9-1 from left to right.
        # When flipped (viewing from Black's side), 1-9 from left to right.
        for i in range(9):
            if self.flipped:
                file_num = i + 1
            else:
                file_num = 9 - i
            label = font.render(str(file_num), True, (80, 80, 80))
            x = i * self.square_size + self.square_size // 2 - 4
            surface.blit(label, (x, -2))
        # Rank labels a-i (top to bottom); reversed when flipped.
        ranks = "abcdefghi"
        for i in range(9):
            idx = (8 - i) if self.flipped else i
            label = font.render(ranks[idx], True, (80, 80, 80))
            x = self.board_pixel_w + 2
            y = i * self.square_size + self.square_size // 2 - 6
            surface.blit(label, (x, y))

    # ---- hand bar (above / below board) ----

    def _hand_layout(self, state, side, area_rect):
        """Compute tile positions for hand bar. Returns (tile_sz, pieces, start_x, tile_y, spacing)."""
        hand = state.hand_pieces(side)
        pieces = [(pt, c) for pt, c in sorted(hand.items()) if c > 0]
        tile_sz = self.hand_renderer._piece_size
        spacing = 8
        total_w = len(pieces) * (tile_sz + spacing) - spacing if pieces else 0
        start_x = (area_rect.width - total_w) // 2
        tile_y = (area_rect.height - tile_sz) // 2
        return tile_sz, pieces, start_x, tile_y, spacing

    def draw_hand(self, surface: pygame.Surface, state: GameState,
                  side: int, area_rect: pygame.Rect):
        """Draw hand pieces as tiles in a horizontal bar."""
        pygame.draw.rect(surface, (50, 48, 46), area_rect, border_radius=4)
        pygame.draw.rect(surface, (80, 75, 70), area_rect, 1, border_radius=4)

        tile_sz, pieces, start_x, tile_y, spacing = \
            self._hand_layout(state, side, area_rect)
        if not pieces:
            return

        count_font = pygame.font.SysFont("Arial", 13, bold=True)
        for i, (pt, count) in enumerate(pieces):
            x = area_rect.x + start_x + i * (tile_sz + spacing)
            y = area_rect.y + tile_y
            code = (pt + 1) if side == WHITE else -(pt + 1)
            tile = self.hand_renderer.render_shogi_piece(
                code, is_mini=False, flipped_board=self.flipped)
            surface.blit(tile, (x, y))
            if count > 1:
                txt = count_font.render(str(count), True, (255, 255, 255))
                bx = x + tile_sz - 4
                by = y + tile_sz - 10
                pygame.draw.circle(surface, (180, 50, 50), (bx, by), 9)
                surface.blit(txt, (bx - txt.get_width() // 2,
                                   by - txt.get_height() // 2))

    def hand_piece_at(self, x: int, y: int, state: GameState,
                      side: int, area_rect: pygame.Rect) -> Optional[int]:
        """Check if (x, y) relative to area_rect hits a hand piece tile."""
        tile_sz, pieces, start_x, tile_y, spacing = \
            self._hand_layout(state, side, area_rect)
        for i, (pt, _) in enumerate(pieces):
            px = start_x + i * (tile_sz + spacing)
            if px <= x < px + tile_sz and tile_y <= y < tile_y + tile_sz:
                return pt
        return None

    def hand_piece_center(self, state, side, piece_type, area_rect):
        tile_sz, pieces, start_x, tile_y, spacing = \
            self._hand_layout(state, side, area_rect)
        for i, (pt, _) in enumerate(pieces):
            if pt == piece_type:
                cx = area_rect.x + start_x + i * (tile_sz + spacing) + tile_sz // 2
                cy = area_rect.y + tile_y + tile_sz // 2
                return (cx, cy)
        return None
