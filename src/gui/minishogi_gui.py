"""Mini Shogi (5x5) board renderer for Pygame."""

from typing import Optional

import pygame

from src.games.base import GameConfig, GameState, WHITE, BLACK
from src.gui.board_renderer import BoardRenderer
from src.gui.pieces import PieceRenderer
from src.gui.themes import Theme


MINISHOGI_CONFIG = GameConfig(
    name="minishogi", board_height=5, board_width=5,
    num_piece_types=9, has_drops=True, has_promotion=True,
)


class MiniShogiRenderer(BoardRenderer):
    """Renderer for 5x5 mini shogi."""

    def __init__(self, theme: Theme, square_size: int = 90):
        super().__init__(MINISHOGI_CONFIG, theme, square_size)
        self.piece_renderer = PieceRenderer(square_size)
        self.hand_renderer = PieceRenderer(42)

    def _display_rank(self, rank: int) -> int:
        return rank

    def _display_file(self, file: int) -> int:
        return self.config.board_width - 1 - file

    def _rank_from_display(self, display_rank: int) -> int:
        return display_rank

    def _file_from_display(self, display_file: int) -> int:
        return self.config.board_width - 1 - display_file

    def _draw_squares(self, surface: pygame.Surface):
        surface.fill(self.theme.light_sq)
        for i in range(self.config.board_width + 1):
            x = i * self.square_size
            pygame.draw.line(surface, (0, 0, 0), (x, 0),
                             (x, self.board_pixel_h), 1)
        for i in range(self.config.board_height + 1):
            y = i * self.square_size
            pygame.draw.line(surface, (0, 0, 0), (0, y),
                             (self.board_pixel_w, y), 1)

    def _draw_pieces(self, surface: pygame.Surface, state: GameState):
        board = state.board_array()
        for sq in range(25):
            piece = board[sq]
            if piece == 0:
                continue
            x, y = self.sq_to_pixel(sq)
            piece_surf = self.piece_renderer.render_shogi_piece(piece, is_mini=True)
            px = x + (self.square_size - piece_surf.get_width()) // 2
            py = y + (self.square_size - piece_surf.get_height()) // 2
            surface.blit(piece_surf, (px, py))

    def _draw_coordinates(self, surface: pygame.Surface):
        font = pygame.font.SysFont("Arial", 12)
        for i in range(5):
            file_num = 5 - i
            label = font.render(str(file_num), True, (80, 80, 80))
            x = i * self.square_size + self.square_size // 2 - 4
            surface.blit(label, (x, -2))
        ranks = "abcde"
        for i in range(5):
            label = font.render(ranks[i], True, (80, 80, 80))
            x = self.board_pixel_w + 2
            y = i * self.square_size + self.square_size // 2 - 6
            surface.blit(label, (x, y))

    # ---- hand bar (above / below board) ----

    def _hand_layout(self, state, side, area_rect):
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
            tile = self.hand_renderer.render_shogi_piece(code, is_mini=True)
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
