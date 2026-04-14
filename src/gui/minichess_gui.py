"""Los Alamos mini chess (6x6) board renderer for Pygame."""

import pygame

from src.games.base import GameConfig, GameState
from src.gui.board_renderer import BoardRenderer
from src.gui.pieces import PieceRenderer
from src.gui.themes import Theme


MINICHESS_CONFIG = GameConfig(
    name="minichess", board_height=6, board_width=6,
    num_piece_types=4, has_drops=False, has_promotion=True,
)


class MiniChessRenderer(BoardRenderer):
    """Renderer for Los Alamos 6x6 chess."""

    def __init__(self, theme: Theme, square_size: int = 90):
        super().__init__(MINICHESS_CONFIG, theme, square_size)
        self.piece_renderer = PieceRenderer(square_size)

    def _display_rank(self, rank: int) -> int:
        return self.config.board_height - 1 - rank

    def _display_file(self, file: int) -> int:
        return file

    def _draw_pieces(self, surface: pygame.Surface, state: GameState):
        board = state.board_array()
        for sq in range(36):
            piece = board[sq]
            if piece == 0:
                continue
            x, y = self.sq_to_pixel(sq)
            piece_surf = self.piece_renderer.render_minichess_piece(piece)
            px = x + (self.square_size - piece_surf.get_width()) // 2
            py = y + (self.square_size - piece_surf.get_height()) // 2
            surface.blit(piece_surf, (px, py))

    def _draw_coordinates(self, surface: pygame.Surface):
        font = pygame.font.SysFont("Arial", 12)
        files = "abcdef"
        h = self.config.board_height
        for i in range(6):
            is_light_bottom = ((h - 1 + i) % 2 == 0)
            is_light_left = ((h - 1 - i) % 2 == 0)
            file_color = self.theme.dark_sq if is_light_bottom else self.theme.light_sq
            rank_color = self.theme.dark_sq if is_light_left else self.theme.light_sq

            file_char = files[5 - i] if self.flipped else files[i]
            label = font.render(file_char, True, file_color)
            x = i * self.square_size + self.square_size - 12
            y = self.board_pixel_h - 14
            surface.blit(label, (x, y))

            rank_num = (i + 1) if self.flipped else (6 - i)
            label = font.render(str(rank_num), True, rank_color)
            y = i * self.square_size + 2
            surface.blit(label, (2, y))
