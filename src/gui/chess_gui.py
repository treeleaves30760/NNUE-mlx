"""Chess board renderer for Pygame."""

import pygame

from src.games.base import GameConfig, GameState
from src.gui.board_renderer import BoardRenderer
from src.gui.pieces import PieceRenderer
from src.gui.themes import Theme


CHESS_CONFIG = GameConfig(
    name="chess", board_height=8, board_width=8,
    num_piece_types=5, has_drops=False, has_promotion=True,
)


class ChessRenderer(BoardRenderer):
    """Renderer for standard 8x8 chess."""

    def __init__(self, theme: Theme, square_size: int = 80):
        super().__init__(CHESS_CONFIG, theme, square_size)
        self.piece_renderer = PieceRenderer(square_size)

    def _display_rank(self, rank: int) -> int:
        # Flip: rank 0 (a1) at bottom -> display row 7
        return self.config.board_height - 1 - rank

    def _display_file(self, file: int) -> int:
        return file

    def _draw_pieces(self, surface: pygame.Surface, state: GameState):
        board = state.board_array()
        for sq in range(64):
            piece = board[sq]
            if piece == 0:
                continue
            x, y = self.sq_to_pixel(sq)
            piece_surf = self.piece_renderer.render_chess_piece(piece)
            # Center the piece in the square
            px = x + (self.square_size - piece_surf.get_width()) // 2
            py = y + (self.square_size - piece_surf.get_height()) // 2
            surface.blit(piece_surf, (px, py))

    def _draw_coordinates(self, surface: pygame.Surface):
        font = pygame.font.SysFont("Arial", 12)
        files = "abcdefgh"
        for i in range(8):
            # Pick color that contrasts with the square color
            is_light_bottom = ((7 + i) % 2 == 0)
            is_light_left = ((7 - i + 0) % 2 == 0)
            file_color = self.theme.dark_sq if is_light_bottom else self.theme.light_sq
            rank_color = self.theme.dark_sq if is_light_left else self.theme.light_sq

            # File labels in bottom-right of each bottom-row square.
            # When flipped, files run h..a instead of a..h.
            file_char = files[7 - i] if self.flipped else files[i]
            label = font.render(file_char, True, file_color)
            x = i * self.square_size + self.square_size - 12
            y = self.board_pixel_h - 14
            surface.blit(label, (x, y))

            # Rank labels in top-left of each left-column square.
            # When flipped, top row shows rank 1 instead of rank 8.
            rank_num = (i + 1) if self.flipped else (8 - i)
            label = font.render(str(rank_num), True, rank_color)
            y = i * self.square_size + 2
            surface.blit(label, (2, y))
