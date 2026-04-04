"""Piece rendering for chess (Unicode emoji) and shogi (kanji text)."""

import pygame

# ---------------------------------------------------------------------------
# Chess Unicode symbols (rendered via "Apple Symbols" font on macOS)
# White pieces are outline glyphs, black pieces are filled glyphs.
# ---------------------------------------------------------------------------

# Chess: 1=Pawn, 2=Knight, 3=Bishop, 4=Rook, 5=Queen, 6=King
CHESS_WHITE = {6: "\u2654", 5: "\u2655", 4: "\u2656", 3: "\u2657", 2: "\u2658", 1: "\u2659"}
CHESS_BLACK = {6: "\u265a", 5: "\u265b", 4: "\u265c", 3: "\u265d", 2: "\u265e", 1: "\u265f"}

# Los Alamos: 1=Pawn, 2=Knight, 3=Rook, 4=Queen, 5=King (no bishop)
MC_WHITE = {5: "\u2654", 4: "\u2655", 3: "\u2656", 2: "\u2658", 1: "\u2659"}
MC_BLACK = {5: "\u265a", 4: "\u265b", 3: "\u265c", 2: "\u265e", 1: "\u265f"}

# ---------------------------------------------------------------------------
# Shogi kanji
# ---------------------------------------------------------------------------

SHOGI_KANJI = {
    1: "\u6b69", 2: "\u9999", 3: "\u6842", 4: "\u9280",
    5: "\u91d1", 6: "\u89d2", 7: "\u98db", 8: "\u738b",
    9: "\u3068", 10: "\u674f", 11: "\u572d", 12: "\u5168",
    13: "\u99ac", 14: "\u9f8d",
}

MINISHOGI_KANJI = {
    1: "\u6b69", 2: "\u9280", 3: "\u91d1", 4: "\u89d2",
    5: "\u98db", 6: "\u738b",
    7: "\u3068", 8: "\u5168", 9: "\u99ac", 10: "\u9f8d",
}

# Shogi tile colors
_SENTE_BG = (255, 248, 230)
_GOTE_BG = (60, 60, 60)
_SENTE_FG = (20, 20, 20)
_GOTE_FG = (230, 230, 230)
_SENTE_PROMOTED = (200, 30, 30)
_GOTE_PROMOTED = (255, 100, 100)


def _find_cjk_font(size: int) -> pygame.font.Font:
    """Find a font that supports CJK characters on macOS."""
    for name in ["Hiragino Sans", "Hiragino Kaku Gothic Pro",
                 "Hiragino Kaku Gothic ProN", "Arial Unicode MS"]:
        try:
            f = pygame.font.SysFont(name, size)
            if f.render("\u738b", True, (0, 0, 0)).get_width() > 4:
                return f
        except Exception:
            continue
    return pygame.font.SysFont(None, size)


class PieceRenderer:
    """Renders chess pieces as emoji symbols, shogi pieces as kanji on tiles."""

    def __init__(self, square_size: int):
        self.square_size = square_size
        self._piece_size = int(square_size * 0.82)
        # Apple Symbols is the font that correctly renders chess Unicode glyphs
        emoji_size = int(square_size * 0.72)
        self._emoji_font = pygame.font.SysFont("Apple Symbols", emoji_size)
        self._cjk_font = _find_cjk_font(int(square_size * 0.50))
        self._small_cjk_font = _find_cjk_font(int(square_size * 0.35))

    # ------------------------------------------------------------------ chess

    def render_chess_piece(self, piece_code: int) -> pygame.Surface:
        """Render a chess piece using Unicode emoji glyph."""
        abs_code = abs(piece_code)
        if piece_code > 0:
            char = CHESS_WHITE.get(abs_code, "?")
        else:
            char = CHESS_BLACK.get(abs_code, "?")
        return self._emoji_font.render(char, True, (0, 0, 0))

    def render_minichess_piece(self, piece_code: int) -> pygame.Surface:
        """Render a Los Alamos chess piece using Unicode emoji glyph."""
        abs_code = abs(piece_code)
        if piece_code > 0:
            char = MC_WHITE.get(abs_code, "?")
        else:
            char = MC_BLACK.get(abs_code, "?")
        return self._emoji_font.render(char, True, (0, 0, 0))

    # ------------------------------------------------------------------ shogi

    def render_shogi_piece(self, piece_code: int, is_mini: bool = False) -> pygame.Surface:
        """Render a shogi piece as kanji on a pentagonal tile."""
        abs_code = abs(piece_code)
        is_sente = piece_code > 0
        lookup = MINISHOGI_KANJI if is_mini else SHOGI_KANJI
        char = lookup.get(abs_code, "?")

        is_promoted = (not is_mini and abs_code >= 9) or (is_mini and abs_code >= 7)
        if is_sente:
            bg, fg = _SENTE_BG, (_SENTE_PROMOTED if is_promoted else _SENTE_FG)
        else:
            bg, fg = _GOTE_BG, (_GOTE_PROMOTED if is_promoted else _GOTE_FG)

        return self._kanji_on_tile(char, bg, fg, flip=not is_sente)

    def _kanji_on_tile(self, char: str, bg, fg, flip: bool) -> pygame.Surface:
        size = self._piece_size
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        m = 2
        points = [
            (size // 2, m), (size - m, size // 4),
            (size - m, size - m), (m, size - m), (m, size // 4),
        ]
        pygame.draw.polygon(surf, bg, points)
        pygame.draw.polygon(surf, (120, 120, 120), points, 2)

        text = self._cjk_font.render(char, True, fg)
        surf.blit(text, ((size - text.get_width()) // 2,
                         (size - text.get_height()) // 2 + 2))
        if flip:
            surf = pygame.transform.rotate(surf, 180)
        return surf

    def render_hand_piece(self, piece_code: int, count: int,
                          is_mini: bool = False) -> pygame.Surface:
        """Render a hand piece label with count."""
        lookup = MINISHOGI_KANJI if is_mini else SHOGI_KANJI
        char = lookup.get(piece_code, "?")
        text = f"{char}{count}" if count > 1 else char
        return self._small_cjk_font.render(text, True, (220, 220, 220))
