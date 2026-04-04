"""Color themes and board styles for the GUI."""


class Theme:
    """A visual theme for the game board."""

    def __init__(self, name, light_sq, dark_sq, highlight, selected,
                 bg_color, text_color, hand_bg):
        self.name = name
        self.light_sq = light_sq
        self.dark_sq = dark_sq
        self.highlight = highlight  # legal move indicator
        self.selected = selected    # selected piece highlight
        self.bg_color = bg_color
        self.text_color = text_color
        self.hand_bg = hand_bg      # shogi hand piece area


CLASSIC = Theme(
    name="Classic",
    light_sq=(240, 217, 181),
    dark_sq=(181, 136, 99),
    highlight=(130, 180, 100, 160),
    selected=(255, 255, 100, 160),
    bg_color=(49, 46, 43),
    text_color=(220, 220, 220),
    hand_bg=(60, 56, 52),
)

SHOGI_WOOD = Theme(
    name="Shogi Wood",
    light_sq=(222, 184, 135),
    dark_sq=(222, 184, 135),  # shogi boards are uniform color
    highlight=(130, 180, 100, 160),
    selected=(255, 255, 100, 160),
    bg_color=(49, 46, 43),
    text_color=(220, 220, 220),
    hand_bg=(180, 150, 100),
)

DEFAULT_CHESS_THEME = CLASSIC
DEFAULT_SHOGI_THEME = SHOGI_WOOD
