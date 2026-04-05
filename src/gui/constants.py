"""GUI constants: palette colors, game/mode options, and config."""

# Palette
BG = (49, 46, 43)
PANEL_BG = (58, 55, 52)
TEXT = (220, 220, 220)
TEXT_DIM = (140, 140, 140)
TEXT_MUTED = (100, 100, 100)
ACCENT = (100, 160, 100)
ACCENT_HI = (120, 180, 120)
BTN = (70, 90, 70)
BTN_HI = (90, 120, 90)
BTN_SEL = (60, 130, 60)
BTN_BORDER = (100, 140, 100)
SEP = (80, 75, 70)

# Game options
GAMES = [
    ("Chess (8\u00d78)", "chess"),
    ("Los Alamos (6\u00d76)", "minichess"),
    ("Shogi (9\u00d79)", "shogi"),
    ("Mini Shogi (5\u00d75)", "minishogi"),
]

MODES = [
    ("Human vs Human", "human-vs-human"),
    ("Human vs AI", "human-vs-ai"),
    ("AI vs AI", "ai-vs-ai"),
]

MODE_SHORT = {
    "human-vs-human": "HvH",
    "human-vs-ai": "HvAI",
    "ai-vs-ai": "AvA",
}

DEPTH_MIN, DEPTH_MAX = 1, 10
TIME_STEPS = [1, 2, 3, 5, 10, 15, 30]


def time_idx(sec: int) -> int:
    if sec in TIME_STEPS:
        return TIME_STEPS.index(sec)
    return 3  # default to 5s
