"""GUI constants: palette colors, game/mode options, and config."""

# ---------------------------------------------------------------- Palette v2
# Darker, more modern base with a blue-leaning accent so the score/eval
# colors (green = White, red = Black) stand out against the chrome.
BG = (24, 26, 32)
BG_GRAD_TOP = (34, 38, 48)
BG_GRAD_BOTTOM = (20, 22, 28)
PANEL_BG = (36, 40, 50)
PANEL_BG_ALT = (30, 34, 42)
PANEL_BORDER = (54, 60, 74)

TEXT = (232, 234, 240)
TEXT_DIM = (160, 168, 184)
TEXT_MUTED = (104, 112, 128)

ACCENT = (88, 140, 220)
ACCENT_HI = (116, 168, 244)
ACCENT_DIM = (60, 100, 160)

BTN = (50, 56, 70)
BTN_HI = (70, 80, 100)
BTN_SEL = (88, 140, 220)
BTN_BORDER = (74, 84, 104)
SEP = (54, 60, 74)

# Side tint colors for eval display.
WHITE_ADV = (220, 224, 232)   # near-white for White advantage
BLACK_ADV = (28, 30, 36)      # deep black for Black advantage
EVAL_GOOD = (120, 200, 140)   # positive score tint
EVAL_BAD = (220, 110, 110)    # negative score tint
EVAL_NEUTRAL = (180, 180, 190)

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

# Player side options (HvAI). Label maps to the WHITE/BLACK constants
# defined in src.games.base; we can't import there without a cycle so
# we use string keys here and resolve downstream.
PLAYER_SIDES = [
    ("Sente / White", "white"),
    ("Gote / Black", "black"),
]

DEPTH_MIN, DEPTH_MAX = 1, 10
TIME_STEPS = [1, 2, 3, 5, 10, 15, 30]


def time_idx(sec: int) -> int:
    if sec in TIME_STEPS:
        return TIME_STEPS.index(sec)
    return 3  # default to 5s
