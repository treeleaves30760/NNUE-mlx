"""Global configuration and game registry."""

# Game name constants
CHESS = "chess"
MINICHESS = "minichess"
SHOGI = "shogi"
MINISHOGI = "minishogi"

ALL_GAMES = [CHESS, MINICHESS, SHOGI, MINISHOGI]


def create_game(game_name: str):
    """Factory function to create initial game state by name."""
    if game_name == CHESS:
        from src.games.chess_pc import initial_state
        return initial_state()
    elif game_name == MINICHESS:
        from src.games.minichess import initial_state
        return initial_state()
    elif game_name == SHOGI:
        from src.games.shogi import initial_state
        return initial_state()
    elif game_name == MINISHOGI:
        from src.games.minishogi import initial_state
        return initial_state()
    else:
        raise ValueError(f"Unknown game: {game_name}. Choose from {ALL_GAMES}")
