"""Board interaction mixin: click handling for GameSession."""

from typing import Optional

from src.games.base import WHITE, BLACK
from src.gui.panel import eval_white_pov


class _BoardInteractionMixin:
    """Mixin providing _handle_click and _handle_board_click for GameSession."""

    def _handle_click(self, pos) -> Optional[str]:
        mx, my = pos

        # Panel clicks
        for key, rect in self.panel_rects.items():
            if not rect.collidepoint(mx, my):
                continue
            if key == "hint":
                self.app.analysis_on = not self.app.analysis_on
                if self.app.analysis_on:
                    self._launch_analysis()
                else:
                    self._cancel_analysis()
                    self.hint_moves = []
                    self.hint_depth = 0
                    self.hint_done = False
            elif key == "restart":
                self._restart()
            elif key == "menu":
                self._cancel_analysis()
                return "menu"
            return None

        # Guard: not interactable
        if self.state.is_terminal() or self.ai_thinking:
            return None
        is_human = (
            self.app.mode == "human-vs-human"
            or (self.app.mode == "human-vs-ai"
                and self.state.side_to_move() != self.ai_side)
        )
        if not is_human:
            return None

        self._handle_board_click(mx, my)
        return None

    def _handle_board_click(self, mx, my):
        # Hand piece click (shogi)
        if self.has_hand and self.selected_sq is None:
            side = self.state.side_to_move()
            hr = (self.sente_hand_rect if side == WHITE
                  else self.gote_hand_rect)
            if hr and hr.collidepoint(mx, my):
                hp = self.renderer.hand_piece_at(
                    mx - hr.x, my - hr.y, self.state, side, hr)
                if hp is not None:
                    self.selected_hand_piece = hp
                    self.selected_sq = None
                    self.legal_targets = [
                        m.to_sq for m in self.state.legal_moves()
                        if m.drop_piece == hp
                    ]
                    return

        bx, by = mx - self.board_x, my - self.board_y
        clicked_sq = self.renderer.pixel_to_sq(bx, by)
        if clicked_sq is None:
            return

        if self.selected_hand_piece is not None:
            if clicked_sq in self.legal_targets:
                for m in self.state.legal_moves():
                    if (m.drop_piece == self.selected_hand_piece
                            and m.to_sq == clicked_sq):
                        self.state = self.state.make_move(m)
                        self.score_history.append(
                            eval_white_pov(self.state))
                        self.hint_moves = []
                        break
            self.selected_hand_piece = None
            self.selected_sq = None
            self.legal_targets = []

        elif self.selected_sq is None:
            board = self.state.board_array()
            piece = board[clicked_sq]
            if piece != 0:
                own = ((piece > 0 and self.state.side_to_move() == WHITE)
                       or (piece < 0
                           and self.state.side_to_move() == BLACK))
                if own:
                    self.selected_sq = clicked_sq
                    self.legal_targets = [
                        m.to_sq for m in self.state.legal_moves()
                        if m.from_sq == clicked_sq
                    ]

        elif clicked_sq == self.selected_sq:
            self.selected_sq = None
            self.legal_targets = []

        elif clicked_sq in self.legal_targets:
            matching = [m for m in self.state.legal_moves()
                        if m.from_sq == self.selected_sq
                        and m.to_sq == clicked_sq]
            if len(matching) == 1:
                self.state = self.state.make_move(matching[0])
                self.score_history.append(eval_white_pov(self.state))
            elif len(matching) > 1:
                promo = [m for m in matching if m.promotion is not None]
                self.state = self.state.make_move(
                    promo[0] if promo else matching[0])
                self.score_history.append(eval_white_pov(self.state))
            self.hint_moves = []
            self.selected_sq = None
            self.legal_targets = []
        else:
            self.selected_sq = clicked_sq
            self.legal_targets = [
                m.to_sq for m in self.state.legal_moves()
                if m.from_sq == clicked_sq
            ]
