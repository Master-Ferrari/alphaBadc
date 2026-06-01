"""Ядро игры BADC (порт slop/game.js на Python).

Поле ``cols x rows``, буквы общие (a/b/c/d). У каждого игрока своё слово.
Состояние полностью описывается доской + счётом + номером хода, поэтому годится
как среда для AlphaZero: ``legal_moves``, ``apply``, ``clone``, ``is_terminal``,
``winner`` и кодирование позиции.

Внутреннее представление клетки (int8):
    -1 — пусто, 0..3 — a/b/c/d, 4 — "e" (порча).
Ход — пара ``(index, letter)`` или его id ``index * len(LETTERS) + letter``.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .config import DIRS, GameConfig, LETTERS, WORD_LEN, WORDS_BY_ID

EMPTY = -1
CORRUPT_ID = len(LETTERS)  # 4 == "e"
NUM_LETTERS = len(LETTERS)

Move = Tuple[int, int]  # (cell_index, letter_id)


def build_lines(cols: int, rows: int) -> List[List[int]]:
    """Все "окна" длиной WORD_LEN на поле — списки индексов клеток."""
    lines: List[List[int]] = []
    for dx, dy in DIRS:
        for row in range(rows):
            for col in range(cols):
                cells = []
                ok = True
                for k in range(WORD_LEN):
                    c = col + dx * k
                    r = row + dy * k
                    if c < 0 or c >= cols or r < 0 or r >= rows:
                        ok = False
                        break
                    cells.append(r * cols + c)
                if ok:
                    lines.append(cells)
    return lines


def build_neighbors(cols: int, rows: int) -> List[List[int]]:
    """8-соседство каждой клетки в пределах поля (для "взрыва")."""
    out: List[List[int]] = []
    for i in range(cols * rows):
        c = i % cols
        r = i // cols
        nb = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nb.append(nr * cols + nc)
        out.append(nb)
    return out


class BADCGame:
    """Партия BADC. API сознательно близок к slop/game.js."""

    def __init__(self, config: Optional[GameConfig] = None) -> None:
        self.config = config or GameConfig()
        self.config.validate()
        self.cols = self.config.cols
        self.rows = self.config.rows
        self.num_cells = self.cols * self.rows
        self.lines = build_lines(self.cols, self.rows)
        self.neighbors = build_neighbors(self.cols, self.rows)

        # Слова игроков (индексы букв).
        self.words: List[List[int]] = [
            list(WORDS_BY_ID[pid]) for pid in range(1, self.config.player_count + 1)
        ]
        # Кэш: для каждой клетки — линии, проходящие через неё.
        self._cell_lines: List[List[List[int]]] = [[] for _ in range(self.num_cells)]
        for line in self.lines:
            for idx in line:
                self._cell_lines[idx].append(line)

        self.reset()

    # ---- базовое состояние ------------------------------------------------
    def reset(self) -> "BADCGame":
        self.board = np.full(self.num_cells, EMPTY, dtype=np.int8)
        self.scores = [0.0] * self.config.player_count
        self.turn = 0
        self.winner: Optional[int] = None  # индекс игрока (0-based), -1 = ничья, None = идёт
        self.win_type: Optional[str] = None  # "instant" | "limit" | "score"
        self.winning_line: Optional[List[int]] = None
        # Последний ход каждого игрока: клетка и turn-метка (для "my last" / "opp last").
        self.last_move_by: List[Optional[int]] = [None] * self.config.player_count
        self.last_move_turn: List[int] = [-1] * self.config.player_count
        return self

    @property
    def current_player(self) -> int:
        return self.turn % self.config.player_count

    @property
    def done(self) -> bool:
        return self.winner is not None

    # ---- ходы -------------------------------------------------------------
    def empty_cells(self) -> List[int]:
        return [i for i in range(self.num_cells) if self.board[i] == EMPTY]

    def legal_moves(self) -> List[Move]:
        moves: List[Move] = []
        for i in range(self.num_cells):
            if self.board[i] == EMPTY:
                for l in range(NUM_LETTERS):
                    moves.append((i, l))
        return moves

    def legal_mask(self) -> np.ndarray:
        """Булева маска размера num_cells*NUM_LETTERS легальных действий."""
        mask = np.zeros(self.num_cells * NUM_LETTERS, dtype=bool)
        for i in range(self.num_cells):
            if self.board[i] == EMPTY:
                mask[i * NUM_LETTERS:(i + 1) * NUM_LETTERS] = True
        return mask

    @staticmethod
    def move_to_action(move: Move) -> int:
        return move[0] * NUM_LETTERS + move[1]

    @staticmethod
    def action_to_move(action: int) -> Move:
        return (action // NUM_LETTERS, action % NUM_LETTERS)

    def rule_for(self, owner: int, mover: int) -> str:
        return self.config.own_word if owner == mover else self.config.opp_word

    def _completions(self, board: np.ndarray) -> List[Tuple[int, List[int]]]:
        """Все выложенные слова: [(player_index, line)]."""
        out: List[Tuple[int, List[int]]] = []
        for line in self.lines:
            if board[line[0]] == EMPTY:
                continue
            for p in range(self.config.player_count):
                word = self.words[p]
                ok = True
                for k in range(WORD_LEN):
                    if board[line[k]] != word[k]:
                        ok = False
                        break
                if ok:
                    out.append((p, line))
                    break
        return out

    def has_live_line(self, board: Optional[np.ndarray] = None) -> bool:
        """Есть ли линия, которую ещё может кто-то выложить."""
        b = self.board if board is None else board
        for line in self.lines:
            for p in range(self.config.player_count):
                word = self.words[p]
                alive = True
                for k in range(WORD_LEN):
                    v = b[line[k]]
                    if v != EMPTY and v != word[k]:
                        alive = False
                        break
                if alive:
                    return True
        return False

    def apply(self, move: Move) -> dict:
        """Применить ход текущего игрока. Меняет состояние, возвращает событие."""
        if self.done:
            raise RuntimeError("game over")
        index, letter = move
        if self.board[index] != EMPTY:
            raise ValueError("cell occupied")
        mover = self.current_player

        self.board[index] = letter
        self.last_move_by[mover] = index
        self.last_move_turn[mover] = self.turn
        completions = self._completions(self.board)

        instant_winner: Optional[int] = None
        instant_line: Optional[List[int]] = None
        clear_set: set = set()
        corrupt_set: set = set()

        for player, line in completions:
            rule = self.rule_for(player, mover)
            if rule == "off":
                # приоритет своему слову ходящего
                if instant_winner is None or player == mover:
                    instant_winner = player
                    instant_line = line
            else:
                self.scores[mover] += 1.0 if player == mover else 0.5
                if rule == "corrupt":
                    corrupt_set.update(line)
                else:
                    for idx in line:
                        clear_set.add(idx)
                        if rule == "explode":
                            clear_set.update(self.neighbors[idx])

        if instant_winner is not None:
            self.winner = instant_winner
            self.winning_line = instant_line
            self.win_type = "instant"
            return {"type": "win", "player": instant_winner, "line": instant_line}

        # только что поставленная буква не стирается никогда
        corrupt_set.discard(index)
        clear_set.discard(index)
        for idx in corrupt_set:
            self.board[idx] = CORRUPT_ID
        for idx in clear_set:  # стирание приоритетнее порчи
            self.board[idx] = EMPTY

        # победа по лимиту счёта
        if self.config.score_limit is not None:
            best = None
            for p in range(self.config.player_count):
                if self.scores[p] >= self.config.score_limit and (
                    best is None or self.scores[p] > self.scores[best]
                ):
                    best = p
            if best is not None:
                self.winner = best
                self.win_type = "limit"
                self.winning_line = None
                return {"type": "win", "player": best}

        self.turn += 1
        if (
            len(self.empty_cells()) == 0
            or self.turn >= self.config.max_turns
            or not self.has_live_line()
        ):
            self._finish_by_score()
        return {
            "type": "continue",
            "cleared": list(clear_set),
            "corrupted": list(corrupt_set),
        }

    def _finish_by_score(self) -> None:
        best = -1.0
        winners: List[int] = []
        for p in range(self.config.player_count):
            if self.scores[p] > best:
                best = self.scores[p]
                winners = [p]
            elif self.scores[p] == best:
                winners.append(p)
        self.winner = winners[0] if len(winners) == 1 else -1  # -1 == ничья
        self.win_type = "score"
        self.winning_line = None

    # ---- AlphaZero-помощники ---------------------------------------------
    def result_for(self, player: int) -> float:
        """Исход терминальной партии с точки зрения игрока: +1/0/-1."""
        if not self.done:
            raise RuntimeError("game not finished")
        if self.winner == -1:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def clone(self) -> "BADCGame":
        g = BADCGame.__new__(BADCGame)
        g.config = self.config
        g.cols = self.cols
        g.rows = self.rows
        g.num_cells = self.num_cells
        g.lines = self.lines
        g.neighbors = self.neighbors
        g.words = self.words
        g._cell_lines = self._cell_lines
        g.board = self.board.copy()
        g.scores = list(self.scores)
        g.turn = self.turn
        g.winner = self.winner
        g.win_type = self.win_type
        g.winning_line = self.winning_line
        g.last_move_by = list(self.last_move_by)
        g.last_move_turn = list(self.last_move_turn)
        return g

    # ---- отображение ------------------------------------------------------
    def render_text(self) -> str:
        chars = LETTERS + ["e"]
        rows_txt = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                v = int(self.board[r * self.cols + c])
                row.append("." if v == EMPTY else chars[v])
            rows_txt.append(" ".join(row))
        header = (
            f"turn={self.turn} player={self.current_player + 1} "
            f"scores={['%.1f' % s for s in self.scores]}"
        )
        return header + "\n" + "\n".join(rows_txt)
