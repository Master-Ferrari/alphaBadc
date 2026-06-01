"""Кодирование позиции BADC в тензор для нейросети AlphaZero.

Позиция кодируется ВСЕГДА с точки зрения игрока, который сейчас ходит, чтобы одна
и та же сеть работала за любого игрока (перспективно-инвариантно). План-каналы на
сетке ``rows x cols``:

  * NUM_LETTERS  — занятость каждой буквой (a/b/c/d);
  * 1            — клетки порчи "e";
  * 1            — пустые клетки;
  * WORD_LEN*NUM_LETTERS — слово ХОДЯЩЕГО (one-hot позиция×буква, broadcast);
  * WORD_LEN*NUM_LETTERS — объединение слов СОПЕРНИКОВ (broadcast);
  * 2            — нормированные счёт ходящего и максимум среди соперников;
  * 1            — прогресс партии (turn / max_turns);
  * 1            — клетка ПОСЛЕДНЕГО хода ходящего (one-hot локация);
  * 1            — клетка ПОСЛЕДНЕГО хода последнего соперника (one-hot локация).

Действие — id клетки*NUM_LETTERS + буква (всего num_cells*NUM_LETTERS).
"""
from __future__ import annotations

import numpy as np

from .config import GameConfig, WORD_LEN
from .game import BADCGame, CORRUPT_ID, EMPTY, NUM_LETTERS, WORDS_BY_ID


def num_channels() -> int:
    # +2 в конце: локации последнего хода ходящего и последнего хода соперника.
    return NUM_LETTERS + 1 + 1 + 2 * (WORD_LEN * NUM_LETTERS) + 2 + 1 + 2


def _location_plane(cell, rows: int, cols: int) -> np.ndarray:
    """Плоскость rows×cols с 1.0 в клетке ``cell`` (пусто, если cell is None)."""
    plane = np.zeros((rows, cols), dtype=np.float32)
    if cell is not None:
        plane[cell // cols, cell % cols] = 1.0
    return plane


def action_size(config: GameConfig) -> int:
    return config.cols * config.rows * NUM_LETTERS


def _word_planes(word, rows, cols) -> np.ndarray:
    """WORD_LEN*NUM_LETTERS broadcast-плоскостей для одного слова."""
    planes = np.zeros((WORD_LEN * NUM_LETTERS, rows, cols), dtype=np.float32)
    for pos, letter in enumerate(word):
        planes[pos * NUM_LETTERS + letter, :, :] = 1.0
    return planes


def encode_state(game: BADCGame) -> np.ndarray:
    """Вернуть тензор (rows, cols, channels) для текущей позиции."""
    rows, cols = game.rows, game.cols
    board = game.board.reshape(rows, cols)
    mover = game.current_player

    planes = []
    for letter in range(NUM_LETTERS):
        planes.append((board == letter).astype(np.float32))
    planes.append((board == CORRUPT_ID).astype(np.float32))
    planes.append((board == EMPTY).astype(np.float32))

    mover_word = game.words[mover]
    for p in _word_planes(mover_word, rows, cols):
        planes.append(p)

    # объединение слов соперников
    opp = np.zeros((WORD_LEN * NUM_LETTERS, rows, cols), dtype=np.float32)
    for pi in range(game.config.player_count):
        if pi == mover:
            continue
        opp = np.maximum(opp, _word_planes(game.words[pi], rows, cols))
    for p in opp:
        planes.append(p)

    # счёт (нормировка на score_limit либо на безопасную константу)
    denom = game.config.score_limit if game.config.score_limit else 8.0
    my_score = game.scores[mover] / denom
    opp_score = max(
        [game.scores[pi] for pi in range(game.config.player_count) if pi != mover],
        default=0.0,
    ) / denom
    planes.append(np.full((rows, cols), my_score, dtype=np.float32))
    planes.append(np.full((rows, cols), opp_score, dtype=np.float32))

    progress = game.turn / max(1, game.config.max_turns)
    planes.append(np.full((rows, cols), progress, dtype=np.float32))

    # Последний ход ходящего и последний ход последнего соперника (по turn-метке).
    my_last = game.last_move_by[mover]
    opp_last, best_turn = None, -1
    for p in range(game.config.player_count):
        if p == mover:
            continue
        if game.last_move_turn[p] > best_turn:
            best_turn = game.last_move_turn[p]
            opp_last = game.last_move_by[p]
    planes.append(_location_plane(my_last, rows, cols))
    planes.append(_location_plane(opp_last, rows, cols))

    tensor = np.stack(planes, axis=0)  # (C, rows, cols)
    return np.transpose(tensor, (1, 2, 0))  # (rows, cols, C) для Keras channels_last
