"""Конфигурация игры BADC и обучения AlphaZero.

Правила полностью настраиваемые (см. ``rules.md``):
  own_word / opp_word ∈ {"off", "on", "explode", "corrupt"}
    off     — мгновенная победа владельца слова (классический режим).
    on      — очки + стираются клетки слова, КРОМЕ только что поставленной.
    explode — то же + стираются клетки в радиусе 1 вокруг слова.
    corrupt — то же, но клетки превращаются в "e" (порча, навсегда блокирует).
  Очки выложившему ход: +1 за СВОЁ слово, +0.5 за выкладывание ЧУЖОГО.

Если хотя бы одно правило != "off" — включается режим счёта: партия идёт до
``score_limit`` очков (или до предела ``max_turns`` / заполнения поля / мёртвой
позиции, тогда побеждает лидер по очкам).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Канонические параметры доски (как в slop/rules.md и game.js).
COLS = 9
ROWS = 5
WORD_LEN = 4
LETTERS = ["a", "b", "c", "d"]
CORRUPT = "e"  # буква "порчи": не входит ни в одно слово, блокирует клетку

# Только половина "розы ветров": →, ↓, ↘, ↗.
DIRS = [(1, 0), (0, 1), (1, 1), (1, -1)]

# Целевые слова игроков как индексы букв (a=0, b=1, c=2, d=3).
# 3-й/4-й игрок могут поменяться — см. геймдизайн.
WORDS_BY_ID = {
    1: [0, 1, 2, 3],  # a b c d
    2: [1, 0, 3, 2],  # b a d c
    3: [1, 2, 0, 3],  # b c a d
    4: [3, 2, 1, 0],  # d c b a
}

RULE_CHOICES = ("off", "on", "explode", "corrupt")


@dataclass
class GameConfig:
    """Все настройки одной партии BADC."""

    player_count: int = 2
    own_word: str = "explode"
    opp_word: str = "on"
    score_limit: Optional[float] = 4.0
    max_turns: int = 600
    cols: int = COLS
    rows: int = ROWS

    def validate(self) -> None:
        if not (2 <= self.player_count <= 4):
            raise ValueError("player_count must be in [2, 4]")
        if self.own_word not in RULE_CHOICES:
            raise ValueError(f"own_word must be one of {RULE_CHOICES}")
        if self.opp_word not in RULE_CHOICES:
            raise ValueError(f"opp_word must be one of {RULE_CHOICES}")

    @property
    def scoring_mode(self) -> bool:
        """Режим счёта: хотя бы одно правило стирания включено."""
        return self.own_word != "off" or self.opp_word != "off"


@dataclass
class AZConfig:
    """Гиперпараметры AlphaZero (self-play + обучение сети)."""

    # MCTS
    num_simulations: int = 80
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.5
    dirichlet_frac: float = 0.25
    temperature: float = 1.0
    temp_moves: int = 12  # после стольких ходов температура -> ~0 (argmax по визитам)

    # Сеть
    channels: int = 64
    blocks: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_coef: float = 1.0

    # Self-play / обучение
    games_per_iter: int = 32
    epochs_per_iter: int = 2
    batch_size: int = 256
    replay_size: int = 50_000
    min_replay: int = 1_000

    def __post_init__(self) -> None:
        if self.num_simulations < 1:
            raise ValueError("num_simulations must be >= 1")
