"""Генерация партий self-play и буфер воспроизведения для AlphaZero.

Один проход self-play играет партию сетью+MCTS против самой себя и собирает
обучающие примеры ``(state, pi, z)``:
  * state — закодированная позиция (с точки зрения ходящего);
  * pi    — нормированное распределение визитов MCTS по действиям (цель policy);
  * z     — итог партии для игрока, который ходил в этой позиции (+1/0/-1).
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np

from .config import AZConfig, GameConfig
from .encoding import encode_state
from .game import BADCGame
from .mcts import MCTS, select_action

Example = Tuple[np.ndarray, np.ndarray, int]  # (state, pi, player)


@dataclass
class GameStats:
    winner: int  # индекс игрока, -1 = ничья
    turns: int
    scores: List[float]
    win_type: str
    mean_policy_gap: float = 0.0  # средний KL(pi_MCTS || policy_net) по ходам партии


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buf)

    def add(self, state: np.ndarray, pi: np.ndarray, z: float) -> None:
        self.buf.append((state, pi, z))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        states = np.stack([b[0] for b in batch]).astype(np.float32)
        pis = np.stack([b[1] for b in batch]).astype(np.float32)
        zs = np.array([b[2] for b in batch], dtype=np.float32)
        return states, pis, zs

    def save(self, path: str) -> int:
        """Сохранить буфер в сжатый .npz (states/pis/zs). Вернуть число примеров."""
        if not self.buf:
            return 0
        states = np.stack([b[0] for b in self.buf]).astype(np.float32)
        pis = np.stack([b[1] for b in self.buf]).astype(np.float32)
        zs = np.array([b[2] for b in self.buf], dtype=np.float32)
        np.savez_compressed(path, states=states, pis=pis, zs=zs)
        return len(self.buf)

    def load(self, path: str) -> int:
        """Загрузить примеры из .npz (с учётом maxlen). Вернуть число загруженных."""
        with np.load(path) as data:
            states, pis, zs = data["states"], data["pis"], data["zs"]
            for s, p, z in zip(states, pis, zs):
                self.buf.append((s.astype(np.float32), p.astype(np.float32), float(z)))
        return len(self.buf)


def play_game(net, game_config: GameConfig, az_config: AZConfig) -> Tuple[List[Tuple[np.ndarray, np.ndarray, int]], GameStats]:
    """Сыграть одну партию self-play. Вернуть (examples, stats)."""
    game = BADCGame(game_config)
    mcts = MCTS(net, az_config)
    history: List[Example] = []
    gaps: List[float] = []

    while not game.done:
        visits, prior = mcts.run(game, add_noise=True, return_prior=True)
        total = visits.sum()
        if total > 0:
            pi = visits / total
            m = pi > 0
            # KL(pi || prior): насколько поиск «исправил» сырую политику сети.
            gaps.append(float(np.sum(pi[m] * (np.log(pi[m]) - np.log(prior[m] + 1e-12)))))
        else:
            pi = game.legal_mask().astype(np.float32)
        history.append((encode_state(game), pi, game.current_player))

        temp = az_config.temperature if game.turn < az_config.temp_moves else 0.0
        action = select_action(visits, temp)
        game.apply(BADCGame.action_to_move(action))

    examples: List[Tuple[np.ndarray, np.ndarray, int]] = []
    for state, pi, player in history:
        z = game.result_for(player)
        examples.append((state, pi, z))

    stats = GameStats(
        winner=game.winner if game.winner is not None else -1,
        turns=game.turn,
        scores=list(game.scores),
        win_type=game.win_type or "unknown",
        mean_policy_gap=float(np.mean(gaps)) if gaps else 0.0,
    )
    return examples, stats
