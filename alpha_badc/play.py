"""Инференс: загрузка чекпоинта и игра обученной сети с визуализацией.

По умолчанию — текстовый рендер доски в консоль (работает всегда). С флагом
``--render`` рисуется доска через pygame (буквы a/b/c/d + "e"-порча, подсветка
последнего хода). Оцениваемая сеть играет за игрока 1 против выбранного
соперника (net / heuristic / random).
"""
from __future__ import annotations

import os
import time
from typing import Optional

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

from .arena import HeuristicAgent, NetAgent, RandomAgent
from .config import AZConfig, GameConfig
from .encoding import action_size, num_channels
from .game import BADCGame, CORRUPT_ID, EMPTY
from .network import AZNetwork

try:
    import pygame
except ModuleNotFoundError:  # pragma: no cover
    pygame = None

_LETTER_COLORS = {
    0: (0, 200, 240),    # a
    1: (240, 220, 0),    # b
    2: (80, 160, 255),   # c
    3: (240, 150, 0),    # d
    CORRUPT_ID: (120, 120, 120),  # e
}


def _make_opponent(kind: str, net: AZNetwork, az_cfg: AZConfig, sims: int):
    if kind == "random":
        return RandomAgent()
    if kind == "heuristic":
        return HeuristicAgent()
    if kind == "net":
        return NetAgent(net, az_cfg, simulations=sims)
    raise ValueError(f"unknown opponent: {kind}")


def run_play(
    checkpoint: str,
    game_cfg: GameConfig,
    az_cfg: AZConfig,
    opponent: str = "heuristic",
    episodes: int = 5,
    sims: int = 80,
    render: bool = False,
    fps: int = 2,
) -> None:
    net = AZNetwork((game_cfg.rows, game_cfg.cols, num_channels()), action_size(game_cfg), az_cfg)
    net.load(checkpoint)
    print(f"[infer] loaded {checkpoint}")

    net_agent = NetAgent(net, az_cfg, simulations=sims)
    opp_agent = _make_opponent(opponent, net, az_cfg, sims)
    agents = [net_agent, opp_agent]  # сеть — игрок 0

    viz = _PygameView(game_cfg) if render else None
    net_wins = draws = 0
    try:
        for ep in range(1, episodes + 1):
            game = BADCGame(game_cfg)
            last_move: Optional[int] = None
            while not game.done:
                action = agents[game.current_player % 2](game)
                last_move = BADCGame.action_to_move(action)[0]
                game.apply(BADCGame.action_to_move(action))
                if viz:
                    viz.draw(game, last_move)
                    time.sleep(1.0 / max(1, fps))
                else:
                    print(game.render_text())
                    print("-" * (game.cols * 2))
            result = "draw" if game.winner == -1 else f"player {game.winner + 1}"
            if game.winner == 0:
                net_wins += 1
            elif game.winner == -1:
                draws += 1
            print(f"[game {ep}/{episodes}] winner={result} type={game.win_type} "
                  f"scores={['%.1f' % s for s in game.scores]} turns={game.turn}")
    except KeyboardInterrupt:
        print("[infer] interrupted")
    finally:
        if viz:
            viz.close()
    print(f"[infer] net winrate={net_wins / max(1, episodes):.2f} draws={draws}")


class _PygameView:
    def __init__(self, game_cfg: GameConfig, cell: int = 64):
        if pygame is None:
            raise ModuleNotFoundError("pygame required for --render (pip install pygame)")
        pygame.init()
        pygame.font.init()
        self.cell = cell
        self.cols, self.rows = game_cfg.cols, game_cfg.rows
        self.screen = pygame.display.set_mode((self.cols * cell, self.rows * cell + 40))
        pygame.display.set_caption("AlphaBADC")
        self.font = pygame.font.SysFont("Consolas", int(cell * 0.6))
        self.small = pygame.font.SysFont("Consolas", 20)
        self.letters = ["a", "b", "c", "d", "e"]

    def draw(self, game: BADCGame, last_move: Optional[int]) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
        self.screen.fill((18, 18, 24))
        for r in range(self.rows):
            for c in range(self.cols):
                idx = r * self.cols + c
                v = int(game.board[idx])
                rect = pygame.Rect(c * self.cell, r * self.cell, self.cell, self.cell)
                pygame.draw.rect(self.screen, (35, 35, 45), rect)
                if idx == last_move:
                    pygame.draw.rect(self.screen, (90, 90, 30), rect)
                pygame.draw.rect(self.screen, (60, 60, 75), rect, 1)
                if v != EMPTY:
                    surf = self.font.render(self.letters[v], True, _LETTER_COLORS.get(v, (230, 230, 230)))
                    self.screen.blit(surf, surf.get_rect(center=rect.center))
        info = f"P{game.current_player + 1} turn={game.turn} scores={['%.1f' % s for s in game.scores]}"
        self.screen.blit(self.small.render(info, True, (220, 220, 220)), (8, self.rows * self.cell + 8))
        pygame.display.flip()

    def close(self) -> None:
        if pygame is not None:
            pygame.quit()
