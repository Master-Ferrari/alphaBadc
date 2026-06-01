"""PUCT-MCTS в стиле AlphaZero.

Поиск разворачивает дерево, направляя выбор формулой PUCT
    a* = argmax_a  Q(s, a) + c_puct * P(s, a) * sqrt(sum_b N(s, b)) / (1 + N(s, a)),
оценивает листья нейросетью (приоры P и ценность v) и распространяет ценность
вверх с инверсией знака на каждом полуходу (negamax; двухигровая нулевая сумма).

Результат поиска — распределение визитов по корню, оно служит обучающей целью
``pi`` для policy-головы и из него же (с температурой) выбирается ход.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from .config import AZConfig
from .encoding import encode_state
from .game import BADCGame


class _Node:
    __slots__ = ("game", "player", "is_expanded", "legal", "P", "N", "W", "children")

    def __init__(self, game: BADCGame):
        self.game = game
        self.player = game.current_player
        self.is_expanded = False
        self.legal: List[int] = []
        self.P: Dict[int, float] = {}
        self.N: Dict[int, int] = {}
        self.W: Dict[int, float] = {}
        self.children: Dict[int, "_Node"] = {}

    def q(self, action: int) -> float:
        n = self.N.get(action, 0)
        return 0.0 if n == 0 else self.W[action] / n

    def visit_sum(self) -> int:
        return sum(self.N.values())


class MCTS:
    def __init__(self, network, config: AZConfig):
        self.net = network
        self.cfg = config

    def run(self, root_game: BADCGame, add_noise: bool = True, return_prior: bool = False):
        """Вернуть массив визитов по действиям (длина action_size).

        При ``return_prior=True`` дополнительно возвращает сырой приор сети в корне
        (нормированный по легальным ходам, ДО dirichlet-шума) — для policy-gap.
        """
        root = _Node(root_game.clone())
        self._expand(root)

        action_size = root_game.cols * root_game.rows * 4
        prior = np.zeros(action_size, dtype=np.float32)
        for a in root.legal:
            prior[a] = root.P[a]

        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.cfg.num_simulations):
            self._simulate(root)

        visits = np.zeros(action_size, dtype=np.float32)
        for a, n in root.N.items():
            visits[a] = n
        if return_prior:
            return visits, prior
        return visits

    # ---- одна симуляция ---------------------------------------------------
    def _simulate(self, root: _Node) -> None:
        path: List[tuple] = []  # (node, action)
        node = root
        while node.is_expanded and not node.game.done:
            action = self._select(node)
            path.append((node, action))
            child = node.children.get(action)
            if child is None:
                child_game = node.game.clone()
                child_game.apply(BADCGame.action_to_move(action))
                child = _Node(child_game)
                node.children[action] = child
            node = child

        # лист: терминал или оценка сетью
        if node.game.done:
            leaf_value = node.game.result_for(node.player) if node.game.winner != -1 else 0.0
            leaf_player = node.player
        else:
            leaf_value = self._expand(node)
            leaf_player = node.player

        # backup с инверсией знака по полуходам (2 игрока)
        for node_i, action in path:
            signed = leaf_value if node_i.player == leaf_player else -leaf_value
            node_i.N[action] = node_i.N.get(action, 0) + 1
            node_i.W[action] = node_i.W.get(action, 0.0) + signed

    def _select(self, node: _Node) -> int:
        total = node.visit_sum()
        sqrt_total = math.sqrt(total + 1e-8)
        best_a, best_u = -1, -float("inf")
        for a in node.legal:
            p = node.P.get(a, 0.0)
            u = node.q(a) + self.cfg.c_puct * p * sqrt_total / (1 + node.N.get(a, 0))
            if u > best_u:
                best_u, best_a = u, a
        return best_a

    def _expand(self, node: _Node) -> float:
        """Развернуть лист сетью. Вернуть оценку ценности (перспектива хода)."""
        probs, value = self.net.predict(encode_state(node.game))
        mask = node.game.legal_mask()
        legal = np.nonzero(mask)[0]
        node.legal = legal.tolist()

        masked = probs * mask
        s = masked.sum()
        if s > 1e-8:
            masked = masked / s
        else:  # сеть не дала веса легальным — равномерно
            masked = mask.astype(np.float32) / max(1, mask.sum())
        for a in node.legal:
            node.P[a] = float(masked[a])
            node.N[a] = 0
            node.W[a] = 0.0
        node.is_expanded = True
        return value

    def _add_dirichlet_noise(self, root: _Node) -> None:
        if not root.legal:
            return
        noise = np.random.dirichlet([self.cfg.dirichlet_alpha] * len(root.legal))
        frac = self.cfg.dirichlet_frac
        for a, n in zip(root.legal, noise):
            root.P[a] = (1 - frac) * root.P[a] + frac * float(n)


def select_action(visits: np.ndarray, temperature: float) -> int:
    """Выбрать действие из распределения визитов с температурой."""
    if temperature <= 1e-3:
        return int(np.argmax(visits))
    logits = np.power(visits, 1.0 / temperature)
    total = logits.sum()
    if total <= 0:
        return int(np.argmax(visits))
    probs = logits / total
    return int(np.random.choice(len(probs), p=probs))
