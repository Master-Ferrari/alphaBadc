"""Батч-self-play: много партий «в ширину», листья всех партий — одним NN-вызовом.

Зачем: на GPU MCTS с ``predict`` на batch-1 неэффективен (launch-overhead съедает
выгоду). Здесь ведётся ``num_games`` партий одновременно, и на каждом раунде
симуляций позиции-листья со ВСЕХ активных партий собираются и оцениваются одним
``net.predict_batch`` → батч в сотни позиций, видеокарта загружена.

Важно про корректность: батчевание идёт ПО ПАРТИЯМ, не внутри одного дерева.
Каждое дерево на каждом раунде делает РОВНО ОДНУ симуляцию, поэтому поиск остаётся
точным AlphaZero-MCTS (negamax-бэкап, без virtual loss и без аппроксимаций). По
сравнению с :func:`alpha_badc.selfplay.play_game` отличается только группировкой
NN-вызовов; формат результата идентичен: ``list[(examples, stats)]``.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from .config import AZConfig, GameConfig
from .encoding import encode_state
from .game import BADCGame
from .mcts import select_action
from .selfplay import GameStats

# Округляем размер NN-батча вверх до кратного _BUCKET, чтобы у tf.function было
# мало разных входных форм (иначе ретрейс на каждый размер по мере убывания партий).
_BUCKET = 32


class _BNode:
    __slots__ = ("game", "player", "is_expanded", "legal", "P", "N", "W", "children")

    def __init__(self, game: BADCGame):
        self.game = game
        self.player = game.current_player
        self.is_expanded = False
        self.legal: List[int] = []
        self.P: Dict[int, float] = {}
        self.N: Dict[int, int] = {}
        self.W: Dict[int, float] = {}
        self.children: Dict[int, "_BNode"] = {}

    def q(self, action: int) -> float:
        n = self.N.get(action, 0)
        return 0.0 if n == 0 else self.W[action] / n

    def visit_sum(self) -> int:
        return sum(self.N.values())


class _GameCtx:
    __slots__ = ("game", "root", "history", "gaps")

    def __init__(self, game: BADCGame):
        self.game = game
        self.root: "_BNode | None" = None
        self.history: List[tuple] = []  # (state, pi, player)
        self.gaps: List[float] = []


class BatchedSelfPlay:
    """Ведёт ``num_games`` self-play партий одновременно, батчуя NN по партиям."""

    def __init__(self, net, az_cfg: AZConfig):
        self.net = net
        self.cfg = az_cfg

    # ---- NN-оценка пачки листьев (с паддингом до кратного _BUCKET) ----------
    def _predict(self, states: np.ndarray):
        n = states.shape[0]
        pad = (-n) % _BUCKET
        if pad:
            states = np.concatenate(
                [states, np.zeros((pad, *states.shape[1:]), dtype=states.dtype)], axis=0
            )
        probs, values = self.net.predict_batch(states)
        return probs[:n], values[:n]

    def _expand_batch(self, nodes: List["_BNode"]) -> np.ndarray:
        """Развернуть пачку листьев одним NN-вызовом. Вернуть их value (длина len(nodes))."""
        if not nodes:
            return np.zeros(0, dtype=np.float32)
        states = np.stack([encode_state(n.game) for n in nodes]).astype(np.float32)
        probs, values = self._predict(states)
        for k, node in enumerate(nodes):
            mask = node.game.legal_mask()
            node.legal = np.nonzero(mask)[0].tolist()
            masked = probs[k] * mask
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
        return values

    def _select(self, node: "_BNode") -> int:
        total = node.visit_sum()
        sqrt_total = math.sqrt(total + 1e-8)
        c = self.cfg.c_puct
        best_a, best_u = -1, -float("inf")
        for a in node.legal:
            p = node.P.get(a, 0.0)
            u = node.q(a) + c * p * sqrt_total / (1 + node.N.get(a, 0))
            if u > best_u:
                best_u, best_a = u, a
        return best_a

    @staticmethod
    def _backup(path: List[tuple], value: float, leaf_player: int) -> None:
        for node, action in path:
            signed = value if node.player == leaf_player else -value
            node.N[action] = node.N.get(action, 0) + 1
            node.W[action] = node.W.get(action, 0.0) + signed

    def _add_noise(self, root: "_BNode") -> None:
        if not root.legal:
            return
        noise = np.random.dirichlet([self.cfg.dirichlet_alpha] * len(root.legal))
        frac = self.cfg.dirichlet_frac
        for a, n in zip(root.legal, noise):
            root.P[a] = (1 - frac) * root.P[a] + frac * float(n)

    # ---- основной проход ----------------------------------------------------
    def play(self, game_cfg: GameConfig, num_games: int) -> List[Tuple[list, GameStats]]:
        A = game_cfg.cols * game_cfg.rows * 4
        cfg = self.cfg
        ctxs = [_GameCtx(BADCGame(game_cfg)) for _ in range(num_games)]
        results: List[Tuple[list, GameStats] | None] = [None] * num_games
        active = list(range(num_games))

        # Внешний цикл — по ХОДАМ: на каждом шаге все активные партии делают один ход.
        while active:
            # 1) свежие корни + разворот одним батчем (как root в обычном MCTS).
            for i in active:
                ctxs[i].root = _BNode(ctxs[i].game.clone())
            self._expand_batch([ctxs[i].root for i in active])

            # сырые приоры корня (до шума) для policy-gap + dirichlet-шум.
            priors: Dict[int, np.ndarray] = {}
            for i in active:
                r = ctxs[i].root
                pri = np.zeros(A, dtype=np.float32)
                for a in r.legal:
                    pri[a] = r.P[a]
                priors[i] = pri
                self._add_noise(r)

            # 2) симуляции: раунд = одна сим на каждую партию, листья — одним батчем.
            for _ in range(cfg.num_simulations):
                leaves: List[tuple] = []  # (ctx_idx, leaf_node, path)
                for i in active:
                    node = ctxs[i].root
                    path: List[tuple] = []
                    while node.is_expanded and not node.game.done:
                        a = self._select(node)
                        path.append((node, a))
                        child = node.children.get(a)
                        if child is None:
                            cg = node.game.clone()
                            cg.apply(BADCGame.action_to_move(a))
                            child = _BNode(cg)
                            node.children[a] = child
                        node = child
                    if node.game.done:
                        val = node.game.result_for(node.player) if node.game.winner != -1 else 0.0
                        self._backup(path, val, node.player)
                    else:
                        leaves.append((i, node, path))
                if leaves:
                    vals = self._expand_batch([lf[1] for lf in leaves])
                    for (i, node, path), v in zip(leaves, vals):
                        self._backup(path, float(v), node.player)

            # 3) выбор хода по визитам корня + запись обучающего примера.
            for i in active:
                ctx = ctxs[i]
                r = ctx.root
                visits = np.zeros(A, dtype=np.float32)
                for a, n in r.N.items():
                    visits[a] = n
                total = visits.sum()
                if total > 0:
                    pi = visits / total
                    m = pi > 0
                    ctx.gaps.append(
                        float(np.sum(pi[m] * (np.log(pi[m]) - np.log(priors[i][m] + 1e-12))))
                    )
                else:
                    pi = ctx.game.legal_mask().astype(np.float32)
                ctx.history.append((encode_state(ctx.game), pi, ctx.game.current_player))

                temp = cfg.temperature if ctx.game.turn < cfg.temp_moves else 0.0
                action = select_action(visits, temp)
                ctx.game.apply(BADCGame.action_to_move(action))

            # 4) финализировать завершившиеся партии.
            for i in list(active):
                g = ctxs[i].game
                if not g.done:
                    continue
                ctx = ctxs[i]
                examples = [(s, pi, g.result_for(p)) for (s, pi, p) in ctx.history]
                stats = GameStats(
                    winner=g.winner if g.winner is not None else -1,
                    turns=g.turn,
                    scores=list(g.scores),
                    win_type=g.win_type or "unknown",
                    mean_policy_gap=float(np.mean(ctx.gaps)) if ctx.gaps else 0.0,
                )
                results[i] = (examples, stats)
            active = [i for i in active if not ctxs[i].game.done]

        return [r for r in results if r is not None]


def batched_self_play(net, game_cfg: GameConfig, az_cfg: AZConfig, num_games: int
                      ) -> List[Tuple[list, GameStats]]:
    """Удобная обёртка: один проход батч-self-play, формат как у play_game."""
    return BatchedSelfPlay(net, az_cfg).play(game_cfg, num_games)


# ===========================================================================
# Шардинг: N процессов, каждый ведёт свою долю партий батчем на GPU.
#
# Один процесс упирается в однопоточный Python-MCTS (GPU простаивает). Здесь
# доля партий раскидывается по нескольким процессам — MCTS-обвязка идёт по разным
# ядрам, GPU кормится из нескольких источников. Каждый воркер ограничивает свою
# память GPU (на 8 ГБ N процессов + обучение должны ужиться).
# ===========================================================================
import multiprocessing as mp  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

_BWORKER: dict = {}


def enable_gpu_memory_growth() -> None:
    """Не давать процессу жадно захватывать всю VRAM (иначе воркеры не уместятся)."""
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass  # GPU уже инициализирован


def _split_games(total: int, parts: int) -> List[int]:
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def _bworker_init(input_shape, action_size, az_cfg, gpu_mem_mb: int) -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            if gpu_mem_mb > 0:
                tf.config.set_logical_device_configuration(
                    gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_mem_mb)]
                )
            else:
                tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass

    from .network import AZNetwork

    _BWORKER["net"] = AZNetwork(input_shape, action_size, az_cfg)
    _BWORKER["cfg"] = az_cfg


def _bworker_play(weights_path: str, n_games: int, game_cfg, seed: int):
    import random

    import numpy as _np

    random.seed(seed)
    _np.random.seed(seed % (2**32 - 1))

    net = _BWORKER["net"]
    net.load(weights_path)
    return BatchedSelfPlay(net, _BWORKER["cfg"]).play(game_cfg, n_games)


class ParallelBatchedSelfPlay:
    """Пул процессов батч-self-play на GPU. Интерфейс как у ParallelSelfPlay."""

    def __init__(self, input_shape, action_size, az_cfg, num_workers: int,
                 tmp_dir: str, gpu_mem_mb: int = 0):
        self.num_workers = max(1, num_workers)
        self._tmp = str(Path(tmp_dir) / f"_bworker_tmp_{os.getpid()}.weights.h5")
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(
            processes=self.num_workers,
            initializer=_bworker_init,
            initargs=(input_shape, action_size, az_cfg, gpu_mem_mb),
        )

    def play(self, net, games: int, game_cfg, az_cfg, base_seed: int
             ) -> List[Tuple[list, GameStats]]:
        net.save(self._tmp)
        chunks = _split_games(games, self.num_workers)
        tasks = [
            (self._tmp, n, game_cfg, base_seed + i * 100003)
            for i, n in enumerate(chunks) if n > 0
        ]
        results: List[Tuple[list, GameStats]] = []
        for chunk in self.pool.starmap(_bworker_play, tasks):
            results.extend(chunk)
        return results

    def close(self) -> None:
        try:
            self.pool.close()
            self.pool.join()
        finally:
            try:
                os.remove(self._tmp)
            except OSError:
                pass
