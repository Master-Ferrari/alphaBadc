"""Smoke-тест без TensorFlow: движок BADC, кодирование, MCTS, self-play, арена.

Использует заглушку сети (равномерные приоры, нулевая ценность), чтобы проверить
всю логику кроме обучения. Запуск: python scripts/smoke_no_tf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from alpha_badc.arena import HeuristicAgent, RandomAgent, evaluate
from alpha_badc.config import AZConfig, GameConfig
from alpha_badc.encoding import action_size, encode_state, num_channels
from alpha_badc.game import BADCGame
from alpha_badc.selfplay import play_game


class StubNet:
    def __init__(self, asize):
        self.asize = asize

    def predict(self, state):
        return np.ones(self.asize, dtype=np.float32) / self.asize, 0.0


def test_engine():
    cfg = GameConfig(player_count=2, own_word="explode", opp_word="on", score_limit=4.0)
    g = BADCGame(cfg)
    assert encode_state(g).shape == (cfg.rows, cfg.cols, num_channels())
    assert action_size(cfg) == cfg.rows * cfg.cols * 4
    moves = 0
    while not g.done and moves < 500:
        legal = g.legal_moves()
        assert legal
        g.apply(legal[np.random.randint(len(legal))])
        moves += 1
    assert g.done
    print(f"[ok] engine: terminal in {g.turn} turns, winner={g.winner}, type={g.win_type}")


def test_instant_mode():
    # классический режим: первое слово -> мгновенная победа
    cfg = GameConfig(player_count=2, own_word="off", opp_word="off", score_limit=None)
    g = BADCGame(cfg)
    # игрок 0 (слово a b c d) выкладывает a,b,c,d в верхней строке по очереди,
    # но ходы чередуются — проверим лишь, что движок отрабатывает мгновенную победу
    # через прямую расстановку.
    g.board[0:3] = [0, 1, 2]  # a b c _ (клетка 3 пустая)
    g.turn = 0  # ход игрока 0
    res = g.apply((3, 3))  # ставим d в клетку 3 -> a b c d
    assert res["type"] == "win" and g.win_type == "instant", res
    print("[ok] instant win mode works")


def test_mcts_selfplay():
    cfg = GameConfig(player_count=2, own_word="explode", opp_word="on", score_limit=4.0, max_turns=200)
    az = AZConfig(num_simulations=16)
    net = StubNet(action_size(cfg))
    examples, stats = play_game(net, cfg, az)
    assert len(examples) > 0
    s0, pi0, z0 = examples[0]
    assert s0.shape == (cfg.rows, cfg.cols, num_channels())
    assert abs(pi0.sum() - 1.0) < 1e-4
    assert z0 in (-1.0, 0.0, 1.0)
    print(f"[ok] self-play: {len(examples)} examples, winner={stats.winner}, turns={stats.turns}")


def test_arena():
    cfg = GameConfig(player_count=2, own_word="explode", opp_word="on", score_limit=4.0, max_turns=200)
    wr = evaluate(HeuristicAgent(), RandomAgent(), cfg, n_games=6)
    print(f"[ok] arena: heuristic vs random winrate={wr:.2f}")
    assert 0.0 <= wr <= 1.0


if __name__ == "__main__":
    np.random.seed(0)
    test_engine()
    test_instant_mode()
    test_mcts_selfplay()
    test_arena()
    print("ALL SMOKE TESTS PASSED")
