"""Smoke-тест TensorFlow-части: сборка сети, predict, train_on_batch, 1 итерация.

Запуск: .venv/Scripts/python scripts/smoke_tf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from alpha_badc.config import AZConfig, GameConfig
from alpha_badc.encoding import action_size, encode_state, num_channels
from alpha_badc.game import BADCGame
from alpha_badc.network import AZNetwork


def main():
    cfg = GameConfig(player_count=2, own_word="explode", opp_word="on", score_limit=4.0, max_turns=120)
    az = AZConfig(num_simulations=8, channels=16, blocks=2, games_per_iter=2,
                  batch_size=16, min_replay=8, epochs_per_iter=1, replay_size=2000)
    asize = action_size(cfg)
    net = AZNetwork((cfg.rows, cfg.cols, num_channels()), asize, az)
    print(f"[ok] model built: params={net.model.count_params()} action_size={asize}")

    g = BADCGame(cfg)
    probs, value = net.predict(encode_state(g))
    assert probs.shape == (asize,) and abs(probs.sum() - 1.0) < 1e-3
    print(f"[ok] predict: probs.sum={probs.sum():.3f} value={value:.3f}")

    states = np.stack([encode_state(g) for _ in range(16)]).astype(np.float32)
    pis = np.random.dirichlet(np.ones(asize), size=16).astype(np.float32)
    zs = np.random.uniform(-1, 1, size=16).astype(np.float32)
    m = net.train_on_batch(states, pis, zs)
    print(f"[ok] train_on_batch: {m}")

    save_path = "checkpoints/_smoke.weights.h5"
    Path("checkpoints").mkdir(exist_ok=True)
    net.save(save_path)
    net2 = AZNetwork((cfg.rows, cfg.cols, num_channels()), asize, az)
    net2.load(save_path)
    print("[ok] save/load weights")

    print("TF SMOKE PASSED")


if __name__ == "__main__":
    main()
