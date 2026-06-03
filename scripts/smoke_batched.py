"""Смоук батч-self-play: корректность формата + базовая работоспособность.

Гоняет несколько партий «в ширину» на маленьких симуляциях и сверяет, что формат
результата совпадает с selfplay.play_game (examples=(state,pi,z) + GameStats).
"""
import numpy as np

from alpha_badc.config import GameConfig, AZConfig
from alpha_badc.encoding import action_size, num_channels
from alpha_badc.network import AZNetwork
from alpha_badc.batched_selfplay import batched_self_play
from alpha_badc.selfplay import GameStats

gc = GameConfig()
ac = AZConfig()
ac.num_simulations = 8  # быстрый смоук

net = AZNetwork((gc.rows, gc.cols, num_channels()), action_size(gc), ac)
A = action_size(gc)

N_GAMES = 6
results = batched_self_play(net, gc, ac, N_GAMES)

assert len(results) == N_GAMES, f"ожидалось {N_GAMES} партий, получено {len(results)}"
total_examples = 0
for k, (examples, stats) in enumerate(results):
    assert isinstance(stats, GameStats), "stats не GameStats"
    assert stats.turns > 0, "пустая партия"
    assert len(examples) == stats.turns, f"examples({len(examples)}) != turns({stats.turns})"
    s0, pi0, z0 = examples[0]
    assert s0.shape == (gc.rows, gc.cols, num_channels()), f"плохая форма state {s0.shape}"
    assert pi0.shape == (A,), f"плохая форма pi {pi0.shape}"
    assert z0 in (-1.0, 0.0, 1.0), f"неожиданный z={z0}"
    total_examples += len(examples)

winners = [st.winner for _, st in results]
print(f"games={N_GAMES} total_examples={total_examples} "
      f"winners={winners} mean_turns={np.mean([st.turns for _, st in results]):.1f}")
print("ALL_OK")
