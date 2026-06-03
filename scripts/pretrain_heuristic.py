"""Претрейн (behavioral cloning эвристики) для AlphaBADC.

Идея: сильная эвристика быстро (без MCTS и без сети) играет тысячи партий; на каждой
позиции записываем (state, ход_эвристики, исход_партии). Затем supervised-обучаем сеть
— policy предсказывает ход эвристики (cross-entropy к one-hot), value предсказывает
исход (MSE) — переиспользуя AZNetwork.train_on_batch. Результат: чекпоинт уровня
~эвристики, с которого запускается обычный self-play:

    python main.py train --resume-from checkpoints/az_badc_v3_pretrain.weights.h5 --run-name az_badc_v3pre ...

Генерация партий параллелится по CPU-ядрам (воркеры НЕ импортируют TF — поэтому лёгкие
и быстро стартуют). Для разнообразия позиций — epsilon-разведка: иногда играется
случайный ход, но цель policy всегда = лучший ход эвристики в данной позиции.

Запуск (16 ядер):
    python scripts/pretrain_heuristic.py --games 5000 --workers 16 --epochs 4
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from typing import List

import numpy as np

# чтобы alpha_badc импортировался при запуске из scripts/ (и в spawn-воркерах)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpha_badc.config import AZConfig, GameConfig  # noqa: E402  (без TF)


def _split(total: int, parts: int) -> List[int]:
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def _gen_worker(task):
    """Сгенерировать партии эвристики и сохранить шард на диск. БЕЗ TensorFlow."""
    n, game_cfg, epsilon, seed, shard_path = task
    import random

    import numpy as _np
    from alpha_badc.arena import HeuristicAgent
    from alpha_badc.encoding import encode_state
    from alpha_badc.game import BADCGame

    random.seed(seed)
    _np.random.seed(seed % (2**32 - 1))
    agent = HeuristicAgent()

    states, actions, zs = [], [], []
    for _ in range(n):
        g = BADCGame(game_cfg)
        history = []  # (state, heuristic_best_action, player)
        while not g.done:
            best = agent(g)  # цель policy = лучший ход эвристики
            history.append((encode_state(g), best, g.current_player))
            if random.random() < epsilon:  # разведка для разнообразия позиций
                act = BADCGame.move_to_action(random.choice(g.legal_moves()))
            else:
                act = best
            g.apply(BADCGame.action_to_move(act))
        for state, action, player in history:
            states.append(state)
            actions.append(action)
            zs.append(float(g.result_for(player)))

    _np.savez_compressed(
        shard_path,
        states=_np.asarray(states, dtype=_np.float32),
        actions=_np.asarray(actions, dtype=_np.int32),
        zs=_np.asarray(zs, dtype=_np.float32),
    )
    return shard_path, len(zs)


def parse():
    p = argparse.ArgumentParser(description="Претрейн (BC эвристики) для AlphaBADC.")
    p.add_argument("--games", type=int, default=5000, help="всего партий эвристики")
    p.add_argument("--workers", type=int, default=16, help="процессов генерации")
    p.add_argument("--epsilon", type=float, default=0.15, help="доля случайных ходов (разведка)")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--kernel-size", type=int, default=3, help="ядро ствола (обычно 3)")
    p.add_argument("--head-kernel", type=int, default=4, help="ядро голов (4 ≈ длина слова BADC)")
    p.add_argument("--out", type=str, default="checkpoints/az_badc_v3_pretrain.weights.h5")
    p.add_argument("--tmp-dir", type=str, default="checkpoints")
    p.add_argument("--reuse-shards", action="store_true",
                   help="не генерировать заново — взять уже лежащие шарды из --tmp-dir")
    p.add_argument("--keep-shards", action="store_true",
                   help="не удалять шарды после обучения (на случай повторного прогона)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--post-eval-games", type=int, default=20, help="контрольный eval после обучения (0=skip)")
    p.add_argument("--eval-sims", type=int, default=80)
    # правила игры — дефолты как в v3
    p.add_argument("--own-word", type=str, default="explode")
    p.add_argument("--opp-word", type=str, default="on")
    p.add_argument("--score-limit", type=float, default=4.0)
    return p.parse_args()


def main():
    args = parse()
    game_cfg = GameConfig(own_word=args.own_word, opp_word=args.opp_word,
                          score_limit=args.score_limit)
    game_cfg.validate()
    az_cfg = AZConfig()
    az_cfg.channels = args.channels
    az_cfg.blocks = args.blocks
    az_cfg.kernel = args.kernel_size
    az_cfg.head_kernel = args.head_kernel
    az_cfg.learning_rate = args.lr

    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ---- 0) импорт TF СРАЗУ: если его нет в окружении — падаем за секунды,
    #         а не после часа генерации (защита от потери шардов). ----
    from alpha_badc.encoding import action_size, num_channels
    from alpha_badc.network import AZNetwork

    # детерминированные пути шардов (не зависят от PID) — нужны для --reuse-shards
    chunks = _split(args.games, args.workers)
    shard_paths = [os.path.join(args.tmp_dir, f"_pretrain_shard_{i}.npz")
                   for i in range(len(chunks))]

    # ---- 1) генерация партий эвристики (параллельно, без TF) ----
    if args.reuse_shards and all(os.path.exists(p) for p in shard_paths):
        shards = [(p, 0) for p in shard_paths]
        print(f"[gen] --reuse-shards: переиспользую {len(shards)} существующих шардов из {args.tmp_dir}")
    else:
        tasks = [
            (n, game_cfg, args.epsilon, args.seed + i * 100003, shard_paths[i])
            for i, n in enumerate(chunks) if n > 0
        ]
        print(f"[gen] {args.games} партий эвристики на {len(tasks)} воркерах (epsilon={args.epsilon})...")
        t0 = time.time()
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=max(1, args.workers)) as pool:
            shards = pool.map(_gen_worker, tasks)
        total = sum(c for _, c in shards)
        print(f"[gen] готово: {total:,} позиций за {time.time()-t0:.0f}с")

    # ---- загрузка шардов (НЕ удаляем здесь — только после успешного save) ----
    S, Ac, Z = [], [], []
    for path, _ in shards:
        with np.load(path) as d:
            S.append(d["states"]); Ac.append(d["actions"]); Z.append(d["zs"])
    states = np.concatenate(S); actions = np.concatenate(Ac); zs = np.concatenate(Z)
    del S, Ac, Z

    # ---- 2) supervised-обучение ----
    A = action_size(game_cfg)
    net = AZNetwork((game_cfg.rows, game_cfg.cols, num_channels()), A, az_cfg)
    N = len(zs)
    idx = np.arange(N)
    print(f"[train] supervised: {N:,} примеров, {args.epochs} эпох, batch={args.batch_size}")
    for ep in range(args.epochs):
        np.random.shuffle(idx)
        tl = pl = vl = 0.0
        nb = 0
        for st in range(0, N, args.batch_size):
            b = idx[st:st + args.batch_size]
            bpi = np.zeros((len(b), A), dtype=np.float32)
            bpi[np.arange(len(b)), actions[b]] = 1.0
            m = net.train_on_batch(states[b], bpi, zs[b])
            tl += m["total_loss"]; pl += m["policy_loss"]; vl += m["value_loss"]; nb += 1
        print(f"[train] эпоха {ep+1}/{args.epochs}: total={tl/nb:.4f} policy={pl/nb:.4f} value={vl/nb:.4f}")

    # ---- сохранение чекпоинта (формат как у trainer: .weights.h5 + .meta.json) ----
    net.save(args.out)
    with open(args.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"iteration": 0, "global_step": 0, "pretrain": "heuristic_bc"}, f)
    print(f"[ok] чекпоинт сохранён: {args.out}")

    # ---- шарды удаляем ТОЛЬКО теперь (чекпоинт уже на диске) ----
    if not args.keep_shards:
        for path, _ in shards:
            try:
                os.remove(path)
            except OSError:
                pass

    # ---- 3) контрольный eval (опц.) ----
    if args.post_eval_games > 0:
        from alpha_badc.arena import HeuristicAgent, NetAgent, RandomAgent, evaluate
        ea = NetAgent(net, az_cfg, simulations=args.eval_sims)
        wr_rand = evaluate(ea, RandomAgent(), game_cfg, args.post_eval_games)
        wr_heur = evaluate(ea, HeuristicAgent(), game_cfg, args.post_eval_games)
        print(f"[eval] после претрейна (sims={args.eval_sims}, {args.post_eval_games} партий): "
              f"vs_random={wr_rand:.3f}  vs_heuristic={wr_heur:.3f}")

    print("[done] Дальше: main.py train --resume-from " + args.out + " --run-name az_badc_v3pre ...")


if __name__ == "__main__":
    main()
