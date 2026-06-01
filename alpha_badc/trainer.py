"""Итеративное обучение AlphaZero для BADC.

Один цикл (iteration):
  1. self-play: сеть играет ``games_per_iter`` партий сама с собой → примеры в буфер;
  2. обучение: ``epochs_per_iter`` проходов по сэмплам буфера (policy + value loss);
  3. периодически — оценка силы на арене (vs random / vs эвристика);
  4. логирование (CSV + TensorBoard) и чекпоинт.

Формат логов намеренно повторяет tetris_poo (CSV `train_metrics.csv` с осью
`global_step` + TensorBoard), чтобы графики ИИ BADC и ИИ тетриса строились одним
и тем же скриптом и сравнивались бок о бок.
"""
from __future__ import annotations

import csv
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional

import numpy as np

from .arena import HeuristicAgent, NetAgent, RandomAgent, evaluate
from .config import AZConfig, GameConfig
from .encoding import action_size, num_channels
from .game import BADCGame
from .network import AZNetwork
from .parallel_selfplay import ParallelSelfPlay, default_workers
from .selfplay import ReplayBuffer, play_game


@dataclass
class TrainConfig:
    total_iters: int = 100
    eval_every: int = 5
    eval_games: int = 20
    eval_sims: int = 40
    save_every: int = 5
    log_every: int = 1
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    run_name: str = "az_badc"
    resume_from: Optional[str] = None
    anchor_checkpoint: Optional[str] = None  # замороженный чекпоинт-якорь для winrate_vs_anchor
    num_workers: int = 0  # параллельный self-play: 0 = авто (2/3 ядер), 1 = последовательно


CSV_HEADER = [
    "iteration",
    "global_step",
    "games_per_sec",
    "mean_game_len",
    "mean_score",
    "draw_rate",
    "first_player_winrate",
    "buffer_size",
    "total_loss",
    "policy_loss",
    "value_loss",
    "winrate_vs_random",
    "winrate_vs_heuristic",
    "policy_gap",
    "eval_len_vs_random",
    "winrate_vs_anchor",
]


def set_seed(seed: int) -> None:
    import random

    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _input_shape(game_cfg: GameConfig):
    return (game_cfg.rows, game_cfg.cols, num_channels())


def train(train_cfg: TrainConfig, game_cfg: GameConfig, az_cfg: AZConfig) -> None:
    from tensorflow.summary import create_file_writer, scalar

    set_seed(train_cfg.seed)
    game_cfg.validate()
    if game_cfg.player_count != 2:
        print("[warn] AlphaZero-обучение настроено на 2 игроков; "
              f"player_count={game_cfg.player_count} даст приближённый negamax-бэкап.")

    net = AZNetwork(_input_shape(game_cfg), action_size(game_cfg), az_cfg)
    buffer = ReplayBuffer(az_cfg.replay_size)

    Path(train_cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    run_log_dir = Path(train_cfg.log_dir) / train_cfg.run_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = run_log_dir / "tensorboard"
    writer = create_file_writer(str(tb_dir))

    csv_path = run_log_dir / "train_metrics.csv"
    csv_exists = csv_path.exists()
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(CSV_HEADER)

    global_step = 0  # суммарно сыгранных ходов self-play
    iteration = 0

    if train_cfg.resume_from:
        if not os.path.exists(train_cfg.resume_from):
            raise FileNotFoundError(
                f"--resume-from указывает на несуществующий файл: {train_cfg.resume_from}. "
                "Проверь путь (напр. *_latest сохраняется только ПОСЛЕ завершения прогона)."
            )
        net.load(train_cfg.resume_from)
        meta_path = train_cfg.resume_from + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            global_step = int(meta.get("global_step", 0))
            iteration = int(meta.get("iteration", 0))
        print(f"[resume] loaded {train_cfg.resume_from} global_step={global_step} iteration={iteration}")
        buf_path = train_cfg.resume_from + ".buffer.npz"
        if os.path.exists(buf_path):
            n = buffer.load(buf_path)
            print(f"[resume] loaded replay buffer ({n} examples) from {buf_path}")
        else:
            print(f"[resume] нет сохранённого буфера ({buf_path}); буфер наберётся заново")

    random_agent = RandomAgent()
    heuristic_agent = HeuristicAgent()
    anchor_agent = None
    if train_cfg.anchor_checkpoint:
        if not os.path.exists(train_cfg.anchor_checkpoint):
            print(f"[warn] --anchor-checkpoint не найден: {train_cfg.anchor_checkpoint}; "
                  "winrate_vs_anchor останется пустым")
        else:
            anchor_net = AZNetwork(_input_shape(game_cfg), action_size(game_cfg), az_cfg)
            anchor_net.load(train_cfg.anchor_checkpoint)
            anchor_agent = NetAgent(anchor_net, az_cfg, simulations=train_cfg.eval_sims)
            print(f"[anchor] winrate_vs_anchor против замороженного {train_cfg.anchor_checkpoint}")
    start_time = time.time()

    recent_len: Deque[int] = deque(maxlen=200)
    recent_score: Deque[float] = deque(maxlen=200)
    recent_gap: Deque[float] = deque(maxlen=200)

    print(
        f"[train] run={train_cfg.run_name} channels={num_channels()} "
        f"action_size={action_size(game_cfg)} sims={az_cfg.num_simulations} "
        f"games/iter={az_cfg.games_per_iter} rules(own={game_cfg.own_word},opp={game_cfg.opp_word}) "
        f"score_limit={game_cfg.score_limit}"
    )

    last_eval = {
        "winrate_vs_random": float("nan"),
        "winrate_vs_heuristic": float("nan"),
        "eval_len_vs_random": float("nan"),
        "winrate_vs_anchor": float("nan"),
    }

    # Параллельный self-play (опционально).
    pool = None
    workers = train_cfg.num_workers if train_cfg.num_workers else default_workers()
    if workers > 1:
        pool = ParallelSelfPlay(
            _input_shape(game_cfg), action_size(game_cfg), az_cfg,
            num_workers=workers, tmp_dir=train_cfg.checkpoint_dir,
        )
        print(f"[parallel] self-play на {workers} воркерах")
    else:
        print("[parallel] последовательный self-play (workers=1)")

    try:
        while iteration < train_cfg.total_iters:
            iteration += 1
            iter_start = time.time()

            # 1) self-play (параллельно или последовательно — результат идентичен по форме)
            draws = 0
            first_wins = 0
            games = az_cfg.games_per_iter
            if pool is not None:
                results = pool.play(net, games, game_cfg, az_cfg, base_seed=global_step + iteration)
            else:
                results = [play_game(net, game_cfg, az_cfg) for _ in range(games)]
            for examples, stats in results:
                for state, pi, z in examples:
                    buffer.add(state, pi, z)
                global_step += stats.turns
                recent_len.append(stats.turns)
                recent_score.append(float(max(stats.scores)))
                recent_gap.append(stats.mean_policy_gap)
                if stats.winner == -1:
                    draws += 1
                elif stats.winner == 0:
                    first_wins += 1

            games_per_sec = games / max(1e-6, time.time() - iter_start)
            draw_rate = draws / games
            first_player_winrate = first_wins / games

            # 2) обучение
            losses = {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
            n_updates = 0
            if len(buffer) >= az_cfg.min_replay:
                steps_per_epoch = max(1, len(buffer) // az_cfg.batch_size)
                for _ in range(az_cfg.epochs_per_iter):
                    for _ in range(steps_per_epoch):
                        states, pis, zs = buffer.sample(az_cfg.batch_size)
                        m = net.train_on_batch(states, pis, zs)
                        for k in losses:
                            losses[k] += m[k]
                        n_updates += 1
            if n_updates:
                for k in losses:
                    losses[k] /= n_updates

            # 3) оценка
            if iteration % train_cfg.eval_every == 0 and len(buffer) >= az_cfg.min_replay:
                eval_agent = NetAgent(net, az_cfg, simulations=train_cfg.eval_sims)
                wr_rand, len_rand = evaluate(
                    eval_agent, random_agent, game_cfg, train_cfg.eval_games, return_stats=True
                )
                last_eval["winrate_vs_random"] = wr_rand
                last_eval["eval_len_vs_random"] = len_rand
                last_eval["winrate_vs_heuristic"] = evaluate(
                    eval_agent, heuristic_agent, game_cfg, train_cfg.eval_games
                )
                if anchor_agent is not None:
                    last_eval["winrate_vs_anchor"] = evaluate(
                        eval_agent, anchor_agent, game_cfg, train_cfg.eval_games
                    )

            mean_len = float(np.mean(recent_len)) if recent_len else 0.0
            mean_score = float(np.mean(recent_score)) if recent_score else 0.0
            mean_gap = float(np.mean(recent_gap)) if recent_gap else 0.0

            # 4) логирование
            if iteration % train_cfg.log_every == 0:
                with writer.as_default():
                    scalar("selfplay/games_per_sec", games_per_sec, step=global_step)
                    scalar("selfplay/mean_game_len", mean_len, step=global_step)
                    scalar("selfplay/mean_score", mean_score, step=global_step)
                    scalar("selfplay/draw_rate", draw_rate, step=global_step)
                    scalar("selfplay/first_player_winrate", first_player_winrate, step=global_step)
                    scalar("selfplay/buffer_size", len(buffer), step=global_step)
                    scalar("selfplay/policy_gap", mean_gap, step=global_step)
                    scalar("loss/total_loss", losses["total_loss"], step=global_step)
                    scalar("loss/policy_loss", losses["policy_loss"], step=global_step)
                    scalar("loss/value_loss", losses["value_loss"], step=global_step)
                    if not np.isnan(last_eval["winrate_vs_random"]):
                        scalar("eval/winrate_vs_random", last_eval["winrate_vs_random"], step=global_step)
                        scalar("eval/winrate_vs_heuristic", last_eval["winrate_vs_heuristic"], step=global_step)
                        scalar("eval/len_vs_random", last_eval["eval_len_vs_random"], step=global_step)
                    if not np.isnan(last_eval["winrate_vs_anchor"]):
                        scalar("eval/winrate_vs_anchor", last_eval["winrate_vs_anchor"], step=global_step)
                writer.flush()

                csv_writer.writerow([
                    iteration,
                    global_step,
                    f"{games_per_sec:.3f}",
                    f"{mean_len:.2f}",
                    f"{mean_score:.3f}",
                    f"{draw_rate:.3f}",
                    f"{first_player_winrate:.3f}",
                    len(buffer),
                    f"{losses['total_loss']:.6f}",
                    f"{losses['policy_loss']:.6f}",
                    f"{losses['value_loss']:.6f}",
                    f"{last_eval['winrate_vs_random']:.3f}",
                    f"{last_eval['winrate_vs_heuristic']:.3f}",
                    f"{mean_gap:.6f}",
                    f"{last_eval['eval_len_vs_random']:.2f}",
                    f"{last_eval['winrate_vs_anchor']:.3f}",
                ])
                csv_file.flush()

                print(
                    f"[train] iter={iteration} step={global_step} "
                    f"g/s={games_per_sec:.2f} len={mean_len:.1f} score={mean_score:.2f} "
                    f"draw={draw_rate:.2f} loss={losses['total_loss']:.4f} "
                    f"(p={losses['policy_loss']:.3f} v={losses['value_loss']:.3f}) "
                    f"vs_rand={last_eval['winrate_vs_random']:.2f} "
                    f"vs_heur={last_eval['winrate_vs_heuristic']:.2f} "
                    f"vs_anchor={last_eval['winrate_vs_anchor']:.2f} buf={len(buffer)} "
                    f"gap={mean_gap:.3f}"
                )

            if iteration % train_cfg.save_every == 0:
                _save_checkpoint(net, train_cfg, iteration, global_step, latest=False)
    finally:
        _save_checkpoint(net, train_cfg, iteration, global_step, latest=True)
        latest_path = str(Path(train_cfg.checkpoint_dir) / f"{train_cfg.run_name}_latest.weights.h5")
        try:
            n_buf = buffer.save(latest_path + ".buffer.npz")
            print(f"[checkpoint] saved replay buffer ({n_buf} examples) -> {latest_path}.buffer.npz")
        except Exception as e:  # буфер не критичен — не валим сохранение весов
            print(f"[warn] не удалось сохранить буфер: {e}")
        if pool is not None:
            pool.close()
        writer.close()
        csv_file.close()
        print(f"[done] total time {time.time() - start_time:.1f}s")


def _save_checkpoint(net: AZNetwork, train_cfg: TrainConfig, iteration: int, global_step: int, latest: bool) -> None:
    suffix = "latest" if latest else f"iter{iteration}"
    path = str(Path(train_cfg.checkpoint_dir) / f"{train_cfg.run_name}_{suffix}.weights.h5")
    net.save(path)
    with open(path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"iteration": iteration, "global_step": global_step}, f)
    print(f"[checkpoint] saved {path}")
