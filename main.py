"""CLI AlphaBADC: обучение AlphaZero, оценка на арене и инференс.

Примеры:
  python main.py train --total-iters 100 --run-name az_badc
  python main.py eval  --checkpoint checkpoints/az_badc_latest.weights.h5 --opponent heuristic
  python main.py infer --checkpoint checkpoints/az_badc_latest.weights.h5 --render
"""
from __future__ import annotations

import argparse

from alpha_badc.config import AZConfig, GameConfig, RULE_CHOICES


def add_game_args(parser: argparse.ArgumentParser) -> None:
    g = GameConfig()
    parser.add_argument("--players", type=int, default=g.player_count, help="2..4 (обучение настроено на 2)")
    parser.add_argument("--own-word", choices=RULE_CHOICES, default=g.own_word)
    parser.add_argument("--opp-word", choices=RULE_CHOICES, default=g.opp_word)
    parser.add_argument("--score-limit", type=float, default=g.score_limit, help="лимит очков (0 = выкл)")
    parser.add_argument("--max-turns", type=int, default=g.max_turns)


def add_az_args(parser: argparse.ArgumentParser) -> None:
    a = AZConfig()
    parser.add_argument("--simulations", type=int, default=a.num_simulations)
    parser.add_argument("--c-puct", type=float, default=a.c_puct)
    parser.add_argument("--channels", type=int, default=a.channels)
    parser.add_argument("--blocks", type=int, default=a.blocks)
    parser.add_argument("--kernel-size", type=int, default=a.kernel, help="размер ядра свёрток ствола (обычно 3)")
    parser.add_argument("--head-kernel", type=int, default=a.head_kernel,
                        help="размер ядра в головах (4 ≈ длина слова BADC)")
    parser.add_argument("--learning-rate", type=float, default=a.learning_rate)
    parser.add_argument("--games-per-iter", type=int, default=a.games_per_iter)
    parser.add_argument("--epochs-per-iter", type=int, default=a.epochs_per_iter)
    parser.add_argument("--batch-size", type=int, default=a.batch_size)
    parser.add_argument("--min-replay", type=int, default=a.min_replay, help="порог буфера до старта обучения")
    parser.add_argument("--replay-size", type=int, default=a.replay_size)


def build_game_cfg(args) -> GameConfig:
    return GameConfig(
        player_count=args.players,
        own_word=args.own_word,
        opp_word=args.opp_word,
        score_limit=(None if args.score_limit in (0, 0.0) else args.score_limit),
        max_turns=args.max_turns,
    )


def build_az_cfg(args) -> AZConfig:
    cfg = AZConfig(
        num_simulations=args.simulations,
        c_puct=args.c_puct,
        channels=args.channels,
        blocks=args.blocks,
        kernel=getattr(args, "kernel_size", 3),
        head_kernel=getattr(args, "head_kernel", 3),
        learning_rate=args.learning_rate,
    )
    cfg.games_per_iter = getattr(args, "games_per_iter", cfg.games_per_iter)
    cfg.epochs_per_iter = getattr(args, "epochs_per_iter", cfg.epochs_per_iter)
    cfg.batch_size = getattr(args, "batch_size", cfg.batch_size)
    cfg.min_replay = getattr(args, "min_replay", cfg.min_replay)
    cfg.replay_size = getattr(args, "replay_size", cfg.replay_size)
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaZero training/eval/inference for BADC.")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train", help="Обучение AlphaZero через self-play.")
    add_game_args(p_train)
    add_az_args(p_train)
    p_train.add_argument("--total-iters", type=int, default=100)
    p_train.add_argument("--eval-every", type=int, default=5)
    p_train.add_argument("--eval-games", type=int, default=20)
    p_train.add_argument("--eval-sims", type=int, default=40)
    p_train.add_argument("--save-every", type=int, default=5)
    p_train.add_argument("--run-name", type=str, default="az_badc")
    p_train.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p_train.add_argument("--log-dir", type=str, default="logs")
    p_train.add_argument("--resume-from", type=str, default=None)
    p_train.add_argument("--resume-partial", action="store_true",
                         help="частичная загрузка весов при смене числа входных каналов "
                              "(переносит stem-conv по старым каналам, новые — заново)")
    p_train.add_argument("--anchor-checkpoint", type=str, default=None,
                         help="замороженный чекпоинт для метрики winrate_vs_anchor")
    p_train.add_argument("--workers", type=int, default=0,
                         help="параллельный self-play: 0 = авто (2/3 ядер), 1 = последовательно")
    p_train.add_argument("--batched-selfplay", action="store_true",
                         help="GPU-режим: партии «в ширину», листья всех партий — одним predict_batch "
                              "(с --workers N шардит на N GPU-процессов; иначе 1 процесс)")
    p_train.add_argument("--selfplay-gpu-mem-mb", type=int, default=0,
                         help="лимит VRAM на GPU-воркера батч-self-play, МБ (0 = memory_growth)")
    p_train.add_argument("--seed", type=int, default=42)

    p_eval = sub.add_parser("eval", help="Оценить чекпоинт на арене.")
    add_game_args(p_eval)
    add_az_args(p_eval)
    p_eval.add_argument("--checkpoint", type=str, required=True)
    p_eval.add_argument("--opponent", choices=["random", "heuristic", "net"], default="heuristic")
    p_eval.add_argument("--games", type=int, default=40)
    p_eval.add_argument("--eval-sims", type=int, default=80)

    p_infer = sub.add_parser("infer", help="Игра обученной сети с визуализацией.")
    add_game_args(p_infer)
    add_az_args(p_infer)
    p_infer.add_argument("--checkpoint", type=str, required=True)
    p_infer.add_argument("--opponent", choices=["random", "heuristic", "net"], default="heuristic")
    p_infer.add_argument("--episodes", type=int, default=5)
    p_infer.add_argument("--infer-sims", type=int, default=80)
    p_infer.add_argument("--render", action="store_true")
    p_infer.add_argument("--fps", type=int, default=2)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    game_cfg = build_game_cfg(args)
    az_cfg = build_az_cfg(args)

    if args.mode == "train":
        from alpha_badc.trainer import TrainConfig, train

        train_cfg = TrainConfig(
            total_iters=args.total_iters,
            eval_every=args.eval_every,
            eval_games=args.eval_games,
            eval_sims=args.eval_sims,
            save_every=args.save_every,
            run_name=args.run_name,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            resume_from=args.resume_from,
            resume_partial=args.resume_partial,
            anchor_checkpoint=args.anchor_checkpoint,
            num_workers=args.workers,
            batched_selfplay=args.batched_selfplay,
            selfplay_gpu_mem_mb=args.selfplay_gpu_mem_mb,
            seed=args.seed,
        )
        train(train_cfg, game_cfg, az_cfg)
        return

    if args.mode == "eval":
        from alpha_badc.arena import HeuristicAgent, NetAgent, RandomAgent, evaluate
        from alpha_badc.encoding import action_size, num_channels
        from alpha_badc.network import AZNetwork

        net = AZNetwork((game_cfg.rows, game_cfg.cols, num_channels()), action_size(game_cfg), az_cfg)
        net.load(args.checkpoint)
        eval_agent = NetAgent(net, az_cfg, simulations=args.eval_sims)
        opp = {"random": RandomAgent(), "heuristic": HeuristicAgent(),
               "net": NetAgent(net, az_cfg, simulations=args.eval_sims)}[args.opponent]
        wr = evaluate(eval_agent, opp, game_cfg, args.games)
        print(f"[eval] winrate vs {args.opponent} = {wr:.3f} over {args.games} games")
        return

    if args.mode == "infer":
        from alpha_badc.play import run_play

        run_play(
            checkpoint=args.checkpoint,
            game_cfg=game_cfg,
            az_cfg=az_cfg,
            opponent=args.opponent,
            episodes=args.episodes,
            sims=args.infer_sims,
            render=args.render,
            fps=args.fps,
        )
        return

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
