from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Описания метрик AlphaBADC (тот же формат, что в tetris_poo, для сравнимости).
METRIC_DESCRIPTIONS: dict[str, str] = {
    "games_per_sec": "Скорость self-play (партий в секунду).",
    "mean_game_len": "Средняя длина партии в ходах (как долго идёт игра до победы/лимита).",
    "mean_score": "Средний максимум очков в партии (динамика набора очков в режиме счёта).",
    "draw_rate": "Доля ничьих среди партий self-play.",
    "first_player_winrate": "Доля побед игрока, ходящего первым (баланс/преимущество первого хода).",
    "second_player_winrate": "Доля побед второго игрока (1 - first_player_winrate - draw_rate).",
    "buffer_size": "Размер буфера воспроизведения (накоплено обучающих примеров).",
    "total_loss": "Суммарный лосс (policy + value + регуляризация).",
    "policy_loss": "Cross-entropy между визитами MCTS (pi) и предсказанной политикой.",
    "value_loss": "MSE между предсказанной ценностью и итогом партии z.",
    "winrate_vs_random": "Доля побед сети против случайного агента (главный индикатор силы на старте).",
    "winrate_vs_heuristic": "Доля побед сети против эвристического бота (главный индикатор качества).",
    "score_per_move": "Очков за ход (mean_score / mean_game_len): эффективность набора счёта.",
    "sec_per_move": "Время на один ход self-play, сек (1 / (games_per_sec * mean_game_len)): "
                    "отделяет удлинение партий от просадки производительности железа.",
    "policy_gap": "Средний KL(pi_MCTS || policy_net) в корне: насколько поиск ещё исправляет "
                  "сырую политику сети. Падение к полу = сеть впитала почти всё, что даёт MCTS.",
    "eval_len_vs_random": "Средняя длина партии (в ходах) в eval против random: "
                          "косвенный индикатор силы — сильная сеть быстрее доводит партию до развязки.",
    "winrate_vs_anchor": "Доля побед против ЗАМОРОЖЕННОГО чекпоинта-якоря. >0.5 = текущая сеть "
                         "переросла якорь; неограниченный индикатор абсолютного прогресса.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot training curves from train_metrics.csv")
    parser.add_argument("--csv", type=str, required=True, help="Path to train_metrics.csv")
    parser.add_argument("--out-dir", type=str, default="logs/plots", help="Directory to save generated plots")
    parser.add_argument("--smooth-window", type=int, default=10, help="Rolling window for smoothing")
    return parser


def plot_metric(df: pd.DataFrame, x_col: str, y_col: str, out_dir: Path, smooth_window: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[x_col], df[y_col], label="raw", alpha=0.45, linewidth=1.2)
    smooth = df[y_col].rolling(window=max(1, smooth_window), min_periods=1).mean()
    ax.plot(df[x_col], smooth, label=f"rolling_mean({smooth_window})", linewidth=2.0)
    ax.set_title(y_col)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{y_col}.png", dpi=140)
    plt.close(fig)


# Серии для совмещённой панели «баланс сторон» (первый/второй/ничья).
_BALANCE_SERIES = [
    ("first_player_winrate", "first"),
    ("second_player_winrate", "second"),
    ("draw_rate", "draw"),
]


def _draw_balance(ax, df: pd.DataFrame, x_col: str, smooth_window: int) -> None:
    """Нарисовать на оси совмещённый баланс: сглаженные линии + бледный raw."""
    for col, label in _BALANCE_SERIES:
        if col not in df.columns:
            continue
        smooth = df[col].rolling(window=max(1, smooth_window), min_periods=1).mean()
        line = ax.plot(df[x_col], smooth, linewidth=2.0, label=label)[0]
        ax.plot(df[x_col], df[col], color=line.get_color(), alpha=0.2, linewidth=1.0)
    ax.set_title("player_balance (first / second / draw)")
    ax.set_xlabel(x_col)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend()


def plot_player_balance(df: pd.DataFrame, x_col: str, out_dir: Path, smooth_window: int) -> None:
    """Отдельный график баланса первого/второго игрока и ничьих."""
    if "first_player_winrate" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    _draw_balance(ax, df, x_col, smooth_window)
    fig.tight_layout()
    fig.savefig(out_dir / "player_balance.png", dpi=140)
    plt.close(fig)


def plot_overview(df: pd.DataFrame, x_col: str, out_dir: Path, smooth_window: int) -> None:
    key_metrics = [
        "winrate_vs_anchor",
        "winrate_vs_random",
        "player_balance",
        "total_loss",
        "value_loss",
        "mean_game_len",
        "mean_score",
        "score_per_move",
    ]

    def _available(m: str) -> bool:
        if m == "player_balance":
            return "first_player_winrate" in df.columns
        return m in df.columns

    existing = [m for m in key_metrics if _available(m)]
    if not existing:
        return

    rows = (len(existing) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, metric in enumerate(existing):
        ax = axes[i]
        if metric == "player_balance":
            _draw_balance(ax, df, x_col, smooth_window)
            continue
        ax.plot(df[x_col], df[metric], alpha=0.45, linewidth=1.2, label="raw")
        smooth = df[metric].rolling(window=max(1, smooth_window), min_periods=1).mean()
        ax.plot(df[x_col], smooth, linewidth=2.0, label="smoothed")
        ax.set_title(metric)
        ax.set_xlabel(x_col)
        ax.grid(True, alpha=0.25)
        ax.legend()

    for j in range(len(existing), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "overview.png", dpi=140)
    plt.close(fig)


def write_explanations(df: pd.DataFrame, out_dir: Path) -> None:
    lines: list[str] = ["# Train Metrics: краткие пояснения", ""]
    for col in df.columns:
        if col in ("iteration", "global_step"):
            continue
        desc = METRIC_DESCRIPTIONS.get(col, "Описание не задано.")
        lines.append(f"- `{col}`: {desc}")
    text = "\n".join(lines) + "\n"
    (out_dir / "metric_explanations.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    # Производные метрики (деление на 0 -> NaN, такие точки не рисуются).
    if {"mean_score", "mean_game_len"}.issubset(df.columns):
        denom = df["mean_game_len"].where(df["mean_game_len"] != 0)
        df["score_per_move"] = df["mean_score"] / denom
    if {"games_per_sec", "mean_game_len"}.issubset(df.columns):
        rate = (df["games_per_sec"] * df["mean_game_len"])
        df["sec_per_move"] = 1.0 / rate.where(rate != 0)
    if {"first_player_winrate", "draw_rate"}.issubset(df.columns):
        df["second_player_winrate"] = 1.0 - df["first_player_winrate"] - df["draw_rate"]

    x_col = "global_step" if "global_step" in df.columns else "iteration"
    if x_col not in df.columns:
        raise ValueError("CSV must contain `global_step` or `iteration` column.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in df.columns:
        if col in ("iteration", "global_step"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            plot_metric(df, x_col, col, out_dir, args.smooth_window)

    plot_player_balance(df, x_col, out_dir, args.smooth_window)
    plot_overview(df, x_col, out_dir, args.smooth_window)
    write_explanations(df, out_dir)

    print(f"[ok] plots saved to: {out_dir}")
    print(f"[ok] explanations saved to: {out_dir / 'metric_explanations.md'}")


if __name__ == "__main__":
    main()
