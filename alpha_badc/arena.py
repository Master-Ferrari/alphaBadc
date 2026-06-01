"""Оценка силы сети: матчи против эталонных соперников.

Агенты получают объект :class:`BADCGame` и возвращают action-id. Реализованы:
  * :class:`RandomAgent`    — равномерно случайный легальный ход;
  * :class:`HeuristicAgent` — жадная эвристика (порт идей slop/ai.js);
  * :class:`NetAgent`       — текущая сеть + MCTS (без шума, argmax по визитам).

:func:`evaluate` гоняет N партий, чередуя, кто ходит первым, и возвращает долю
побед оцениваемого агента (ничья = 0.5).
"""
from __future__ import annotations

import random
from typing import Callable

import numpy as np

from .config import AZConfig, GameConfig
from .game import BADCGame, EMPTY, NUM_LETTERS, WORD_LEN
from .mcts import MCTS, select_action

Agent = Callable[[BADCGame], int]

# ценность линии по числу совпавших букв (как WEIGHT в ai.js)
_LINE_WEIGHT = [0, 1, 8, 64, 1000]


class RandomAgent:
    def __call__(self, game: BADCGame) -> int:
        moves = game.legal_moves()
        mv = random.choice(moves)
        return BADCGame.move_to_action(mv)


class HeuristicAgent:
    """Жадная оценка хода: своё развитие минус потенциал соперников + мгновенные исходы."""

    def _line_value(self, board, line, word) -> int:
        count = 0
        for k in range(WORD_LEN):
            v = board[line[k]]
            if v == EMPTY:
                continue
            if v != word[k]:
                return 0
            count += 1
        return _LINE_WEIGHT[count]

    def __call__(self, game: BADCGame) -> int:
        me = game.current_player
        best_action, best_score = -1, -float("inf")
        for i in range(game.num_cells):
            if game.board[i] != EMPTY:
                continue
            for letter in range(NUM_LETTERS):
                # быстрая мгновенная победа / симуляция через движок
                g2 = game.clone()
                g2.apply((i, letter))
                if g2.done and g2.winner == me:
                    return BADCGame.move_to_action((i, letter))
                score = self._positional(g2, me) + random.random() * 0.01
                if score > best_score:
                    best_score, best_action = score, BADCGame.move_to_action((i, letter))
        return best_action

    def _positional(self, g: BADCGame, me: int) -> float:
        if g.done and g.winner is not None:
            if g.winner == me:
                return 1e7
            if g.winner != -1:
                return -1e7
        s = 0.0
        for p in range(g.config.player_count):
            s += (1.0 if p == me else -1.0) * g.scores[p] * 1000.0
        my_word = g.words[me]
        for line in g.lines:
            s += self._line_value(g.board, line, my_word)
            opp_max = 0
            for p in range(g.config.player_count):
                if p == me:
                    continue
                opp_max = max(opp_max, self._line_value(g.board, line, g.words[p]))
            s -= 0.8 * opp_max
        return s


class NetAgent:
    def __init__(self, net, az_config: AZConfig, simulations: int | None = None):
        cfg = az_config
        if simulations is not None:
            cfg = AZConfig(**{**az_config.__dict__})
            cfg.num_simulations = simulations
        self.mcts = MCTS(net, cfg)

    def __call__(self, game: BADCGame) -> int:
        visits = self.mcts.run(game, add_noise=False)
        return select_action(visits, temperature=0.0)


def play_match(agent_first: Agent, agent_second: Agent, game_config: GameConfig):
    """Сыграть партию. agent_first ходит за игрока 0.

    Вернуть ``(winner, turns)``: индекс победителя (-1 ничья) и длину партии в ходах.
    """
    game = BADCGame(game_config)
    agents = [agent_first, agent_second]
    while not game.done:
        action = agents[game.current_player % 2](game)
        game.apply(BADCGame.action_to_move(action))
    winner = game.winner if game.winner is not None else -1
    return winner, game.turn


def evaluate(eval_agent: Agent, opponent: Agent, game_config: GameConfig,
             n_games: int = 20, return_stats: bool = False):
    """Доля побед eval_agent (ничья = 0.5), чередуя право первого хода.

    При ``return_stats=True`` вернуть ``(winrate, mean_game_len)``.
    """
    wins = 0.0
    lengths: list[int] = []
    for i in range(n_games):
        if i % 2 == 0:
            winner, turns = play_match(eval_agent, opponent, game_config)
            eval_is = 0
        else:
            winner, turns = play_match(opponent, eval_agent, game_config)
            eval_is = 1
        lengths.append(turns)
        if winner == eval_is:
            wins += 1.0
        elif winner == -1:
            wins += 0.5
    winrate = wins / max(1, n_games)
    if return_stats:
        mean_len = sum(lengths) / len(lengths) if lengths else 0.0
        return winrate, mean_len
    return winrate
