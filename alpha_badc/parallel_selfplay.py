"""Параллельный self-play на нескольких CPU-процессах.

Self-play — узкое место обучения (~95% времени) и при этом «эмбэрэссингли
параллелен»: партии независимы. Здесь — пул постоянных воркеров, каждый держит
СВОЮ копию сети и подгружает свежие веса из файла в начале каждой итерации.

Особенности (важны на Windows со start-method ``spawn``):
  * TF и сеть импортируются ЛЕНИВО, уже ВНУТРИ воркера, после ограничения числа
    потоков (intra/inter op = 1), иначе N воркеров × дефолтные потоки = тормоза;
  * живая TF-модель не передаётся между процессами — только путь к весам;
  * у каждого воркера свой seed, чтобы партии различались.

Главный процесс (trainer) не меняется по сути: он получает тот же список
``(examples, stats)``, что и при последовательном self-play, и дальше всё как
прежде — буфер, обучение, чекпоинты, логи.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from typing import List, Tuple

# В этом модуле НЕ импортируем tensorflow/network на верхнем уровне:
# при spawn дочерний процесс заново импортирует модуль, и TF подтянулся бы
# раньше, чем мы успеем ограничить потоки. Все тяжёлые импорты — внутри функций.

_WORKER: dict = {}  # per-process кэш: собранная сеть и её конфиг


def default_workers() -> int:
    """Две трети логических ядер (минимум 1)."""
    n = os.cpu_count() or 4
    return max(1, int(n * 2 / 3))


def _split_games(total: int, parts: int) -> List[int]:
    """Раздать ``total`` партий по ``parts`` воркерам максимально ровно."""
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def _worker_init(input_shape, action_size, az_cfg) -> None:
    # 1) Прибиваем потоки ДО импорта TF.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    import tensorflow as tf

    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass  # рантайм уже инициализирован — не критично

    from .network import AZNetwork

    _WORKER["net"] = AZNetwork(input_shape, action_size, az_cfg)


def _worker_play(weights_path: str, n_games: int, game_cfg, az_cfg, seed: int):
    """Загрузить свежие веса и сыграть ``n_games`` партий self-play."""
    import random

    import numpy as np

    from .selfplay import play_game

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    net = _WORKER["net"]
    net.load(weights_path)

    return [play_game(net, game_cfg, az_cfg) for _ in range(n_games)]


class ParallelSelfPlay:
    """Постоянный пул воркеров для self-play.

    Использование::

        pool = ParallelSelfPlay(input_shape, action_size, az_cfg, num_workers, tmp_dir)
        results = pool.play(net, games, game_cfg, az_cfg, base_seed)  # list[(examples, stats)]
        ...
        pool.close()
    """

    def __init__(self, input_shape, action_size, az_cfg, num_workers: int, tmp_dir: str):
        self.num_workers = max(1, num_workers)
        # Уникальное имя на процесс: два параллельных прогона в одной папке не столкнутся.
        self._tmp_weights = str(Path(tmp_dir) / f"_worker_tmp_{os.getpid()}.weights.h5")
        ctx = mp.get_context("spawn")  # явно: одинаково на Win/Linux
        self.pool = ctx.Pool(
            processes=self.num_workers,
            initializer=_worker_init,
            initargs=(input_shape, action_size, az_cfg),
        )

    def play(self, net, games: int, game_cfg, az_cfg, base_seed: int
             ) -> List[Tuple[list, object]]:
        """Сыграть ``games`` партий параллельно. Возвращает list[(examples, stats)]."""
        # Главный процесс пишет текущие веса; воркеры их прочитают.
        net.save(self._tmp_weights)

        chunks = _split_games(games, self.num_workers)
        tasks = [
            (self._tmp_weights, n, game_cfg, az_cfg, base_seed + i * 100003)
            for i, n in enumerate(chunks)
            if n > 0
        ]
        results: List[Tuple[list, object]] = []
        for chunk in self.pool.starmap(_worker_play, tasks):
            results.extend(chunk)
        return results

    def close(self) -> None:
        try:
            self.pool.close()
            self.pool.join()
        finally:
            try:
                os.remove(self._tmp_weights)
            except OSError:
                pass
