

venv
```
wsl
source .venv-wsl/bin/activate
```

претрейн
```
python scripts/pretrain_heuristic.py --games 5000 --workers 16 --epochs 4 \
  --head-kernel 4 --channels 64 --blocks 5 \
  --keep-shards \
  --out checkpoints/az_badc_v3_pretrain.weights.h5
```

cpu на четверной голове
```
python main.py train --resume-from checkpoints/az_badc_v3_pretrain.weights.h5 \
  --run-name az_badc_v3pre --workers 16 --head-kernel 4 --channels 64 --blocks 5 \
  --total-iters 200
```












# AlphaBADC

Состязательное обучение ИИ для настольной игры **BADC** методом **AlphaZero**
(self-play + MCTS + нейросеть на **TensorFlow**). Правила игры полностью
настраиваемые — см. [`rules.md`](rules.md).

Проект сделан как «брат» [`tetris_poo`](../tetris_poo): тот же формат логов и тот
же скрипт построения графиков, чтобы кривые обучения ИИ BADC и ИИ тетриса можно
было сравнивать бок о бок.

## Что это за игра

BADC — «пять-в-ряд наоборот» на поле 9×5 четырьмя общими буквами `a/b/c/d`.
У каждого игрока своё целевое слово; кто первым выложит свой паттерн в линию —
получает очки/побеждает. Доп. правила (`ownWord`/`oppWord`: `off`/`on`/`explode`/
`corrupt`) и лимит очков делают режим либо «мгновенная победа», либо «по очкам».
Подробно — в [`rules.md`](rules.md).

## Архитектура

```
alpha_badc/
  config.py     # GameConfig (правила) + AZConfig (гиперпараметры AlphaZero)
  game.py       # движок BADC (порт slop/game.js): apply/clone/legal_moves/winner
  encoding.py   # кодирование позиции в тензор планов (с точки зрения ходящего)
  network.py    # TensorFlow/Keras: residual-ствол → policy + value головы
  mcts.py       # PUCT-MCTS (negamax-бэкап, двухигровая нулевая сумма)
  selfplay.py   # генерация партий self-play + буфер воспроизведения
  arena.py      # оценка: random / эвристика / net-vs-net, win-rate
  trainer.py    # итеративный цикл self-play → обучение → оценка → логи/чекпоинты
  play.py       # инференс с текстовым / pygame-рендером
main.py         # CLI: train | eval | infer
scripts/plot_train_metrics.py   # графики (тот же стиль, что у tetris_poo)
```

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Быстрый старт

### 1) Обучение

```bash
python main.py train --total-iters 100 --run-name az_badc
```

Один цикл: сеть играет `--games-per-iter` партий сама с собой, учится на буфере и
раз в `--eval-every` циклов оценивается на арене. Во время обучения:
- чекпоинты пишутся в `checkpoints/` (`*.weights.h5` + `*.meta.json`);
- метрики — в `logs/<run-name>/train_metrics.csv`;
- TensorBoard-логи — в `logs/<run-name>/tensorboard`.

Настройка правил прямо из CLI (как в `rules.md`):

```bash
python main.py train --own-word explode --opp-word on --score-limit 4 --players 2
```

### 2) Графики

```bash
MPLBACKEND=Agg python scripts/plot_train_metrics.py \
  --csv logs/az_badc_v3/train_metrics.csv --out-dir logs/plots_v3
```

`MPLBACKEND=Agg` — headless-бэкенд: обязателен под WSL/сервером без дисплея
(иначе matplotlib пытается открыть GUI-окно). На десктопе с экраном можно опустить.

Зависимости (`matplotlib`, `pandas`) ставятся из `requirements.txt`. Если видишь
`ModuleNotFoundError: No module named 'matplotlib'` — окружение собрано не полностью,
доустанови: `pip install -r requirements.txt`.

> ⚠️ Окружения `.venv` / `.venv-wsl` в этом проекте — **WSL/Linux** (см. `pyvenv.cfg`:
> `home = /usr/bin`). Из Windows PowerShell их нельзя активировать через
> `.venv\Scripts\activate` — запускай через WSL:
> `wsl -e bash -lc "cd <проект> && source .venv-wsl/bin/activate && MPLBACKEND=Agg python scripts/plot_train_metrics.py --csv ... --out-dir ..."`

Скрипт идентичен тетрисному: строит каждую метрику + `overview.png` +
`metric_explanations.md`. Чтобы сравнить с тетрисом — постройте оба CSV одним и
тем же скриптом и смотрите `overview.png` рядом.

### 3) Оценка чекпоинта

```bash
python main.py eval --checkpoint checkpoints/az_badc_latest.weights.h5 --opponent heuristic --games 40
```

### 4) Инференс (игра сети + визуализация)

```bash
# текстовый рендер в консоль
python main.py infer --checkpoint checkpoints/az_badc_latest.weights.h5 --opponent heuristic
# pygame-окно
python main.py infer --checkpoint checkpoints/az_badc_latest.weights.h5 --render --fps 2
```

## Основные параметры

`train`:
- правила: `--own-word`, `--opp-word` (`off|on|explode|corrupt`), `--score-limit`, `--players`, `--max-turns`
- AlphaZero: `--simulations`, `--c-puct`, `--channels`, `--blocks`, `--learning-rate`
- цикл: `--total-iters`, `--games-per-iter`, `--epochs-per-iter`, `--batch-size`
- оценка/сохранение: `--eval-every`, `--eval-games`, `--eval-sims`, `--save-every`
- `--resume-from checkpoints/az_badc_latest.weights.h5`

## Замечание про число игроков

Движок поддерживает 2–4 игроков, но AlphaZero-бэкап реализован для **двух**
игроков (negamax с инверсией знака). При `--players > 2` обучение использует
приближённый бэкап — корректнее всего обучать на `--players 2`, а многопользова-
тельские партии гонять в `eval`/`infer`.

## Resume обучения

```bash
python main.py train --resume-from checkpoints/az_badc_latest.weights.h5 --run-name az_badc
```

## Очистка

```bash
rm -rf checkpoints/ logs/
```
