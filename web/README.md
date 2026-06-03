# AlphaBADC — serverless веб-бандл

Игра против обученной сети **целиком в браузере**: сеть, кодировка позиции и
PUCT-MCTS портированы на чистый JS, forward сети выполняется вручную (без
TensorFlow.js / ONNX / WASM). Бэкенд не нужен — это статические файлы.

## Состав

| Файл | Что это |
|------|---------|
| `index.html` | UI (порт интерфейса из `webapp.py`) |
| `net.js` | ручной forward сети (conv+BN-fold, residual, две головы) |
| `game.js` | порт `alpha_badc/game.py` |
| `encoding.js` | порт `alpha_badc/encoding.py` |
| `mcts.js` | порт `alpha_badc/mcts.py` |
| `app.js` | склейка UI ↔ игра ↔ MCTS |
| `model.json` + `model.bin` | веса (≈3.6 МБ float32) — **артефакт сборки** |
| `parity.json`, `parity_test.js` | проверка совпадения JS-инференса с Python |

## Сборка весов

Из корня проекта, в окружении проекта (`wsl` → `source .venv-wsl/bin/activate`):

```bash
PYTHONPATH=. python scripts/export_web.py \
    --checkpoint checkpoints/az_badc_v3pre_nobonus_iter260.weights.h5 \
    --channels 64 --blocks 5 --head-kernel 4
```

Скрипт складывает BatchNorm в свёртки, выгружает `web/model.bin` + `web/model.json`
и тестовые позиции `web/parity.json`. Параметры архитектуры берутся из чекпоинта
(этот: 43 входных канала, ствол 64×5 блоков, ядро голов 4×4).

## Проверка паритета (Node)

```bash
node web/parity_test.js
```

Сравнивает JS-кодировку и JS-инференс с эталоном из Python. Норма — `Δ < 1e-4`
(фактически ~1e-7, разница только от округления float32).

## Запуск

`fetch` не работает с `file://`, поэтому нужен любой статический сервер:

```bash
cd web && python -m http.server 8080
# открыть http://localhost:8080
```

Для деплоя — выложить каталог `web/` на любой статический хостинг
(GitHub Pages, Netlify, S3 и т.п.).

## Заметки

* Инференс: ~0.4 с/ход при 80 симуляциях MCTS (селектор в UI: 20/80/160/320).
* Оппонент играет как `arena.NetAgent`: без dirichlet-шума, argmax по визитам.
* Поддержан режим `GameConfig` по умолчанию (2 игрока, `own_word=explode`,
  `opp_word=on`, `score_limit=4`). Другие правила потребуют правок в `game.js`.
* `model.bin` — бинарный артефакт; при желании добавьте `web/model.bin` в
  `.gitignore` и пересобирайте скриптом.
```
