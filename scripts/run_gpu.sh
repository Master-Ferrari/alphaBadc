#!/usr/bin/env bash
# Запуск GPU-обучения в WSL.
#
# TF 2.21 не прописывает каталоги pip-колёс nvidia-*-cu12 в путь загрузчика, и
# dlopen CUDA-библиотек падает ("Cannot dlopen some GPU libraries" -> CPU). Здесь
# мы добавляем эти каталоги в LD_LIBRARY_PATH, убеждаемся, что GPU виден, и только
# потом стартуем обучение. Все аргументы прокидываются в `main.py train`.
#
# Использование (venv уже активирован: source .venv-wsl/bin/activate):
#   bash scripts/run_gpu.sh --run-name az_badc_gpu --channels 128 --blocks 10 \
#     --games-per-iter 256 --batch-size 1024 --simulations 160 \
#     --batched-selfplay --total-iters 1000 --eval-games 100 --eval-every 10
set -euo pipefail
cd "$(dirname "$0")/.."

SP=$(python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$(ls -d "$SP"/nvidia/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"

python -c 'import tensorflow as tf; \
g = tf.config.list_physical_devices("GPU"); \
assert g, "GPU не виден TF — проверь nvidia-smi и колёса nvidia-*-cu12"; \
print("[gpu] OK:", g)'

exec python main.py train "$@"
