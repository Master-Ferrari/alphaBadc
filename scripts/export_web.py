"""Экспорт чекпоинта AlphaBADC в самодостаточный веб-бандл (без ML-рантайма).

Forward сети в браузере выполняется вручную на чистом JS (см. web/net.js),
поэтому здесь:
  1. строим модель и грузим веса чекпоинта;
  2. складываем каждый BatchNorm в предшествующую свёртку (conv без bias),
     получая conv с bias — JS реализует только conv+relu/add/dense;
  3. выгружаем все веса одним float32-блобом web/model.bin + манифест
     web/model.json (имя слоя -> shape/offset);
  4. выгружаем несколько случайных позиций с эталонными (policy, value)
     в web/parity.json — для проверки совпадения JS-инференса с Python.

Запуск (в WSL-окружении проекта):
    python scripts/export_web.py \
        --checkpoint checkpoints/az_badc_v3pre_nobonus_iter260.weights.h5 \
        --channels 64 --blocks 5 --head-kernel 4
"""
from __future__ import annotations

import argparse
import json
import os
import struct

import numpy as np

from alpha_badc.config import AZConfig, GameConfig
from alpha_badc.encoding import action_size, encode_state, num_channels
from alpha_badc.game import BADCGame
from alpha_badc.network import AZNetwork


def _bn_fold(conv_w, bn_layer):
    """Сложить BatchNorm в предшествующую conv (use_bias=False).

    Возвращает (W', b') такие, что conv(W')+b' == bn(conv(W)).
    Keras BN: y = gamma*(x-mean)/sqrt(var+eps) + beta.
    """
    gamma, beta, mean, var = (w for w in bn_layer.get_weights())
    eps = float(bn_layer.epsilon)
    scale = gamma / np.sqrt(var + eps)          # (out,)
    w_folded = conv_w * scale.reshape(1, 1, 1, -1)
    b_folded = beta - mean * scale
    return w_folded.astype(np.float32), b_folded.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--head-kernel", type=int, default=4)
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--out", default="web")
    ap.add_argument("--parity-positions", type=int, default=8)
    args = ap.parse_args()

    game_cfg = GameConfig()
    az_cfg = AZConfig(channels=args.channels, blocks=args.blocks,
                      kernel=args.kernel, head_kernel=args.head_kernel)
    in_c = num_channels()
    net = AZNetwork((game_cfg.rows, game_cfg.cols, in_c), action_size(game_cfg), az_cfg)
    net.load_partial(args.checkpoint, zero_new_channels=True)
    model = net.model
    layers = {l.name: l for l in model.layers}

    blob = bytearray()
    manifest = {}

    def add(name, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        manifest[name] = {"shape": list(arr.shape), "offset": len(blob) // 4,
                          "size": int(arr.size)}
        blob.extend(arr.tobytes())

    # --- ствол: stem + residual-блоки ---
    w, b = _bn_fold(layers["stem_conv"].get_weights()[0], layers["stem_bn"])
    add("stem.w", w); add("stem.b", b)
    for i in range(args.blocks):
        wa, ba = _bn_fold(layers[f"res{i}_a_conv"].get_weights()[0], layers[f"res{i}_a_bn"])
        add(f"res{i}.a.w", wa); add(f"res{i}.a.b", ba)
        wb, bb = _bn_fold(layers[f"res{i}_b_conv"].get_weights()[0], layers[f"res{i}_b_bn"])
        add(f"res{i}.b.w", wb); add(f"res{i}.b.b", bb)

    # --- policy head ---
    wp, bp = _bn_fold(layers["policy_head_conv"].get_weights()[0], layers["policy_head_bn"])
    add("policy.conv.w", wp); add("policy.conv.b", bp)
    pw, pb = layers["policy_logits"].get_weights()
    add("policy.fc.w", pw); add("policy.fc.b", pb)

    # --- value head ---
    wv, bv = _bn_fold(layers["value_head_conv"].get_weights()[0], layers["value_head_bn"])
    add("value.conv.w", wv); add("value.conv.b", bv)
    vw, vb = layers["value_dense"].get_weights()
    add("value.fc1.w", vw); add("value.fc1.b", vb)
    ow, ob = layers["value"].get_weights()
    add("value.fc2.w", ow); add("value.fc2.b", ob)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "model.bin"), "wb") as f:
        f.write(blob)
    meta = {
        "input_shape": [game_cfg.rows, game_cfg.cols, in_c],
        "channels": args.channels, "blocks": args.blocks,
        "kernel": args.kernel, "head_kernel": args.head_kernel,
        "action_size": action_size(game_cfg),
        "weights": manifest,
    }
    with open(os.path.join(args.out, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] {len(blob)//4} float32 -> {args.out}/model.bin "
          f"({len(blob)/1e6:.2f} MB), манифест model.json")

    # --- паритет: случайные позиции + эталонные (state, probs, value) ---
    rng = np.random.default_rng(0)
    cases = []
    for _ in range(args.parity_positions):
        g = BADCGame(game_cfg)
        steps = int(rng.integers(0, 18))
        actions = []
        for _ in range(steps):
            if g.done:
                break
            moves = g.legal_moves()
            mv = moves[int(rng.integers(0, len(moves)))]
            actions.append(BADCGame.move_to_action(mv))
            g.apply(mv)
        if g.done:
            continue
        st = encode_state(g)            # (rows, cols, C)
        probs, value = net.predict(st)  # softmax probs, scalar
        cases.append({
            "actions": actions,  # последовательность action-id от старта (для JS-воспроизведения)
            "state": st.reshape(-1).astype(np.float32).round(6).tolist(),
            "probs": probs.astype(np.float32).round(6).tolist(),
            "value": round(float(value), 6),
        })
    with open(os.path.join(args.out, "parity.json"), "w") as f:
        json.dump({"shape": list(encode_state(BADCGame(game_cfg)).shape),
                   "cases": cases}, f)
    print(f"[export] {len(cases)} паритет-позиций -> {args.out}/parity.json")


if __name__ == "__main__":
    main()
