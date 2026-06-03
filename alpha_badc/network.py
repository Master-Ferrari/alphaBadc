"""Нейросеть AlphaZero на TensorFlow/Keras.

Общий residual-ствол → две головы:
  * policy — логиты по всем действиям (num_cells*NUM_LETTERS);
  * value  — скаляр в [-1, 1] (tanh): ожидаемый исход с точки зрения ходящего.

Класс :class:`AZNetwork` инкапсулирует модель, предсказание (одиночное и
батчевое для MCTS), один шаг обучения и сохранение/загрузку весов.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import AZConfig


def _conv_block(x, channels: int, l2: float, name: str, kernel: int = 3):
    x = layers.Conv2D(
        channels, kernel, padding="same", use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2), name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def _residual_block(x, channels: int, l2: float, name: str, kernel: int = 3):
    shortcut = x
    y = _conv_block(x, channels, l2, f"{name}_a", kernel=kernel)
    y = layers.Conv2D(
        channels, kernel, padding="same", use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2), name=f"{name}_b_conv",
    )(y)
    y = layers.BatchNormalization(name=f"{name}_b_bn")(y)
    y = layers.Add(name=f"{name}_add")([shortcut, y])
    return layers.ReLU(name=f"{name}_out")(y)


def build_model(input_shape: Tuple[int, int, int], action_size: int, cfg: AZConfig) -> keras.Model:
    l2 = cfg.weight_decay
    k = getattr(cfg, "kernel", 3)            # ядро ствола
    hk = getattr(cfg, "head_kernel", 3)      # ядро голов (4 ≈ длина слова BADC)
    inputs = keras.Input(shape=input_shape, name="state")
    x = _conv_block(inputs, cfg.channels, l2, "stem", kernel=k)
    for i in range(cfg.blocks):
        x = _residual_block(x, cfg.channels, l2, f"res{i}", kernel=k)

    # policy head
    p = _conv_block(x, 32, l2, "policy_head", kernel=hk)
    p = layers.Flatten(name="policy_flatten")(p)
    policy = layers.Dense(
        action_size, kernel_regularizer=keras.regularizers.l2(l2), name="policy_logits"
    )(p)

    # value head
    v = _conv_block(x, 32, l2, "value_head", kernel=hk)
    v = layers.Flatten(name="value_flatten")(v)
    v = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(l2), name="value_dense")(v)
    value = layers.Dense(1, activation="tanh", name="value")(v)

    return keras.Model(inputs=inputs, outputs=[policy, value], name="alpha_badc_net")


def _read_stem_in_channels(path: str, trunk_channels: int) -> int:
    """Прочитать число входных каналов stem-conv из .weights.h5.

    Stem — единственное 3x3-ядро формы (3, 3, C, trunk_channels) с
    C != trunk_channels (у остальных свёрток вход равен ширине ствола).
    """
    import h5py

    candidates: list[int] = []
    with h5py.File(path, "r") as f:
        def visit(_name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 4:
                s = tuple(obj.shape)
                if s[0] == 3 and s[1] == 3 and s[3] == trunk_channels and s[2] != trunk_channels:
                    candidates.append(s[2])
        f.visititems(visit)
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise ValueError(
            f"Не удалось однозначно определить число входных каналов в {path} "
            f"(кандидаты={unique}); ожидался ровно один stem-conv (3,3,C,{trunk_channels})."
        )
    return unique[0]


class AZNetwork:
    def __init__(self, input_shape: Tuple[int, int, int], action_size: int, config: AZConfig):
        self.config = config
        self.action_size = action_size
        self.model = build_model(input_shape, action_size, config)
        self.optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)

    # ---- инференс ---------------------------------------------------------
    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Одна позиция (rows, cols, C) -> (policy_probs[action_size], value)."""
        logits, value = self.predict_batch(state[None, ...])
        return logits[0], float(value[0])

    def predict_batch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Батч позиций -> (softmax-probs [N, action_size], values [N])."""
        logits, value = self._forward(tf.convert_to_tensor(states, dtype=tf.float32))
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        return probs, value.numpy().reshape(-1)

    @tf.function(reduce_retracing=True)
    def _forward(self, states):
        return self.model(states, training=False)

    # ---- обучение ---------------------------------------------------------
    @tf.function(reduce_retracing=True)
    def _train_step(self, states, target_pi, target_z):
        with tf.GradientTape() as tape:
            logits, value = self.model(states, training=True)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            policy_loss = -tf.reduce_mean(tf.reduce_sum(target_pi * log_probs, axis=-1))
            value_loss = tf.reduce_mean(tf.square(target_z - tf.squeeze(value, axis=-1)))
            reg_loss = tf.add_n(self.model.losses) if self.model.losses else 0.0
            total = policy_loss + self.config.value_loss_coef * value_loss + reg_loss
        grads = tape.gradient(total, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return total, policy_loss, value_loss

    def train_on_batch(self, states, target_pi, target_z) -> Dict[str, float]:
        total, p_loss, v_loss = self._train_step(
            tf.convert_to_tensor(states, dtype=tf.float32),
            tf.convert_to_tensor(target_pi, dtype=tf.float32),
            tf.convert_to_tensor(target_z, dtype=tf.float32),
        )
        return {
            "total_loss": float(total),
            "policy_loss": float(p_loss),
            "value_loss": float(v_loss),
        }

    # ---- сохранение -------------------------------------------------------
    def save(self, path: str) -> None:
        self.model.save_weights(path)

    def load(self, path: str) -> None:
        self.model.load_weights(path)

    def load_partial(self, path: str, *, zero_new_channels: bool = False) -> None:
        """Загрузить веса из чекпоинта с ДРУГИМ числом входных каналов.

        Применимо, когда расходится единственный слой — входной conv (stem):
        его ядро (3, 3, C, K) имеет другое C, а всё глубже работает с K
        каналами и переносится один-в-один. Старое ядро ложится в первые
        C_old срезов нового; оставшиеся срезы (новые плоскости входа)
        сохраняют исходную инициализацию, либо обнуляются при
        ``zero_new_channels=True``. Предполагается, что новые каналы дописаны
        В КОНЕЦ кодировки (см. encode_state) — иначе соответствие срезов
        нарушится молча.
        """
        new_c = int(self.model.input_shape[-1])
        old_c = _read_stem_in_channels(path, trunk_channels=self.config.channels)
        if old_c == new_c:
            self.model.load_weights(path)
            return

        rows, cols = self.model.input_shape[1], self.model.input_shape[2]
        old_model = build_model((rows, cols, old_c), self.action_size, self.config)
        old_model.load_weights(path)  # формы совпадают -> штатная загрузка

        stem_name = "stem_conv"
        for new_layer in self.model.layers:
            if new_layer.name == stem_name or not new_layer.weights:
                continue
            new_layer.set_weights(old_model.get_layer(new_layer.name).get_weights())

        new_stem = self.model.get_layer(stem_name)
        old_stem = old_model.get_layer(stem_name)
        kernel = new_stem.get_weights()[0].copy()      # (3, 3, new_c, K), инициализация
        old_kernel = old_stem.get_weights()[0]         # (3, 3, old_c, K), обученное
        keep = min(old_c, new_c)
        kernel[:, :, :keep, :] = old_kernel[:, :, :keep, :]
        if zero_new_channels and new_c > old_c:
            kernel[:, :, old_c:, :] = 0.0
        new_stem.set_weights([kernel])  # stem без bias -> только ядро
        print(f"[partial] stem-conv: перенесено {keep} вх. каналов, "
              f"{max(0, new_c - old_c)} новых "
              f"({'обнулены' if zero_new_channels else 'случайная инициализация'})")

    def clone_weights_to(self, other: "AZNetwork") -> None:
        other.model.set_weights(self.model.get_weights())
